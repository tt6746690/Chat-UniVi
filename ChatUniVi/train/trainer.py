import os
import torch
from transformers import Trainer
from typing import Optional
import torch.distributed as dist
import wandb

try:
    from rosemary import parse_kv_from_string
except:
    pass


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


## imports for exposing `training_steps`

from transformers.utils import is_sagemaker_mp_enabled, is_apex_available
if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

    from .trainer_pt_utils import smp_forward_backward, smp_forward_only, smp_gather, smp_nested_concat
else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_apex_available():
    from apex import amp

from transformers.modeling_utils import unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


class ChatUniViTrainer(Trainer):
    def _save_checkpoint(self, model, trial, metrics=None):
        if 0 and getattr(self.args, 'tune_mm_mlp_adapter', False):
            from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
            checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

            run_dir = self._get_output_dir(trial=trial)
            output_dir = os.path.join(run_dir, checkpoint_folder)

            # Only save Adapter
            keys_to_match = ['mm_projector', "ctm", "block"]
            if getattr(self.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])

            weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

            if self.args.local_rank == 0 or self.args.local_rank == -1:
                self.model.config.save_pretrained(output_dir)
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        else:
            super(ChatUniViTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if 0 and getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(ChatUniViTrainer, self)._save(output_dir, state_dict)


    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss
    

    def training_step(self, model, inputs):
        """reference: https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/trainer.py#L2628
        """
        model.train()
        inputs = self._prepare_inputs(inputs)

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)
        
        # wpq: to get `ModelOutputs` instead of tuple.
        inputs.update({'return_dict': True})

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)


        ## wpq: log per-expert LM loss
        if outputs.losses is not None: # m3 type training
            if self.is_world_process_zero() and 'wandb' in self.args.report_to:
                log_dict = {}
                losses_lm_reduced = torch.mean(outputs.losses_lm, 0) # (1,) unweighted lm loss
                for k in range(outputs.losses.numel()):
                    log_dict.update({f'moe/loss_lm_{k}': losses_lm_reduced[k].item()})


        ### wpq: MoE load balancing loss.
        if self.model.get_model().is_m3_moe:
            # use `self.model` to access llama model.
            kvs = parse_kv_from_string(self.model.config.config.get('moe', None))

            # compute the number of tokens used for different types of m3 training.
            m3_kvs = parse_kv_from_string(self.model.config.config.get('matryoshka_vis_token_scale', None))
            tokscale_list = eval(m3_kvs.get('numtoks'))
            if m3_kvs['ver'] == 'v0':
                numtoks_list = tokscale_list
            elif m3_kvs['ver'] == 'v1':
                # tokscale_list = [144], eventscale_list = [1, 4]
                # -> numtoks_list = [144, 4*144]
                eventscale_list = eval(m3_kvs.get('numevents', 1))
                numtoks_list = [tokscale_list[0]*x for x in eventscale_list]

            # (micro-bsz, K)
            gating_prob = outputs.gating_prob
            device, dtype = gating_prob.device, gating_prob.dtype
            assert(gating_prob.shape[1] == len(numtoks_list))

            with torch.no_grad():
                # gather `gating_prob`` (micro-bsz, K) -> (B, K) where K is number of experts.
                gating_prob_list = [torch.zeros_like(outputs.gating_prob) for _ in range(self.args.world_size)]
                dist.all_gather(gating_prob_list, outputs.gating_prob)
                batch_gating_prob = torch.cat(gating_prob_list, dim=0)
                B, K = batch_gating_prob.shape
                # (K,)
                batch_per_expert_gating_prob = torch.mean(gating_prob, dim=0)
                # (K,)
                batch_per_expert_assignment = torch.nn.functional.one_hot(batch_gating_prob.argmax(dim=1), num_classes=K)
                batch_per_expert_assignment = batch_per_expert_assignment.sum(dim=0).float() / B
                batch_per_expert_assignment = batch_per_expert_assignment.to(device).to(dtype)

            if self.is_world_process_zero() and 'wandb' in self.args.report_to:
                for k in range(K):
                    log_dict.update({f'moe/avg_gating_prob_{k}': batch_per_expert_gating_prob[k].item()})
                for k in range(K):
                    log_dict.update({f'moe/avg_expert_assignment_{k}': batch_per_expert_assignment[k].item()})


            moe_objective_type = kvs.get('obj', 'weightedlm')
            if moe_objective_type.startswith('bounderr'):
                margin = float(kvs.get('margin', 0))
                # (micro-bsz, K)
                gating_prob_argmax = compute_gating_prob_argmax(gating_prob, kvs)
                # assume token scale sorted, largeest token scale at the end.
                # (micro-bsz, K)
                losses_lm = outputs.losses_lm
                # (micro-bsz)
                losses_argmaxscale = (losses_lm * gating_prob_argmax).sum(1)
                losses_maxtokscale = losses_lm[:, -1]
                losses_diff = losses_argmaxscale - losses_maxtokscale
                if moe_objective_type == 'bounderr':
                    loss = torch.clamp(losses_diff - margin, min=0).mean()
                elif moe_objective_type == 'bounderrsq':
                    loss = torch.square(torch.clamp(losses_diff - margin, min=0)).mean()

                if self.is_world_process_zero() and 'wandb' in self.args.report_to:
                    log_dict.update({
                        'moe_bounderr/loss_argmaxscale_avg': losses_argmaxscale.mean().item(),
                        'moe_bounderr/loss_maxscale_avg': losses_maxtokscale.mean().item(),
                        'moe_bounderr/loss_diff_avg': losses_diff.mean().item(),
                    })
            elif moe_objective_type == 'weightedlm':
                pass

            ## compute switch transformer load balance loss: https://dl.acm.org/doi/pdf/10.5555/3586589.3586709
            if kvs.get('loadb', None) == 'switch':
                alpha = float(kvs['alpha'])
                per_expert_cost_type = kvs.get('costt', 'count')
                per_expert_cost = get_per_expert_cost(per_expert_cost_type, batch_per_expert_assignment, numtoks_list, device, dtype)
                # (K,), (K,) -> (,)
                loss_switch = alpha * K * (per_expert_cost * torch.mean(gating_prob, dim=0)).sum()
                loss += loss_switch
                
                if self.is_world_process_zero() and 'wandb' in self.args.report_to:
                    log_dict.update({'moe_load/loss_switch': loss_switch.item(),})
                    for k in range(K):
                        log_dict.update({f'moe_load/cost_{k}': per_expert_cost[k].item()})
            elif kvs.get('loadb', None) == 'argmaxcost':
                # apply expert specific cost to argmax of `gating_prob`
                alpha = float(kvs['alpha'])
                per_expert_cost_type = kvs.get('costt')
                # since `argmaxcost` normalized to [0,1], therefore, select target value within [0,1] suffices.
                target_value = kvs.get('tval', None)
                numtoks_margin = kvs.get('tmargin', None)
                # (K,)
                per_expert_cost = get_per_expert_cost(per_expert_cost_type, batch_per_expert_assignment, numtoks_list, device, dtype)
                if not moe_objective_type.startswith('bounderr'): # already initialized `gating_prob_argmax`
                    # (micro-bsz, K)
                    gating_prob_argmax = compute_gating_prob_argmax(gating_prob, kvs)
                # (1,) micro-batch cost
                # since cost sums to 1, therefore sum wrt expert dimension
                argmaxcost = (gating_prob_argmax * per_expert_cost.reshape(-1, K)).sum(1).mean()
                # (1,) batch average cost
                # if just use micro-batch cost in loss, too noisy since micro-bsz is quite small (e.g., 4)
                with torch.no_grad():
                    argmaxcost_list = [torch.zeros_like(argmaxcost.unsqueeze(0)) for _ in range(self.args.world_size)]
                    dist.all_gather(argmaxcost_list, argmaxcost.unsqueeze(0))
                    argmaxcost_list = torch.cat(argmaxcost_list, dim=0)
                    batch_argmaxcost = argmaxcost_list.mean()
                if target_value is not None:
                    loss_argmaxcost = alpha * torch.square(batch_argmaxcost - argmaxcost.detach() + argmaxcost - numtoks_margin - target_value)
                else:
                    loss_argmaxcost = alpha * torch.clamp(batch_argmaxcost - argmaxcost.detach() + argmaxcost - numtoks_margin, min=0)
                loss += loss_argmaxcost
                if self.is_world_process_zero() and 'wandb' in self.args.report_to:
                    log_dict.update({'moe_load/loss_argmaxcost': loss_argmaxcost.item(),})
                    for k in range(K):
                        log_dict.update({f'moe_load/cost_{k}': per_expert_cost[k].item()})
            elif kvs.get('loadb', None) == 'betalogprob':
                if K != 2:
                    raise ValueError(f'#tokscale = {K} not supported.')
                alpha = float(kvs['alpha'])
                beta_alpha = float(kvs['ba'])
                beta_beta = float(kvs['bb'])
                beta_dist = torch.distributions.Beta(beta_alpha, beta_beta)
                log_prob = beta_dist.log_prob(gating_prob[:,1])
                loss_beta_logprob = alpha * log_prob.sum()
                loss += loss_beta_logprob
                if self.is_world_process_zero() and 'wandb' in self.args.report_to:
                    log_dict.update({'moe_load/loss_beta_logprob': loss_beta_logprob.item(),})

         # log once/batch if assume no gradient accumulation.
        if self.is_world_process_zero() and 'wandb' in self.args.report_to:
            wandb.log(log_dict)

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps



def compute_gating_prob_argmax(gating_prob, kvs):
    # if hard=True, taking argmax and therefore `tau` does not really matter.
    tau = float(kvs.get('tau', 1))
    hard = bool(kvs.get('hard', True))
    # (micro-bsz, K)
    gating_prob_argmax = torch.nn.functional.gumbel_softmax(gating_prob, tau=tau, hard=hard, dim=1)
    return gating_prob_argmax


def get_per_expert_cost(per_expert_cost_type, batch_per_expert_assignment, numtoks_list, device, dtype):
    if per_expert_cost_type == 'count': # default used in switch transformers
        per_expert_cost = batch_per_expert_assignment
    elif per_expert_cost_type == 'numtoks':
        per_expert_cost = torch.tensor(numtoks_list, device=device, dtype=dtype)
        per_expert_cost = per_expert_cost / per_expert_cost.sum()
    elif per_expert_cost_type == 'lognumtoks':
        per_expert_cost = torch.tensor(numtoks_list, device=device, dtype=dtype)
        per_expert_cost = torch.log(per_expert_cost+1) # add 1 to prevent cost(tokscale=1)=0
        per_expert_cost = per_expert_cost / per_expert_cost.sum()
    elif per_expert_cost_type == 'count*numtoks':
        per_expert_cost = batch_per_expert_assignment
        per_expert_cost_2 = torch.tensor(numtoks_list, device=device, dtype=dtype)
        per_expert_cost_2 = per_expert_cost_2 / per_expert_cost_2.sum()
        per_expert_cost *= per_expert_cost_2
        per_expert_cost = per_expert_cost / per_expert_cost.sum()
    elif per_expert_cost_type == 'count*lognumtoks':
        per_expert_cost = batch_per_expert_assignment
        per_expert_cost_2 = torch.tensor(numtoks_list, device=device, dtype=dtype)
        per_expert_cost_2 = torch.log(per_expert_cost_2+1) # add 1 to prevent cost(tokscale=1)=0
        per_expert_cost_2 = per_expert_cost_2 / per_expert_cost_2.sum()
        per_expert_cost *= per_expert_cost_2
        per_expert_cost = per_expert_cost / per_expert_cost.sum()
    else:
        raise ValueError(f'per_expert_cost_type={per_expert_cost_type} not supported.')
    return per_expert_cost
