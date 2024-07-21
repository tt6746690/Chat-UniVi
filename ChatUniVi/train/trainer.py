import os
import torch
from transformers import Trainer
from typing import Optional
import torch.distributed as dist

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
        

        with self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            
        # use `self.model` to access llama model.
        if hasattr(self.model.config, 'config') and self.model.config.config.get('moe', None):
            kvs = parse_kv_from_string(self.model.config.config.get('moe', None))
            ## compute switch transformer load balance loss: https://dl.acm.org/doi/pdf/10.5555/3586589.3586709
            if kvs['loadb'] == 'switch':
                alpha = float(kvs['alpha'])
                # gating_prob: (micro-bsz, K) where K is number of experts.
                gating_prob_idx = 3
                assert(outputs[gating_prob_idx].shape[1] == 5)
                if self.is_local_process_zero():
                    gating_prob_list = [torch.zeros_like(outputs[gating_prob_idx]) for _ in range(self.args.world_size)]
                    dist.gather(outputs[gating_prob_idx], gating_prob_list, dst=0)
                    # (B, K)
                    gating_prob = torch.cat(gating_prob_list, dim=0)
                    B, K = gating_prob.shape
                    per_expert_assignment = torch.nn.functional.one_hot(gating_prob.argmax(dim=1), num_classes=K)
                    per_expert_assignment = per_expert_assignment.sum(dim=0).float() / B
                    per_expert_gating_prob = torch.mean(gating_prob, dim=0)
                    # (K,), (K,) -> (,)
                    loss_switch = alpha * K * (per_expert_assignment * per_expert_gating_prob).sum()
                    # scale by world_size since DDP average gradients and `loss_switch` computed on rank=1 process only.
                    loss_switch = self.args.world_size * loss_switch
                    loss += loss_switch
                else:
                    dist.gather(outputs[gating_prob_idx], dst=0)
                

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