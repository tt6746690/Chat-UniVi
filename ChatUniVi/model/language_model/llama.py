from typing import List, Optional, Tuple, Union
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ChatUniVi.model.arch import MetaModel, ChatUniViMetaForCausalLM
try:
    from rosemary import parse_kv_from_string, create_string_from_kv
except:
    pass



def lm_loss(logits, labels, lm_loss_type='micro'):
    """Compute LM loss.
        default huggingface's implementation uses `micro` average.
    """
    vocab_size = logits.shape[-1]
    # typical LM loss
    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        if lm_loss_type == 'micro':
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        elif lm_loss_type == 'macro':
            loss_fct_noreduce = CrossEntropyLoss(reduction='none')
            shift_labels = shift_labels.to(shift_logits.device)
            losses = loss_fct_noreduce(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            losses = losses.view(-1, shift_labels.shape[-1])
            valid_mask = (shift_labels != -100)
            # (B,)
            loss = (losses * valid_mask).sum(1) / (valid_mask.sum(1) + 1e-8)
            # (1,)
            loss = loss.mean()
        else:
            raise ValueError(f'invalid lm_loss_type = {lm_loss_type}')

    return loss


def lm_loss_weighted(logits, labels, sample_weights, lm_loss_type='micro'):
    """Compute LM loss weighted by `sample_weights`.
        `sample_weights`    (B,)
    """
    vocab_size = logits.shape[-1]
    # LM loss weighted by gating prob
    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        # (B, seq_len-1, vocab_size)
        shift_logits = logits[..., :-1, :].contiguous()
        # (B, seq_len-1)
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct_noreduce = CrossEntropyLoss(reduction='none')
        # Enable model/pipeline parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        sample_weights = sample_weights.to(shift_logits.device)
        losses = loss_fct_noreduce(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        # (B, seq_len-1)
        losses = losses.view(-1, logits.shape[1]-1)
        valid_mask = (shift_labels != -100)
        
        if lm_loss_type == 'micro':
            loss = (losses * valid_mask).sum(1)
            # (B,)
            loss = loss * sample_weights.reshape_as(loss)
            # (1,)
            loss = loss.sum() / (valid_mask.sum() + 1e-8)
        elif lm_loss_type == 'macro':
            loss = (losses * valid_mask).sum(1) / (valid_mask.sum(1) + 1e-8)
            # (B,)
            loss = loss * sample_weights.reshape_as(loss)
            # (1,)
            loss = loss.mean()
        else:
            raise ValueError(f'invalid lm_loss_type = {lm_loss_type}')
    return loss


def lm_loss_unreduced(logits, labels, lm_loss_type='micro'):
    """Compute LM loss in unreduced form such that 
        when taking mean, equal in value to reduced loss.
    """
    vocab_size = logits.shape[-1]
    # typical LM loss
    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct_noreduce = CrossEntropyLoss(reduction='none')
        shift_labels = shift_labels.to(shift_logits.device)
        losses = loss_fct_noreduce(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
        losses = losses.view(-1, shift_labels.shape[-1])
        valid_mask = (shift_labels != -100)
        if lm_loss_type == 'micro':
            loss = (losses * valid_mask).sum(1)
            loss = loss * loss.shape[0] / (valid_mask.sum() + 1e-8)
        elif lm_loss_type == 'macro':
            loss = (losses * valid_mask).sum(1) / (valid_mask.sum(1) + 1e-8)
        else:
            raise ValueError(f'invalid lm_loss_type = {lm_loss_type}')

    return loss


@dataclass
class CausalLMOutputWithPastWithGatingProb(CausalLMOutputWithPast):
    losses: Optional[torch.FloatTensor] = None
    losses_lm: Optional[torch.FloatTensor] = None
    gating_prob: Optional[torch.FloatTensor] = None



class ChatUniViConfig(LlamaConfig):
    model_type = "ChatUniVi"


class ChatUniViLlamaModel(MetaModel, LlamaModel):
    config_class = ChatUniViConfig

    def __init__(self, config: LlamaConfig):
        super(ChatUniViLlamaModel, self).__init__(config)


class ChatUniViLlamaForCausalLM(LlamaForCausalLM, ChatUniViMetaForCausalLM):
    config_class = ChatUniViConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = ChatUniViLlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward_single_matryoshka(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        matryoshka_vis_token_scale: Optional[str] = None,
    ) -> Union[Tuple, CausalLMOutputWithPastWithGatingProb]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels, position_ids, gating_prob = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images, matryoshka_vis_token_scale=matryoshka_vis_token_scale)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)

        #### wpq: 
        if gating_prob is not None:
            kvs = parse_kv_from_string(matryoshka_vis_token_scale)
            if kvs['ver'] == 'v0':
                if kvs['numtoks'] == 'gateprobargmax':
                    # reached only during inference, and `gating_prob` not useful after this point
                    # since it only participates in weighting the loss.
                    gating_prob_k = None
                else:
                    # reached only during training
                    tokscale_list = self.get_model().tokscale_list
                    k = tokscale_list.index(kvs['numtoks'])
                    # (B, K) -> (B,)
                    gating_prob_k = gating_prob[:, k]
            elif kvs['ver'] == 'v1':
                if kvs['numevents'] == 'gateprobargmax':
                    gating_prob_k = None
                else:
                    # reached only during training
                    eventscale_list = self.get_model().eventscale_list
                    k = eventscale_list.index(kvs['numevents'])
                    # (B, K) -> (B,)
                    gating_prob_k = gating_prob[:, k]
        else:
            gating_prob_k = None
        ####

        lm_loss_type = self.get_model().config.config.get('lm_loss_type', 'micro')

        loss_lm = lm_loss_unreduced(logits, labels, lm_loss_type)
        if gating_prob_k is not None:
            loss = lm_loss_weighted(logits, labels, gating_prob_k, lm_loss_type)
        else:
            loss = lm_loss(logits, labels, lm_loss_type)


        if not return_dict:
            output = (logits,) + outputs[1:] + (None, loss_lm, gating_prob,)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPastWithGatingProb(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            losses=None,
            losses_lm=loss_lm,
            gating_prob=gating_prob,
        )


    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        matryoshka_vis_token_scale: Optional[str] = None,
    ) -> Union[Tuple, CausalLMOutputWithPastWithGatingProb]:


        if self.training and self.get_model().is_m3:
            matryoshka_vis_token_scale = self.config.config['matryoshka_vis_token_scale']
            kvs = parse_kv_from_string(matryoshka_vis_token_scale)

            if kvs['ver'] == 'v0':
                # 'ver=v0_numtoks=[144,576]' -> ['ver=v0_numtoks=144', 'ver=v0_numtoks=576']
                num_toks = eval(kvs['numtoks']) # str -> List
                matryoshka_vis_token_scale = []
                for num_tok in num_toks:
                    kvs['numtoks'] = str(num_tok)
                    matryoshka_vis_token_scale.append( create_string_from_kv(kvs) )
            elif kvs['ver'] == 'v1':
                # 'ver=v1_numtoks=[144]_numevents=[1,4]' -> ['ver=v1_numtoks=144_numevents=1', 'ver=v1_numtoks=144_numevents=4']
                num_toks = eval(kvs['numtoks'])
                assert(len(num_toks) == 1)
                num_events = eval(kvs['numevents'])
                matryoshka_vis_token_scale = []
                for num_event in num_events:
                    kvs['numevents'] = str(num_event)
                    kvs['numtoks'] = str(num_toks[0])
                    matryoshka_vis_token_scale.append( create_string_from_kv(kvs) )
            else:
                raise ValueError(f"[ChatUniVi.model.language_model.llama.py] {kvs['ver']} not implemented.")

            # print("The model is in training mode.")
            loss = 0
            losses_accumulate = [] # can be weighted or not.
            losses_lm_accumulate = [] # unweighted lm loss
            logits_accumulate = []
            for matryoshka_vis_token_scale_element in matryoshka_vis_token_scale:
                outputs = self.forward_single_matryoshka(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    past_key_values = past_key_values,
                    inputs_embeds = inputs_embeds,
                    labels = labels,
                    use_cache = use_cache,
                    output_attentions = output_attentions,
                    output_hidden_states = output_hidden_states,
                    images = images,
                    return_dict = return_dict,
                    matryoshka_vis_token_scale= matryoshka_vis_token_scale_element
                )
                loss_item = outputs.loss
                loss_lm_term = outputs.losses_lm
                logits = outputs.logits
                if outputs.gating_prob is None:
                    loss_item = loss_item/len(matryoshka_vis_token_scale)
                    loss_lm_term = loss_lm_term/len(matryoshka_vis_token_scale)
                losses_accumulate.append(loss_item)
                losses_lm_accumulate.append(loss_lm_term)
                logits_accumulate.append(logits)
            logits = torch.cat(logits_accumulate, dim = 1)
            losses = torch.stack(losses_accumulate)
            losses_lm = torch.stack(losses_lm_accumulate).T # (B, K)
            loss = losses.sum()
            # [(B,), ...] -> (B, K)
            # gating_prob = torch.stack(gating_prob_accumulate).T
            # wpq: only logits & loss is the avg of the different scales.
            #      `past_key_values`, `hidden_states`, `attentions` are from last scale only.
            #      this is ok since this conditional block is used in training only that just needs `loss`.
            #                 1 scale              multiple scales
            # loss:   (1,)                     ->  (1,)             
            # logits: (B, seq_len, vocab_size) ->  (B, seq_len_1+...+seq_len_K, vocab_size) where K=#token scales
            if not return_dict:
                output = (logits,) + outputs[1:] + (losses, losses_lm, outputs.gating_prob)
                return (loss,) + output if loss is not None else output
            return CausalLMOutputWithPastWithGatingProb(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
                losses=losses,
                losses_lm=losses_lm,
                gating_prob=outputs.gating_prob,
            )
            
        else:
            # print("The model is in evaluation mode or trained without matryoshka_vis_token_scale.")
            return self.forward_single_matryoshka(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                images=images,
                return_dict=return_dict,
                matryoshka_vis_token_scale=matryoshka_vis_token_scale,
            )


    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
                "matryoshka_vis_token_scale": kwargs.get("matryoshka_vis_token_scale", None),
            }
        )

        return model_inputs

AutoConfig.register("ChatUniVi", ChatUniViConfig)
AutoModelForCausalLM.register(ChatUniViConfig, ChatUniViLlamaForCausalLM)
