from typing import List, Optional, Tuple, Union
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from ChatUniVi.model.arch import MetaModel, ChatUniViMetaForCausalLM


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
    ) -> Union[Tuple, CausalLMOutputWithPast]:

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
            from rosemary import parse_kv_from_string
            kvs = parse_kv_from_string(matryoshka_vis_token_scale)
            if kvs['ver'] == 'v0':
                if kvs['numtoks'] == 'gateprobargmax':
                    # reached only during inference, and `gating_prob` not useful after this point
                    # since it only participates in weighting the loss.
                    gating_prob = None 
                else:
                    token_scales = eval(parse_kv_from_string(self.model.config.config['matryoshka_vis_token_scale'])['numtoks'])
                    k = token_scales.index(kvs['numtoks'])
                    # (B, K) -> (B,)
                    gating_prob = gating_prob[:, k]
        ####
            
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            # (B, seq_len-1, vocab_size)
            shift_logits = logits[..., :-1, :].contiguous()
            # (B, seq_len-1)
            shift_labels = labels[..., 1:].contiguous()

            if gating_prob is None:
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = loss_fct(shift_logits, shift_labels)
            else:
                # Flatten the tokens
                loss_fct_noreduce = CrossEntropyLoss(reduction='none')
                L = logits.shape[1]-1
                # Enable model/pipeline parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                gating_prob = gating_prob.to(shift_logits.device)

                losses = loss_fct_noreduce(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
                # (B, seq_len-1)
                losses = losses.view(-1, L)
                valid_mask = (shift_labels != -100)
                if valid_mask.any():
                    loss = (losses * valid_mask).sum(1) / valid_mask.sum(1)
                else:
                    loss = torch.tensor(0.0, dtype=shift_logits.dtype, device=shift_logits.device)
                # (B,)
                loss = loss * gating_prob.reshape_as(loss)
                # (1,)
                loss = loss.mean()

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        ), gating_prob


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
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        

        if self.training and hasattr(self.config, "config") and self.config.config['matryoshka_vis_token_scale'] is not None:
            from rosemary import parse_kv_from_string, create_string_from_kv
            matryoshka_vis_token_scale = self.config.config['matryoshka_vis_token_scale']
            kvs = parse_kv_from_string(matryoshka_vis_token_scale)
            if kvs['ver'] == 'v0':
                num_toks = eval(kvs['numtoks']) # str -> List
                matryoshka_vis_token_scale = []
                for num_tok in num_toks:
                    kvs['numtoks'] = str(num_tok)
                    matryoshka_vis_token_scale.append( create_string_from_kv(kvs) )
            else:
                raise ValueError(f"[ChatUniVi.model.language_model.llama.py] {kvs['ver']} not implemented.")

            # print("The model is in training mode.")
            loss = 0
            logits_accumulate = []
            for matryoshka_vis_token_scale_element in matryoshka_vis_token_scale:
                outputs, gating_prob = self.forward_single_matryoshka(
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
                logits = outputs.logits
                if gating_prob is None:
                    loss += loss_item/len(matryoshka_vis_token_scale)
                else:
                    loss += loss_item
                logits_accumulate.append(logits)
                # wpq: not sure why this is needed:
                # assert len(outputs) == 1, 'len(outputs) == 1 is False'
            logits = torch.cat(logits_accumulate, dim = 1)
                    
            # wpq: only logits & loss is the avg of the different scales.
            #      `past_key_values`, `hidden_states`, `attentions` are from last scale only.
            #      this is ok since this conditional block is used in training only that just needs `loss`.
            #                 1 scale              multiple scales
            # loss:   (1,)                     ->  (1,)             
            # logits: (B, seq_len, vocab_size) ->  (B, seq_len_1+...+seq_len_K, vocab_size) where K=#token scales
            if not return_dict:
                output = (logits,) + outputs[1:]
                return (loss,) + output if loss is not None else output
            return CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
            
        else:
            # print("The model is in evaluation mode or trained without matryoshka_vis_token_scale.")
            outputs, gating_prob = self.forward_single_matryoshka(
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
            return outputs


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
