import re
from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch
import torch.nn.functional as F
from litellm import ChatCompletionMessageToolCall
from litellm.types.utils import Function
from torch.func import jacrev, vmap
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
)
from transformers.generation.utils import (
    BeamSearchScorer,
    ConstrainedBeamSearchScorer,
    DisjunctiveConstraint,
    GenerateOutput,
    GenerationConfig,
    GenerationMode,
    LogitsProcessorList,
    PhrasalConstraint,
    StoppingCriteriaList,
    dist,
    inspect,
    is_deepspeed_zero3_enabled,
    is_fsdp_managed_module,
    logger,
    warnings,
)
from transformers.models.llama.modeling_llama import (
    BaseModelOutputWithPast,
    Cache,
    CausalLMOutputWithPast,
    KwargsForCausalLM,
    Unpack,
    auto_docstring,
    can_return_tuple,
)

if TYPE_CHECKING:
    from transformers.generation.streamers import BaseStreamer
    from transformers.modeling_utils import PreTrainedModel

from .base_local_model import BaseLocalModel
from .model_config import ModelConfig
from .utils import FINEGRAINED, PRODUCT_PROBS, SUM_SQUARES, parse_json_string


class LlamaForCausalLM_GBC(LlamaForCausalLM):
    def __init__(self, config: LlamaConfig, gradient_strategy: int = SUM_SQUARES):
        super().__init__(config)
        self.initial_inputs_embeds = None
        self.gradient_strategy = gradient_strategy

    def set_initial_input_embed(self, inputs_embeds: List[torch.FloatTensor]):
        """
        Set the initial inputs_embeds for the model.
        """
        self.initial_inputs_embeds = inputs_embeds

    def reset_initial_input_embed(self):
        self.initial_inputs_embeds = None

    def set_gradient_strategy(self, strategy: int):
        """
        Set the gradient strategy for the model.
        """
        self.gradient_strategy = strategy

    def calculate_gradient(
        self,
        inputs_embeds: List[torch.FloatTensor],
        logits: torch.FloatTensor,
        log_probs: torch.FloatTensor,
    ):
        """
        Calculate the gradient of the logits with respect to the inputs_embeds.
        """

        # Calculate the gradient of the logits with respect to the inputs_embeds
        gradients = list()
        for batch_index in range(logits.shape[0]):
            if self.gradient_strategy == FINEGRAINED:
                # Compute the gradient of the logits with respect to the inputs_embeds
                batch_grads = list()
                for token_index in range(logits.shape[1]):
                    grads = torch.autograd.grad(
                        outputs=logits[batch_index][token_index],
                        inputs=inputs_embeds[batch_index],
                        retain_graph=True,
                    )[0]
                    batch_grads.append(grads)
                # Concatenate the gradients for all tokens in the batch
                batch_grads = torch.stack(batch_grads, dim=0)  # (output_sequence_length, input_sequence_length, hidden_size)
                gradients.append(batch_grads)
            elif self.gradient_strategy == SUM_SQUARES:
                logits_sum_squares = logits[batch_index].pow(2).sum()
                # Compute the gradient of the sum of squares of logits with respect to the inputs_embeds
                grads = torch.autograd.grad(
                    outputs=logits_sum_squares,
                    inputs=inputs_embeds[batch_index],
                    retain_graph=True,
                )[0]  # (input_sequence_length, hidden_size)
                gradients.append(grads)
            elif self.gradient_strategy == PRODUCT_PROBS:
                probs_product = log_probs[batch_index].sum().exp()  # Product of probabilities
                # Compute the gradient of the product of probabilities with respect to the inputs_embeds
                grads = torch.autograd.grad(
                    outputs=probs_product,
                    inputs=inputs_embeds[batch_index],
                    retain_graph=True,
                )[0] # (input_sequence_length, hidden_size)
                gradients.append(grads)
            else:
                raise NotImplementedError(f"Gradient strategy {self.gradient_strategy} is not implemented.")
        # Concatenate the gradients for all batches
        gradients = torch.stack(gradients, dim=0)
        # If FINEGRAINED: (batch_size, output_sequence_length, input_sequence_length, hidden_size)
        # If SUM_SQUARES: (batch_size, input_sequence_length, hidden_size)
        # If PRODUCT_PROBS: (batch_size, input_sequence_length, hidden_size)
        return gradients

    @can_return_tuple
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # Record the initial inputs_embeds
        inputs_embeds = inputs_embeds if inputs_embeds is not None else self.model.embed_tokens(input_ids)
        if self.initial_inputs_embeds is None:
            initial_inputs_embeds = list()
            for i in range(inputs_embeds.shape[0]):
                input_embed = inputs_embeds[i, :, :].clone().detach()
                input_embed.requires_grad_(True)
                initial_inputs_embeds.append(input_embed)
            self.set_initial_input_embed(initial_inputs_embeds)
            inputs_embeds = torch.stack(self.initial_inputs_embeds, dim=0)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        custom_generate: Optional[str] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # 0. If requested, load an arbitrary generation recipe from the Hub and run it instead
        if custom_generate is not None:
            trust_remote_code = kwargs.pop("trust_remote_code", None)
            # Get all `generate` arguments in a single variable. Custom functions are responsible for handling them:
            # they receive the same inputs as `generate`, only with `model` instead of `self`. They can access to
            # methods from `GenerationMixin` through `model`.
            global_keys_to_exclude = {"self", "kwargs"}
            generate_arguments = {key: value for key, value in locals().items() if key not in global_keys_to_exclude}
            generate_arguments.update(kwargs)

            custom_generate_function = self.load_custom_generate(
                custom_generate, trust_remote_code=trust_remote_code, **kwargs
            )
            return custom_generate_function(model=self, **generate_arguments)

        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        assistant_tokenizer = kwargs.pop("assistant_tokenizer", None)  # only used for assisted generation

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        generation_config.return_dict_in_generate = True  # Always return dict in generate
        generation_config.output_logits = True  # Always return logits in generate
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_assistant(assistant_model, tokenizer, assistant_tokenizer)

        # 2. Set generation parameters if not already defined
        if synced_gpus is None:
            synced_gpus = (is_deepspeed_zero3_enabled() or is_fsdp_managed_module(self)) and dist.get_world_size() > 1

        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # decoder-only models must use left-padding for batched generation.
        if not self.config.is_encoder_decoder:
            # If `input_ids` was given, check if the last id in any sequence is `pad_token_id`
            # Note: If using, `inputs_embeds` this check does not work, because we want to be more hands-off.
            if (
                generation_config._pad_token_tensor is not None
                and batch_size > 1
                and len(inputs_tensor.shape) == 2
                and torch.sum(inputs_tensor[:, -1] == generation_config._pad_token_tensor) > 0
            ):
                logger.warning(
                    "A decoder-only architecture is being used, but right-padding was detected! For correct "
                    "generation results, please set `padding_side='left'` when initializing the tokenizer."
                )

        # 4. Define other model kwargs
        # decoder-only models with inputs_embeds forwarding must use caching (otherwise we can't detect whether we are
        # generating the first new token or not, and we only want to use the embeddings for the first new token)
        if not self.config.is_encoder_decoder and model_input_name == "inputs_embeds":
            generation_config.use_cache = True

        if not kwargs_has_attention_mask and requires_attention_mask and accepts_attention_mask:
            model_kwargs["attention_mask"] = self._prepare_attention_mask_for_generation(
                inputs_tensor, generation_config, model_kwargs
            )
        elif kwargs_has_attention_mask:
            # TODO (joao): generalize this check with other types of inputs
            if model_input_name == "input_ids" and len(model_kwargs["attention_mask"].shape) > 2:
                raise ValueError("`attention_mask` passed to `generate` must be 2D.")

        if self.config.is_encoder_decoder and "encoder_outputs" not in model_kwargs:
            # if model is encoder decoder encoder_outputs are created and added to `model_kwargs`
            model_kwargs = self._prepare_encoder_decoder_kwargs_for_generation(
                inputs_tensor, model_kwargs, model_input_name, generation_config
            )

        # 5. Prepare `input_ids` which will be used for auto-regressive generation
        if self.config.is_encoder_decoder:
            input_ids, model_kwargs = self._prepare_decoder_input_ids_for_generation(
                batch_size=batch_size,
                model_input_name=model_input_name,
                model_kwargs=model_kwargs,
                decoder_start_token_id=generation_config._decoder_start_token_tensor,
                device=inputs_tensor.device,
            )
        else:
            input_ids = inputs_tensor if model_input_name == "input_ids" else model_kwargs.pop("input_ids")

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, tokenizer)

        if streamer is not None:
            streamer.put(input_ids.cpu())

        # 6. Prepare `max_length` depending on other stopping criteria.
        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        # If the model supports `logits_to_keep` in forward(), set it to 1 to avoid computing the whole
        # logit matrix. This can save a lot of memory during the first forward pass. Note that assisted decoding
        # dynamically overrides this value as it can need more than the last token logits
        if self._supports_logits_to_keep() and "logits_to_keep" not in model_kwargs:
            model_kwargs["logits_to_keep"] = 1

        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)

        # 7. Prepare the cache.
        # - `model_kwargs` may be updated in place with a cache as defined by the parameters in `generation_config`.
        # - different models have a different cache name expected by the model (default = "past_key_values")
        # - `max_length`, prepared above, is used to determine the maximum cache length
        max_cache_length = generation_config.max_length - 1
        if (
            inputs_tensor.shape[1] != input_ids_length
            and model_input_name == "inputs_embeds"
            and not self.config.is_encoder_decoder
        ):
            max_cache_length += inputs_tensor.shape[1]
        self._prepare_cache_for_generation(
            generation_config, model_kwargs, assistant_model, batch_size, max_cache_length, device
        )

        # 8. determine generation mode
        generation_mode = generation_config.get_generation_mode(assistant_model)

        if streamer is not None and (generation_config.num_beams > 1):
            raise ValueError(
                "`streamer` cannot be used with beam search (yet!). Make sure that `num_beams` is set to 1."
            )

        if self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation."
                " Please make sure that you have put `input_ids` to the"
                f" correct device by calling for example input_ids = input_ids.to('{self.device.type}') before"
                " running `.generate()`.",
                UserWarning,
            )

        # 9. prepare logits processors and stopping criteria
        prepared_logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=inputs_tensor,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            logits_processor=logits_processor,
            device=inputs_tensor.device,
            model_kwargs=model_kwargs,
            negative_prompt_ids=negative_prompt_ids,
            negative_prompt_attention_mask=negative_prompt_attention_mask,
        )
        prepared_stopping_criteria = self._get_stopping_criteria(
            generation_config=generation_config, stopping_criteria=stopping_criteria, tokenizer=tokenizer, **kwargs
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache

        # 10. go into different generation modes
        if generation_mode == GenerationMode.ASSISTED_GENERATION:
            if generation_config.num_return_sequences > 1:
                raise ValueError(
                    "num_return_sequences has to be 1 when doing assisted generate, "
                    f"but is {generation_config.num_return_sequences}."
                )
            if batch_size > 1:
                raise ValueError("assisted generate is only supported for batch_size = 1")
            if not model_kwargs["use_cache"]:
                raise ValueError("assisted generate requires `use_cache=True`")
            if generation_config.cache_implementation in ["static", "hybrid", "sliding_window"]:
                raise ValueError("assisted generate is not supported with Static cache classes`")
            if self._is_stateful:
                # In assisted generation we need the ability to confirm whether the model would pick certain tokens,
                # which is not possible with stateful models (they can't reset to a previous subset of generated text)
                raise ValueError(
                    f"assisted generation is not supported with stateful models, such as {self.__class__.__name__}"
                )

            # 11. Get the candidate generator, given the parameterization
            candidate_generator = self._get_candidate_generator(
                generation_config=generation_config,
                input_ids=input_ids,
                inputs_tensor=inputs_tensor,
                assistant_model=assistant_model,
                logits_processor=logits_processor,
                target_tokenizer=tokenizer,
                assistant_tokenizer=assistant_tokenizer,
                model_kwargs=model_kwargs,
            )

            # 12. run assisted generate
            result = self._assisted_decoding(
                input_ids,
                candidate_generator=candidate_generator,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )
        elif generation_mode == GenerationMode.DOLA_GENERATION:
            if self._is_stateful:
                # DoLa decoding was not designed for stateful models, and would require some changes
                raise ValueError(
                    f"dola decoding is not supported with stateful models, such as {self.__class__.__name__}"
                )
            result = self._dola_decoding(
                input_ids,
                dola_layers=generation_config.dola_layers,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONTRASTIVE_SEARCH:
            if not model_kwargs["use_cache"]:
                raise ValueError("Contrastive search requires `use_cache=True`")
            if self._is_stateful:
                # Just like assisted generation, we need to be able to rollback to a previous state (see comment above)
                raise ValueError(
                    f"contrastive search is not supported with stateful models, such as {self.__class__.__name__}"
                )

            result = self._contrastive_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.SAMPLE, GenerationMode.GREEDY_SEARCH):
            # 11. expand input_ids with `num_return_sequences` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_return_sequences,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )

            # 12. run sample (it degenerates to greedy search when `generation_config.do_sample=False`)
            result = self._sample(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                streamer=streamer,
                **model_kwargs,
            )

        elif generation_mode in (GenerationMode.BEAM_SAMPLE, GenerationMode.BEAM_SEARCH):
            # 11. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 12. run beam sample
            result = self._beam_search(
                input_ids,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.GROUP_BEAM_SEARCH:
            # 11. prepare beam search scorer
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                num_beam_groups=generation_config.num_beam_groups,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._group_beam_search(
                input_ids,
                beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        elif generation_mode == GenerationMode.CONSTRAINED_BEAM_SEARCH:
            final_constraints = []
            if generation_config.constraints is not None:
                final_constraints = generation_config.constraints

            if generation_config.force_words_ids is not None:

                def typeerror():
                    raise ValueError(
                        "`force_words_ids` has to either be a `List[List[List[int]]]` or `List[List[int]]` "
                        f"of positive integers, but is {generation_config.force_words_ids}."
                    )

                if (
                    not isinstance(generation_config.force_words_ids, list)
                    or len(generation_config.force_words_ids) == 0
                ):
                    typeerror()

                for word_ids in generation_config.force_words_ids:
                    if isinstance(word_ids[0], list):
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any(not isinstance(token_ids, list) for token_ids in word_ids):
                            typeerror()
                        if any(
                            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
                            for token_ids in word_ids
                        ):
                            typeerror()

                        constraint = DisjunctiveConstraint(word_ids)
                    else:
                        if not isinstance(word_ids, list) or len(word_ids) == 0:
                            typeerror()
                        if any((not isinstance(token_id, int) or token_id < 0) for token_id in word_ids):
                            typeerror()

                        constraint = PhrasalConstraint(word_ids)
                    final_constraints.append(constraint)

            # 11. prepare beam search scorer
            constrained_beam_scorer = ConstrainedBeamSearchScorer(
                constraints=final_constraints,
                batch_size=batch_size,
                num_beams=generation_config.num_beams,
                device=inputs_tensor.device,
                length_penalty=generation_config.length_penalty,
                do_early_stopping=generation_config.early_stopping,
                num_beam_hyps_to_keep=generation_config.num_return_sequences,
                max_length=generation_config.max_length,
            )
            # 12. interleave input_ids with `num_beams` additional sequences per batch
            input_ids, model_kwargs = self._expand_inputs_for_generation(
                input_ids=input_ids,
                expand_size=generation_config.num_beams,
                is_encoder_decoder=self.config.is_encoder_decoder,
                **model_kwargs,
            )
            # 13. run beam search
            result = self._constrained_beam_search(
                input_ids,
                constrained_beam_scorer=constrained_beam_scorer,
                logits_processor=prepared_logits_processor,
                stopping_criteria=prepared_stopping_criteria,
                generation_config=generation_config,
                synced_gpus=synced_gpus,
                **model_kwargs,
            )

        # Convert to legacy cache format if requested
        if (
            generation_config.return_legacy_cache is True
            and hasattr(result, "past_key_values")
            and getattr(result.past_key_values, "to_legacy_cache") is not None
        ):
            result.past_key_values = result.past_key_values.to_legacy_cache()

        # Compute the gradient of the logits with respect to the inputs_embeds
        input_ids = result.sequences  # (batch_size, total_sequence_length)
        logits = torch.stack(result.logits, dim=1)  # (batch_size, new_sequence_length, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, new_sequence_length, vocab_size)
        picked_input_ids = input_ids[:, -logits.shape[1]:]  # (batch_size, new_sequence_length)
        picked_logits = logits.gather(2, picked_input_ids.unsqueeze(-1)).squeeze(-1)  # (batch_size, new_sequence_length)
        picked_log_probs = log_probs.gather(2, picked_input_ids.unsqueeze(-1)).squeeze(-1)  # (batch_size, new_sequence_length)
        gradients = self.calculate_gradient(self.initial_inputs_embeds, picked_logits, picked_log_probs)
        result.gradients = gradients
        # If FINEGRAINED: (batch_size, output_sequence_length, input_sequence_length, hidden_size)
        # If SUM_SQUARES: (batch_size, input_sequence_length, hidden_size)
        # If PRODUCT_PROBS: (batch_size, input_sequence_length, hidden_size)
        result.embeds = torch.stack(self.initial_inputs_embeds, dim=0)  # (batch_size, input_sequence_length, hidden_size)

        # Reset initial input embedding to None
        self.reset_initial_input_embed()
        
        return result

class LlamaModel(BaseLocalModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.model = LlamaForCausalLM_GBC.from_pretrained(
            config.model_path,
            device_map="auto",
            torch_dtype="auto",
            quantization_config=self.quantization_config,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        with open(self.config.chat_template_path, "r", encoding="utf-8") as f:
            chat_template = f.read()
        self.tokenizer.chat_template = chat_template
        self.model.eval()
        self.model.set_gradient_strategy(self.gradient_strategy)

    def _get_tool_calls(self, output: str) -> Optional[List[ChatCompletionMessageToolCall]]:
        """
        Extract tool calls from the output string.
        """
        processed_output = parse_json_string(output)
        if isinstance(processed_output, list):
            tool_calls = list()
            for call in processed_output:
                if isinstance(call, dict) and "name" in call and ("arguments" in call or "parameters" in call):
                    # Create a tool call from the dictionary
                    tool_calls.append(ChatCompletionMessageToolCall(
                        function=Function(
                            name=call.get("name", None),
                            arguments=call.get("arguments", dict()) or call.get("parameters", dict())
                        )
                    ))
            if tool_calls:
                return tool_calls
            else:
                return None
        elif isinstance(processed_output, dict):
            if "name" in processed_output and ("arguments" in processed_output or "parameters" in processed_output):
                # Create a single tool call from the dictionary
                return [ChatCompletionMessageToolCall(
                    function=Function(
                        name=processed_output.get("name", None),
                        arguments=processed_output.get("arguments", dict()) or processed_output.get("parameters", dict())
                    )
                )]
            else:
                return None
        else:
            return None