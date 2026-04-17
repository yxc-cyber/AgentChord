from typing import TYPE_CHECKING, Callable, List, Optional, Union

import torch
import torch.nn.functional as F
from litellm import ChatCompletionMessageToolCall
from litellm.types.utils import Function
from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM
from transformers.generation.utils import (
    GENERATION_MODES_MAPPING,
    GenerateOutput,
    GenerationConfig,
    GenerationMixin,
    LogitsProcessorList,
    StoppingCriteriaList,
    TransformersKwargs,
    inspect,
    logger,
    warnings,
)
from transformers.models.llama.modeling_llama import (
    BaseModelOutputWithPast,
    Cache,
    CausalLMOutputWithPast,
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
        inputs_embeds: List[list],
        logits: torch.FloatTensor,
        log_probs: torch.FloatTensor,
        gradient_blocks: Optional[list] = None,
    ):
        """
        Calculate the gradient of the logits with respect to the inputs_embeds.
        """
        # inputs_embeds: now expected to be a list (len=batch) of lists (len=input_sequence_length) of per-token tensors
        # Build a per-batch list of required indices (if gradient_blocks provided) or all indices
        batch_size = logits.shape[0]
        device = logits.device
        dtype = logits.dtype

        # Helper to expand blocks into a set of indices
        def blocks_to_indices(blocks, seq_len):
            if blocks is None:
                return set(range(seq_len))
            idxs = set()
            for b in blocks:
                if not (isinstance(b, (list, tuple)) and len(b) == 2):
                    continue
                start, end = int(b[0]), int(b[1])
                # inclusive end as requested
                start = max(0, start)
                end = min(seq_len - 1, end)
                if start <= end:
                    idxs.update(range(start, end + 1))
            return idxs

        gradients_per_batch = []
        for bidx in range(batch_size):
            input_token_tensors = inputs_embeds[bidx]  # list of per-token tensors shape (hidden_size,)
            input_seq_len = len(input_token_tensors)
            # gradient_blocks may be either:
            # - None => all tokens
            # - a single list of (start,end) tuples => apply to all batch elements
            # - a list of lists where each element corresponds to a batch sample
            if gradient_blocks is None:
                per_batch_blocks = None
            elif isinstance(gradient_blocks, (list, tuple)) and len(gradient_blocks) == batch_size:
                per_batch_blocks = gradient_blocks[bidx]
            else:
                per_batch_blocks = gradient_blocks
            # indices which need gradients for this batch
            required_indices = blocks_to_indices(per_batch_blocks, input_seq_len)

            if self.gradient_strategy == FINEGRAINED:
                # We need gradients for each output token w.r.t. all input tokens.
                out_len = logits.shape[1]
                # prepare container (out_len, input_seq_len, hidden_size)
                hidden_size = input_token_tensors[0].shape[0] if input_seq_len > 0 else 0
                batch_grads = torch.zeros((out_len, input_seq_len, hidden_size), device=device, dtype=dtype)

                # Build list of tensors that are requested for grad (ordered by index)
                grad_inputs = [input_token_tensors[i] for i in range(input_seq_len) if i in required_indices]
                grad_input_indices = [i for i in range(input_seq_len) if i in required_indices]

                for out_i in range(out_len):
                    out_scalar = logits[bidx][out_i]
                    if len(grad_inputs) == 0:
                        # nothing requires grad -> leave zeros
                        continue
                    grads_tuple = torch.autograd.grad(
                        outputs=out_scalar,
                        inputs=tuple(grad_inputs),
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                    # grads_tuple corresponds to grad_inputs order
                    for gi, gi_idx in enumerate(grad_input_indices):
                        g = grads_tuple[gi]
                        if g is None:
                            continue
                        # g should have shape (hidden_size,)
                        batch_grads[out_i, gi_idx, :] = g

                gradients_per_batch.append(batch_grads)

            elif self.gradient_strategy == SUM_SQUARES:
                logits_sum_squares = logits[bidx].pow(2).sum()
                grad_inputs = [input_token_tensors[i] for i in range(input_seq_len) if i in required_indices]
                grad_input_indices = [i for i in range(input_seq_len) if i in required_indices]

                if len(grad_inputs) == 0:
                    # No requested grads -> return zero tensor
                    hidden_size = input_token_tensors[0].shape[0] if input_seq_len > 0 else 0
                    grads_full = torch.zeros((input_seq_len, hidden_size), device=device, dtype=dtype)
                else:
                    grads_tuple = torch.autograd.grad(
                        outputs=logits_sum_squares,
                        inputs=tuple(grad_inputs),
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                    hidden_size = grad_inputs[0].shape[0]
                    grads_full = torch.zeros((input_seq_len, hidden_size), device=device, dtype=dtype)
                    for gi, gi_idx in enumerate(grad_input_indices):
                        g = grads_tuple[gi]
                        if g is None:
                            continue
                        grads_full[gi_idx, :] = g

                gradients_per_batch.append(grads_full)

            elif self.gradient_strategy == PRODUCT_PROBS:
                probs_product = log_probs[bidx].sum().exp()
                grad_inputs = [input_token_tensors[i] for i in range(input_seq_len) if i in required_indices]
                grad_input_indices = [i for i in range(input_seq_len) if i in required_indices]

                if len(grad_inputs) == 0:
                    hidden_size = input_token_tensors[0].shape[0] if input_seq_len > 0 else 0
                    grads_full = torch.zeros((input_seq_len, hidden_size), device=device, dtype=dtype)
                else:
                    grads_tuple = torch.autograd.grad(
                        outputs=probs_product,
                        inputs=tuple(grad_inputs),
                        retain_graph=False,
                        create_graph=False,
                        allow_unused=True,
                    )
                    hidden_size = grad_inputs[0].shape[0]
                    grads_full = torch.zeros((input_seq_len, hidden_size), device=device, dtype=dtype)
                    for gi, gi_idx in enumerate(grad_input_indices):
                        g = grads_tuple[gi]
                        if g is None:
                            continue
                        grads_full[gi_idx, :] = g

                gradients_per_batch.append(grads_full)

            else:
                raise NotImplementedError(f"Gradient strategy {self.gradient_strategy} is not implemented.")

        # Stack batches appropriately.
        # For FINEGRAINED entries, each element is (out_len, in_len, hidden_size) -> stack -> (batch, out_len, in_len, hidden_size)
        # For SUM_SQUARES/PRODUCT_PROBS entries, each is (in_len, hidden_size) -> stack -> (batch, in_len, hidden_size)
        if self.gradient_strategy == FINEGRAINED:
            gradients = torch.stack(gradients_per_batch, dim=0)
        else:
            gradients = torch.stack(gradients_per_batch, dim=0)

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
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        # Record the initial inputs_embeds
        with torch.no_grad():
            inputs_embeds = inputs_embeds if inputs_embeds is not None else self.model.embed_tokens(input_ids)
        # gradient_blocks may be passed via kwargs (from generate)
        gradient_blocks = kwargs.get("gradient_blocks", None)
        dummy_weights = kwargs.get("dummy_weights", False)

        if self.initial_inputs_embeds is None and not dummy_weights:
            # We will build per-batch per-token leaf tensors so we can selectively require_grad only for requested tokens
            initial_inputs_embeds = list()
            batch_size = inputs_embeds.shape[0]
            seq_len = inputs_embeds.shape[1]
            for i in range(batch_size):
                per_token_list = list()
                # determine per-sample blocks: gradient_blocks can be per-batch (list of lists)
                if gradient_blocks is None:
                    per_sample_blocks = None
                elif isinstance(gradient_blocks, (list, tuple)) and len(gradient_blocks) == batch_size:
                    per_sample_blocks = gradient_blocks[i]
                else:
                    per_sample_blocks = gradient_blocks

                for t in range(seq_len):
                    token_embed = inputs_embeds[i, t, :].clone().detach()
                    # Determine if this token index requires grad
                    need_grad = True
                    if per_sample_blocks is not None:
                        # per_sample_blocks is a list of (start, end) tuples (inclusive)
                        in_block = False
                        for blk in per_sample_blocks:
                            if not (isinstance(blk, (list, tuple)) and len(blk) == 2):
                                continue
                            start, end = int(blk[0]), int(blk[1])
                            if start <= t <= end:
                                in_block = True
                                break
                        need_grad = in_block
                    # Set requires_grad flag for this per-token tensor
                    token_embed.requires_grad_(need_grad)
                    per_token_list.append(token_embed)
                initial_inputs_embeds.append(per_token_list)
            # save per-token lists
            self.set_initial_input_embed(initial_inputs_embeds)
            # build the stacked inputs_embeds tensor to feed the model; stacking preserves graph links to per-token tensors
            inputs_embeds = torch.stack([torch.stack(per_token_list, dim=0) for per_token_list in self.initial_inputs_embeds], dim=0)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            # input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
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
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], list[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        use_model_defaults: Optional[bool] = None,
        custom_generate: Optional[Union[str, Callable]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        # Extract gradient blocks
        gradient_blocks = kwargs.pop("gradient_blocks", None)
        dummy_weights = kwargs.pop("dummy_weights", False)

        # 0. If requested, load an arbitrary generation recipe from the Hub and run it instead
        trust_remote_code = kwargs.pop("trust_remote_code", None)

        if custom_generate is not None and isinstance(custom_generate, str):
            # Get all `generate` arguments in a single variable. Custom functions are responsible for handling them:
            # they receive the same inputs as `generate`, with `model` instead of `self` and excluding the arguments to
            # trigger the custom generation. They can access to methods from `GenerationMixin` through `model`.
            global_keys_to_exclude = {
                "self",
                "kwargs",
                "global_keys_to_exclude",
                "trust_remote_code",
                "custom_generate",
            }
            generate_arguments = {key: value for key, value in locals().items() if key not in global_keys_to_exclude}
            generate_arguments.update(kwargs)

            custom_generate_function = self.load_custom_generate(
                custom_generate, trust_remote_code=trust_remote_code, **kwargs
            )
            return custom_generate_function(model=self, **generate_arguments)

        # 1. Handle kwargs, `generation_config`, validate them and obtain generation mode
        generation_mode_kwargs = self._extract_generation_mode_kwargs(
            custom_generate,
            kwargs,
            synced_gpus,
            assistant_model,
            streamer,
        )

        generation_config, model_kwargs = self._prepare_generation_config(
            generation_config, use_model_defaults, **kwargs
        )
        generation_mode = generation_config.get_generation_mode(assistant_model)
        if isinstance(custom_generate, Callable):
            decoding_method = custom_generate
        else:
            # type() required to access the unbound class-level method
            decoding_method = getattr(type(self), GENERATION_MODES_MAPPING[generation_mode])

        generation_config.return_dict_in_generate = True  # Always return dict in generate
        generation_config.output_logits = True  # Always return logits in generate
        self._validate_model_kwargs(model_kwargs.copy())
        self._validate_generation_mode(generation_mode, generation_config, generation_mode_kwargs)
        model_kwargs["gradient_blocks"] = gradient_blocks
        model_kwargs["dummy_weights"] = dummy_weights

        # Deprecation-related step: set Hub repo for deprecated strategies.
        # NOTE: This must come after initializing generation_config, since we need it to determine if this is a deprecated mode.
        # It must also be before any preparation steps, since Hub repos expect to be loaded before preparation steps.
        # TODO joao, manuel: remove this in v4.62.0
        if deprecated_mode_repo := self._get_deprecated_gen_repo(generation_mode, trust_remote_code, custom_generate):
            return GenerationMixin.generate(
                self,
                inputs=inputs,
                generation_config=generation_config,
                logits_processor=logits_processor,
                stopping_criteria=stopping_criteria,
                prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
                assistant_model=assistant_model,
                negative_prompt_ids=negative_prompt_ids,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                use_model_defaults=use_model_defaults,
                custom_generate=deprecated_mode_repo,
                trust_remote_code=trust_remote_code,
                **generation_mode_kwargs,
                **kwargs,
            )

        # 2. Set generation parameters if not already defined
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()

        accepts_attention_mask = "attention_mask" in set(inspect.signature(self.forward).parameters.keys())
        requires_attention_mask = "encoder_outputs" not in model_kwargs
        kwargs_has_attention_mask = model_kwargs.get("attention_mask", None) is not None

        # 3. Define model inputs
        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(
            inputs, generation_config.bos_token_id, model_kwargs
        )
        # Some generation modes (e.g. assisted) need `inputs_tensor` to rerun encoder.forward()
        if "inputs_tensor" in inspect.signature(decoding_method).parameters.keys():
            generation_mode_kwargs["inputs_tensor"] = inputs_tensor
        batch_size = inputs_tensor.shape[0]

        device = inputs_tensor.device
        self._prepare_special_tokens(generation_config, kwargs_has_attention_mask, device=device)

        # Tail-window prefill: if gradient_blocks exist, prefill the prefix up to the earliest
        # required-grad token (across the whole batch) under no_grad to build KV cache, then
        # run gradient-enabled forward only on the remaining tail. This reduces activation
        # memory in proportion to tail length.
        gb = model_kwargs.get("gradient_blocks", None)
        # Compute global earliest index among all blocks; support both per-batch and global lists.
        # If a per-sample list is present but empty for a sample, treat it as "full prefill" for that sample
        # (i.e., its earliest index is seq_len_full), so it does not constrain the global earliest.
        seq_len_full = inputs_tensor.shape[1]

        def earliest_from_blocks(blocks, seq_len_full_val):
            earliest = None
            saw_any_sample = False
            saw_any_block = False
            if blocks is None:
                return None
            # per-sample list-of-lists
            saw_any_sample = True
            for sample_blocks in blocks:
                if sample_blocks:
                    saw_any_block = True
                    for blk in sample_blocks:
                        if isinstance(blk, (list, tuple)) and len(blk) == 2:
                            s = int(blk[0])
                            earliest = s if earliest is None else min(earliest, s)
                else:
                    # Empty list -> no gradients for this sample: treat earliest as seq_len_full
                    earliest = seq_len_full_val if earliest is None else min(earliest, seq_len_full_val)
            # If we saw per-sample lists but none had blocks, set to full length
            if saw_any_sample and not saw_any_block and earliest is None:
                earliest = seq_len_full_val
            return earliest

        earliest_idx = earliest_from_blocks(gb, seq_len_full)
        # Only apply when we have input_ids and a non-zero prefix exists
        if earliest_idx is not None and isinstance(earliest_idx, int) and earliest_idx > 0 and model_input_name == "input_ids":
            seq_len_full = inputs_tensor.shape[1]
            prefix_len = min(max(0, earliest_idx), seq_len_full)
            tail_len = seq_len_full - prefix_len
            if tail_len > 0:
                # Prefill prefix under no_grad to build past_key_values
                prefix_ids = inputs_tensor[:, :prefix_len].to(self.device)
                if "attention_mask" in model_kwargs and model_kwargs["attention_mask"] is not None:
                    prefix_mask = model_kwargs["attention_mask"][:, :prefix_len].to(self.device)
                else:
                    prefix_mask = torch.ones(prefix_ids.shape, device=self.device, dtype=torch.long)

                with torch.no_grad():
                    prefill_out = self.model(
                        input_ids=prefix_ids,
                        attention_mask=prefix_mask,
                        use_cache=True,
                    )
                    past_key_values = prefill_out.past_key_values

                # Trim to tail ids for the rest of generation
                inputs_tensor = inputs_tensor[:, prefix_len:]  # (batch_size, input_tail_length)
                model_input_name = "input_ids"
                model_kwargs["past_key_values"] = past_key_values
                model_kwargs["cache_position"] = torch.arange(prefix_len, prefix_len + tail_len, device=self.device)

                # Adjust gradient_blocks to tail-relative coordinates and drop blocks before prefix
                def adjust_blocks(blocks):
                    adjusted = []
                    for blk in blocks or []:
                        if not (isinstance(blk, (list, tuple)) and len(blk) == 2):
                            continue
                        s, e = int(blk[0]), int(blk[1])
                        # keep overlap with [prefix_len, seq_len_full-1]
                        if e < prefix_len:
                            continue
                        s2 = max(0, s - prefix_len)
                        e2 = min(tail_len - 1, e - prefix_len)
                        if s2 <= e2:
                            adjusted.append([s2, e2])
                    return adjusted

                model_kwargs["gradient_blocks"] = [adjust_blocks(sample) for sample in gb]

                # Ensure attention_mask covers full original sequence (prefix+tail)
                if "attention_mask" not in model_kwargs or model_kwargs["attention_mask"] is None:
                    model_kwargs["attention_mask"] = torch.ones((batch_size, seq_len_full), device=inputs_tensor.device, dtype=torch.long)

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

        # Expand inputs depending on the generation mode
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=input_ids,
            expand_size=max(generation_config.num_beams, generation_config.num_return_sequences),
            is_encoder_decoder=self.config.is_encoder_decoder,
            **model_kwargs,
        )

        if generation_config.token_healing:
            input_ids = self.heal_tokens(input_ids, generation_mode_kwargs.get("tokenizer"))

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
            generation_config, model_kwargs, generation_mode, batch_size, max_cache_length
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

        # 8. prepare logits processors and stopping criteria
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
            generation_config=generation_config,
            stopping_criteria=stopping_criteria,
            tokenizer=generation_mode_kwargs.get("tokenizer"),
        )

        # Set model_kwargs `use_cache` so we can use it later in forward runs
        model_kwargs["use_cache"] = generation_config.use_cache

        # 9. Call generation mode
        result = decoding_method(
            self,
            input_ids,
            logits_processor=prepared_logits_processor,
            stopping_criteria=prepared_stopping_criteria,
            generation_config=generation_config,
            **generation_mode_kwargs,
            **model_kwargs,
        )

        # Convert to legacy cache format if requested
        if (
            generation_config.return_legacy_cache is True
            and hasattr(result, "past_key_values")
            and getattr(result.past_key_values, "to_legacy_cache") is not None
        ):
            result.past_key_values = result.past_key_values.to_legacy_cache()

        # Keep returned sequence aligned with original input length when tail-window prefill is used.
        if earliest_idx is not None and isinstance(earliest_idx, int) and earliest_idx > 0:
            result.sequences = torch.cat([prefix_ids, result.sequences], dim=1)

        # Compute the gradient of the logits with respect to the inputs_embeds unless dummy weights are requested.
        if dummy_weights:
            result.gradients = None
            result.embeds = None
        else:
            input_ids = result.sequences  # (batch_size, input_tail_length + output_sequence_length)
            logits = torch.stack(result.logits, dim=1)  # (batch_size, output_sequence_length, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1)  # (batch_size, output_sequence_length, vocab_size)
            picked_input_ids = input_ids[:, -logits.shape[1]:]  # (batch_size, output_sequence_length)
            picked_logits = logits.gather(2, picked_input_ids.unsqueeze(-1)).squeeze(-1)  # (batch_size, output_sequence_length)
            picked_log_probs = log_probs.gather(2, picked_input_ids.unsqueeze(-1)).squeeze(-1)  # (batch_size, output_sequence_length)
            # Calculate gradients. Pass the same gradient_blocks that were provided to generate (if any)
            gradients = self.calculate_gradient(
                self.initial_inputs_embeds,
                picked_logits,
                picked_log_probs,
                gradient_blocks=model_kwargs.get("gradient_blocks", None),
            )
            result.gradients = gradients
            # If FINEGRAINED: (batch_size, output_sequence_length, input_tail_length, hidden_size)
            # If SUM_SQUARES: (batch_size, input_tail_length, hidden_size)
            # If PRODUCT_PROBS: (batch_size, input_tail_length, hidden_size)
            # Reconstruct embeds from per-token lists into a tensor for reference
            result.embeds = torch.stack([torch.stack(per_token_list, dim=0) for per_token_list in self.initial_inputs_embeds], dim=0)
            # (batch_size, input_tail_length, hidden_size)

            # Adjust the values so that they correspond to the original input sequence length
            if earliest_idx is not None and isinstance(earliest_idx, int) and earliest_idx > 0:
                # We prefixed up to earliest_idx under no_grad, so we need to pad the gradients and embeds
                batch_size = result.gradients.shape[0]
                hidden_size = result.embeds.shape[2]
                if self.gradient_strategy == FINEGRAINED:
                    out_len = result.gradients.shape[1]
                    prefix_pad = torch.zeros((batch_size, out_len, earliest_idx, hidden_size), device=result.gradients.device, dtype=result.gradients.dtype)
                    result.gradients = torch.cat([prefix_pad, result.gradients], dim=2)
                else:
                    prefix_pad = torch.zeros((batch_size, earliest_idx, hidden_size), device=result.gradients.device, dtype=result.gradients.dtype)
                    result.gradients = torch.cat([prefix_pad, result.gradients], dim=1)
                # Embeds padding
                embed_prefix_pad = torch.zeros((batch_size, earliest_idx, hidden_size), device=result.embeds.device, dtype=result.embeds.dtype)
                result.embeds = torch.cat([embed_prefix_pad, result.embeds], dim=1)

        # The final sizes are:
        # 1. gradients:
        #    - FINEGRAINED: (batch_size, output_sequence_length, input_sequence_length, hidden_size)
        #    - SUM_SQUARES/PRODUCT_PROBS: (batch_size, input_sequence_length, hidden_size)
        # 2. embeds: (batch_size, input_sequence_length, hidden_size)
        # 3. sequences: (batch_size, total_sequence_length)

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