from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch import Tensor, nn

from .address import StructuralAddress


@dataclass(frozen=True)
class StructuralWindow:
    """A window of chunk-level structure mapped onto token positions."""

    text: str
    token_addresses: tuple[StructuralAddress, ...]
    chunk_address: StructuralAddress


class TokenLevelSGAMBias(nn.Module):
    """Additive SGAM bias over token-level structural addresses.

    Input addresses are expected to have shape ``(B, L, 4)`` and represent
    ``(m, d, p, s)`` for every token in the sequence.
    """

    def __init__(
        self,
        *,
        depth_weight: float = 0.0,
        parent_weight: float = 0.0,
        sibling_weight: float = 0.0,
        max_depth: int = 20,
    ) -> None:
        super().__init__()
        self.depth_weight = nn.Parameter(torch.tensor(float(depth_weight)))
        self.parent_weight = nn.Parameter(torch.tensor(float(parent_weight)))
        self.sibling_weight = nn.Parameter(torch.tensor(float(sibling_weight)))
        self.max_depth = max_depth

    def _normalize_distance(self, value: Tensor, *, max_value: int) -> Tensor:
        if max_value <= 0:
            return value
        return value / float(max_value)

    def forward(self, addresses: Tensor, attention_mask: Tensor | None = None) -> Tensor:
        if addresses.dim() != 3 or addresses.shape[-1] != 4:
            raise ValueError("addresses must have shape (B, L, 4)")

        diffs = addresses[:, :, None, :] - addresses[:, None, :, :]
        depth_term = self._normalize_distance(diffs[..., 1].abs(), max_value=self.max_depth) * self.depth_weight
        parent_term = (diffs[..., 2] != 0).float() * self.parent_weight
        sibling_term = self._normalize_distance(diffs[..., 3].abs(), max_value=max(1, self.max_depth)) * self.sibling_weight
        bias = depth_term + parent_term + sibling_term

        if attention_mask is not None:
            valid = attention_mask
            if valid.dim() == 2:
                valid = valid.unsqueeze(1) * valid.unsqueeze(2)
            elif valid.dim() == 3:
                if valid.shape[1] == 1:
                    valid = valid.squeeze(1)
                elif valid.shape[-1] == 1:
                    valid = valid.squeeze(-1)
                valid = valid.unsqueeze(1) * valid.unsqueeze(2)
            elif valid.dim() == 4:
                if valid.shape[1] == 1 and valid.shape[2] == 1:
                    valid = valid.squeeze(1).squeeze(1)
                    valid = valid.unsqueeze(1) * valid.unsqueeze(2)
                elif valid.shape[1] == 1:
                    valid = valid.squeeze(1)
                elif valid.shape[2] == 1:
                    valid = valid.squeeze(2)
            bias = bias * valid.to(dtype=bias.dtype)
        return bias.unsqueeze(1)


def build_token_level_addresses(
    chunk_texts: Sequence[str],
    chunk_addresses: Sequence[StructuralAddress],
    *,
    tokenizer: Any | None = None,
    max_length: int = 512,
) -> list[Tensor]:
    """Map a chunk-level structural address to each token in the window.

    This is the operational version of the design decision: SGAM must be token-level,
    and all tokens in the same chunk inherit the chunk's structural coordinates.
    """
    if len(chunk_texts) != len(chunk_addresses):
        raise ValueError("chunk_texts and chunk_addresses must have the same length")

    if tokenizer is None:
        outputs: list[Tensor] = []
        for address in chunk_addresses:
            token_address = torch.tensor([address.as_vector()], dtype=torch.float32)
            outputs.append(token_address)
        return outputs

    windows: list[Tensor] = []
    for text, address in zip(chunk_texts, chunk_addresses):
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            return_offsets_mapping=True,
            add_special_tokens=True,
        )
        token_count = len(encoded["input_ids"])
        base = torch.tensor([address.as_vector() for _ in range(token_count)], dtype=torch.float32)
        windows.append(base)
    return windows


def _bert_attention_mask_to_additive(mask: Tensor | None) -> Tensor | None:
    if mask is None:
        return None

    input_is_binary = mask.dtype == torch.bool or not mask.is_floating_point()
    mask = mask.to(dtype=torch.float32)
    is_binary = bool(torch.all((mask == 0) | (mask == 1)))
    if mask.dim() == 2:
        mask = mask[:, None, None, :]
    elif mask.dim() == 3:
        mask = mask[:, None, :, :]
    elif mask.dim() == 4 and not input_is_binary and (not is_binary or bool(torch.all(mask == 0))):
        return mask

    if not is_binary:
        raise ValueError("non-binary attention masks must be 4D additive masks")
    return (1.0 - mask) * -1e9


def _bert_attention_mask_to_valid(mask: Tensor | None, *, batch_size: int, key_length: int, device: torch.device) -> Tensor:
    if mask is None:
        return torch.ones((batch_size, key_length), dtype=torch.bool, device=device)

    if mask.dim() == 2:
        valid = mask > 0
    elif mask.dim() == 3:
        valid = mask[:, 0, :] > -1 if mask.is_floating_point() and torch.any(mask < 0) else mask[:, 0, :] > 0
    elif mask.dim() == 4:
        key_mask = mask[:, 0, 0, :]
        valid = key_mask > -1 if mask.is_floating_point() else key_mask > 0
    else:
        raise ValueError("attention_mask must have 2, 3, or 4 dimensions")

    if valid.shape != (batch_size, key_length):
        raise ValueError(
            f"attention mask key shape must be {(batch_size, key_length)}, got {tuple(valid.shape)}"
        )
    return valid


def force_eager_bert_attention(model: Any) -> Any:
    """Force the eager BERT attention path so the additive SGAM bias is actually in the compute graph."""
    if hasattr(model, "config"):
        config = getattr(model, "config")
        for attribute in ("_attn_implementation", "attn_implementation"):
            if hasattr(config, attribute):
                setattr(config, attribute, "eager")
    return model


def attach_sgam_to_bert_attention(
    model: Any,
    *,
    token_addresses: Tensor | None = None,
    shared: bool = True,
    layer_indices: Sequence[int] | None = None,
) -> Any:
    """Attach a token-level SGAM bias to each BERT attention layer.

    The implementation keeps the base BERT model intact except for the additive-pre-softmax
    bias term. The default is a single shared bias module across all 12 layers.
    """
    model = force_eager_bert_attention(model)
    if not hasattr(model, "encoder") or not hasattr(model.encoder, "layer"):
        raise TypeError("model must be a Hugging Face-style BERT model with encoder.layer")

    layer_list = list(model.encoder.layer)
    if layer_indices is None:
        selected = range(len(layer_list))
    else:
        selected = layer_indices

    if token_addresses is not None:
        token_addresses = token_addresses.to(dtype=torch.float32)

    shared_module: nn.Module | None = TokenLevelSGAMBias() if shared else None
    for idx, layer in enumerate(layer_list):
        if idx not in selected:
            continue
        if shared:
            layer.attention.self.sgam = shared_module
        else:
            layer.attention.self.sgam = TokenLevelSGAMBias()
        layer.attention.self.sgam_addresses = token_addresses

        original = layer.attention.self.forward

        def patched_forward(
            self_module: Any,
            hidden_states: Tensor,
            attention_mask: Tensor | None = None,
            head_mask: Tensor | None = None,
            encoder_hidden_states: Tensor | None = None,
            encoder_attention_mask: Tensor | None = None,
            past_key_value: tuple[Tensor, Tensor] | None = None,
            output_attentions: bool = False,
        ) -> tuple[Tensor, ...]:
            is_cross_attention = encoder_hidden_states is not None

            if is_cross_attention and past_key_value is not None:
                key_layer = past_key_value[0]
                value_layer = past_key_value[1]
                attention_mask = encoder_attention_mask
            elif is_cross_attention:
                key_layer = self_module.transpose_for_scores(self_module.key(encoder_hidden_states))
                value_layer = self_module.transpose_for_scores(self_module.value(encoder_hidden_states))
                attention_mask = encoder_attention_mask
            elif past_key_value is not None:
                key_layer = self_module.transpose_for_scores(self_module.key(hidden_states))
                value_layer = self_module.transpose_for_scores(self_module.value(hidden_states))
                key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
                value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
            else:
                key_layer = self_module.transpose_for_scores(self_module.key(hidden_states))
                value_layer = self_module.transpose_for_scores(self_module.value(hidden_states))

            query_layer = self_module.transpose_for_scores(self_module.query(hidden_states))
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) / math.sqrt(self_module.attention_head_size)

            if attention_mask is not None:
                additive_mask = _bert_attention_mask_to_additive(attention_mask)
                if additive_mask is not None:
                    attention_scores = attention_scores + additive_mask.to(device=attention_scores.device, dtype=attention_scores.dtype)

            if hasattr(self_module, "sgam") and self_module.sgam is not None and self_module.sgam_addresses is not None:
                batch_addresses = self_module.sgam_addresses.to(device=hidden_states.device, dtype=hidden_states.dtype)
                valid_attention_mask = _bert_attention_mask_to_valid(
                    attention_mask,
                    batch_size=batch_addresses.shape[0],
                    key_length=key_layer.shape[2],
                    device=hidden_states.device,
                )
                bias = self_module.sgam(batch_addresses, valid_attention_mask)
                if bias is not None:
                    attention_scores = attention_scores + bias.to(device=attention_scores.device, dtype=attention_scores.dtype)

            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            dropout = getattr(self_module, "dropout", None)
            if dropout is None:
                dropout = nn.Identity()
            attention_probs = dropout(attention_probs)
            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            context_layer = torch.matmul(attention_probs, value_layer)
            context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
            new_context_layer_shape = context_layer.size()[:-2] + (self_module.all_head_size,)
            context_layer = context_layer.view(*new_context_layer_shape)
            outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
            return outputs

        layer.attention.self.forward = patched_forward.__get__(layer.attention.self, type(layer.attention.self))

    return model


def make_windowed_sgam_demo(
    texts: Sequence[str],
    chunk_addresses: Sequence[StructuralAddress],
    *,
    max_tokens: int = 512,
) -> tuple[list[Tensor], list[Tensor]]:
    """Small helper for an SGAM sanity check over a multi-chunk window."""
    windows: list[Tensor] = []
    masks: list[Tensor] = []
    if len(texts) != len(chunk_addresses):
        raise ValueError("texts and chunk_addresses must match")

    for text, address in zip(texts, chunk_addresses):
        token_count = min(len(text.split()), max_tokens)
        addresses = torch.tensor([address.as_vector() for _ in range(token_count)], dtype=torch.float32)
        mask = torch.ones(token_count, dtype=torch.float32)
        windows.append(addresses)
        masks.append(mask)

    return windows, masks


def zero_weight_equivalence_gate(
    model: Any,
    *,
    input_ids: Tensor,
    attention_mask: Tensor | None = None,
    token_addresses: Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-4,
) -> dict[str, float | bool]:
    """Compare the baseline model against the same model with zeroed SGAM weights.

    The zero-weight gate is the real system-level sanity check for the SGAM hook: if the module
    is attached correctly and the bias is truly additive/pre-softmax, setting all SGAM weights to
    zero should preserve the stock BERT outputs within numerical tolerance.
    """
    if attention_mask is None:
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)

    model.eval()
    with torch.inference_mode():
        baseline = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        attach_sgam_to_bert_attention(model, token_addresses=token_addresses, shared=True)
        for layer in model.encoder.layer:
            module = layer.attention.self
            if hasattr(module, "sgam") and module.sgam is not None:
                module.sgam.depth_weight.data.zero_()
                module.sgam.parent_weight.data.zero_()
                module.sgam.sibling_weight.data.zero_()
        patched = model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
    max_abs_diff = float((baseline - patched).abs().max().item())
    passed = bool(torch.allclose(baseline, patched, atol=atol, rtol=rtol))
    return {"max_abs_diff": max_abs_diff, "passed": passed}
