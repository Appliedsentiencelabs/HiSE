import torch
from torch import nn
from transformers import BertConfig, BertModel

from hise.address import StructuralAddress
from hise.bert_sgam import (
    TokenLevelSGAMBias,
    attach_sgam_to_bert_attention,
    build_token_level_addresses,
    zero_weight_equivalence_gate,
)


class DummySelfAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.query = nn.Linear(4, 4, bias=False)
        self.key = nn.Linear(4, 4, bias=False)
        self.value = nn.Linear(4, 4, bias=False)
        self.attention_head_size = 2
        self.all_head_size = 4
        self.sgam = None
        self.sgam_addresses = None
        self.dropout = nn.Dropout(0.0)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, dim = x.shape
        head_dim = dim // 2
        x = x.view(batch, length, 2, head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor | None = None, *, output_attentions: bool = False, **_: object) -> torch.Tensor:
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.attention_head_size ** 0.5)
        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1)
        if self.sgam is not None and self.sgam_addresses is not None:
            bias = self.sgam(self.sgam_addresses.to(device=hidden_states.device), attention_mask)
            scores = scores + bias.to(device=hidden_states.device, dtype=scores.dtype)
        probs = nn.functional.softmax(scores, dim=-1)
        context = torch.matmul(probs, v)
        context = context.permute(0, 2, 1, 3).contiguous().view(hidden_states.shape[0], hidden_states.shape[1], self.all_head_size)
        return context


class DummyLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = type("Attention", (), {"self": DummySelfAttention()})()


class DummyBertModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = type("Encoder", (), {"layer": [DummyLayer()]})()


def test_token_level_sgam_bias_is_zero_for_same_structure() -> None:
    addresses = torch.tensor([[[0.0, 1.0, 10.0, 0.0], [0.0, 1.0, 10.0, 0.0], [1.0, 2.0, 20.0, 0.0]]], dtype=torch.float32)
    bias = TokenLevelSGAMBias(depth_weight=1.0, parent_weight=2.0, sibling_weight=3.0)(addresses)
    assert bias.shape == (1, 1, 3, 3)
    assert bias[0, 0, 0, 1].item() == 0.0


def test_build_token_level_addresses_keeps_chunk_structure() -> None:
    addr = StructuralAddress(sequence_position=0, depth=2, parent_id=123, sibling_index=1)
    built = build_token_level_addresses(["A quick test"], [addr])
    assert len(built) == 1
    assert built[0].shape[-1] == 4
    assert tuple(built[0][0].tolist()) == (0.0, 2.0, 123.0, 1.0)


def test_attach_sgam_to_bert_attention_registers_bias() -> None:
    model = DummyBertModel()
    addresses = torch.tensor([[[0.0, 1.0, 10.0, 0.0], [0.0, 1.0, 10.0, 0.0], [1.0, 2.0, 20.0, 0.0]]], dtype=torch.float32)
    attach_sgam_to_bert_attention(model, token_addresses=addresses, shared=True)
    encoder = model.encoder.layer[0].attention.self
    assert encoder.sgam is not None
    assert encoder.sgam_addresses is not None
    x = torch.randn(1, 3, 4)
    out = encoder.forward(x, attention_mask=torch.ones(1, 3, dtype=torch.float32))
    assert out[0].shape == (1, 3, 4)


def test_real_bert_zero_weight_equivalence_gate() -> None:
    config = BertConfig(vocab_size=99, hidden_size=16, num_hidden_layers=2, num_attention_heads=2, intermediate_size=32, attn_implementation="eager")
    model = BertModel(config)
    assert type(model.encoder.layer[0].attention.self).__name__ == "BertSelfAttention"
    input_ids = torch.randint(0, config.vocab_size, (2, 8))
    attention_mask = torch.ones_like(input_ids)
    token_addresses = torch.tensor(
        [
            [[0.0, 1.0, 10.0, 0.0], [0.0, 1.0, 10.0, 0.0], [1.0, 2.0, 11.0, 1.0], [1.0, 2.0, 11.0, 1.0], [2.0, 3.0, 12.0, 0.0], [2.0, 3.0, 12.0, 0.0], [3.0, 4.0, 13.0, 0.0], [3.0, 4.0, 13.0, 0.0]],
            [[0.0, 1.0, 20.0, 0.0], [0.0, 1.0, 20.0, 0.0], [1.0, 2.0, 21.0, 1.0], [1.0, 2.0, 21.0, 1.0], [2.0, 3.0, 22.0, 1.0], [2.0, 3.0, 22.0, 1.0], [3.0, 4.0, 23.0, 0.0], [3.0, 4.0, 23.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    result = zero_weight_equivalence_gate(
        model,
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_addresses=token_addresses,
    )
    assert result["passed"] is True
    assert float(result["max_abs_diff"]) < 1e-4
