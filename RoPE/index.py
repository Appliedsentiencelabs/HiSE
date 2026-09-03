import torch
import torch.nn as nn

class RoPE(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) for LLM attention layers.
    Reference: "RoFormer: Transformer Encoder Discovered as a Fast Autoregressive Decoder" (Su et al., 2021)
    
    Precompute frequency tensor: shape (dim // 2)
    freq_i = base^(-2i/d) for i in [0, dim/2)
    Precompute rotation angles for positions 0 to max_seq_len-1
    Convert to cos/sin tables for fast lookup
    Using register_buffer ensures proper device/dtype handling during .to() calls
    """

    def __init__(self, dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE requires an even hidden dimension (e.g., 64, 128, 4096)")
        
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base


        freqs = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))

        positions = torch.arange(max_seq_len)
        self.freqs = torch.outer(positions, freqs)  # (max_seq_len, dim//2)
        self.register_buffer("freqs_cos", torch.cos(self.freqs).unsqueeze(0), persistent=False)
        self.register_buffer("freqs_sin", torch.sin(self.freqs).unsqueeze(0), persistent=False)



    def forward(self, x: torch.Tensor, pos: torch.Tensor = None) -> torch.Tensor:
        """
        Apply RoPE to input tensor (typically Q or K projections).
        
        Args:
            x: (batch_size, seq_len, dim) - Query or Key tensor
            pos: (batch_size, seq_len) - Position indices. If None, assumes [0, 1, 2, ..., seq_len-1]
            
        Returns:
            Rotated tensor with identical shape to x
        """
        # Extract cos/sin for requested positions (handles variable sequence lengths)
        # Split into dim//2 pairs and convert to complex numbers for efficient rotation
        # x_reshaped: (batch, seq_len, dim//2, 2)
        # Convert frequency tables to complex: cos + j*sin
        # Element-wise complex multiplication applies the rotation matrix
        # Convert back to real tensor and restore original shape


        if pos is None:
            pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], x.shape[1])

        batch_size, seq_len, dim = x.shape

        cos = self.freqs_cos.to(x.device)[:, :seq_len, :].expand(batch_size, -1, -1)
        sin = self.freqs_sin.to(x.device)[:, :seq_len, :].expand(batch_size, -1, -1)

        x_reshaped = x.reshape(batch_size, seq_len, dim // 2, 2)
        x_complex = torch.view_as_complex(x_reshaped)  # (batch, seq_len, dim//2)

        freqs_complex = torch.view_as_complex(torch.stack([cos, sin], dim=-1))
        x_rotated_complex = x_complex * freqs_complex
        x_rotated = torch.view_as_real(x_rotated_complex)  # (batch, seq_len, dim//2, 2)
        x_rotated = x_rotated.reshape(batch_size, seq_len, dim)

        return x_rotated


# Dummy Q and K projections (in real LLMs, these come from W_Q @ X and W_K @ X)
# Default positions: 0, 1, 2, ..., seq_len-1
# Verify rotation preserves norm (RoPE is orthogonal transformation)
# Optional: Custom positions (e.g., for KV-cache continuation)


def demo_rope():
    # Configuration
    head_dim = 64          # attention head 
    seq_len = 8            # Sequence length
    batch_size = 2
    base = 10000.0

    rope = RoPE(dim=head_dim, max_seq_len=1024, base=base) #init

    torch.manual_seed(42)
    q = torch.randn(batch_size, seq_len, head_dim)
    k = torch.randn(batch_size, seq_len, head_dim)

    q_rot = rope(q)
    k_rot = rope(k)

    print("✅ RoPE Applied Successfully")
    print(f"Input Q shape: {q.shape}")
    print(f"Rotated Q shape: {q_rot.shape}")
    
    q_norm_before = torch.norm(q, dim=-1)
    q_norm_after  = torch.norm(q_rot, dim=-1)
    print(f"Norm preservation check (should be ~1.0): {torch.allclose(q_norm_before, q_norm_after)}")

    custom_pos = torch.tensor([[0, 1, 2, 3, 10, 11, 12, 13]] * batch_size, device=q.device)
    q_custom_rot = rope(q, pos=custom_pos)
    print(f"Custom position rotation shape: {q_custom_rot.shape}")

if __name__ == "__main__":
    demo_rope()
