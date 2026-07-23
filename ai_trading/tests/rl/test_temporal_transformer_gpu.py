import torch

from ai_trading.rl.models.temporal_transformer import FinancialTemporalTransformer


def test_temporal_transformer_gpu_forward_attention_and_gradient():
    model = FinancialTemporalTransformer(5, d_model=16, nhead=2, num_layers=1, dim_feedforward=32, max_seq_len=12).cuda()
    output, attention = model(torch.randn(2, 8, 5, device="cuda"))
    output.square().mean().backward()
    assert output.shape == (2, 1)
    assert len(attention) == 1
    assert any(parameter.grad is not None for parameter in model.parameters())
