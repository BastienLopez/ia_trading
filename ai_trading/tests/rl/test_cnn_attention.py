import torch

from ai_trading.rl.models.cnn_model import HybridCNNAttention


def test_hybrid_cnn_attention_preserves_and_attends_to_time_axis_on_gpu():
    model = HybridCNNAttention(5, 8, 2, cnn_filters=[8], cnn_kernel_sizes=[3], num_heads=2).cuda()
    output, attention = model(torch.randn(3, 5, 12, device="cuda"))
    assert output.shape == (3, 2)
    assert attention["temporal"].shape[-2:] == (12, 12)
