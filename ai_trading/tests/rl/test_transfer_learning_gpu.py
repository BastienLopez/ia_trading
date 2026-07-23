import torch
from torch.utils.data import DataLoader, TensorDataset

from ai_trading.rl.models.transfer_learning import DomainAdaptation, MarketTransferLearning


def test_transfer_learning_and_domain_adaptation_run_real_gpu_updates():
    torch.manual_seed(7)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.ReLU(), torch.nn.Linear(8, 1))
    features = torch.randn(24, 4)
    targets = (features.sum(dim=1, keepdim=True) * 0.2)
    loader = DataLoader(TensorDataset(features, targets), batch_size=6, shuffle=False)

    transfer = MarketTransferLearning(model, fine_tune_layers=["2"], device="cuda")
    before = model[2].weight.detach().clone()
    history = transfer.fine_tune(loader, loader, epochs=2, early_stopping_patience=2)

    assert len(history["train_loss"]) == 2
    assert torch.isfinite(transfer.predict(features[:2])).all()
    assert not torch.equal(before, model[2].weight.detach())

    adaptation = DomainAdaptation(
        torch.nn.Sequential(torch.nn.Linear(4, 6), torch.nn.ReLU(), torch.nn.Linear(6, 1)),
        adaptation_type="dann",
        device="cuda",
    )
    metrics = adaptation.train_step(features[:8], targets[:8], features[8:16])
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["total_loss"] > 0


def test_coral_rejects_non_estimable_singleton_batches():
    adaptation = DomainAdaptation(torch.nn.Linear(4, 1), adaptation_type="coral", device="cuda")
    try:
        adaptation.train_step(torch.randn(1, 4), torch.randn(1, 1), torch.randn(1, 4))
    except ValueError as error:
        assert "au moins deux" in str(error)
    else:
        raise AssertionError("CORAL doit refuser un batch singleton")
