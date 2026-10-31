from __future__ import annotations

import hashlib
import importlib

import pytest
torch = pytest.importorskip("torch")
nn = torch.nn
reid = importlib.import_module("crosscam.reid")
OSNetConfig = reid.OSNetConfig
OSNetEncoder = reid.OSNetEncoder


class ToyModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(128, 512), nn.BatchNorm1d(512), nn.ReLU())
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, images):
        pooled = images.mean(dim=(2, 3)).repeat(1, 43)[:, :128]
        return self.fc(pooled)


def make_encoder(tmp_path, monkeypatch, *, device="cpu", batch_size=2):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    source_model = ToyModel(num_classes=3)
    checkpoint = tmp_path / "toy.pth"
    torch.save(
        {
            "fc.0.weight": source_model.fc[0].weight,
            "fc.0.bias": source_model.fc[0].bias,
            "fc.1.weight": source_model.fc[1].weight,
            "fc.1.bias": source_model.fc[1].bias,
            "fc.1.running_mean": source_model.fc[1].running_mean,
            "fc.1.running_var": source_model.fc[1].running_var,
            "fc.1.num_batches_tracked": source_model.fc[1].num_batches_tracked,
            "classifier.weight": source_model.classifier.weight,
            "classifier.bias": source_model.classifier.bias,
        },
        checkpoint,
    )
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    encoder = OSNetEncoder(
        OSNetConfig(
            checkpoint_path=checkpoint,
            expected_sha256=digest,
            device=device,
            batch_size=batch_size,
        ),
        model_factory=lambda _: ToyModel(num_classes=3),
    )
    return encoder, checkpoint, digest


def rgb_batch(count=2):
    return torch.rand(count, 3, 256, 128)


def test_cpu_encoder_returns_finite_l2_normalized_512d_features(tmp_path, monkeypatch):
    encoder, _, _ = make_encoder(tmp_path, monkeypatch)
    result = encoder.encode(rgb_batch(3))
    assert result.shape == (3, 512)
    assert result.device.type == "cpu"
    assert torch.isfinite(result).all()
    torch.testing.assert_close(
        torch.linalg.vector_norm(result, dim=1), torch.ones(3), atol=1e-6, rtol=0
    )
    assert encoder.metadata["device"] == "cpu"
    assert encoder.metadata["l2_normalized"] is True


@pytest.mark.parametrize(
    "bad, error",
    [
        (torch.rand(3, 256, 128), ValueError),
        (torch.rand(1, 1, 256, 128), ValueError),
        (torch.rand(0, 3, 256, 128), ValueError),
        (torch.rand(1, 3, 255, 128), ValueError),
        (torch.rand(1, 3, 256, 127), ValueError),
        (torch.ones(1, 3, 256, 128, dtype=torch.uint8), TypeError),
    ],
)
def test_rejects_invalid_shapes_and_dtypes(tmp_path, monkeypatch, bad, error):
    encoder, _, _ = make_encoder(tmp_path, monkeypatch)
    with pytest.raises(error):
        encoder.encode(bad)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -0.01, 1.01])
def test_rejects_nonfinite_or_out_of_range_pixels(tmp_path, monkeypatch, value):
    encoder, _, _ = make_encoder(tmp_path, monkeypatch)
    images = rgb_batch(1)
    images[0, 0, 0, 0] = value
    with pytest.raises(ValueError):
        encoder.encode(images)


def test_batches_are_bounded(tmp_path, monkeypatch):
    encoder, _, _ = make_encoder(tmp_path, monkeypatch, batch_size=2)
    assert encoder.encode(rgb_batch(5)).shape == (5, 512)


def test_rejects_zero_norm_features(tmp_path, monkeypatch):
    encoder, _, _ = make_encoder(tmp_path, monkeypatch)

    class ZeroFeatures(nn.Module):
        def forward(self, images):
            return torch.zeros(images.shape[0], 512)

    encoder._model = ZeroFeatures()
    with pytest.raises(RuntimeError, match="zero-norm"):
        encoder.encode(rgb_batch(1))


def test_missing_checkpoint_fails(tmp_path):
    with pytest.raises(FileNotFoundError):
        OSNetEncoder(OSNetConfig(tmp_path / "missing.pth"))


def test_bad_checkpoint_hash_fails_before_load(tmp_path, monkeypatch):
    _, checkpoint, _ = make_encoder(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="SHA-256"):
        OSNetEncoder(
            OSNetConfig(checkpoint, expected_sha256="0" * 64, device="cpu"),
            model_factory=lambda _: ToyModel(3),
        )


def test_cuda_request_fails_clearly_when_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    _, checkpoint, digest = make_encoder(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="CUDA was requested"):
        OSNetEncoder(
            OSNetConfig(checkpoint, expected_sha256=digest, device="cuda"),
            model_factory=lambda _: ToyModel(3),
        )


def test_auto_device_uses_cpu_fallback(tmp_path, monkeypatch):
    encoder, _, _ = make_encoder(tmp_path, monkeypatch, device="auto")
    assert encoder.device.type == "cpu"


def test_rejects_invalid_expected_hash_and_batch_size(tmp_path):
    with pytest.raises(ValueError, match="expected_sha256"):
        OSNetConfig(tmp_path / "unused", expected_sha256="not-a-hash")
    with pytest.raises(ValueError, match="batch_size"):
        OSNetConfig(tmp_path / "unused", batch_size=0)

