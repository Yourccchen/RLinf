"""Real token packing, compressor updates, and checkpoint round trip."""

import pytest
import torch

from rlinf.serving import rlt_features


def test_video_tokens_follow_view_time_space_order():
    spec = {"source": "video_condition", "temporal": "last", "spatial_pool": [1, 2]}
    first = torch.arange(16.0).reshape(2, 2, 2, 2)
    second = first + 100
    tokens = rlt_features.pack_features({"video_condition": [first, second]}, spec)
    expected = torch.tensor([[5., 13.], [6., 14.], [105., 113.], [106., 114.]])
    assert torch.equal(tokens, expected)
    with pytest.raises(ValueError, match="source"):
        rlt_features.pack_features({}, spec)


def test_feature_training_updates_and_reloads_encoder(tmp_path):
    spec = {"source": "text_condition"}
    paths = []
    for index in range(3):
        path = tmp_path / f"sample_{index}.pt"
        tokens = torch.arange(12.0).reshape(3, 4) / 12 + index
        rlt_features.save_feature_sample(path, tokens, spec, "base-A")
        paths.append(path)
    config = dict(input_dim=4, embed_dim=8, prefix_seq_len=3,
                  num_layers=1, num_heads=2, mlp_ratio=2.)
    checkpoint = tmp_path / "encoder.pt"
    metrics = rlt_features.train_encoder(
        paths, checkpoint, config, steps=3, batch_size=2, lr=1e-3, device="cpu", seed=5
    )
    assert len(metrics) == 3 and all(torch.isfinite(torch.tensor(metrics)))
    model, manifest = rlt_features.load_encoder(checkpoint, device="cpu")
    assert manifest["feature_identity"] == "base-A"
    assert manifest["feature_spec"] == spec
    assert all(not p.requires_grad for p in model.parameters())
    sample = torch.load(paths[0], weights_only=True)["tokens"][None]
    restored, _ = rlt_features.load_encoder(checkpoint, device="cpu")
    assert torch.equal(model.encode_flat(sample), restored.encode_flat(sample))
    earlier = tmp_path / "earlier.pt"
    rlt_features.train_encoder(paths, earlier, config, steps=1, batch_size=2,
                              lr=1e-3, device="cpu", seed=5)
    early_model, _ = rlt_features.load_encoder(earlier)
    assert not torch.equal(early_model.encode_flat(sample), model.encode_flat(sample))
    with pytest.raises(FileExistsError):
        rlt_features.train_encoder(paths, checkpoint, config, steps=1, device="cpu")


def test_mixed_feature_versions_fail_before_training(tmp_path):
    paths = [tmp_path / "a.pt", tmp_path / "b.pt"]
    for path, identity in zip(paths, ["base-A", "base-B"]):
        rlt_features.save_feature_sample(path, torch.ones(2, 4),
                                        {"source": "text_condition"}, identity)
    with pytest.raises(ValueError, match="identity"):
        rlt_features.train_encoder(paths, tmp_path / "out.pt", {}, steps=1, device="cpu")
