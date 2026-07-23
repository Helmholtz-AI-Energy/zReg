"""Smoke tests for train_label_transfer.py — D-02's own definition of done.

Wave-0 test file (47-CONTEXT.md D-02, 47-RESEARCH.md Validation Architecture
MODEL-05/07 row): proves the training loop mechanics work, not that training
converges. Four properties are checked, for BOTH models
(``PointNet2LabelTransfer``, ``EGNNLabelTransfer``):

1. forward + backward succeed (``train_step`` returns a finite loss and model
   parameters receive non-zero gradients);
2. loss decreases over a handful of steps (weak D-02 bar: average loss over
   the first few steps is higher than the average over the last few);
3. a checkpoint saves and reloads identically under
   ``torch.load(..., weights_only=True)`` (state_dict tensors compare equal
   via ``torch.equal``, metadata keys intact);
4. at least one iteration runs on MPS when available (Open Question 2) — a
   local, MPS-aware device fixture is defined in THIS file (Pitfall 4 /
   Open Question 1); ``tests/conftest.py``'s shared CUDA-only ``device``
   fixture is neither imported nor modified. When MPS is unavailable, the
   dedicated MPS test is skipped with an explicit reason rather than
   silently substituting CPU.
"""

# zreg (and scipy) must be imported before torch/torch_geometric on macOS ARM
# to avoid duplicate libomp initialisation (SIGABRT) -- 47-RESEARCH.md
# Pitfall 1, mirrors tests/conftest.py and train_label_transfer.py.
from zreg.dataset import zRegPointCloud  # noqa: F401

import math
import sys
from pathlib import Path

# Defensive repo-root sys.path insertion mirroring tests/conftest.py — this
# file is self-contained even if conftest's own insertion were ever skipped.
_repo_root = Path(__file__).parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import pytest
import torch

from eval.config import EvalConfig
from eval.data_factory import DataFactory
from train_label_transfer import _build_parser, main, resolve_device, save_checkpoint, train_step
from zreg.models import EGNNLabelTransfer, PointNet2LabelTransfer

N_CLASSES = 4
MODEL_NAMES = ["pointnet2", "egnn"]


def _hyperparams(model_name: str) -> dict:
    """Small, fast-for-tests constructor kwargs — not the production defaults."""
    if model_name == "pointnet2":
        return {"n_classes": N_CLASSES, "hidden_dim": 16}
    return {"n_classes": N_CLASSES, "hidden_dim": 16, "n_layers": 2}


def _make_model(model_name: str) -> torch.nn.Module:
    cls = PointNet2LabelTransfer if model_name == "pointnet2" else EGNNLabelTransfer
    return cls(**_hyperparams(model_name))


def _make_factory() -> DataFactory:
    # generate_training_triple/generate_training_set never call load_real()/
    # load_target(), so data_path is never actually read from disk here.
    config = EvalConfig(data_path="unused-tests-do-not-load-real-data")
    return DataFactory(config)


def _available_devices() -> list[str]:
    """cpu always; mps too when torch.backends.mps.is_available() (Pitfall 4)."""
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


@pytest.fixture(params=_available_devices())
def local_device(request) -> str:
    """Local MPS-aware device fixture (Open Question 1).

    Deliberately NOT the shared ``tests/conftest.py::device`` fixture, which
    is CUDA-vs-CPU only (Pitfall 4) and is left untouched. Yields "cpu"
    always, and "mps" too on hosts where MPS is available.
    """
    return request.param


# ---------------------------------------------------------------------------
# resolve_device
# ---------------------------------------------------------------------------


def test_resolve_device_explicit_arg_takes_precedence():
    assert resolve_device("cpu") == torch.device("cpu")


def test_resolve_device_auto_detect_returns_known_backend():
    device = resolve_device(None)
    assert device.type in ("cuda", "mps", "cpu")


def test_resolve_device_falls_back_to_cpu_when_no_accelerator():
    """resolve_device(None) returns cpu when neither CUDA nor MPS is available (line 112)."""
    from unittest.mock import patch
    with patch("torch.cuda.is_available", return_value=False), \
         patch("torch.backends.mps.is_available", return_value=False):
        device = resolve_device(None)
    assert device == torch.device("cpu")


# ---------------------------------------------------------------------------
# Forward + backward (Property 1) — parametrized over model AND local_device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_train_step_forward_backward(model_name, local_device):
    device = torch.device(local_device)
    torch.manual_seed(0)
    model = _make_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    factory = _make_factory()
    triple = factory.generate_training_triple(seed=0, n_classes=N_CLASSES)

    loss = train_step(model, triple, optimizer, device, N_CLASSES)

    assert math.isfinite(loss)
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "no gradients populated after loss.backward()"
    assert any(g.abs().sum().item() > 0 for g in grads), "all gradients are exactly zero"
    print(f"\n[smoke] {model_name} train_step ran on device={device}, loss={loss:.4f}")


# ---------------------------------------------------------------------------
# Loss decreases over a handful of steps (Property 2, D-02's weak bar)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_loss_decreases_over_a_handful_of_steps(model_name):
    device = torch.device("cpu")
    torch.manual_seed(0)
    model = _make_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    factory = _make_factory()
    triples = factory.generate_training_set(range(16), n_classes=N_CLASSES)

    losses = [train_step(model, t, optimizer, device, N_CLASSES) for t in triples]

    first_avg = sum(losses[:4]) / 4
    last_avg = sum(losses[-4:]) / 4
    print(f"\n[smoke] {model_name} losses: {losses}")
    assert last_avg < first_avg, (
        f"{model_name}: loss did not trend downward over {len(losses)} steps "
        f"(first-4 avg={first_avg:.4f}, last-4 avg={last_avg:.4f})"
    )


# ---------------------------------------------------------------------------
# Checkpoint round-trip under weights_only=True (Property 3, Pattern 6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_checkpoint_round_trip(model_name, tmp_path):
    torch.manual_seed(0)
    model = _make_model(model_name)
    hyperparams = _hyperparams(model_name)
    path = tmp_path / f"{model_name}.pt"

    save_checkpoint(model, model_name, hyperparams, epoch=3, path=str(path))
    loaded = torch.load(path, weights_only=True)

    assert loaded["model_class"] == model_name
    assert loaded["epoch"] == 3
    assert loaded["hyperparams"] == hyperparams
    original_state = model.state_dict()
    assert set(original_state.keys()) == set(loaded["model_state_dict"].keys())
    for key, tensor in original_state.items():
        assert torch.equal(tensor, loaded["model_state_dict"][key]), f"tensor mismatch for {key!r}"


# ---------------------------------------------------------------------------
# Explicit MPS iteration (Property 4, Open Question 2)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS unavailable on this host — MPS validation not performed",
)
@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_mps_train_step_runs_on_mps(model_name):
    """Run one train_step with device='mps' and assert a tensor lands on mps.

    Any MPS-specific failure (e.g. an unimplemented kernel) is allowed to
    fail this test loudly — it is a documented finding (Open Question 2),
    never silently worked around by falling back to CPU.
    """
    device = torch.device("mps")
    torch.manual_seed(0)
    model = _make_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    factory = _make_factory()
    triple = factory.generate_training_triple(seed=0, n_classes=N_CLASSES)

    loss = train_step(model, triple, optimizer, device, N_CLASSES)

    assert math.isfinite(loss)
    landed_device = next(model.parameters()).device
    assert landed_device.type == "mps", f"expected mps, got {landed_device.type}"
    print(f"\n[smoke] {model_name} MPS iteration OK, device={landed_device}, loss={loss:.4f}")


# ---------------------------------------------------------------------------
# _build_parser and main (CLI entry point)
# ---------------------------------------------------------------------------


def test_build_parser_accepts_required_args():
    parser = _build_parser()
    args = parser.parse_args(["--model", "pointnet2"])
    assert args.model == "pointnet2"
    assert args.epochs == 1
    assert args.n_seeds == 8
    assert args.n_classes == 6
    assert args.hidden_dim == 32
    assert args.checkpoint_dir == "checkpoints"


def test_build_parser_rejects_unknown_model(capsys):
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--model", "unknown"])


@pytest.mark.parametrize("model_name", MODEL_NAMES)
def test_main_runs_and_saves_checkpoint(model_name, tmp_path):
    rc = main([
        "--model", model_name,
        "--epochs", "1",
        "--n-seeds", "2",
        "--n-classes", "4",
        "--hidden-dim", "8",
        "--checkpoint-dir", str(tmp_path / "ckpts"),
    ])
    assert rc == 0
    assert len(list((tmp_path / "ckpts").glob("*.pt"))) == 1


def test_main_validates_epochs_zero():
    with pytest.raises(ValueError, match="--epochs"):
        main(["--model", "pointnet2", "--epochs", "0", "--checkpoint-dir", "/tmp"])


def test_main_validates_n_seeds_zero():
    with pytest.raises(ValueError, match="--n-seeds"):
        main(["--model", "pointnet2", "--n-seeds", "0", "--checkpoint-dir", "/tmp"])


def test_main_validates_n_classes_one():
    with pytest.raises(ValueError, match="--n-classes"):
        main(["--model", "pointnet2", "--n-classes", "1", "--checkpoint-dir", "/tmp"])
