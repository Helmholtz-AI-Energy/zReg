"""Training entry point for eGNN/PointNet++ label-transfer models (Phase 47).

Usage::

    python train_label_transfer.py --model pointnet2 --epochs 3 --n-seeds 4 \
        --device cpu --checkpoint-dir checkpoints

    python train_label_transfer.py --model egnn --epochs 3 --n-seeds 4 \
        --device cpu --checkpoint-dir checkpoints

Mirrors ``run_eval.py``'s repo-root/argparse CLI convention (47-RESEARCH.md
Open Question 3): ``_repo_root``/``_src_root`` sys.path injection,
``_build_parser()``, ``main(argv) -> int``, ``if __name__ == "__main__":
sys.exit(main())``.

Definition of done for this phase (47-CONTEXT.md D-02): this entry point
delivers device-agnostic model classes, a training loop, and checkpoint
save/load code, smoke-tested only (a handful of iterations on CPU/MPS/small
data). It does NOT run a full/converged training run — that happens on the
user's external CUDA cluster, consuming whatever checkpoint this script's
``--checkpoint-dir`` produces.

Device handling (D-01, 47-RESEARCH.md Pattern 2): ``resolve_device`` is
MPS-aware (cuda -> mps -> cpu); no bare CUDA-only device call (the dunder
``cuda`` tensor method) exists anywhere in this file. The process-global
device-configuration helper documented in ``src/zreg/config.py`` is
intentionally NOT used here -- it sets a process-wide default device, which
is the wrong tool for a script that threads ``device`` explicitly through
every call site.

Training loop (47-RESEARCH.md Pattern 5): per-triple (batch-size-1) forward
passes over ``DataFactory.generate_training_set(seeds, n_classes)`` -- Phase
46's synthetic training-triple pipeline. ``train_step`` builds a joint
(source+target-concatenated) cloud, one-hot-encodes source labels into an
``n_classes``-wide feature block with a trailing "unknown" flag bit for
target rows, and computes a masked cross-entropy loss over
``logits[n_source:]`` (the target-only supervised subset) vs.
``triple.target_labels`` -- read exclusively via the ``TrainingTriple``
label properties, never the raw cell-identity field on the target cloud
(label-vs-id discipline, 45-DESIGN.md).

Checkpoint format (47-RESEARCH.md Pattern 6): a plain dict of
``model_state_dict`` (``model.state_dict()``), ``model_class`` (str
dispatch key for Phase 48), ``hyperparams`` (dict of primitives), and
``epoch`` (int) -- no custom classes, so Phase 48's mandated
``torch.load(..., weights_only=True)`` needs no ``add_safe_globals``
registration.
"""

# sys.path injection — train_label_transfer.py is at repo root (mirrors
# run_eval.py's convention exactly).
import sys
from pathlib import Path

_repo_root = Path(__file__).parent
if str(_repo_root) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(_repo_root))
_src_root = _repo_root / "src"
if str(_src_root) not in sys.path:  # pragma: no cover
    sys.path.insert(0, str(_src_root))

# zreg (a scipy-importing module) must be imported before torch/torch_geometric
# on macOS ARM to avoid a duplicate libomp initialisation SIGABRT
# (47-RESEARCH.md Pitfall 1) — mirrors eval/data_factory.py and
# src/zreg/models/{pointnet2,egnn}.py's own import-order guard.
from zreg.dataset import zRegPointCloud  # noqa: F401

import argparse
from datetime import datetime

import torch
import torch.nn.functional as F

from eval.config import EvalConfig
from eval.data_factory import DataFactory
from zreg.models import EGNNLabelTransfer, PointNet2LabelTransfer

__all__ = ["resolve_device", "train_step", "save_checkpoint", "main"]

MODEL_REGISTRY = {"pointnet2": PointNet2LabelTransfer, "egnn": EGNNLabelTransfer}


# ---------------------------------------------------------------------------
# Device resolution
# ---------------------------------------------------------------------------


def resolve_device(arg: str | None) -> torch.device:
    """MPS-aware device resolution (47-RESEARCH.md Pattern 2).

    Never calls the process-global device-configuration helper documented in
    ``src/zreg/config.py`` (an opt-in utility, not a per-module convention)
    and never calls a bare CUDA-only device method — D-01.

    Parameters
    ----------
    arg : str or None
        Explicit ``--device`` CLI value (e.g. ``"cpu"``, ``"cuda"``,
        ``"mps"``). When ``None``, auto-detects: CUDA -> MPS -> CPU.

    Returns
    -------
    torch.device
        The resolved device.
    """
    if arg is not None:
        return torch.device(arg)
    if torch.cuda.is_available():  # pragma: no cover
        return torch.device("cuda")
    if torch.backends.mps.is_available():  # pragma: no cover
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Training step (shared, model-agnostic — 47-RESEARCH.md Pattern 5)
# ---------------------------------------------------------------------------


def train_step(model, triple, optimizer, device, n_classes: int) -> float:
    """One masked-cross-entropy training step over a single TrainingTriple.

    Builds the joint (source+target-concatenated) cloud, one-hot-encodes
    source labels into the first ``n_classes`` feature dims, sets the
    trailing "unknown" flag bit for target rows, runs ``model.forward``, and
    computes ``cross_entropy`` over the target-only logit subset
    (``logits[n_source:]``) vs. ``triple.target_labels``. Labels are read
    exclusively via ``TrainingTriple.source_labels``/``target_labels``
    (never the raw cell-identity field on the target cloud — label-vs-id
    discipline, 45-DESIGN.md).

    Parameters
    ----------
    model : torch.nn.Module
        ``PointNet2LabelTransfer`` or ``EGNNLabelTransfer`` instance, already
        moved to ``device``.
    triple : eval.types.TrainingTriple
        One (source, target) training example.
    optimizer : torch.optim.Optimizer
        Optimizer stepping ``model.parameters()``.
    device : torch.device
        Device to run the forward/backward pass on.
    n_classes : int
        Number of label classes (joint-cloud feature width is
        ``n_classes + 1``).

    Returns
    -------
    float
        The scalar loss value for this step.
    """
    source_pos = triple.source_cloud["pos"].to(device)
    source_labels = triple.source_labels.to(device).long()
    target_pos = triple.target_cloud["pos"].to(device)
    target_labels = triple.target_labels.to(device).long()

    n_source = source_pos.shape[0]
    joint_pos = torch.cat([source_pos, target_pos], dim=0)
    joint_feat = torch.zeros(joint_pos.shape[0], n_classes + 1, device=device)
    joint_feat[:n_source, :n_classes] = F.one_hot(source_labels, num_classes=n_classes).float()
    joint_feat[n_source:, -1] = 1.0  # "unknown" flag for target rows

    logits = model(joint_pos, joint_feat)  # (n_joint, n_classes)
    target_logits = logits[n_source:]
    loss = F.cross_entropy(target_logits, target_labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Checkpoint I/O (47-RESEARCH.md Pattern 6)
# ---------------------------------------------------------------------------


def save_checkpoint(model, model_class: str, hyperparams: dict, epoch: int, path: str) -> None:
    """Save a ``weights_only=True``-compatible checkpoint (Pattern 6).

    Writes a plain dict — ``model_state_dict`` + primitive metadata, never a
    custom class or the model object itself — so Phase 48's mandated
    ``torch.load(..., weights_only=True)`` load needs no ``add_safe_globals``
    registration (T-47-07).

    Parameters
    ----------
    model : torch.nn.Module
        The model whose ``state_dict()`` is saved.
    model_class : str
        Dispatch key for Phase 48 — ``"pointnet2"`` or ``"egnn"``.
    hyperparams : dict
        Plain-Python constructor kwargs (e.g. ``n_classes``, ``hidden_dim``)
        needed to reconstruct the model before loading the state dict.
    epoch : int
        The epoch number this checkpoint was saved at.
    path : str
        Destination file path. Parent directories are created if missing.
    """
    ckpt = {
        "model_state_dict": model.state_dict(),
        "model_class": model_class,
        "hyperparams": dict(hyperparams),
        "epoch": epoch,
    }
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, dest)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser (V5 input validation)."""
    p = argparse.ArgumentParser(
        description="Train a PointNet++ or eGNN label-transfer model (Phase 47 smoke-test scaffold)"
    )
    p.add_argument(
        "--model",
        required=True,
        choices=["pointnet2", "egnn"],
        help="Model architecture to train",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Device override (cpu/cuda/mps). Auto-detected (cuda->mps->cpu) if omitted.",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of passes over the training seed set (default 1)",
    )
    p.add_argument(
        "--n-seeds",
        type=int,
        default=8,
        help="Number of training seeds (triples) per epoch (default 8)",
    )
    p.add_argument(
        "--n-classes",
        type=int,
        default=6,
        help="Number of label classes (default 6, matches DataFactory's default)",
    )
    p.add_argument(
        "--hidden-dim",
        type=int,
        default=32,
        help="Model hidden feature width (default 32)",
    )
    p.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory to write checkpoint .pt files (default 'checkpoints')",
    )
    return p


def main(argv=None) -> int:
    """Parse arguments, run a per-triple training loop, and save a checkpoint.

    Parameters
    ----------
    argv : list[str] or None
        Argument list. ``None`` reads from ``sys.argv[1:]`` (production); an
        explicit list is used by tests.

    Returns
    -------
    int
        0 on success.

    Raises
    ------
    ValueError
        If ``--epochs`` or ``--n-seeds`` is not a positive integer (V5 input
        validation, beyond what argparse's ``choices``/``type`` alone catch).
    """
    args = _build_parser().parse_args(argv)

    if args.epochs < 1:
        raise ValueError(f"--epochs must be >= 1, got {args.epochs}")
    if args.n_seeds < 1:
        raise ValueError(f"--n-seeds must be >= 1, got {args.n_seeds}")
    if args.n_classes < 2:
        raise ValueError(f"--n-classes must be >= 2, got {args.n_classes}")

    device = resolve_device(args.device)

    hyperparams: dict = {"n_classes": args.n_classes, "hidden_dim": args.hidden_dim}
    if args.model == "egnn":
        hyperparams["n_layers"] = 4  # 47-RESEARCH.md Pattern 4 Assumption A1 default

    model_cls = MODEL_REGISTRY[args.model]
    model = model_cls(**hyperparams).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Assumption A2

    # DataFactory only needs a syntactically-valid EvalConfig here —
    # generate_training_set/generate_training_triple never call load_real()/
    # load_target(), so data_path is never actually read from disk.
    config = EvalConfig(data_path="unused-training-entry-point-does-not-load-real-data")
    factory = DataFactory(config)

    epoch = 0
    for epoch in range(args.epochs):
        seeds = range(epoch * args.n_seeds, (epoch + 1) * args.n_seeds)
        triples = factory.generate_training_set(seeds, n_classes=args.n_classes)
        for triple in triples:
            train_step(model, triple, optimizer, device, args.n_classes)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    checkpoint_path = str(Path(args.checkpoint_dir) / f"{args.model}_{timestamp}.pt")
    save_checkpoint(model, args.model, hyperparams, epoch, checkpoint_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
