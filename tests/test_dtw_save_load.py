"""DTW save/load round-trips callable distance metrics (CPD-07, U2-8).

On baseline 6c1c37f ``DynamicTimeWarping.save()`` stored the distance-metric callable
itself in the ``.pt`` config, so ``torch.load(path, weights_only=True)`` (used by
``load()``) raised ``UnpicklingError``; ``load()`` also dropped the config entirely.

After the fix a durable module-level callable is stored as
``{"callable": "module:qualname"}`` and restored by importing its module; lambdas,
local functions, ``__main__`` callables, ``nn.Module`` instances and references that do
not resolve back to the same object are rejected with ``TypeError`` before any file is
written. ``load()`` is for trusted files only (importing a module executes code).
"""

import importlib
import os
import pickle
import subprocess
import sys
import types
from pathlib import Path

import pytest

import zreg  # noqa: F401  (import zreg before torch)
import torch

from zreg.algorithms.dtw import DynamicTimeWarping
from zreg.core.dataset import zRegPointCloud
from zreg.distance_metrics import euclidean_distance, SlicedWassersteinDistance


REPO_ROOT = Path(__file__).resolve().parent.parent
EUCLIDEAN_REF = f"{euclidean_distance.__module__}:{euclidean_distance.__qualname__}"


@pytest.fixture
def small_trajectory_pair():
    """Create very small trajectories for fast testing."""
    gen = torch.Generator().manual_seed(0)
    x = {}
    for i in range(3):
        x[i] = zRegPointCloud(
            pos=torch.randn(10, 3, generator=gen),
            label=torch.rand(10, 3, generator=gen),
            id=torch.arange(10),
        )
    y = {}
    for i in range(3):
        y[i] = zRegPointCloud(
            pos=torch.randn(10, 3, generator=gen),
            label=torch.rand(10, 3, generator=gen),
            id=torch.arange(10),
        )
    return x, y


def _computed(x, y, metric):
    dtw_obj = DynamicTimeWarping(x, y, distance_metric=metric, downsample_method=None, cpd_type=None)
    dtw_obj.compute()
    return dtw_obj


def _assert_nothing_written(path):
    assert not path.exists()
    assert not Path(str(path) + ".transforms.pkl").exists()


class TestCallableMetricRoundTrip:
    def test_save_load_callable_metric_round_trip(self, small_trajectory_pair, tmp_path):
        """Fails on 6c1c37f: torch.load(weights_only=True) raises UnpicklingError."""
        x, y = small_trajectory_pair
        path = tmp_path / "r.pt"
        _computed(x, y, euclidean_distance).save(path)
        raw = torch.load(path, weights_only=True)
        assert raw["config"]["distance_metric"] == {"callable": EUCLIDEAN_REF}
        loaded = DynamicTimeWarping.load(path)
        assert loaded.config["distance_metric"] is euclidean_distance

    def test_save_load_callable_metric_fresh_process(self, small_trajectory_pair, tmp_path):
        """Fails on 6c1c37f: the fresh interpreter cannot load the file (UnpicklingError)."""
        x, y = small_trajectory_pair
        path = tmp_path / "r.pt"
        _computed(x, y, euclidean_distance).save(path)
        code = (
            "import sys\n"
            "import zreg\n"
            "from zreg.algorithms.dtw import DynamicTimeWarping\n"
            "res = DynamicTimeWarping.load(sys.argv[1])\n"
            "fn = res.config['distance_metric']\n"
            "print(f'{fn.__module__}:{fn.__qualname__}')\n"
        )
        env = os.environ.copy()
        src = str(REPO_ROOT / "src")
        env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        proc = subprocess.run(
            [sys.executable, "-c", code, str(path)],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert proc.returncode == 0, proc.stderr
        assert proc.stdout.strip().splitlines()[-1] == "zreg.distance_metrics.general:euclidean_distance"

    def test_save_load_metric_list_round_trip(self, small_trajectory_pair, tmp_path):
        """Fails on 6c1c37f (UnpicklingError on the callable inside the list)."""
        x, y = small_trajectory_pair
        path = tmp_path / "r.pt"
        _computed(x, y, ["euclidean", euclidean_distance]).save(path)
        loaded = DynamicTimeWarping.load(path)
        metric = loaded.config["distance_metric"]
        assert metric[0] == "euclidean"
        assert metric[1] is euclidean_distance
        assert len(metric) == 2

    def test_save_load_string_metric_config(self, small_trajectory_pair, tmp_path):
        """Fails on 6c1c37f: DTWResult had no config field."""
        x, y = small_trajectory_pair
        path = tmp_path / "r.pt"
        _computed(x, y, "euclidean").save(path)
        loaded = DynamicTimeWarping.load(path)
        assert loaded.config["distance_metric"] == "euclidean"
        assert loaded.config["cpd_type"] is None

    def test_load_legacy_file_without_config(self, small_trajectory_pair, tmp_path):
        """Files without a config key still load; config is None."""
        x, y = small_trajectory_pair
        path = tmp_path / "r.pt"
        _computed(x, y, "euclidean").save(path)
        data = torch.load(path, weights_only=True)
        del data["config"]
        torch.save(data, path)
        loaded = DynamicTimeWarping.load(path)
        assert loaded.config is None
        assert loaded.warping_path[0] == (0, 0)


class TestNonDurableMetricRejected:
    def test_save_rejects_lambda_before_writing(self, small_trajectory_pair, tmp_path):
        """Fails on 6c1c37f: save() silently pickled the lambda (no TypeError)."""
        x, y = small_trajectory_pair
        dtw_obj = _computed(x, y, lambda a, b: euclidean_distance(a, b))
        path = tmp_path / "r.pt"
        with pytest.raises(TypeError, match="distance_metric"):
            dtw_obj.save(path)
        _assert_nothing_written(path)

    def test_save_rejects_local_function_before_writing(self, small_trajectory_pair, tmp_path):
        """Fails on 6c1c37f: local functions were pickled (no TypeError)."""

        def local_metric(a, b):
            return euclidean_distance(a, b)

        assert "<locals>" in local_metric.__qualname__
        x, y = small_trajectory_pair
        dtw_obj = _computed(x, y, local_metric)
        path = tmp_path / "r.pt"
        with pytest.raises(TypeError, match="distance_metric"):
            dtw_obj.save(path)
        _assert_nothing_written(path)

    def test_save_rejects_module_instance_before_writing(self, small_trajectory_pair, tmp_path):
        """Fails on 6c1c37f: nn.Module metric instances were pickled (no TypeError)."""
        x, y = small_trajectory_pair
        metric = SlicedWassersteinDistance(num_projs=10)
        assert isinstance(metric, torch.nn.Module)
        dtw_obj = DynamicTimeWarping(x, y, distance_metric=metric, downsample_method="random", cpd_type=None)
        dtw_obj.compute()
        path = tmp_path / "r.pt"
        with pytest.raises(TypeError, match="distance_metric"):
            dtw_obj.save(path)
        _assert_nothing_written(path)

    def test_save_rejects_main_module_callable_before_writing(self, small_trajectory_pair, tmp_path, monkeypatch):
        """A __main__ callable is rejected even though its reference resolves in-process.

        Fails on 6c1c37f: save() wrote the file without complaint.
        """
        fn = types.FunctionType(
            euclidean_distance.__code__,
            euclidean_distance.__globals__,
            "euclidean_distance_main",
            euclidean_distance.__defaults__,
        )
        fn.__module__ = "__main__"
        fn.__qualname__ = "euclidean_distance_main"
        monkeypatch.setattr(sys.modules["__main__"], "euclidean_distance_main", fn, raising=False)
        x, y = small_trajectory_pair
        dtw_obj = _computed(x, y, fn)
        path = tmp_path / "r.pt"
        with pytest.raises(TypeError, match="distance_metric"):
            dtw_obj.save(path)
        _assert_nothing_written(path)


class TestMetricRefHelpers:
    def test_metric_ref_helpers_direct(self):
        """Fails on 6c1c37f: helpers do not exist (ImportError inside the test)."""
        from zreg.algorithms.dtw.core import _metric_to_ref, _ref_to_metric

        ref = _metric_to_ref(euclidean_distance)
        assert ref == {"callable": "zreg.distance_metrics.general:euclidean_distance"}
        assert _ref_to_metric(ref) is euclidean_distance
        assert _ref_to_metric("euclidean") == "euclidean"
        assert _metric_to_ref("euclidean") == "euclidean"

    @pytest.mark.parametrize(
        "ref",
        [
            {"callable": "no_colon_here"},
            {"callable": "a:b:c"},
            {"callable": "zreg_no_such_module_xyz:f"},
            {"callable": "zreg.distance_metrics.general:does_not_exist"},
        ],
    )
    def test_ref_to_metric_malformed_reference_rejected(self, ref):
        """Fails on 6c1c37f: helper does not exist (ImportError inside the test)."""
        from zreg.algorithms.dtw.core import _ref_to_metric

        with pytest.raises(ValueError, match="callable reference"):
            _ref_to_metric(ref)


class TestLoadMetricResolution:
    """WR-02: an unresolvable metric reference does not make the result unloadable,
    and ``resolve_metric=False`` reads the file without importing anything."""

    def _save_with_ref(self, x, y, path, ref):
        _computed(x, y, euclidean_distance).save(path)
        data = torch.load(path, weights_only=True)
        data["config"]["distance_metric"] = ref
        torch.save(data, path)

    def test_unresolvable_reference_keeps_tensors_and_reference(
        self, small_trajectory_pair, tmp_path, caplog
    ):
        """Fails before WR-02: load() raised ValueError and the tensors were lost."""
        x, y = small_trajectory_pair
        path = tmp_path / "r.pt"
        ref = {"callable": "zreg.distance_metrics.general:renamed_since_saving"}
        self._save_with_ref(x, y, path, ref)
        expected = torch.load(path, weights_only=True)
        with caplog.at_level("WARNING", logger="zreg.algorithms.dtw.core"):
            loaded = DynamicTimeWarping.load(path)
        assert loaded.config["distance_metric"] == ref
        assert torch.equal(loaded.cost_matrix, expected["cost_matrix"])
        assert loaded.warping_path == expected["warping_path"]
        assert any("not restored" in r.getMessage() for r in caplog.records)
        with pytest.raises(ValueError, match="callable reference"):
            DynamicTimeWarping.resolve_metric(loaded.config["distance_metric"])

    def test_resolve_metric_false_imports_nothing(
        self, small_trajectory_pair, tmp_path, monkeypatch
    ):
        """Fails before WR-02: load() always imported the module named in the file."""
        x, y = small_trajectory_pair
        mod_dir = tmp_path / "mods"
        mod_dir.mkdir()
        mod_name = "zreg_wr02_side_effect_mod"
        (mod_dir / f"{mod_name}.py").write_text(
            "IMPORTED = True\n\n\ndef metric(a, b):\n    return 0.0\n"
        )
        monkeypatch.syspath_prepend(str(mod_dir))
        monkeypatch.delitem(sys.modules, mod_name, raising=False)
        path = tmp_path / "r.pt"
        ref = {"callable": f"{mod_name}:metric"}
        self._save_with_ref(x, y, path, ref)

        loaded = DynamicTimeWarping.load(path, resolve_metric=False)
        assert mod_name not in sys.modules
        assert loaded.config["distance_metric"] == ref
        assert loaded.warping_path[0] == (0, 0)

        fn = DynamicTimeWarping.resolve_metric(loaded.config["distance_metric"])
        assert mod_name in sys.modules
        assert fn is sys.modules[mod_name].metric
        monkeypatch.delitem(sys.modules, mod_name, raising=False)

    def test_resolve_metric_false_does_not_unpickle_companion(
        self, small_trajectory_pair, tmp_path, monkeypatch
    ):
        """WR-02 (iteration 2): the safe mode also skips the ``.transforms.pkl``.

        Before, ``load(path, resolve_metric=False)`` still unpickled the companion,
        so a crafted ``__reduce__`` ran code although the docstring promised that
        nothing is imported.
        """
        x, y = small_trajectory_pair
        mod_dir = tmp_path / "mods"
        mod_dir.mkdir()
        mod_name = "zreg_wr02_pickle_payload_mod"
        marker = tmp_path / "payload_ran"
        (mod_dir / f"{mod_name}.py").write_text(
            "import pathlib\n\n\n"
            "def mark(p):\n"
            "    pathlib.Path(p).write_text('ran')\n"
            "    return {}\n\n\n"
            "class Payload:\n"
            "    def __init__(self, p):\n"
            "        self.p = p\n\n"
            "    def __reduce__(self):\n"
            "        return (mark, (self.p,))\n"
        )
        monkeypatch.syspath_prepend(str(mod_dir))
        monkeypatch.delitem(sys.modules, mod_name, raising=False)
        path = tmp_path / "r.pt"
        _computed(x, y, euclidean_distance).save(path)
        mod = importlib.import_module(mod_name)
        Path(str(path) + ".transforms.pkl").write_bytes(
            pickle.dumps(mod.Payload(str(marker)))
        )
        del mod
        monkeypatch.delitem(sys.modules, mod_name, raising=False)

        loaded = DynamicTimeWarping.load(path, resolve_metric=False)
        assert not marker.exists()
        assert mod_name not in sys.modules
        assert loaded.stored_transforms == {}
        assert loaded.warping_path[0] == (0, 0)

        # The trusted paths still unpickle the companion.
        assert DynamicTimeWarping.load_stored_transforms(path) == {}
        assert marker.exists()
        marker.unlink()
        DynamicTimeWarping.load(path)
        assert marker.exists()
        monkeypatch.delitem(sys.modules, mod_name, raising=False)

    def test_default_load_still_restores_callable(self, small_trajectory_pair, tmp_path):
        """CPD-07 contract unchanged: a resolvable reference is restored by default."""
        x, y = small_trajectory_pair
        path = tmp_path / "r.pt"
        _computed(x, y, euclidean_distance).save(path)
        assert DynamicTimeWarping.load(path).config["distance_metric"] is euclidean_distance
        assert DynamicTimeWarping.load(path, resolve_metric=False).config[
            "distance_metric"
        ] == {"callable": EUCLIDEAN_REF}
