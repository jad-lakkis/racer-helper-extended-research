"""
Offline checks for trackmania_rl/driver_context.py and the stateful centerline tracker. Run from linesight/:  python tests/test_driver_context.py

Uses the committed driver packages under style_vectors/<track>/packages. The package build test needs the (not committed)
style_vectors/<track>/profiles and is skipped with a message when they are missing; the reward test also needs pygbx and the replays.
"""

import dataclasses
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "linesight"))
sys.path.insert(0, str(ROOT / "linesight" / "tests"))
from trackmania_rl.driver_context import (  # noqa: E402
    LOOK_AHEAD_M,
    N_REFERENCE_FEATURES,
    DriverPackage,
)
from trackmania_rl.style_features import CenterlineTracker, project_on_centerline  # noqa: E402

STYLE_DIR = ROOT / "style_vectors" / "Hockolicious"
PACKAGE = STYLE_DIR / "packages" / "-wally-.npz"


def test_committed_package_loads():
    pkg = DriverPackage.load(PACKAGE)
    assert pkg.style.name == "-wally-" and pkg.style.track == "Hockolicious" and len(pkg.style.corners) == 19
    assert len(pkg.refs) == 2 and pkg.style.best_lap_time_ms == 54040 == min(r.lap_time_ms for r in pkg.refs)
    assert pkg.style.half_width == 16.0 and pkg.style.centerline_file.endswith("ESL-Hockolicious_0.5m_cl2.npy")
    r = pkg.refs[0]
    assert r.s.shape == r.offset.shape == r.speed.shape and abs(np.diff(r.s).mean() - 2.0) < 1e-3
    assert 0 < r.speed.max() < 400 and np.abs(r.offset).max() < 30
    assert all(lo <= hi for d in pkg.style.ranges.values() for lo, hi in d.values())


def test_save_load_round_trip():
    pkg = DriverPackage.load(PACKAGE)
    with tempfile.TemporaryDirectory() as tmp:
        pkg.save(Path(tmp) / "x.npz")
        again = DriverPackage.load(Path(tmp) / "x.npz")
    assert again.style.to_meta() == pkg.style.to_meta()
    for a, b in zip(pkg.refs, again.refs):
        assert a.lap_id == b.lap_id and a.lap_time_ms == b.lap_time_ms
        assert np.allclose(a.offset, b.offset, atol=1e-4) and np.allclose(a.speed, b.speed, atol=1e-3)


def test_building_from_the_extraction_gives_the_committed_package():
    if not (STYLE_DIR / "profiles").exists():
        print("   skipped: needs style_vectors/Hockolicious/profiles (written by extract_style_vectors.py)")
        return
    built = DriverPackage.build(STYLE_DIR, "-wally-", k=5)
    committed = DriverPackage.load(PACKAGE)
    assert built.style.to_meta() == committed.style.to_meta()
    assert [r.lap_id for r in built.refs] == [r.lap_id for r in committed.refs]
    assert np.allclose(built.refs[0].offset, committed.refs[0].offset, atol=1e-4)


def test_reference_features():
    pkg = DriverPackage.load(PACKAGE)
    ref = pkg.refs[0]
    p, e = 1234.0, 3.0
    f = pkg.reference_features(ref, p, e)
    assert f.shape == (N_REFERENCE_FEATURES,) == (len(pkg.feature_names()),) and f.dtype == np.float32 and np.isfinite(f).all()
    n = len(LOOK_AHEAD_M)
    for i, d in enumerate(LOOK_AHEAD_M):
        assert np.isclose(f[i], np.interp(p + d, ref.s, ref.offset) / 16.0, atol=1e-6)
        assert np.isclose(f[n + i], (np.interp(p + d, ref.s, ref.speed) - 150.0) / 100.0, atol=1e-6)
    assert np.isclose(f[-1], (3.0 - np.interp(p, ref.s, ref.offset)) / 16.0, atol=1e-6)
    # on the reference line: zero error; the end of the lap holds the last value
    on_line = pkg.reference_features(ref, p, float(ref.offset_at(p)))
    assert abs(on_line[-1]) < 1e-6
    end = pkg.reference_features(ref, ref.s[-1] + 500.0, 0.0)
    assert np.isclose(end[0], ref.offset[-1] / 16.0, atol=1e-6) and np.isclose(end[n], (ref.speed[-1] - 150.0) / 100.0, atol=1e-6)


def test_each_rollout_gets_a_reference_and_all_are_used():
    pkg = DriverPackage.load(PACKAGE)
    rng = np.random.default_rng(0)
    seen = {pkg.sample_reference(rng).lap_id for _ in range(50)}
    assert seen == {r.lap_id for r in pkg.refs}


def test_stateful_tracker_matches_the_whole_lap_projection():
    pkg = DriverPackage.load(PACKAGE)
    cl = pkg.style.centerline
    rng = np.random.default_rng(1)
    # a car that drives the centerline at 60 m/s with a wobbling offset and some noise
    s = np.arange(0.0, cl.s[-1] - 5, 2.4)
    idx = np.searchsorted(cl.s, s)
    off = 6 * np.sin(s / 40.0) + rng.normal(0, 0.3, len(s))
    xz = cl.pts[idx][:, [0, 2]] + cl.normal[idx] * off[:, None]
    pos = np.stack([xz[:, 0], np.zeros(len(s)), xz[:, 1]], axis=1)
    v = np.full(len(s), 60.0)
    p_all, e_all = project_on_centerline(pos, v, cl, 0.04)
    tracker = CenterlineTracker(cl)
    got = np.array([tracker.step(pos[i], v[i], 0.04) for i in range(len(pos))])
    assert np.array_equal(got[:, 0], p_all) and np.array_equal(got[:, 1], e_all)
    assert np.abs(p_all - s).max() < 2.5 and np.abs(e_all - off).max() < 1.0  # and it actually recovers where the car is
    tracker.reset()
    assert tracker.prev == 0


def test_line_term_of_the_reward_is_the_distance_to_the_reference_line():
    try:
        from test_style_constraint import _constraint, _human_lap_as_a_rollout
    except SystemExit:
        return
    lap = _human_lap_as_a_rollout()
    if lap is None:
        print("   skipped: needs pygbx and replays/tmnf_exchange/Hockolicious")
        return
    pos, speed, actions, race_time = lap
    sc = _constraint()
    pkg = DriverPackage.load(PACKAGE)
    own = pkg.refs[0]  # the replay is this lap
    assert own.lap_time_ms == 54040
    shifted = dataclasses.replace(own, offset=own.offset + 6.0)  # the same line, 6 m to one side
    centre = sc.score(pos, speed, actions, True, race_time)
    on_line = sc.score(pos, speed, actions, True, race_time, reference=own)
    off_line = sc.score(pos, speed, actions, True, race_time, reference=shifted)
    w, n = sc.weights.w_line, len(pos) - 1
    # the lap runs along its own line (40 ms resampling of a 100 ms record and the 2 m profile grid leave under a metre of error)
    gap = (on_line.base_reward.sum() - off_line.base_reward.sum()) / (w * n)
    assert 4.5 < gap <= 6.0 + 1e-6, gap  # shifting the line by 6 m costs about 6 m per step (less where the lap is already off its own line)
    assert on_line.base_reward.sum() > off_line.base_reward.sum()
    assert np.allclose(on_line.c_style, centre.c_style) and on_line.style_total == centre.style_total  # the style cost does not depend on the line


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
