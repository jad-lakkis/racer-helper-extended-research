# Style vectors from human replays

Turns `.Replay.Gbx` laps into per-corner driver style vectors, then clusters laps into styles.
Implements the style vector `z = (b, f, l, s)` of the problem formulation.

```
pip install pygbx scipy pandas scikit-learn matplotlib requests     # pygbx needs liblzo2-dev
python scripts/tools/fetch_tmx_replays.py 414041 --max-time 70 --out ../replays/tmnf_exchange/Hockolicious   # optional: other drivers' replays
cd linesight
python scripts/extract_style_vectors.py ../replays --out ../style_vectors
python scripts/analyze_style_vectors.py ../style_vectors
```

New track: `--track-name MyTrack --centerline maps/MyTrack_0.5m_cl.npy` (0.5 m centerline from `scripts/tools/gbx_to_vcp.py`).
Known tracks (Bahrain, Hockolicious) are picked from the replay filename.

## What is measured

Corners are found once per track from the centerline curvature (`|k| > 1/150 m`, same-sign runs merged
across gaps under 25 m, at least 20 m and 20 degrees), so corner *k* means the same segment in every lap.

| symbol | column(s) in `corners.csv` | definition |
|---|---|---|
| b | `b` | fraction of corner time braking (10 ms input timeline) |
| f | `f` | fraction of corner time with accelerate pressed and brake released |
| l | `l_entry`, `l_apex`, `l_exit` | lateral offset from the centerline / (track width / 2); +1 = inside edge of the corner |
| s | `s` | steering speed: total variation of the steering command per second (full-lock sweeps/s) |

Extra columns: `b_approach`, `brake_onset_m` (braking in the 120 m before entry), `coast`, `v_entry`, `v_apex`,
`v_min`, `v_exit`, `v_entry_ratio`, `steer_reversals`, `corner_time_s`.

`laps.csv` has one row per valid lap: the means of the above plus `braking_aggression`,
`oversteer_understeer_score`, `corner_entry_speed_ratio` (from `extract_driver_profile.py`) and
`risk_tolerance_proxy` (mean absolute offset / 15 m, the same normalisation as the risk reward).
`style_ranges.json` holds min / max / mean / std of z per corner from each driver's 5 fastest laps.

Clustering needs at least 8 laps on a track. Style features exclude lap time; `clusters.json` reports their
Spearman correlation with lap time so skill-confounded features are visible, and gives each cluster's mean
RL conditioning values.

## Caveats

- Steering is stored two ways: digital `SteerLeft`/`SteerRight`, or `Steer` events (analog for pad / wheel, -1/0/+1 for keyboard).
  Both are decoded to one signal in [-1, 1]; `laps.csv` has `steer_input` (analog / digital). Digital drivers still jump between full lock
  values, so `s` is not strictly comparable across the two groups. About a third of the Hockolicious replays are analog.
- Replays that take large cuts leave the centerline route and are skipped (`skipped.csv`). On Hockolicious that is the 20 fastest laps (49.5 to 53.5 s).
- Line offsets are relative to the centerline file. For Bahrain that file was built from a driver's own lap,
  so offsets are relative to that line, not the geometric road centre.
- Track width is assumed to be 32 m (`--track-width`). Raw offsets in metres are in `style_vectors/<track>/profiles/*.npz`.
- Laps with respawns, or that cover under 90 % of the centerline, are skipped.
- Position is sampled every 100 ms, so corners shorter than about 0.3 s are noisy.

## Clustering drivers

```
python scripts/cluster_driver_styles.py ../style_vectors --me jadlakkisjad [--raw]
```
See `style_vectors/README.md` for the method and the Hockolicious result (no distinct style groups after removing speed and steering device).
