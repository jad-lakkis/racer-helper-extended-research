# Style vectors: ESL-Hockolicious

594 valid laps from 374 drivers (public replays from tmnf.exchange, track 414041, plus the two local laps of `jadlakkisjad`).
Every lap has one style vector per corner for 19 corners: `b` braking fraction, `f` full-throttle fraction,
`l` line (−1 outside edge … +1 inside edge, 32 m road assumed), `s` steering speed (full-lock sweeps per second),
plus entry speed and brake onset. No clustering has been run. Definitions and caveats are in
[docs/style-vectors.md](../docs/style-vectors.md).

## Files

| file | content |
|---|---|
| `Hockolicious/corners.csv` | one row per lap and corner (11,286 rows): the style vector and extras |
| `Hockolicious/laps.csv` | one row per lap: lap-level means, `steer_input` (analog / digital), RL conditioning values |
| `Hockolicious/drivers.csv`, `driver_style_vectors.csv` | one row per driver: best lap, number of laps, and the driver's mean style vector (feature × corner) |
| `Hockolicious/style_ranges.json` | min / max / mean / std of each style value per corner from each driver's 5 fastest laps |
| `Hockolicious/corners_def.json` | the 19 corners (start, apex, end in metres along the centerline) |
| `Hockolicious/skipped.csv` | the 22 replays not used, with the reason |
| `figures/` | the plots below |

## Median style by driver best-lap band

Median of the per-driver means. 374 drivers; the 20 fastest laps (49.5 to 53.5 s) take big cuts, are not on the centerline route and are not included.

| best lap (s) | drivers | b | f | l_apex | s | entry / peak speed | brake onset (m before corner) |
|:--|--:|--:|--:|--:|--:|--:|--:|
| < 54 | 17 | 0.13 | 0.87 | 0.30 | 0.79 | 0.80 | 26.7 |
| 54–56 | 147 | 0.13 | 0.87 | 0.25 | 1.13 | 0.81 | 26.1 |
| 56–58 | 81 | 0.12 | 0.86 | 0.20 | 1.68 | 0.81 | 24.9 |
| 58–62 | 94 | 0.09 | 0.87 | 0.19 | 2.17 | 0.81 | 21.0 |
| ≥ 62 | 35 | 0.05 | 0.86 | 0.14 | 2.42 | 0.79 | 14.0 |
| **jadlakkisjad** (56.34 s) | 1 | 0.14 | 0.82 | 0.23 | **4.19** | 0.84 | **42.1** |

Faster drivers brake more, take a tighter line at the apex and steer less. Full-throttle fraction is flat across bands.
Steering speed and brake onset are where `jadlakkisjad` is furthest from drivers of similar time.

## Figures

**Style features against lap time** (one dot per lap, star = `jadlakkisjad`)
![features vs time](figures/Hockolicious_features_vs_time.png)

**Mean style vector per corner for each best-lap band** (bottom row of each panel = `jadlakkisjad`)
![bands per corner](figures/Hockolicious_time_band_corner_styles.png)

**Band minus `jadlakkisjad` per corner** (red = the band has more of it than you)
![bands minus me](figures/Hockolicious_time_band_vs_jadlakkisjad.png)

**Speed, line and braking along the track per lap-time band** (black = `jadlakkisjad`)
![band profiles](figures/Hockolicious_band_profiles.png)

**Corner segmentation**
![corners](figures/Hockolicious_corners_map.png)

Regenerate: `cd linesight && python scripts/extract_style_vectors.py ../replays --out ../style_vectors && python scripts/analyze_style_vectors.py ../style_vectors`
