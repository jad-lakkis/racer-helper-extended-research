# Tracks

- `monza/`: Monza from the Trackmania 2020 map 271697: geometry extracted from the map file (lap length 5,796.6 m against the official 5,793 m).
  No replays are available for it.
- `tmnf_real_circuit_candidates.csv`: TMNF tracks on tmnf.exchange named after real circuits, with their replay counts
  and the path length of the fastest replay compared with the real lap length (`linesight/scripts/tools/find_real_circuits_tmx.py`,
  `tmx_circuit_path_lengths.py`). The path ratio is only a screening: a ratio near 1 means one lap of about the right length,
  above 1.2 usually means several laps, and below 0.9 a shortened track.

Result: no TMNF track has both many replays and a real-circuit layout. The most replayed track (ESL-Hockolicious, 635 replays) is
fictional. The most replayed real-circuit recreation is ESL - Suzuka with 34 replays (path 0.62 of the real lap, so it is scaled down),
then Monaco GP Single (23, ratio 0.93), WCG09-Laguna-1 (20, ratio 1.23), Nürburgring-like track (16), Barcelona (15), Silverstone (13),
InterlagosGP (12, ratio 0.99). Everything else has 11 replays or fewer. These are replay counts; see the corrected driver counts below.

## Which TMNF track matches a real circuit? (`compare/`)

`linesight/scripts/tools/compare_tmnf_to_real.py` takes the first lap of the fastest replay of a TMNF track, fits it onto the real outline
(bacinger/f1-circuits, built from OpenStreetMap) over every start point, both directions and a mirror, with one rotation and one scale factor,
and numbers the corners on both in lap order. `compare/overview.png` shows all 21 candidates, `compare/results.csv` has the numbers and
`compare/<name>_<id>.png` the three-panel figure for each (TMNF layout, real outline, overlay).

| rank by shape | TMNF track | drivers (replays) | size vs real | shape gap | corners found (TMNF / real) | in Assetto Corsa (from memory) |
|--:|---|--:|--:|--:|--:|---|
| 1 | InterlagosGP [Unlimiter 2.0] (12851169) | **3** (12) | 1.01 (same size) | 0.35 % (15 m) | 11 / 14 | mod only |
| 2 | F1 GPs: Catalunya (1482521) | 13 (13) | 1.51 (two thirds of real) | 1.6 % (72 m) | 14 / 13 | Barcelona is DLC |
| 3 | Monza GP (5169102) | 5 (11) | 2.16 (less than half size) | 2.2 % (126 m) | 10 / 10 | official |

**Replays are not drivers.** TMX keeps every improving personal best of a driver, so InterlagosGP's 12 replays are 10 successive laps of one driver
(KRONGTEKK_, 53.16 s down to 49.61 s on one day) and 2 other drivers. The site's replay page shows only the 3 current records. The driver counts
above are the ones to use. By drivers, Monaco GP Single has 18 (1.14x, 2.9 % gap) and Catalunya 13.

Next best: GT14_Spa_Francorchamps (5 drivers, 2.06x, 1.5 %, official AC), F1 Hungaroring (9 replays, 1.10x, 2.0 %, but the layout is visibly
stretched), Monaco GP Single (23 replays, 1.14x, 2.9 %, loose shape).

How to read it: "size vs real" is the factor that scales the TMNF lap up to the real size (1.00 = a 1:1 recreation); "shape gap" is the RMS
distance between the two outlines after the best fit, in metres and as a share of the real lap. It is measured on the fastest replay's driven
line, not on the road centre, so a few metres of it are the racing line. The real outlines have only 100 to 300 points per circuit.
Corner counts come from a curvature threshold, so chicanes can be merged and the numbers do not follow the official turn numbers.
Laguna Seca is not here: it is not an F1 circuit, so there is no outline in that source.
