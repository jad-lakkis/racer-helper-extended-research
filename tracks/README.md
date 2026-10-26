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
InterlagosGP (12, ratio 0.99). Everything else has 11 replays or fewer.
