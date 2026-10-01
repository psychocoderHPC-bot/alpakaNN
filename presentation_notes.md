# Heat-closure presentation notes (template)

This is a reproducibility scaffold, not a report of completed solver results. Populate only from validated run artifacts. The renderer requires `manifest.csv` and `frame_%06d.csv` with `x,y,u`; it plots temperature in fixed [0,1] range and nearest physical-time matched frames. It does not infer coefficients or metrics from temperatures.

## Five-minute outline
1. Problem (45 s): introduce unit-square plate; left hot boundary, right cold boundary, insulated top/bottom. Show geometry only if generated from the specified material constants.
2. Baseline (60 s): show `comparison_*.png`; describe physical time and front only if visible in actual data.
3. Integration (60 s): feature packing → device inference → bounded coefficient → conservative update; replace this conceptual sequence only with implementation evidence.
4. Evidence (90 s): insert measured temperature/coefficient errors and timings with commands and artifact paths. Current comparison renderer shows only temperature and absolute field difference.
5. Portability/outlook (45 s): list tested backends and skipped backends from actual logs; no speedup or portability claims without tests.

## Commands
`python3 tools/render_heat_closure.py --reference RESULTS --comparison NN_RESULTS --output presentation_frames`
`python3 tools/make_heat_closure_video.py --frames presentation_frames --output heat_closure_demo.mp4`

Video cards explicitly disclose missing coefficient/metrics evidence. Replace those only when exported, verified evidence is available. Inputs are frame-held for accelerated playback; no numerical interpolation is performed.
