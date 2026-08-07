# KickClip Scene-wide Target Selection V1

This package turns a fully reviewed Action Spotting scene into a portable,
shot-ordered gallery of camera-cut-free local player tracklets. It never
declares a real-world identity and never links candidates across a cut without
an explicit user decision.

## Contract

1. `discover` accepts only a reviewed R2 shot-boundary artifact with exact
   frame coverage and no pending cuts.
2. Frozen V1 RF-DETR detections and frozen V2 Stage 3-B0 local association are
   reused without changing their source, checkpoints, ranking metric, or gates.
3. Representative observations are optimized for recognition; initialization
   observations are independently validated for tracking.
4. `select` creates an immutable `target_selection_rNNNN.json` revision and a
   diverse reference set from one shot-local tracklet.
5. `propose-earlier` uses the R2 safe Sports OSNet loader and frozen V2 B1
   metric. Rank 1 is still routed to user confirmation.
6. `prepare-tracking` creates a cache-isolated R3 launch contract. It never
   fabricates pre-anchor boxes or claims the target was absent before the
   selected anchor.

## CLI

```text
run_scene_target_selection.py discover
run_scene_target_selection.py select
run_scene_target_selection.py propose-earlier
run_scene_target_selection.py confirm-earlier
run_scene_target_selection.py start-selected-shot
run_scene_target_selection.py prepare-tracking
```

All files intended for API/UI handoff use paths relative to the scene output
root. Absolute runtime paths are kept only in backend-private execution
metadata.

## Safety

- No frame-0 target default.
- No cross-shot IoU or motion identity link.
- No retrieval-score auto confirmation.
- No jersey OCR or team color identity decision.
- No bbox before the confirmed/selected tracking anchor.
- Changing the selection revision or candidate changes the tracking cache key.

