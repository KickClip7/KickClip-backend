# Event Candidate Ranking V1.1.2 Contract Fix

V1.1.2 is an additive contract revision. V1.1 and V1.1.1 source files and
manifests remain frozen. The V1.1 ranking policy is reused without changing
weights. Status remains `PROVISIONAL_SHADOW_ONLY`, automatic target
confirmation remains false, and the production recommendation UI remains
blocked.

## Contract corrections

- Annotation accepts an explicit V1.1/V1.1.1/V1.1.2 ranking artifact
  allowlist. Every independent review and final annotation records ranking
  package, schema version, source-manifest SHA, feature-schema SHA, and policy
  SHA. Evaluation rejects identity mismatches.
- Shot review accepts only `REVIEWED_PASS` and `CONFIRMED`. All other or blank
  states fail and are reported in `unapproved_review_count` and
  `unapproved_shot_ids`.
- Ranking preparation cross-checks the current `scene_candidates.json` SHA,
  the discovery-time stored SHA, and the candidate manifest-declared SHA.
- Timeline event label/time and scene bounds are server resolved. The event
  must be inside valid scene bounds, and scene duration must agree with the
  scene video within the configured tolerance.
- The V1.1.2 request model forbids extra fields. Client event label/time and
  scene bounds therefore return HTTP 422 instead of being ignored.
- The verifier validates the input document, every candidate feature row, and
  the complete ranking output against Draft 2020-12 JSON Schemas.
- Ball trajectories are selected and reset per `(shot_id, frame)`.
  Candidate-frame coverage and trajectory continuity are distinct; continuity
  uses selected valid ball pairs. Pre-event trend is restricted to the event
  shot and event-near window. Low-coverage derived values remain null.
- `broadcast.post_event_shots_with_appearance` is null and explicitly
  deprecated because a shot-local candidate cannot establish cross-shot
  identity.

## Representative Goal status

The workspace does not contain the immutable 11-shot/170-candidate
`scene_candidates.json`, matching reviewed shot boundaries and detections, or
the Project/HighlightRevision/scene relationship needed to resolve and execute
the representative run. Action Spotting event-candidate artifacts are not a
substitute for scene player candidates.

The representative Top 5 and Recall metrics remain `NOT_RUN`; no candidate,
frame, shirt number, or human ground truth was injected into production
features.
