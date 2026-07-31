# Event Candidate Ranking V1.1.2a Compatibility Hotfix

V1.1.2a is additive. V1.1, V1.1.1, and V1.1.2 files and manifests are frozen,
and the V1.1 ranking policy is reused without weight changes.

## Compatibility corrections

- Reviewed-shot parsing accepts `end_frame`, `end_frame_inclusive`,
  `last_frame`, and `frame_end`, plus `review_status`, `review_state`,
  `status`, and `boundary_status`. Approval remains restricted to
  `REVIEWED_PASS` and `CONFIRMED`.
- Ball selection assigns a trajectory segment within each shot whenever
  `continuity_reset_gap_frames` is exceeded. Reset-gap pairs are excluded from
  continuity, velocity, and direction calculations.
- Candidate ball output distinguishes selected observation count, valid
  continuity pair count, reset-gap pair count, passing pair count, coverage,
  and trajectory continuity.
- Preparation freezes `discovery_id`, discovery artifact root, scene ID, and
  candidate manifest SHA. Execution writes only below that frozen root and
  rejects a changed discovery as `STALE_DISCOVERY_INPUT`.
- The cache fingerprint includes detections SHA, discovery ID, scene ID and
  bounds, and shortlist size.
- Candidate ball ranges/counts, evidence frame types, shortlist decision
  values, and shortlist/all-candidate item contracts are validated by
  Draft 2020-12 JSON Schema.
- V1.1.2 ranking artifacts complete independent review, final annotation, and
  evaluation while cross-version review mixing is rejected.

The fixture `tests/fixtures/r2_r3_reviewed_shot_artifact.json` preserves the R3
shot field shape found in the repository contract fixture. A production
reviewed-shot artifact was not present in this workspace, which is recorded in
the fixture provenance rather than hidden.

## Representative run

The immutable 11-shot/170-candidate discovery snapshot and its complete
server-side relationship are still absent. The representative Top 5 and
Recall metrics remain `NOT_RUN`. Automatic target confirmation is false and
the production recommendation UI remains blocked.
