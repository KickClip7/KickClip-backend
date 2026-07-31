# Event Candidate Ranking V1.1 Raw Feature Bridge

Package: `target_centric_tracking_event_candidate_ranking_v1_1`

Status: `PROVISIONAL_SHADOW_ONLY`

Automatic target confirmation: `false`

Production UI recommendation activation: `BLOCKED`

The existing Event Candidate Ranking V1 implementation, V1 policy/schema,
V1/V2 tracking, R2/R3, and Scene Target Selection algorithms are inputs to this
additive package and are not replaced by it.

## Supported event contract

Canonical event classes are `goal`, `shot`, `free_kick`, and `corner`.
Aliases are frozen in `event_candidate_ranking_policy.json`. Other event
classes, including Foul, Card, and Penalty, produce
`UNSUPPORTED_EVENT_CLASS`, an empty shortlist, and the complete scene gallery
fallback.

Event time is in source-video seconds. Candidate observation time is in
scene-local seconds. Conversion is:

```text
event_scene_local_time_sec = event_time_sec - scene_start_sec
```

## Immutable input and cache

The V1.1 endpoint is:

```text
POST /api/v1/projects/{project_id}/highlight/revisions/{revision_id}/event-candidate-rankings/v1.1
```

Before the durable task is enqueued, the backend resolves and hashes:

- `scene_candidates.json`
- detections artifact
- source video
- shot-boundary artifact
- scene candidate manifest
- candidate package manifest
- V1.1 source manifest, policy, and feature schema
- event ID, canonical input label, and source-video event time

The complete server-side freeze material is part of the task payload and
therefore its idempotency key. A source or policy hash change cannot reuse an
earlier completed task.

All input paths are project-storage-relative. Each path must resolve within
the storage root, must be a regular non-symlink file, must have an allowed
suffix, and must match the recorded SHA-256.

## Feature behavior

Features are computed from the actual candidate observation sequence.
Unsupported or unavailable features remain `null`; they are not silently
replaced with zero.

The output contains global and per-candidate availability for temporal,
visual, motion, broadcast, ball, and field-context groups. Evidence frame
indices and source names are recorded per candidate.

`local_track_duration_ratio` is a temporal feature. It is not called repeated
broadcast focus. Cross-shot identity is never inferred. Broadcast focus
features are observation/shot proposals only.

Ball detections are audited from the immutable detections artifact.
No matching class produces `BALL_SIGNAL_UNAVAILABLE`; insufficient matching
observation coverage produces `BALL_SIGNAL_LOW_COVERAGE`.

## Human annotation and evaluation

Independent reviews, finalization, and evaluation are available under:

```text
POST /api/v1/event-candidate-rankings/v1.1/artifacts/{artifact_id}/annotation-reviews
POST /api/v1/event-candidate-rankings/v1.1/artifacts/{artifact_id}/annotations/finalize
GET  /api/v1/event-candidate-rankings/v1.1/artifacts/{artifact_id}/evaluation
```

A final annotation must select one approved review or prove explicit
consensus among multiple identical reviews. Reviewer labels are never unioned.
Metrics remain `NOT_RUN` until annotation status is `COMPLETE`.

## Verifier states

- `FULL_SCENE_SELECTION_TRACKING_E2E_VERIFIED` covers selection-assisted
  tracking only.
- `EVENT_RANKING_SHADOW_RUNTIME_VERIFIED` requires V1.1 source, policy,
  schema, fixture, and dependency hashes plus the synthetic raw-feature smoke.
- `FULL_EVENT_RECOMMENDATION_E2E_VERIFIED` stays false until an approved
  labeled dataset has completed Recall evaluation.
