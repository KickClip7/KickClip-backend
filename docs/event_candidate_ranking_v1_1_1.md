# Event Candidate Ranking V1.1.1 Safety Revision

V1.1.1 is an additive safety revision. The V1.1 source, manifest, policy, and
results remain frozen. Its ranking policy SHA-256 remains:

```text
7359adf60fba7128080bcd59895074f883a5c18f456ad16f1510d92b8af21e44
```

## Server-resolved request

The endpoint accepts only `event_id`, `scene_id`, and `shortlist_size`:

```text
POST /api/v1/projects/{project_id}/highlight/revisions/{revision_id}/event-candidate-rankings/v1.1.1
```

The server resolves event label, timestamp, confidence, and scene bounds from
`TimelineEvent`. It verifies the Project Match, HighlightRevision selection,
Action Spotting job, and event-to-scene relation before enqueuing the durable
task. Client event labels, timestamps, confidence, and scene bounds are not
accepted.

## Safety changes

- Frame decoding uses a four-frame bounded LRU and sequential decoding for
  consecutive unique frames.
- Local track duplicate scope is `(shot_id, local_tracklet_id)`.
- `local_post_event_focus` is local evidence.
  `cross_shot_repeated_focus` remains `null` with
  `CROSS_SHOT_IDENTITY_NOT_ESTABLISHED`.
- One valid ball detection is selected per frame using confidence,
  size/bounds, and trajectory continuity.
- Low-coverage ball derived values remain `null`.
- Every candidate records `feature_completeness_score`,
  `critical_features_missing`, and one of `READY`, `PARTIAL_FEATURES`, or
  `NO_RELIABLE_SHORTLIST`.
- Shot input must have zero pending review, full unique frame coverage, and
  valid candidate/observation membership.

Automatic target confirmation remains false and the production recommendation
UI remains blocked.

## Representative input status

The connected DB contains real Action Spotting Goal events and source videos,
but no Project, HighlightRevision, scene candidate set, detections artifact, or
shot-boundary artifact. The frozen Scene Target Selection runtime is not
configured or installed. Therefore the exact 11-shot/170-candidate run cannot
be regenerated without expanding or changing the frozen discovery conditions.

The run remains `NOT_RUN`; no provisional Top 5 or Recall metric is fabricated.
