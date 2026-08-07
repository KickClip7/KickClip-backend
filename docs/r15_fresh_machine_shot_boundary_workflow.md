# R15 Fresh-Machine Shot-Boundary Workflow

R15 removes the fresh-event dependency on a copied `scene.mp4`,
`reviewed_shots.json`, and `detections.csv` bundle. Candidate discovery uses a
content-addressed `AUTO_SHOT_BOUNDARIES` artifact after structural validation;
human review is requested only when that gate fails. Target identity is never
confirmed automatically.

## API contract

All routes require the normal bearer session and apply project ownership
scoping. `scene_id` is a required query parameter.

```text
POST /api/v1/projects/{project_id}/highlight/revisions/{revision_id}/events/{event_id}/shot-boundaries/prepare
GET  /api/v1/projects/{project_id}/highlight/revisions/{revision_id}/events/{event_id}/shot-boundaries
PUT  /api/v1/projects/{project_id}/highlight/revisions/{revision_id}/events/{event_id}/shot-boundaries/draft
POST /api/v1/projects/{project_id}/highlight/revisions/{revision_id}/events/{event_id}/shot-boundaries/reset
POST /api/v1/projects/{project_id}/highlight/revisions/{revision_id}/events/{event_id}/shot-boundaries/confirm
POST /api/v1/projects/{project_id}/highlight/revisions/{revision_id}/events/{event_id}/shot-boundaries/detections/retry
GET  /api/v1/projects/{project_id}/highlight/revisions/{revision_id}/events/{event_id}/shot-boundaries/media/contact-sheet
GET  /api/v1/projects/{project_id}/highlight/revisions/{revision_id}/events/{event_id}/shot-boundaries/media/cut/{cut_frame}/{side}
```

Calling prepare with `new_review_revision=true` after confirmation starts a new
review session and later produces a new artifact revision; it never overwrites
the earlier `REVIEWED_SHOT_BOUNDARIES` row.

OpenAPI models are `ShotBoundaryReviewResponse`, `ShotBoundaryDraftUpdate`,
`ShotBoundaryConfirmRequest`, and `ShotBoundaryConfirmResponse`.

## State transitions

```text
scene selected
  -> materialize exact scene video
  -> automatic cut analysis
  -> structural gate PASS
  -> AUTO_SHOT_BOUNDARIES + sampled SCENE_PLAYER_DETECTIONS
  -> candidate preparation

structural gate FAIL
  -> SHOT_BOUNDARY_REVIEW_REQUIRED
  -> draft PUT (optimistic draft_revision)
  -> confirm (human reviewer + note + idempotency key)
  -> REVIEWED_SHOT_BOUNDARIES + candidate preparation

CONFIRMED / detections FAILED_RETRYABLE
  -> detections/retry
CONFIRMED / detections READY
```

`AUTO_SHOT_BOUNDARIES` declares `boundary_origin=AUTO_DETECTED`,
`human_reviewed=false`, and `automatic_target_confirmation=false`. It contains
no `REVIEWED_PASS` state and creates no review decision. Detection is sampled
around the Action Spotting timestamp. A later human confirmation remains an
immutable, separate `REVIEWED_SHOT_BOUNDARIES` artifact.

Before confirmation, rows are sorted by `(start_frame,
end_frame_inclusive, shot_id)`. The server assigns `shot_index` from that
canonical order rather than parsing `shot_id`, and materializes `frame_count`,
`cut_in_frame`, and `cut_out_frame`. The final shot has a null
`cut_out_frame`. Confirm runs the installed frozen Scene Target Selection
`validate_reviewed_shot_boundaries()` contract in an isolated process before
registering the immutable artifact.

Candidate APIs return HTTP 409 only when the automatic structural gate fails:

```json
{
  "code": "SHOT_BOUNDARY_REVIEW_REQUIRED",
  "reason": "AUTOMATIC_BOUNDARY_STRUCTURAL_GATE_FAILED",
  "prepare_review_url": "/api/v1/projects/.../shot-boundaries/prepare?scene_id=...",
  "review_status_url": "/api/v1/projects/.../shot-boundaries?scene_id=...",
  "automatic_target_confirmation": false
}
```

An older or otherwise incompatible confirmed artifact is rejected before a
candidate task is enqueued:

```json
{
  "code": "REVIEWED_SHOT_BOUNDARY_CONTRACT_INVALID",
  "message": "Confirmed shot boundaries are incompatible with Scene Target Selection.",
  "detail": { "reason": "SHOT_INDEXES_NOT_CONTIGUOUS" }
}
```

Candidate task idempotency includes the exact automatic-or-reviewed boundary
artifact ID and SHA plus the sampled detection artifact ID.
After `prepare?new_review_revision=true` and another human confirmation, the
new SHA creates a new candidate task; an earlier failed task remains failed
history.

## Storage and provenance

Files are stored below `STORAGE_ROOT/shot_boundary_reviews` and DB rows contain
project-root-relative paths only. A scene directory is content-scoped by source
SHA so a changed source never overwrites an earlier immutable artifact.

The `SCENE_VIDEO` artifact records source asset/SHA, exact source seconds,
scene SHA, FPS, frame count, and dimensions. Automatic boundaries are stored
content-addressed and never masquerade as a reviewed artifact. RF-DETR output
is cached by scene SHA, model SHA, event-window sampling policy, and event
provenance as `SCENE_PLAYER_DETECTIONS`. Its CSV contains the Scene Target
Selection runtime fields `frame_index`, `time_ms`, `detection_index`, and
`detection_id`.

## Database migration

```bash
source venv311/bin/activate
python -m alembic upgrade head
python -m alembic heads
```

The expected single head is `20260807_0017`. The migration creates
`shot_boundary_review_sessions` and `shot_boundary_review_decisions`.

## Moving an existing project to another computer

An existing human decision cannot be reconstructed from source video or code.
Transfer the PostgreSQL database, storage tree, environment settings, and the
exact model checkpoint together.

On the source machine, stop writers and create consistent snapshots:

```bash
pg_dump --format=custom --no-owner --file=kickclip.dump "$DATABASE_URL"
tar -C "$(dirname "$STORAGE_ROOT")" -czf kickclip-storage.tgz "$(basename "$STORAGE_ROOT")"
shasum -a 256 kickclip.dump kickclip-storage.tgz
```

On the destination machine, verify the two transport hashes before restoring:

```bash
shasum -a 256 kickclip.dump kickclip-storage.tgz
createdb kickclip
pg_restore --no-owner --dbname="$DATABASE_URL" kickclip.dump
tar -C "$(dirname "$STORAGE_ROOT")" -xzf kickclip-storage.tgz
python -m alembic upgrade head
```

Configure `PLAYER_DETECTOR_CHECKPOINT` to the transferred checkpoint and verify
its SHA against the original deployment record. Keep storage at the same
storage-relative keys; absolute source-machine paths are not migrated. Do not
rename another event's artifact IDs or copy its reviewed bundle into this
event. If either DB or storage is missing, start a new review session for a new
event instead of fabricating a successful migration.

## Verification commands

```bash
venv311/bin/python -m pytest -q tests/test_shot_boundary_bootstrap.py
venv311/bin/python -m alembic heads
python3 -m compileall -q app alembic/versions

cd ../KickClip-frontend
npm test
npm run typecheck
npm run lint
npm run build
```
