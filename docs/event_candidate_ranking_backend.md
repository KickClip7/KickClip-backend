# Event-aware player recommendations

KickClip now exposes a shadow-only recommendation layer before the existing
scene-wide player gallery.

## API flow

1. Complete reviewed-shot scene-wide candidate discovery.
2. Create or reuse ranking:

   `POST /api/v1/projects/{project_id}/highlight/revisions/{revision_id}/events/{event_id}/candidate-ranking`

   ```json
   {
     "scene_id": "evt_...",
     "scene_video_asset_id": "asset_...",
     "shot_boundaries_artifact_id": "art_...",
     "detections_artifact_id": "art_..."
   }
   ```

   `detections_artifact_id` is optional. When omitted, the backend reuses the
   server-owned detections artifact recorded by the matching scene discovery.
   If that artifact is unavailable or contains no usable ball rows, ball
   evidence is disabled safely; no zero-distance feature is fabricated.

3. Render the provisional Top 3-5:

   `GET .../events/{event_id}/recommended-players`

4. Keep the fallback:

   `GET .../events/{event_id}/all-player-candidates`

5. When the user presses Select, call the existing immutable
   `POST .../target-selections`. The ranking endpoint never confirms a target
   and never launches tracking.

## UI wording

Use “이 장면에서 중요한 선수 후보” and show reason/risk text. Do not render
the raw recommendation score as identity confidence. When
`NO_RELIABLE_SHORTLIST` is returned, show:

> 중요 선수를 자동으로 좁히기 어려워 전체 선수 후보를 보여드립니다.

The policy remains `PROVISIONAL_SHADOW_ONLY`; production automatic selection
must stay disabled.
