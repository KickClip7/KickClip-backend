# Event Candidate Ranking V1.2 Shadow Shortlist Patch

## Status

`PROVISIONAL_SHADOW_ONLY`

V1.2 is a shortlist-only counterfactual patch over immutable V1.1.2a output.
It does not recompute event features, recommendation scores, or global ranks.
It does not activate automatic target confirmation or the production
recommendation UI.

## Inputs

- Immutable `EVENT_CANDIDATE_RANKING_V1_1_2A_SHADOW` JSON and SHA-256
- Approved shot-boundary artifact
- Shortlist size `5`

Human labels, player name, shirt number, and known answer candidate IDs are not
shortlist inputs. Human annotation is used only after output generation for
counterfactual evaluation.

## Shortlist policy

- At most 2 candidates per shot
- At most 2 candidates from the event-containing/post-event phase
- At least 1, preferably 2 candidates from the pre-event/action phase
- If the event-containing shot starts within 2 seconds before the event, the
  highest-scored eligible candidate in the previous approved shot receives the
  mandatory `CUT_ADJACENT_ACTION_CANDIDATE` slot
- Remaining capacity follows unchanged V1.1.2a global score order
- `READY` and `PARTIAL_FEATURES` candidates are eligible
- Missing phase candidates are reported; unreliable candidates are not forced

`SHOT_CANDIDATE_CAP` records candidates excluded because their shot already has
two selected candidates.

## Phase contract

Shots before the event-containing shot are `PRE_EVENT_ACTION`. The
event-containing shot and later shots are `EVENT_CONTAINING_POST`. This is
camera-shot phase diversity, not cross-shot identity merging.

## Full-class diagnostic

The Representative Goal run separately evaluates RF-DETR classes 0–4 on frames
303–398:

- player
- goalkeeper
- referee
- staff
- ball

The diagnostic is not an input to V1.2 scores or shortlist selection.

## Future ranking evidence artifact

`ranking_evidence_artifact_schema.json` keeps two immutable sources separate:

- player discovery detections: current player/goalkeeper discovery input
- ranking object detections: player/goalkeeper/referee/staff/ball evidence

Connection to scoring remains `false` until a later calibrated revision.

