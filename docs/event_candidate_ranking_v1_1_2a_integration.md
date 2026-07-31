# Event Candidate Ranking V1.1.2a Integration Patch

The compatibility package and its manifest remain frozen. Integration is
implemented outside the package so package/source verification continues to
use the same SHA-256.

At API router bootstrap, the integration module:

- registers the V1.1.2a POST endpoint;
- adds the `EVENT_CANDIDATE_RANKING_V1_1_2A_SHADOW` SceneAITask type and
  executor dispatch;
- binds existing annotation review/finalize/evaluation endpoints to the
  V1.1/V1.1.1/V1.1.2/V1.1.2a compatibility service;
- wraps the scene-target installation verifier with the V1.1.2a verifier and
  the `EVENT_RANKING_COMPATIBILITY_RUNTIME_VERIFIED` component.

`EVENT_RANKING_RUNTIME_VERIFIED` requires the existing V1.1 shadow,
V1.1.1 safety, V1.1.2 contract, and new V1.1.2a compatibility states.
`FULL_EVENT_RECOMMENDATION_E2E_VERIFIED` remains false.

Regression coverage executes:

```text
POST V1.1.2a
→ SceneAITask QUEUED
→ executor claim and adapter dispatch
→ EVENT_CANDIDATE_RANKING_V1_1_2A_SHADOW artifact
→ task COMPLETED
```

The heavy ranking body is replaced by a deterministic fixture adapter in the
HTTP/task E2E; queueing, task persistence, executor dispatch, artifact
persistence, and completion use the real backend code paths.

