class CandidateHandoffR1Error(ValueError):
    code = "CANDIDATE_HANDOFF_R1_ERROR"


class CandidateSelectionProvenanceMismatch(CandidateHandoffR1Error):
    code = "CANDIDATE_SELECTION_PROVENANCE_MISMATCH"


class InsufficientReviewableTargetReference(CandidateHandoffR1Error):
    code = "INSUFFICIENT_REVIEWABLE_TARGET_REFERENCE"


class TrackletIdentityInconsistent(CandidateHandoffR1Error):
    code = "TRACKLET_IDENTITY_INCONSISTENT"


class CandidatePreparationError(CandidateHandoffR1Error):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(f"{code}: {message}")


class CandidateRecommendationNotPrepared(CandidateHandoffR1Error):
    code = "CANDIDATE_RECOMMENDATION_NOT_PREPARED"

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(
            "V1.2 ranking or candidate review bundles are not ready."
        )
