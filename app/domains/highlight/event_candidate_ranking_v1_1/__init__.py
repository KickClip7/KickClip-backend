"""Event Candidate Ranking V1.1 shadow-only raw feature bridge."""

from .contract import (
    PACKAGE_NAME,
    STATUS,
    CandidateObservation,
    CandidateSequence,
    ImmutableCandidateInput,
    canonical_event_label,
)

__all__ = [
    "PACKAGE_NAME",
    "STATUS",
    "CandidateObservation",
    "CandidateSequence",
    "ImmutableCandidateInput",
    "canonical_event_label",
]
