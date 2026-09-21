"""
OceanEmbed - Active Learning: typed result shapes
====================================================
Plain dataclasses (no pydantic/web-framework dependency -- ml/ stays pure
per docs/architecture.md) describing what this package returns, so the
JSON shape served by /api/active-learning/* is defined in exactly one
place rather than assembled ad hoc in the API layer.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, field


@dataclass
class SamplingRecommendation:
    rank: int
    latitude: float
    longitude: float
    priority_score: float
    uncertainty: float
    reason: str
    recommended_platform: str
    recommended_depths_m: list = field(default_factory=list)
    adaptive_sampling: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)
