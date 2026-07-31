from __future__ import annotations

import numpy as np
import pytest

from app.domains.media.preloaded import PreloadedMatchImporter


def test_inspect_feature_accepts_champion_contract(tmp_path) -> None:
    path = tmp_path / "features.npy"
    np.save(path, np.zeros((20, 512), dtype=np.float32))

    shape, dtype = PreloadedMatchImporter._inspect_feature(path)

    assert shape == (20, 512)
    assert dtype == "float32"


def test_inspect_feature_rejects_wrong_dimension(tmp_path) -> None:
    path = tmp_path / "features.npy"
    np.save(path, np.zeros((20, 256), dtype=np.float32))

    with pytest.raises(ValueError, match=r"shape \[T, 512\]"):
        PreloadedMatchImporter._inspect_feature(path)


def test_deterministic_match_id_is_stable() -> None:
    assert (
        PreloadedMatchImporter._match_id("kor_jpn")
        == "match_preloaded_kor_jpn"
    )
