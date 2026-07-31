"""Register root-level MP4/NPY pairs as uploaded, feature-ready matches.

The command is idempotent. Large source files are hard-linked under STORAGE_ROOT
when possible, and a combined [T, 512] NPY is split into the two feature assets
required by the Champion Action Spotting model.

Examples:
    python scripts/import_preloaded_matches.py
    python scripts/import_preloaded_matches.py --only kor_jpn
    python scripts/import_preloaded_matches.py --owner-user-id usr_123
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import select

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.domains.auth.model import User
from app.domains.media.preloaded import (
    PreloadedMatchImporter,
    PreloadedMatchSpec,
)


DEFAULT_SPECS = {
    "fc_bayern_vs_frankfurt_2018_19": PreloadedMatchSpec(
        key="fc_bayern_vs_frankfurt_2018_19",
        video_path=PROJECT_ROOT / "fc_bayern_vs_frankfurt_2018_19.mp4",
        feature_path=PROJECT_ROOT / "fc_bayern_vs_frankfurt_2018_19.npy",
        home_team="FC Bayern Munich",
        away_team="Eintracht Frankfurt",
        competition="Bundesliga",
        season="2018/19",
    ),
    "kor_jpn": PreloadedMatchSpec(
        key="kor_jpn",
        video_path=PROJECT_ROOT / "kor_jpn.mp4",
        feature_path=PROJECT_ROOT / "kor_jpn.npy",
        home_team="대한민국",
        away_team="일본",
    ),
    "real_madrid_vs_girona_2017_2018": PreloadedMatchSpec(
        key="real_madrid_vs_girona_2017_2018",
        video_path=PROJECT_ROOT / "real_madrid_vs_girona_2017_2018.mp4",
        feature_path=PROJECT_ROOT / "real_madrid_vs_girona_2017_2018.npy",
        home_team="Real Madrid",
        away_team="Girona",
        competition="La Liga",
        season="2017/18",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(DEFAULT_SPECS),
        help="Import only the selected key. Repeat to select multiple matches.",
    )
    parser.add_argument("--owner-user-id", default=None)
    parser.add_argument(
        "--link-mode",
        choices=["auto", "hardlink", "copy"],
        default="auto",
    )
    parser.add_argument(
        "--skip-video-sha256",
        action="store_true",
        help="Skip the upload-compatible video hash (the first job will hash it).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = get_settings()
    if settings.is_production:
        raise RuntimeError("Preloaded local matches cannot be imported in production.")

    keys = args.only or list(DEFAULT_SPECS)
    results = []
    with SessionLocal() as db:
        owner_id = args.owner_user_id or _resolve_owner_id(db)
        importer = PreloadedMatchImporter(db)
        for key in keys:
            result = importer.import_match(
                DEFAULT_SPECS[key],
                owner_id=owner_id,
                link_mode=args.link_mode,
                compute_video_sha256=not args.skip_video_sha256,
            )
            results.append(
                {
                    "key": key,
                    "match_id": result.match_id,
                    "owner_id": result.owner_id,
                    "raw_video_asset_id": result.raw_video_asset_id,
                    "feature_asset_ids": result.feature_asset_ids,
                    "video_duration_sec": round(result.video_duration_sec, 3),
                    "feature_shape": list(result.feature_shape),
                    "half1_shape": list(result.half1_shape),
                    "half2_shape": list(result.half2_shape),
                    "feature_fps": result.feature_fps,
                    "video_materialization": result.video_materialization,
                    "feature_materialization": result.feature_materialization,
                }
            )

    print(json.dumps({"imported": results}, ensure_ascii=False, indent=2))
    print()
    print("Action Spotting starts through the existing endpoint:")
    print("  POST /api/v1/action-spotting/matches/{match_id}/jobs")
    print('  {"run_feature_extraction":true,"feature_extraction_mode":"auto"}')
    return 0


def _resolve_owner_id(db) -> str:
    users = list(
        db.scalars(
            select(User)
            .where(User.is_active.is_(True))
            .order_by(User.created_at.asc())
            .limit(2)
        ).all()
    )
    if len(users) == 1:
        return users[0].user_id
    if not users:
        raise ValueError(
            "No active user exists. Register a user or pass --owner-user-id."
        )
    raise ValueError(
        "More than one active user exists. Pass --owner-user-id explicitly."
    )


if __name__ == "__main__":
    raise SystemExit(main())
