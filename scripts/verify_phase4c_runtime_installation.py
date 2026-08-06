from __future__ import annotations

import json

from app.domains.tracking.verifier import (
    get_scene_target_tracking_verifier,
)


def main() -> int:
    status = get_scene_target_tracking_verifier().check(force=True)
    print(f"status={'PASS' if status.available else 'FAIL'}")
    print(f"code={status.code}")
    print(f"message={status.message}")
    print(f"verifier_return_code={status.verifier_return_code}")
    print(
        "components="
        + json.dumps(status.components or {}, ensure_ascii=False)
    )
    return 0 if status.available else 2


if __name__ == "__main__":
    raise SystemExit(main())
