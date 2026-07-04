from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dim) for dim in shape]
    except Exception:
        return None


def _summarize_state_dict(state_dict: dict[str, Any], max_items: int = 40) -> dict[str, Any]:
    items = []
    for index, (key, value) in enumerate(state_dict.items()):
        if index >= max_items:
            break
        items.append(
            {
                "key": key,
                "shape": _shape_of(value),
                "dtype": str(getattr(value, "dtype", "")) or None,
            }
        )
    return {
        "num_tensors": len(state_dict),
        "sample_tensors": items,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect a KickClip highlight checkpoint before copying it into the backend."
    )
    parser.add_argument("checkpoint", help="Path to best.pt/checkpoint.pt")
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)

    try:
        import torch  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "torch is required to inspect a .pt checkpoint. Install torch in this environment first."
        ) from exc

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

    report: dict[str, Any] = {
        "checkpoint_path": checkpoint_path.as_posix(),
        "checkpoint_type": type(checkpoint).__name__,
        "is_dict": isinstance(checkpoint, dict),
    }

    if isinstance(checkpoint, dict):
        report["top_level_keys"] = list(checkpoint.keys())
        for key in ["state_dict", "model_state_dict", "model", "net"]:
            value = checkpoint.get(key)
            if isinstance(value, dict):
                report[key] = _summarize_state_dict(value)

        # Some training scripts save the state_dict itself as the checkpoint dict.
        tensor_like_values = [
            value
            for value in checkpoint.values()
            if _shape_of(value) is not None
        ]
        if tensor_like_values:
            report["checkpoint_dict_as_state_dict"] = _summarize_state_dict(checkpoint)
    else:
        report["repr"] = repr(checkpoint)[:1000]

    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
