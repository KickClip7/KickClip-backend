from uuid import uuid4


def generate_prefixed_id(prefix: str) -> str:
    """Generate a readable unique id.

    예:
    - proj_ab12cd34ef56
    - match_ab12cd34ef56
    - asset_ab12cd34ef56
    """
    return f"{prefix}_{uuid4().hex[:12]}"