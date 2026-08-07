from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any


def render_gallery_html(
    *,
    output_root: Path,
    candidates: dict[str, Any],
    gallery: dict[str, Any],
) -> Path:
    by_id = {
        row["candidate_id"]: row for row in candidates.get("candidates", [])
    }
    sections = []
    for shot in gallery.get("shots", []):
        cards = []
        for candidate_id in shot.get("candidate_ids", []):
            candidate = by_id[candidate_id]
            representative = candidate["representative_observation"]
            visibility = candidate.get(
                "gallery_visibility", "VISIBLE"
            )
            cards.append(
                f"""
                <article class="candidate {('hidden-low-quality' if visibility != 'VISIBLE' else '')}" data-visible="{html.escape(visibility)}">
                  <img src="{html.escape(representative['thumbnail_artifact'])}" alt="candidate" loading="lazy">
                  <video src="{html.escape(candidate['artifacts']['tracklet_review_video'])}" controls preload="none"></video>
                  <p><strong>{html.escape(candidate_id)}</strong></p>
                  <p>frame {candidate['first_frame']}–{candidate['last_frame']} ·
                     trackability {candidate['quality']['trackability_score']:.3f}</p>
                  <button data-candidate-id="{html.escape(candidate_id)}" onclick="selectCandidate(this)">이 선수 선택</button>
                </article>
                """
            )
        sections.append(
            f"""
            <details open>
              <summary>{html.escape(shot['shot_id'])} — {shot['start_time_sec']:.2f}~{shot['end_time_sec']:.2f}s
                ({len(cards)} candidates)</summary>
              <div class="grid">{''.join(cards)}</div>
            </details>
            """
        )
    target = output_root / "candidate_gallery.html"
    target.write_text(
        f"""<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><title>Scene-wide target gallery</title>
<style>
body{{font-family:system-ui;background:#0b1020;color:#eef2ff;margin:24px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(280px,1fr));gap:16px}}
.candidate{{background:#151d34;padding:12px;border-radius:12px}}
img,video{{width:100%;max-height:240px;object-fit:contain;background:#000}}
summary{{font-size:18px;font-weight:700;margin:22px 0 12px}}
button{{padding:10px 14px}} .hidden{{opacity:.55}}
.hidden-low-quality{{display:none}} body.show-hidden .hidden-low-quality{{display:block;opacity:.62}}
#selection-status{{position:sticky;top:8px;background:#16213f;padding:12px;border-radius:10px}}
</style></head><body>
<h1>이 하이라이트에서 추적할 선수를 선택하세요.</h1>
<p>첫 화면에 없는 선수도 선택할 수 있습니다. 점수는 자동 선택에 사용되지 않습니다.</p>
<p><button onclick="document.body.classList.toggle('show-hidden')">숨겨진 낮은 품질 후보 보기/숨기기</button></p>
<p id="selection-status">선택 대기 중</p>
{''.join(sections)}
<script>
async function selectCandidate(button) {{
  const candidateId = button.dataset.candidateId;
  const integration = window.KICKCLIP_TARGET_SELECTION_CONFIG || {{}};
  document.getElementById('selection-status').textContent = '사용자 선택: ' + candidateId;
  window.dispatchEvent(new CustomEvent('kickclip:scene-target-selected', {{detail: {{candidateId}}}}));
  if (integration.endpoint) {{
    const response = await fetch(integration.endpoint, {{
      method: 'POST', headers: {{'Content-Type':'application/json'}},
      body: JSON.stringify({{...(integration.payload || {{}}), candidate_id:candidateId, scene_id:{json.dumps(candidates.get('scene_id'))}}})
    }});
    if (!response.ok) document.getElementById('selection-status').textContent = '선택 저장 실패: ' + await response.text();
  }} else if (navigator.clipboard) {{
    await navigator.clipboard.writeText(candidateId);
  }}
}}
</script>
</body></html>
""",
        encoding="utf-8",
    )
    return target
