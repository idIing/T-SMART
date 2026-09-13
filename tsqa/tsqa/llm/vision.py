import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


VISION_SYSTEM_PROMPT = """\
You are an objective Time Series Topological Sensor. Your sole function is to observe the provided image artifact (e.g., line plot or spectrogram) and return a strict JSON description of its topology and geometry.

You MUST NOT see the original question, answer choices, or attempt to solve the problem.
Output exactly ONE JSON object using the following geometric/topological dictionary:
    - pattern_type: (e.g., "monotonic", "oscillatory", "irregular", "flat")
    - trend_direction: (e.g., "increasing", "decreasing", "mixed", "none")
    - noise_level: (e.g., "low", "medium", "high")
    - dominant_ridges: list of dominant frequencies or spectral ridges
    - anomaly_regions: list of structural breaks, spikes, or anomalous regions (use normalized coordinates or panel indices)
    - phase_alignment: (e.g., "aligned", "offset", "unknown")
    - confidence: float between 0.0 and 1.0 indicating signal clarity
    - summary: a purely descriptive summary (< 30 words)

Do not add commentary, recommendations, or markdown wrapping outside of the JSON. If a feature is obscured, use null.
"""


REQUIRED_VISION_KEYS = [
    "pattern_type",
    "trend_direction",
    "noise_level",
    "dominant_ridges",
    "anomaly_regions",
    "phase_alignment",
    "confidence",
    "summary",
]


def build_vision_user_prompt(
    viz_type=None, layout=None, series_map=None, color_map=None, guidance=None
):
    """Construct a concise user prompt for the vision sensor using only metadata.

    This function never includes the user's question or answer choices.
    """
    viz_hint = f"Viz type: {viz_type}." if viz_type else ""
    layout_hint = f"Layout: {layout}." if layout else ""

    meta = {
        "viz_type": viz_type,
        "layout": layout,
    }
    if series_map:
        meta["series_map"] = series_map
    if color_map:
        meta["color_map"] = color_map

    meta_block = json.dumps(meta, indent=2)
    guidance_block = f"\nGuidance:\n{guidance}\n" if guidance else ""

    return (
        "Analyze the provided image artifact and return the JSON object described in the system prompt."
        + ("\n" + viz_hint if viz_hint else "")
        + ("\n" + layout_hint if layout_hint else "")
        + guidance_block
        + "\nMetadata (use this to reference panels/colors; do NOT answer the user's question):\n"
        + meta_block
        + "\nRespond ONLY with the JSON object."
    )


def _extract_json(text: str):
    if not text:
        return None

    # Optional debug print if you want to see exactly what goes into the extractor
    # print(f"Extracting JSON from: {text}")

    # Remove any ```json or ``` markdown blocks first so they don't trip up regex
    text = re.sub(r"```[A-Za-z]*|```", "", text)

    # Find the outermost JSON object
    brace_m = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_m:
        json_str = brace_m.group()
        try:
            return json.loads(json_str, strict=False)
        except json.JSONDecodeError as e:
            # Maybe the model left trailing commas or used single quotes?
            # We skip advanced repair for now and just return None to log fallback.
            return None
    return None


def analyze_image(
    llm_client,
    image_path: str,
    viz_type: str | None = None,
    layout: str | None = None,
    series_map: dict | None = None,
    color_map: dict | None = None,
    guidance: str | None = None,
) -> dict:
    """Call the vision-capable LLM to produce a structured, sensor-style JSON.

    Args:
        llm_client: object with .generate(system_prompt, user_prompt, artifacts=...)
        image_path: path to the image file to analyze
        viz_type: optional hint ('line_plot' or 'spectrogram')
        layout: optional layout hint (e.g. 'horizontal_stack','vertical_stack','single')
        series_map: optional mapping that describes which series corresponds to which panel or color. Example: {"panel_0": "ts1", "panel_1": "ts2"} or {"#1f77b4": "ts1", "#ff7f0e": "ts2"}
        color_map: optional color hex -> series id mapping

    Returns:
        dict with keys: vision_struct (dict or None), vision_text (summary or raw), raw (raw response)
    """
    # Minimal user prompt: only indicate the artifact metadata. Do NOT include
    # the user's question or answer choices.
    viz_hint = f"\nViz type: {viz_type}." if viz_type else ""
    layout_hint = f"\nLayout: {layout}." if layout else ""

    meta = {}
    meta["viz_type"] = viz_type
    meta["layout"] = layout
    if series_map:
        meta["series_map"] = series_map
    if color_map:
        meta["color_map"] = color_map

    user_prompt = build_vision_user_prompt(
        viz_type, layout, series_map, color_map, guidance
    )

    artifacts = [{"kind": "image", "path": image_path}]

    raw = llm_client.generate(VISION_SYSTEM_PROMPT, user_prompt, artifacts=artifacts)
    struct = _extract_json(raw)
    if struct is None:
        # Fallback: return raw text as 'vision_text' so the pipeline can log it
        return {"vision_struct": None, "vision_text": raw.strip(), "raw": raw}

    # Enforce and normalize output shape: fill missing keys with sensible defaults
    if isinstance(struct, dict):
        normalized = {}
        for k in REQUIRED_VISION_KEYS:
            if k in struct:
                normalized[k] = struct[k]
            else:
                # defaults: null for scalars, [] for list fields
                if k in ("dominant_ridges", "anomaly_regions"):
                    normalized[k] = []
                elif k == "confidence":
                    normalized[k] = 0.0
                else:
                    normalized[k] = None
        summary = normalized.get("summary")
        if isinstance(summary, str):
            normalized["summary"] = summary.strip()
        return {
            "vision_struct": normalized,
            "vision_text": normalized.get("summary"),
            "raw": raw,
        }

    return {"vision_struct": None, "vision_text": raw.strip(), "raw": raw}
