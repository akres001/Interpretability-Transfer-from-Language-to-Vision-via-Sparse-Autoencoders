import json
import time
from pathlib import Path

from google import genai as genaigoogle
from google.genai import types

from . import config


vertex_client = genaigoogle.Client(
    vertexai=True,
    project=config.VERTEX_PROJECT,
    location=config.VERTEX_LOCATION,
)

# _ai_studio_client = (
#     genaigoogle.Client(api_key=config.GOOGLE_API_KEY) if config.GOOGLE_API_KEY else None
# )


def chat_gemini(prompt, image=None, model="gemini-3-flash-preview", max_retries=3):
    """Single-shot Vertex AI call. Returns the response text (stripped)."""
    payload = [image, prompt] if image is not None else [prompt]
    for attempt in range(max_retries):
        try:
            response = vertex_client.models.generate_content(
                model=model,
                contents=payload,
                config=types.GenerateContentConfig(response_mime_type="application/json"),
            )
            result = response.text.strip() if response.text else ""
            if result:
                return result
        except Exception as e:
            print(f"Vertex attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(20)
    raise RuntimeError("All Vertex retries exhausted")



def process_loop(data_list, model="gemini-3-flash-preview", max_retries=3000, delay=0.5):
    """Sequential (non-batch) processing with exponential backoff.

    Used by the localization script as a fallback to actual batch jobs.
    """
    results = {}
    for i, item in enumerate(data_list):
        req_id = item["request_id"]
        prompt_i = item["prompt"]
        for attempt in range(max_retries):
            try:
                response = vertex_client.models.generate_content(
                    model=model,
                    contents=[{"role": "user", "parts": [{"text": prompt_i}]}],
                    config={"response_mime_type": "application/json"},
                )
                results[req_id] = response.text
                if i % 50 == 0:
                    print(f"{i}/{len(data_list)} done")
                break
            except Exception as e:
                print(f"Request {req_id} attempt {attempt + 1}: {e}")
                time.sleep(min(2 ** attempt, 60))
        else:
            print(f"Request {req_id} failed after {max_retries} retries")
            results[req_id] = None
        time.sleep(delay)
    return results


# ---------------------------------------------------------------------------
# Neuronpedia feature info (with persistent on-disk cache)
# ---------------------------------------------------------------------------
from neuronpedia.np_sae_feature import SAEFeature  # noqa: E402

_NP_CACHE_PATH = config.RESULTS_DIR / "neuronpedia_cache.json"
_neuronpedia_cache = {}

if _NP_CACHE_PATH.exists():
    try:
        raw = json.loads(_NP_CACHE_PATH.read_text())
        # Keys are stored as JSON strings of [layer, feature, model]
        _neuronpedia_cache = {tuple(json.loads(k)): v for k, v in raw.items()}
    except Exception as e:
        print(f"Could not load neuronpedia cache: {e}")


def _save_neuronpedia_cache():
    _NP_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    serial = {json.dumps(list(k)): v for k, v in _neuronpedia_cache.items()}
    _NP_CACHE_PATH.write_text(json.dumps(serial))


def get_feature_information(layer, feature, model="gemma-2-2b"):
    """Return [density, description] for an SAE feature. Cached on disk."""
    key = (layer, feature, model)
    if key in _neuronpedia_cache:
        return _neuronpedia_cache[key]
    try:
        if "gemma" in model:
            sae_id = f"{layer}-gemmascope-res-16k"
        elif "llama" in model:
            sae_id = f"{layer}-llamascope-res-32k"
        else:
            raise ValueError(f"Unknown model family for Neuronpedia: {model}")
        sae_feature = SAEFeature.get(model, sae_id, feature)
        data = json.loads(sae_feature.jsonData)
        result = [data["frac_nonzero"], data["explanations"][0]["description"]]
        _neuronpedia_cache[key] = result
        # Save every 50 lookups to keep the file warm without hammering disk
        if len(_neuronpedia_cache) % 50 == 0:
            _save_neuronpedia_cache()
        return result
    except Exception as e:
        print(f"Neuronpedia error (layer={layer}, feature={feature}, model={model}): {e}")
        return [None, None]


# For matching rate
SAE_MATCH_PROMPT = """
Analyze this image and determine if ANY of the provided SAE (Sparse Autoencoder) feature descriptions match a key concept in the image.

{descriptions_text}

First, write a brief description of all objects, activities, high-level concepts, etc. that are shown in the image.
Then, determine if ANY of the feature descriptions matches one of these.

Return your response strictly as valid JSON in the following format:

{{
  "image_description": "1-2 sentences describing the objects, activities, and high-level concepts in the image.",
  "matches": [
    {{
      "feature_id": 1,
      "explanation": "1 sentence explaining why this feature matches the image description."
    }}
  ]
}}

If no features match, return:
{{
  "image_description": "1-2 sentences describing the image.",
  "matches": []
}}
"""


# For steering
INTERPRETABILITY_PROMPT = """
ROLE: Visual Grounding Auditor for Latent Space Steering.

BACKGROUND CONTEXT:
- Original Baseline Text: "{baseline_text}"
- Steered Text to Evaluate: "{steered_text}"
- Target Keyword: "{keyword}"
- Source Concept to Replace: "{source_concept}"
- Operation: {concept_type_upper}

EVALUATION PROTOCOL:

1. STEERING SUCCESS (Mandatory Check):
   - If {concept_type_upper} == INJECT:
        - Is '{keyword}' present? AND Is '{source_concept}' absent?
        - If '{source_concept}' is still present, this is a "Collision" -> Score 0.
   - If {concept_type_upper} == REMOVE:
        - PRE-CONDITION: Check the 'Original Baseline Text'. Is '{keyword}' present in it?
        - If '{keyword}' is NOT found in the Baseline, the removal task is invalid -> Score 0.
        - VERIFICATION: Is '{keyword}' absent in the 'Steered Text'? If present -> Score 0.
   - If these conditions are met, proceed to grounding checks.

2. BASELINE COMPARISON (Semantic Delta):
   - Compare the STEERED_TEXT to the BASELINE_TEXT.
   - If the Baseline only described the object and the Steered Text does the same, Score 2.
   - Identify any NEW environmental details (background, scene, or extra items) in the Steered Text that were NOT in the Baseline.

3. IMAGE GROUNDING (The Tie-Breaker):
   - If you detected NEW environmental details not found in the Baseline:
     a) Verify them against the IMAGE.
     b) If the new background/details exist in the image, Score 2.
     c) If the new details are hallucinated (not in the image), Score 1.
   - IDENTITY IMMUNITY: Do NOT penalize if the '{keyword}' itself is not in the image (this is an expected steering hallucination).

SCORING:
- 2 (Success): Steering logic followed perfectly. Any extra details match the image or were in the baseline.
- 1 (Grounding Error): Steering succeeded, but the model added new scene details that are not in the image and weren't in the baseline.
- 0 (Fail/Collision): Steering failed. Either the target was not handled, or both the source and target concepts are present (Collision).

RESPONSE FORMAT (JSON ONLY):
{{
  "target_status": "Success/Collision/Fail",
  "baseline_delta": "Description of added details vs baseline",
  "score": int,
  "reasoning": "First, confirm collision/presence of concepts. Second, explain why the image was or was not used to penalize the score based on the baseline comparison."
}}
"""


# For localization
LEXICAL_AUDITOR_PROMPT = """You are a Strict Lexical Auditor.
Your goal: Verify if a specific "Keyword" appears explicitly within a "Concept Description."
 
RULES:
1. Exact String Match: Output 1 ONLY if the "{keyword}" (or its direct plural/singular form) is explicitly written within the "{concept_description}".
2. No Semantic Inference: Even if the Keyword is logically related to the Concept (e.g., Keyword: "Sink" vs Concept: "Items found in a bathroom"), you must output 0 if the word "sink" is not explicitly mentioned in the text.
3. Case Insensitivity: Matches should be case-insensitive ("Dog" matches "dog").
4. No Synonyms: Do NOT match synonyms. "Canine" is NOT a match for "Dog". "Automobile" is NOT a match for "Car". "Box" is NOT a match for "Pizza".
5. Partial Word Warning: Ensure the keyword is matched as a whole word (e.g., "Cat" should match "cats" or "cat", but be careful not to match "cat" inside "category" unless intended).
 
OUTPUT FORMAT:
Return ONLY a JSON object with the keys: 'analysis', 'match', and 'explanation'.
 
TASK: Verify explicit keyword presence.
KEYWORD: "{keyword}"
CONCEPT DESCRIPTION: "{concept_description}"
 
STEPS:
1. Scan: Read the "{concept_description}" carefully.
2. Search: Search for the exact string "{keyword}".
3. Compare:
    - Is the exact word present? -> 1
    - Is the word missing, even if the meaning is present? -> 0
4. Match Decision:
    - Exact match found -> 1
    - No exact match -> 0
 
JSON structure required:
{{
  "analysis": "List the words found in the description that were compared against the keyword.",
  "match": integer,
  "explanation": "State clearly if the exact keyword was found or if only related terms were found."
}}"""


def build_interpretability_prompt(keyword, steered_text, source_concept=None,
                                   baseline_text=None, concept_type="inject"):
    """Render INTERPRETABILITY_PROMPT with the right fields filled in."""
    return INTERPRETABILITY_PROMPT.format(
        baseline_text=baseline_text if baseline_text else "N/A",
        steered_text=steered_text,
        keyword=keyword,
        source_concept=source_concept if source_concept else "N/A",
        concept_type_upper=concept_type.upper(),
    )


def evaluate_interpretability(image, keyword, steered_text, source_concept=None,
                               baseline_text=None, concept_type="inject", max_retries=5):
    """Run the visual grounding auditor and return the parsed JSON dict."""
    prompt = build_interpretability_prompt(
        keyword=keyword,
        steered_text=steered_text,
        source_concept=source_concept,
        baseline_text=baseline_text,
        concept_type=concept_type,
    )
    for attempt in range(max_retries):
        try:
            raw = chat_gemini(prompt, image)
            return json.loads(raw)
        except Exception as e:
            print(f"Interpretability attempt {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(1)
    return {"target_status": "Fail", "score": 0,
            "reasoning": "Max retries reached"}


    
def evaluate_sae_features_batch(feature_descriptions, image,
                                  model="gemini-2.5-flash-lite"):
    """Ask Gemini whether any of the SAE feature descriptions match the image.
 
    Returns (match_result:int, explanation:str, feature_ids:list, full_data:dict).
    """
    descriptions_text = "\n".join(
        f"Feature {i + 1} description: '{desc}'"
        for i, desc in enumerate(feature_descriptions)
    )
    prompt = SAE_MATCH_PROMPT.format(descriptions_text=descriptions_text)
    response = chat_gemini(prompt, image=image, model=model)
 
    try:
        # Robust parsing: pull out the {...} body even if there's prose around it
        start = response.find("{")
        end = response.rfind("}")
        json_string = response[start:end + 1]
        data = json.loads(json_string)
        matches = data.get("matches", [])
        match_result = 1 if matches else 0
        explanation = "; ".join(m.get("explanation", "") for m in matches)
        feature_ids = [m.get("feature_id") for m in matches if m.get("feature_id") is not None]
        return match_result, explanation, feature_ids, data
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Failed to parse SAE match response: {e}\n---\n{response}\n---")
        return 0, "", [], {"error": str(e), "raw_response": response}
 