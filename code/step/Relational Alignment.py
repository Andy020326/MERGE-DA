
# =============================================================================
#  LLM CONFIGURATION  ->  *** FILL IN YOUR OWN MODEL / API SETTINGS BELOW ***
# =============================================================================
#  This step calls a large language model. Put your model name, key, endpoint
#  and the actual call logic here. `call_llm` is the single entry point used by
#  the rest of the project.
#
#  Only `torch` / `transformers` are needed for a LOCAL HF model; comment them
#  out if you call a hosted API instead.
# -----------------------------------------------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from openai import OpenAI            # or anthropic / your provider's SDK

LLM_MODEL_NAME = ""    # e.g. "Qwen/Qwen3-235B-A22B", "meta-llama/Meta-Llama-3-8B-Instruct", "gpt-4o"
LLM_API_KEY    = ""    # API key, if you use a hosted model
LLM_API_BASE   = ""    # custom endpoint, leave "" for the provider default


def call_llm(system_prompt: str, user_prompt: str, **kwargs) -> str:
    """Send a (system_prompt, user_prompt) pair to your LLM and return the text.

    Replace the body with your own implementation, e.g. an OpenAI-compatible
    hosted endpoint:

        # client = OpenAI(api_key=LLM_API_KEY, base_url=LLM_API_BASE or None)
        # resp = client.chat.completions.create(
        #     model=LLM_MODEL_NAME,
        #     messages=[{"role": "system", "content": system_prompt},
        #               {"role": "user",   "content": user_prompt}],
        # )
        # return resp.choices[0].message.content.strip()

    or a local Hugging Face model loaded from LLM_MODEL_NAME.
    """
    raise NotImplementedError("Implement call_llm() with your own model / API.")


# =============================================================================
#  GENERAL IMPORTS
# =============================================================================
import json
import os
import random
import re

from tqdm import tqdm


# =============================================================================
#  CONFIGURATION  -  dataset-specific parameters, tune everything here
# =============================================================================
INPUT_FILE  = "train_generated.json"            # graphs produced by Step 2
OUTPUT_FILE = "train_generated_text.json"       # output: graphs + generated text

CORE_EXPRESSIONS_COMMON = "expression_pro"      # common-relation key phrases (Step 1)
CORE_EXPRESSIONS_RARE   = "expression_rare_pro" # rare-relation key phrases (Step 1)

MAX_FEW_SHOT_EXAMPLES   = 3      # few-shot examples per relation
MAX_GENERATION_ATTEMPTS = 5      # LLM retries per relation before falling back
SAVE_INTERVAL           = 10     # write the output file every N items

# Readability gate: a generated text is accepted if its Flesch-Kincaid scores are
# within these distances of the reference (loose thresholds = rarely rejected).
READABILITY_EASE_THRESHOLD  = 
READABILITY_GRADE_THRESHOLD = 

RANDOM_SEED = 42                 # seed for reproducible few-shot sampling


# =============================================================================
#  PROMPTS  -  edit to retune the text generation
# =============================================================================
TEXT_GEN_SYSTEM_PROMPT = ""      # system message sent with every generation call

FEW_SHOT_HEADER = "Here are some examples of relations and their expressions:\n"

GENERATION_PROMPT_TEMPLATE = (
    "{few_shot}Please generate a natural text expression for this relation: {relation}\n\n"
    "Return a JSON with these fields:\n"
    "- 'relation': the relation (i.e., {relation})\n"
    "- 'reference': the reference expression you used\n"
    "- 'generated_text': your generated expression\n\n"
    "Return only JSON, no explanation."
)

# Fallback sentence used when every generation attempt fails.
DEFAULT_TEXT_TEMPLATE = "{head} has a {rel_type} relationship with {tail}."

# Default 'instruction' field for output items that have none.
DEFAULT_INSTRUCTION = "Please analyze the entities and relations in the following text."


# =============================================================================
#  READABILITY (simplified Flesch-Kincaid, English + Chinese)
# =============================================================================
def count_sentences(text):
    sentences = re.split(r"[.!?;。！？；]", text)
    return len([s for s in sentences if s.strip()])


def count_words(text):
    """Count English words and individual Chinese characters."""
    return len(re.findall(r"\b[a-zA-Z]+\b|[\u4e00-\u9fa5]", text))


def count_syllables(text):
    """Approximate syllable count by the number of letters / Chinese characters."""
    return len(re.findall(r"[\u4e00-\u9fa5a-zA-Z]", text))


def flesch_kincaid_readability(text):
    """Return (reading_ease, grade_level) for `text`."""
    num_sentences = count_sentences(text)
    num_words = count_words(text)
    num_syllables = count_syllables(text)

    asl = num_words / num_sentences if num_sentences > 0 else 0      # avg sentence length
    asw = num_syllables / num_words if num_words > 0 else 0          # avg syllables per word

    reading_ease = 206.835 - (1.015 * asl) - (84.6 * asw)
    grade_level = (0.39 * asl) + (11.8 * asw) - 15.59
    return reading_ease, grade_level


def compare_readability(text1, text2,
                        ease_threshold=READABILITY_EASE_THRESHOLD,
                        grade_threshold=READABILITY_GRADE_THRESHOLD):
    """Return (is_similar, ease_diff, grade_diff) for two texts."""
    ease1, grade1 = flesch_kincaid_readability(text1)
    ease2, grade2 = flesch_kincaid_readability(text2)
    ease_diff = abs(ease1 - ease2)
    grade_diff = abs(grade1 - grade2)
    is_similar = ease_diff <= ease_threshold and grade_diff <= grade_threshold
    return is_similar, ease_diff, grade_diff


# =============================================================================
#  FEW-SHOT EXAMPLES + GENERATION
# =============================================================================
def load_few_shot_examples(relation_type, max_examples=MAX_FEW_SHOT_EXAMPLES):
    """Load few-shot examples for a relation from the common then rare folders."""
    relation_file = os.path.join(CORE_EXPRESSIONS_COMMON, f"{relation_type}.json")
    if not os.path.exists(relation_file):
        relation_file = os.path.join(CORE_EXPRESSIONS_RARE, f"{relation_type}.json")
        if not os.path.exists(relation_file):
            return []

    with open(relation_file, "r", encoding="utf-8") as f:
        examples = json.load(f)

    if len(examples) > max_examples:
        examples = random.sample(examples, max_examples)
    return examples


def build_few_shot_block(examples):
    """Render few-shot examples into a prompt prefix."""
    if not examples:
        return ""
    block = FEW_SHOT_HEADER
    for i, example in enumerate(examples):
        rel = example.get("relation", [])
        text = example.get("relevant_part", "") or example.get("expression", "")
        block += f"Example {i + 1} - Relation: {rel}\nExpression: {text}\n\n"
    return block


def generate_relation_with_llm(head, rel_type, tail, few_shot_examples,
                               max_attempts=MAX_GENERATION_ATTEMPTS):
    """Generate a natural-language expression for one relation triple.

    Returns (result_dict, ease_diff, grade_diff). On repeated failure a default
    template sentence is returned instead.
    """
    current_relation = [head, rel_type, tail]
    prompt = GENERATION_PROMPT_TEMPLATE.format(
        few_shot=build_few_shot_block(few_shot_examples), relation=current_relation
    )

    for attempt in range(max_attempts):
        try:
            response = call_llm(TEXT_GEN_SYSTEM_PROMPT, prompt)
            cleaned = response.strip().strip("```json").strip("```").strip()
            result = json.loads(cleaned)

            for field in ("relation", "reference", "generated_text"):
                if field not in result:
                    raise ValueError(f"Missing field: {field}")
            if result["relation"] != current_relation:
                raise ValueError("Relation mismatch")

            reference = result.get("reference", "")
            generated = result.get("generated_text", "")
            if reference:
                similar, ease_diff, grade_diff = compare_readability(generated, reference)
                if not similar and attempt < max_attempts - 1:
                    raise ValueError("Readability mismatch")
            else:
                ease_diff, grade_diff = 0, 0
            return result, ease_diff, grade_diff

        except Exception:
            if attempt == max_attempts - 1:
                fallback_reference = (
                    few_shot_examples[0].get("relevant_part", "") if few_shot_examples else ""
                )
                return {
                    "relation": current_relation,
                    "reference": fallback_reference,
                    "generated_text": DEFAULT_TEXT_TEMPLATE.format(
                        head=head, rel_type=rel_type, tail=tail
                    ),
                }, 0, 0


# =============================================================================
#  MAIN PROCESSING
# =============================================================================
def process_relations_with_llm(input_file, output_file):
    """Generate text for every relation of every item and write the results."""
    print(f"Loading data from {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total items: {len(data)}")

    results = []
    for idx, item in enumerate(tqdm(data, desc="Generating text")):
        try:
            output = json.loads(item["output"])
        except Exception:
            continue

        relations = output.get("relations", [])
        generated_texts = []
        readability_results = []
        relation_details = []

        for rel in relations:
            if len(rel) != 3:
                generated_texts.append("")
                readability_results.append(None)
                relation_details.append(None)
                continue

            head, rel_type, tail = rel
            few_shot_examples = load_few_shot_examples(rel_type, MAX_FEW_SHOT_EXAMPLES)
            gen_result, ease_diff, grade_diff = generate_relation_with_llm(
                head, rel_type, tail, few_shot_examples
            )

            generated_texts.append(gen_result["generated_text"])
            readability_results.append({"ease_diff": ease_diff, "grade_diff": grade_diff})
            relation_details.append({
                "relation": rel,
                "generated_text": gen_result["generated_text"],
                "reference": gen_result.get("reference", ""),
            })

        results.append({
            "instruction": item.get("instruction", DEFAULT_INSTRUCTION),
            "input": item.get("input", ""),
            "output": json.dumps(output, ensure_ascii=False),
            "generated_texts": generated_texts,
            "readability_results": readability_results,
            "relation_details": relation_details,
        })

        # Periodic save so an interrupted run keeps its progress.
        if (idx + 1) % SAVE_INTERVAL == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDone. {len(results)} items saved to {output_file}")


def main():
    random.seed(RANDOM_SEED)
    process_relations_with_llm(INPUT_FILE, OUTPUT_FILE)


if __name__ == "__main__":
    main()