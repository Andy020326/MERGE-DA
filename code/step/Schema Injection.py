
# =============================================================================
#  LLM CONFIGURATION  ->  *** FILL IN YOUR OWN MODEL / API SETTINGS BELOW ***
# =============================================================================
#  Kept here for project-wide consistency. This step only rewrites instructions
#  and does NOT call the LLM, so you may comment out the `torch` / `transformers`
#  imports if you run only this step.
# -----------------------------------------------------------------------------
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
# from openai import OpenAI            # or anthropic / your provider's SDK

LLM_MODEL_NAME = ""    # e.g. "Qwen/Qwen3-235B-A22B", "meta-llama/Meta-Llama-3-8B-Instruct", "gpt-4o"
LLM_API_KEY    = ""    # API key, if you use a hosted model
LLM_API_BASE   = ""    # custom endpoint, leave "" for the provider default


def call_llm(system_prompt: str, user_prompt: str, **kwargs) -> str:
    """Send a (system_prompt, user_prompt) pair to your LLM and return the text.

    Replace the body with your own implementation (hosted API or local HF model).
    """
    raise NotImplementedError("Implement call_llm() with your own model / API.")


# =============================================================================
#  GENERAL IMPORTS
# =============================================================================
import json
import os
import random

from tqdm import tqdm


# =============================================================================
#  CONFIGURATION  -  dataset-specific parameters, tune everything here
# =============================================================================
INPUT_FILE      = "train_event.json"            # event-text items from Step 4
OUTPUT_FILE     = "train_final.json"            # final schema-augmented dataset
STATISTICS_FILE = "relation_statistics.json"    # schema source (from Step 1)

RANDOM_SCHEMA_PROB =      # probability of adding each extra (non-present) type, for diversity
RANDOM_SEED        = 42      # seed for reproducible schema augmentation

# Entity types always included in the schema (dataset-specific; adjust for your data).
COMMON_ENTITY_TYPES = {"PER", "ORG", "LOC", "TIME", "MISC", "NUM"}


# =============================================================================
#  PROMPTS  -  edit to retune the instruction wording
# =============================================================================
# Base instruction used for items that have none.
DEFAULT_INSTRUCTION = "Please analyze the entities and relations in the text."

SCHEMA_INSTRUCTION_TEMPLATE = (
    "{base_instruction}\n\n"
    "Please extract entities and relations according to the following schema:\n"
    "{schema_json}\n\n"
    "Note: You may encounter entities and relations beyond this schema, "
    "extract them as well if present in the text."
)


# =============================================================================
#  CORE LOGIC
# =============================================================================
def load_schema_from_statistics():
    """Build the full entity/relation schema from the Step 1 statistics report."""
    print(f"Loading schema from {STATISTICS_FILE}...")
    with open(STATISTICS_FILE, "r", encoding="utf-8") as f:
        stats = json.load(f)

    entity_types = set()
    relation_types = set()
    for rel_info in stats.get("relations", []):
        relation_types.add(rel_info["relation_name"])
        for pattern in rel_info.get("entity_patterns", []):
            entity_types.add(pattern.get("head_type"))
            entity_types.add(pattern.get("tail_type"))

    entity_types.update(COMMON_ENTITY_TYPES)
    entity_types.discard(None)

    schema = {"entities": sorted(entity_types), "relations": sorted(relation_types)}
    print(f"Schema: {len(schema['entities'])} entity types, {len(schema['relations'])} relation types")
    return schema


def get_relevant_schema(output_data, full_schema):
    """Return the entity/relation types that actually occur in one item."""
    try:
        output_dict = json.loads(output_data)
    except Exception:
        return {"entities": [], "relations": []}

    relevant_entities = [
        et for et in output_dict.get("entities", {}) if et in full_schema["entities"]
    ]
    relevant_relations = list({
        rel[1] for rel in output_dict.get("relations", [])
        if len(rel) >= 3 and rel[1] in full_schema["relations"]
    })
    return {"entities": relevant_entities, "relations": relevant_relations}


def add_random_schema(relevant_schema, full_schema, prob=RANDOM_SCHEMA_PROB):
    """Add random extra types not already present (for diversity), then shuffle."""
    extended = {
        "entities": list(relevant_schema["entities"]),
        "relations": list(relevant_schema["relations"]),
    }
    for entity in full_schema["entities"]:
        if entity not in relevant_schema["entities"] and random.random() < prob:
            extended["entities"].append(entity)
    for relation in full_schema["relations"]:
        if relation not in relevant_schema["relations"] and random.random() < prob:
            extended["relations"].append(relation)
    random.shuffle(extended["entities"])
    random.shuffle(extended["relations"])
    return extended


def build_instruction_with_schema(base_instruction, schema):
    """Append a JSON schema description to the base instruction."""
    schema_json = json.dumps(
        {"entities": schema["entities"], "relations": schema["relations"]},
        ensure_ascii=False, indent=2,
    )
    return SCHEMA_INSTRUCTION_TEMPLATE.format(
        base_instruction=base_instruction, schema_json=schema_json
    )


def process_dataset(input_file, output_file, full_schema):
    """Append a schema description to every item's instruction."""
    print(f"Processing {input_file}...")
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total items: {len(data)}")

    for item in tqdm(data, desc="Adding schema"):
        base_instruction = item.get("instruction", DEFAULT_INSTRUCTION)
        relevant_schema = get_relevant_schema(item.get("output", "{}"), full_schema)
        extended_schema = add_random_schema(relevant_schema, full_schema)
        item["instruction"] = build_instruction_with_schema(base_instruction, extended_schema)
        # Other fields (input, output, evaluation) are left unchanged.

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(data)} items to {output_file}")
    return data


def verify_output(output_file):
    """Print a quick summary of the first few output items."""
    with open(output_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\nVerifying {output_file} ({len(data)} items):")
    for i, item in enumerate(data[:3]):
        print(f"  Item {i + 1}: instruction {len(item.get('instruction', ''))} chars, "
              f"input starts {item.get('input', '')[:60]!r}")
        try:
            output = json.loads(item.get("output", "{}"))
            n_entities = sum(len(v) for v in output.get("entities", {}).values())
            n_relations = len(output.get("relations", []))
            print(f"           entities: {n_entities}, relations: {n_relations}")
        except Exception:
            print("           failed to parse output")


# =============================================================================
#  MAIN
# =============================================================================
def main():
    random.seed(RANDOM_SEED)
    full_schema = load_schema_from_statistics()
    process_dataset(INPUT_FILE, OUTPUT_FILE, full_schema)
    verify_output(OUTPUT_FILE)
    print("\nAll done!")


if __name__ == "__main__":
    main()