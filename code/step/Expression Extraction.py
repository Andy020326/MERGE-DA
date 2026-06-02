
# =============================================================================
#  LLM CONFIGURATION  ->  *** FILL IN YOUR OWN MODEL / API SETTINGS BELOW ***
# =============================================================================
#  This script calls a large language model in Phase 2. Put your model name,
#  key, endpoint and the actual call logic here. `call_llm` is the single entry
#  point used by the rest of the project.
#
#  Only `torch` / `transformers` are needed if you run a LOCAL HF model; comment
#  them out if you call a hosted API instead.
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
from collections import defaultdict, Counter

from tqdm import tqdm


# =============================================================================
#  CONFIGURATION  -  dataset-specific parameters, tune everything here
# =============================================================================
RAW_INPUT_FILE = "train.json"                 # source dataset (a JSON list)

# Phase 1 outputs / Phase 2 inputs.
COMMON_DIR = "expression"                     # common relations (down-sampled)
RARE_DIR   = "expression_rare"                # rare relations (kept in full)

# Phase 2 outputs (instances augmented with an extracted key phrase).
COMMON_PRO_DIR = "expression_pro"
RARE_PRO_DIR   = "expression_rare_pro"

# A relation is "rare" if its frequency < FREQUENCY_THRESHOLD, where
# frequency = (relation instance count) / (total number of items).
FREQUENCY_THRESHOLD = 0.2

# Common relations are down-sampled to (TARGET_COMMON_RATIO * total_items) instances each.
TARGET_COMMON_RATIO = 0.2

RANDOM_SEED = 42                              # seed for reproducible sampling

STATISTICS_REPORT_FILE = "relation_statistics.json"   # full per-relation report
PATTERNS_REPORT_FILE   = "relation_patterns.json"     # compact pattern summary


# =============================================================================
#  PROMPTS  -  edit to retune the key-phrase extraction
# =============================================================================
SYSTEM_PROMPT = (
    "You are a professional information-extraction assistant, skilled at "
    "precisely extracting relation expressions from text."
)

EXTRACTION_PROMPT_TEMPLATE = """Extract the key phrase from the text below that expresses the "{relation_label}" relation.

Relation triple: {relation}
Text: {text}

Requirements:
1. Extract only the most core, most concise phrase containing this relation.
2. The phrase must include both entities and the core expression of the relation.
3. Do not explain or add anything extra; output only the phrase itself.
4. If the relation appears in several places, choose the most typical one.

Key phrase:"""


# =============================================================================
#  PHASE 1  -  relation frequency analysis and instance sampling
# =============================================================================
def build_entity_type_map(entities: dict) -> dict:
    """Map every entity surface form to its entity type."""
    entity_to_type = {}
    for ent_type, ent_list in entities.items():
        for ent in ent_list:
            entity_to_type[ent] = ent_type
    return entity_to_type


def compute_statistics(data):
    """First pass: count relation instances and (head_type, tail_type) patterns.

    Returns relation_instance_counts, relation_entity_patterns, total_items,
    total_relation_instances.
    """
    relation_instance_counts = {}
    relation_entity_patterns = defaultdict(Counter)
    total_items = 0
    total_relation_instances = 0

    for item in tqdm(data, desc="Counting relations"):
        total_items += 1
        output_data = json.loads(item["output"])
        relations = output_data.get("relations", [])
        entity_to_type = build_entity_type_map(output_data.get("entities", {}))

        for relation in relations:
            if len(relation) >= 3:
                head, rel, tail = relation[0], relation[1], relation[2]
                relation_instance_counts[rel] = relation_instance_counts.get(rel, 0) + 1
                total_relation_instances += 1
                head_type = entity_to_type.get(head, "UNKNOWN")
                tail_type = entity_to_type.get(tail, "UNKNOWN")
                relation_entity_patterns[rel][(head_type, tail_type)] += 1

    return relation_instance_counts, relation_entity_patterns, total_items, total_relation_instances


def classify_relations(relation_instance_counts, total_items, threshold):
    """Split relations into common / rare by frequency = instances / total_items."""
    common, rare, frequencies = set(), set(), {}
    for rel, count in relation_instance_counts.items():
        freq = count / total_items
        frequencies[rel] = freq
        (rare if freq < threshold else common).add(rel)
    return common, rare, frequencies


def collect_relation_instances(data):
    """Second pass: gather all (text, relation) instances per relation (no dedup)."""
    relation_texts = defaultdict(list)
    for item in tqdm(data, desc="Collecting instances"):
        output_data = json.loads(item["output"])
        input_text = item["input"]
        for relation in output_data.get("relations", []):
            if len(relation) >= 3:
                relation_texts[relation[1]].append((input_text, tuple(relation)))
    return relation_texts


def safe_filename(name: str) -> str:
    """Turn a relation name into a filesystem-safe file stem."""
    safe = "".join(c for c in name if c.isalnum() or c in (" ", "-", "_")).rstrip()
    return safe or "unknown_relation"


def save_relation_files(relation_texts, rare_relations, target_common_count):
    """Write one JSON file per relation; down-sample common relations.

    Common relations go to COMMON_DIR, rare ones to RARE_DIR.
    Returns a summary dict of counters.
    """
    summary = {
        "common_files": 0,
        "rare_files": 0,
        "common_instances_original": 0,
        "common_instances_sampled": 0,
        "rare_instances": 0,
    }

    for relation_name, texts in relation_texts.items():
        if not texts:
            continue
        stem = safe_filename(relation_name)

        if relation_name in rare_relations:
            target_dir = RARE_DIR
            texts_to_save = texts                       # keep all rare instances
            summary["rare_files"] += 1
            summary["rare_instances"] += len(texts_to_save)
        else:
            target_dir = COMMON_DIR
            if len(texts) > target_common_count:
                texts_to_save = random.sample(texts, target_common_count)
            else:
                texts_to_save = texts
            summary["common_files"] += 1
            summary["common_instances_original"] += len(texts)
            summary["common_instances_sampled"] += len(texts_to_save)

        records = [{"text": t, "relation": list(r)} for t, r in texts_to_save]
        with open(os.path.join(target_dir, f"{stem}.json"), "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

    return summary


def save_reports(relation_instance_counts, relation_entity_patterns, relation_texts,
                 common_relations, rare_relations, total_items, total_relation_instances,
                 target_common_count):
    """Write the full statistics report and the compact pattern summary."""

    # ---- full per-relation report ----
    relations_detail = []
    for relation_name in sorted(relation_instance_counts.keys()):
        patterns = relation_entity_patterns[relation_name].most_common()
        total_patterns = sum(c for _, c in patterns)
        pattern_list = [
            {
                "head_type": h,
                "tail_type": t,
                "count": c,
                "ratio": round(c / total_patterns, 4) if total_patterns else 0,
            }
            for (h, t), c in patterns
        ]

        instance_count = len(relation_texts.get(relation_name, []))
        is_common = relation_name in common_relations
        is_sampled = is_common and instance_count > target_common_count
        sampled_count = target_common_count if is_sampled else instance_count

        relations_detail.append({
            "relation_name": relation_name,
            "instance_count": instance_count,
            "frequency": round(instance_count / total_items, 4),
            "sampled_count": sampled_count,
            "is_sampled": is_sampled,
            "sampled_frequency": round(sampled_count / total_items, 4),
            "type": "common" if is_common else "rare",
            "entity_patterns": pattern_list,
            "total_pattern_instances": total_patterns,
        })

    report_data = {
        "total_items": total_items,
        "total_relation_instances": total_relation_instances,
        "avg_relations_per_item": round(total_relation_instances / total_items, 2),
        "frequency_threshold": FREQUENCY_THRESHOLD,
        "target_common_ratio": TARGET_COMMON_RATIO,
        "target_common_count": target_common_count,
        "common_relations_count": len(common_relations),
        "rare_relations_count": len(rare_relations),
        "common_relations": sorted(common_relations),
        "rare_relations": sorted(rare_relations),
        "relations": relations_detail,
    }
    with open(STATISTICS_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

    # ---- compact pattern summary (top-3 patterns per relation) ----
    relation_patterns = {}
    for rel, pattern_counter in relation_entity_patterns.items():
        total_patterns = sum(pattern_counter.values())
        relation_patterns[rel] = {
            "type": "common" if rel in common_relations else "rare",
            "instance_count": relation_instance_counts[rel],
            "frequency": round(relation_instance_counts[rel] / total_items, 4),
            "top_patterns": [
                {"head": h, "tail": t, "ratio": round(c / total_patterns, 4)}
                for (h, t), c in pattern_counter.most_common(3)
            ],
        }

    quick_stats = {
        "total_items": total_items,
        "total_relation_instances": total_relation_instances,
        "frequency_threshold": FREQUENCY_THRESHOLD,
        "target_common_ratio": TARGET_COMMON_RATIO,
        "rare_relations": sorted(rare_relations),
        "common_relations": sorted(common_relations),
        "relation_patterns": relation_patterns,
    }
    with open(PATTERNS_REPORT_FILE, "w", encoding="utf-8") as f:
        json.dump(quick_stats, f, ensure_ascii=False, indent=2)


def run_phase1_sampling():
    """Analyse the dataset, sample instances, and write per-relation files + reports."""
    for folder in (COMMON_DIR, RARE_DIR):
        os.makedirs(folder, exist_ok=True)

    with open(RAW_INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {RAW_INPUT_FILE}")

    counts, patterns, total_items, total_instances = compute_statistics(data)
    common, rare, _ = classify_relations(counts, total_items, FREQUENCY_THRESHOLD)
    target_common_count = int(total_items * TARGET_COMMON_RATIO)

    print(f"Total items: {total_items} | relation instances: {total_instances} "
          f"| avg/item: {total_instances / total_items:.2f}")
    print(f"Common: {len(common)} | rare: {len(rare)} (threshold={FREQUENCY_THRESHOLD})")
    print(f"Down-sampling common relations to {target_common_count} instances each.")

    relation_texts = collect_relation_instances(data)
    summary = save_relation_files(relation_texts, rare, target_common_count)
    save_reports(counts, patterns, relation_texts, common, rare,
                 total_items, total_instances, target_common_count)

    print(f"Common -> '{COMMON_DIR}': {summary['common_files']} files, "
          f"{summary['common_instances_original']} -> {summary['common_instances_sampled']} instances")
    print(f"Rare   -> '{RARE_DIR}': {summary['rare_files']} files, "
          f"{summary['rare_instances']} instances")


# =============================================================================
#  PHASE 2  -  LLM key-phrase extraction
# =============================================================================
def extract_expression(text: str, relation: list) -> str:
    """Call the LLM to extract the key phrase expressing `relation` in `text`."""
    user_prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        relation=relation, relation_label=relation[1], text=text
    )
    expression = call_llm(SYSTEM_PROMPT, user_prompt)
    return expression.strip().strip('"').strip("'").strip()


def process_relation_file(input_path: str, output_path: str) -> bool:
    """Add an extracted key phrase to every instance in one relation file.

    Supports resume: if `output_path` already exists, processing continues from
    where it stopped. The output is saved after each item so an interrupted run
    loses no progress. Returns True on success, False if the input is unreadable.
    """
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Failed to read {input_path}: {e}")
        return False

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            processed = json.load(f)
    else:
        processed = []

    start = len(processed)
    for item in tqdm(data[start:], desc=os.path.basename(input_path),
                     initial=start, total=len(data)):
        text = item.get("text", "")
        relation = item.get("relation", [])
        if text and relation:
            try:
                expr = extract_expression(text, relation)
            except Exception as e:
                print(f"  LLM call failed: {e}")
                expr = ""
            item["expression"] = expr
            item["relevant_part"] = expr
        # else: invalid instance, kept as-is

        processed.append(item)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(processed, f, ensure_ascii=False, indent=2)

    return True


def process_relation_folder(input_dir: str, output_dir: str, relation_type: str) -> dict:
    """Run key-phrase extraction over every relation file in a folder."""
    if not os.path.isdir(input_dir):
        print(f"Input folder not found: {input_dir}")
        return {"processed": 0, "errors": 0}

    os.makedirs(output_dir, exist_ok=True)
    counters = {"processed": 0, "errors": 0}
    print(f"\n[{relation_type}] '{input_dir}' -> '{output_dir}'")
    for filename in sorted(os.listdir(input_dir)):
        if not filename.endswith(".json"):
            continue
        ok = process_relation_file(os.path.join(input_dir, filename),
                                   os.path.join(output_dir, filename))
        counters["processed" if ok else "errors"] += 1
    return counters


def run_phase2_extraction():
    """Extract key phrases for both common and rare relations."""
    common = process_relation_folder(COMMON_DIR, COMMON_PRO_DIR, "common")
    rare = process_relation_folder(RARE_DIR, RARE_PRO_DIR, "rare")
    print(f"\nExtraction done | common: {common['processed']} files "
          f"({common['errors']} failed) | rare: {rare['processed']} files "
          f"({rare['errors']} failed)")


# =============================================================================
#  MAIN
# =============================================================================
def main():
    random.seed(RANDOM_SEED)

    print("=== Phase 1: relation frequency analysis & sampling ===")
    run_phase1_sampling()

    print("\n=== Phase 2: LLM key-phrase extraction ===")
    run_phase2_extraction()


if __name__ == "__main__":
    main()