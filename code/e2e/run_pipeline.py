
# =============================================================================
#  IMPORTS
# =============================================================================
import argparse
import json
import os
import random
import re
import traceback
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

# Optional heavy deps for a LOCAL HF model, wrapped so that steps which do not
# call the LLM (e.g. steps 2 and 5) can still run without torch/transformers.
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
except ImportError:
    torch = None
    AutoTokenizer = AutoModelForCausalLM = None
# from openai import OpenAI            # or anthropic / your provider's SDK


# =============================================================================
#  LLM CONFIGURATION  ->  *** IMPLEMENT call_llm() BELOW ***
# =============================================================================
#  Steps 1, 3 and 4 reach the model through the single entry point `call_llm`.
#  The model name comes from --llm-model (run.sh); secrets come from the
#  environment so they never sit on the command line: LLM_API_KEY, LLM_API_BASE.
# -----------------------------------------------------------------------------
LLM_MODEL_NAME = ""                                   # overridable via --llm-model
LLM_API_KEY    = os.environ.get("LLM_API_KEY", "")    # export it in run.sh
LLM_API_BASE   = os.environ.get("LLM_API_BASE", "")   # export it in run.sh


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
#  CONFIGURATION  -  defaults (any of these can be overridden from run.sh / CLI)
# =============================================================================
# --- Shared ---
RANDOM_SEED   = 42               # seed re-applied at the start of every step
SAVE_INTERVAL = 10               # steps 3 & 4 write their output file every N items

# --- Data files (the pipeline chains these; defined once, referenced everywhere) ---
ORIGINAL_DATA       = "train.json"                  # external input (you provide this)
STATISTICS_REPORT   = "relation_statistics.json"    # step1 -> steps 2, 5
PATTERNS_REPORT     = "relation_patterns.json"      # step1 (inspection only; not read downstream)
COMMON_DIR          = "expression"                  # step1 phase 1 -> phase 2
RARE_DIR            = "expression_rare"             # step1 phase 1 -> phase 2
COMMON_PRO_DIR      = "expression_pro"              # step1 phase 2 -> step3 few-shot
RARE_PRO_DIR        = "expression_rare_pro"         # step1 phase 2 -> step3 few-shot
GENERATED_FILE      = "train_generated.json"        # step2 -> step3
GENERATED_TEXT_FILE = "train_generated_text.json"   # step3 -> step4
REFERENCE_FILE      = "reference_examples.json"     # step4 phase 1 cache
EVENT_FILE          = "train_event.json"            # step4 -> step5
FINAL_FILE          = "train_final.json"            # final output

# --- Step 1: relation sampling + key-phrase extraction ---
RARE_FREQUENCY_THRESHOLD =    # a relation is "rare" if frequency < this (frequency = instances / items)
TARGET_COMMON_RATIO      =    # down-sample common relations to ratio * total_items instances each

# --- Step 2: rare-relation graph generation ---
GEN_VERIFY_THRESHOLD          =    # verification target base (target freq = this * TARGET_FREQ_MULTIPLIER)
TARGET_FREQ_MULTIPLIER        = 
GENERATION_SCALE              =    # number of generated items = original count * this
RARE_RELS_PER_ITEM            =      # target number of rare relations synthesised per item
MAX_ATTEMPTS_PER_ITEM         = 500   # cap on attempts when synthesising one item
TOP_PATTERNS_PER_RELATION     = 3     # top entity-type patterns reused per relation
PRIORITY_REL_WEIGHT           = 5     # how strongly priority rare relations are favoured
MAX_TOTAL_ATTEMPTS_MULTIPLIER = 5     # global attempt cap = target item count * this
DEFAULT_ENTITY_PATTERNS = [("PER", "ORG"), ("PER", "LOC"), ("ORG", "LOC")]  # fallback patterns

# --- Step 3: per-relation text generation ---
MAX_FEW_SHOT_EXAMPLES       = 3       # few-shot examples per relation
MAX_GENERATION_ATTEMPTS     = 5       # LLM retries per relation before falling back
READABILITY_EASE_THRESHOLD  = 400     # Flesch-Kincaid gate (loose thresholds = rarely rejected)
READABILITY_GRADE_THRESHOLD = 60

# --- Step 4: reference pool + event generation ---
EMBEDDING_MODEL       = "sentence-transformers/all-MiniLM-L6-v2"  # local path or HF name; use multilingual for non-English text
N_CLUSTERS            = 5       # number of clusters
SAMPLES_PER_CLUSTER   = 2       # representative samples kept per cluster
REBUILD_REFERENCES    = False   # set True to rebuild the pool even if REFERENCE_FILE exists
NUM_REFERENCE_SAMPLES = 3       # reference examples drawn from the pool per generation
MAX_EVAL_RETRIES      = 3       # evaluation retries before an item is dropped

# --- Step 5: schema augmentation ---
RANDOM_SCHEMA_PROB  =        # probability of adding each extra (non-present) type, for diversity
COMMON_ENTITY_TYPES = {"PER", "ORG", "LOC", "TIME", "MISC", "NUM"}  # always-included types (dataset-specific)


# =============================================================================
#  PROMPTS  -  edit to retune model behaviour
# =============================================================================
# Shared default 'instruction' field for items that have none.
DEFAULT_INSTRUCTION = "Please analyze the entities and relations in the text."

# --- Step 1: key-phrase extraction ---
KEYPHRASE_SYSTEM_PROMPT = (
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

# --- Step 3: per-relation text generation ---
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

DEFAULT_TEXT_TEMPLATE = "{head} has a {rel_type} relationship with {tail}."   # step 3 fallback sentence

# --- Step 4: event-text generation + evaluation ---
GENERATION_SYSTEM_PROMPT = (
    "You are an expert in writing news reports and biographical articles. "
    "Please generate a coherent text based on the given entities and relations, "
    "following the style of the reference examples."
)

EVALUATION_SYSTEM_PROMPT = (
    "You are an expert evaluator. Check if entities and relations are present in the text."
)

GENERATION_TASK = (
    "Task: Generate a coherent news report or biographical article that includes "
    "ALL the entities and relations listed above.\n\n"
)

GENERATION_REQUIREMENTS = (
    "Requirements:\n"
    "1. Include ALL entities mentioned above\n"
    "2. Express ALL relations naturally in the text\n"
    "3. Follow the style and tone of the reference examples\n"
    "4. Make the text coherent and readable\n"
    "5. Do not explicitly list the relations; integrate them into the narrative\n"
    "6. STRICTLY follow the expressions shown in the 'Relation Expressions' above when "
    "describing each relation. Use the same wording and phrasing as much as possible.\n"
    "7. IGNORE any real-world knowledge or factual connections that may exist between the "
    "entities. Do NOT add relations, facts, or associations that are not explicitly listed "
    "above, even if they are historically or logically true in the real world.\n"
    "8. Only the relations explicitly listed should appear in the text. Treat the given "
    "entities and relations as a closed world - no extra inferred links, background "
    "knowledge, or commonsense connections should be introduced.\n\n"
    "Generated Text:"
)

EVALUATION_PROMPT_HEADER = (
    "Evaluate whether the text contains each entity and relation.\n"
    "For each item, output: \"item: 1\" if present, \"item: 0\" if absent.\n"
    "Be strict but accurate. Only output the list.\n\n"
)

# --- Step 5: schema augmentation ---
SCHEMA_INSTRUCTION_TEMPLATE = (
    "{base_instruction}\n\n"
    "Please extract entities and relations according to the following schema:\n"
    "{schema_json}\n\n"
    "Note: You may encounter entities and relations beyond this schema, "
    "extract them as well if present in the text."
)


# =============================================================================
#  STEP 1  -  relation sampling + LLM key-phrase extraction
# =============================================================================
def build_entity_type_map(entities: dict) -> dict:
    """Map every entity surface form to its entity type."""
    entity_to_type = {}
    for ent_type, ent_list in entities.items():
        for ent in ent_list:
            entity_to_type[ent] = ent_type
    return entity_to_type


def compute_statistics(data):
    """First pass: count relation instances and (head_type, tail_type) patterns."""
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
    """Write one JSON file per relation; down-sample common relations."""
    summary = {
        "common_files": 0, "rare_files": 0,
        "common_instances_original": 0, "common_instances_sampled": 0, "rare_instances": 0,
    }
    for relation_name, texts in relation_texts.items():
        if not texts:
            continue
        stem = safe_filename(relation_name)
        if relation_name in rare_relations:
            target_dir = RARE_DIR
            texts_to_save = texts
            summary["rare_files"] += 1
            summary["rare_instances"] += len(texts_to_save)
        else:
            target_dir = COMMON_DIR
            texts_to_save = (random.sample(texts, target_common_count)
                             if len(texts) > target_common_count else texts)
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
    relations_detail = []
    for relation_name in sorted(relation_instance_counts.keys()):
        patterns = relation_entity_patterns[relation_name].most_common()
        total_patterns = sum(c for _, c in patterns)
        pattern_list = [
            {"head_type": h, "tail_type": t, "count": c,
             "ratio": round(c / total_patterns, 4) if total_patterns else 0}
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
        "frequency_threshold": RARE_FREQUENCY_THRESHOLD,
        "target_common_ratio": TARGET_COMMON_RATIO,
        "target_common_count": target_common_count,
        "common_relations_count": len(common_relations),
        "rare_relations_count": len(rare_relations),
        "common_relations": sorted(common_relations),
        "rare_relations": sorted(rare_relations),
        "relations": relations_detail,
    }
    with open(STATISTICS_REPORT, "w", encoding="utf-8") as f:
        json.dump(report_data, f, ensure_ascii=False, indent=2)

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
        "frequency_threshold": RARE_FREQUENCY_THRESHOLD,
        "target_common_ratio": TARGET_COMMON_RATIO,
        "rare_relations": sorted(rare_relations),
        "common_relations": sorted(common_relations),
        "relation_patterns": relation_patterns,
    }
    with open(PATTERNS_REPORT, "w", encoding="utf-8") as f:
        json.dump(quick_stats, f, ensure_ascii=False, indent=2)


def run_phase1_sampling():
    """Analyse the dataset, sample instances, and write per-relation files + reports."""
    for folder in (COMMON_DIR, RARE_DIR):
        os.makedirs(folder, exist_ok=True)

    with open(ORIGINAL_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} items from {ORIGINAL_DATA}")

    counts, patterns, total_items, total_instances = compute_statistics(data)
    common, rare, _ = classify_relations(counts, total_items, RARE_FREQUENCY_THRESHOLD)
    target_common_count = int(total_items * TARGET_COMMON_RATIO)
    print(f"Total items: {total_items} | relation instances: {total_instances} "
          f"| avg/item: {total_instances / total_items:.2f}")
    print(f"Common: {len(common)} | rare: {len(rare)} (threshold={RARE_FREQUENCY_THRESHOLD})")
    print(f"Down-sampling common relations to {target_common_count} instances each.")

    relation_texts = collect_relation_instances(data)
    summary = save_relation_files(relation_texts, rare, target_common_count)
    save_reports(counts, patterns, relation_texts, common, rare,
                 total_items, total_instances, target_common_count)
    print(f"Common -> '{COMMON_DIR}': {summary['common_files']} files, "
          f"{summary['common_instances_original']} -> {summary['common_instances_sampled']} instances")
    print(f"Rare   -> '{RARE_DIR}': {summary['rare_files']} files, {summary['rare_instances']} instances")


def extract_expression(text: str, relation: list) -> str:
    """Call the LLM to extract the key phrase expressing `relation` in `text`."""
    user_prompt = EXTRACTION_PROMPT_TEMPLATE.format(
        relation=relation, relation_label=relation[1], text=text
    )
    expression = call_llm(KEYPHRASE_SYSTEM_PROMPT, user_prompt)
    return expression.strip().strip('"').strip("'").strip()


def process_relation_file(input_path: str, output_path: str) -> bool:
    """Add an extracted key phrase to every instance in one relation file (resumable)."""
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
          f"({common['errors']} failed) | rare: {rare['processed']} files ({rare['errors']} failed)")


def run_step1():
    random.seed(RANDOM_SEED)
    print("Step 1.1: relation frequency analysis & sampling")
    run_phase1_sampling()
    print("\nStep 1.2: LLM key-phrase extraction")
    run_phase2_extraction()


# =============================================================================
#  STEP 2  -  synthetic rare-relation graph generation
# =============================================================================
def load_statistics():
    """Load the statistics report; return total items, per-relation stats and rare set."""
    if not os.path.exists(STATISTICS_REPORT):
        raise FileNotFoundError(f"Statistics file not found: {STATISTICS_REPORT}")

    with open(STATISTICS_REPORT, "r", encoding="utf-8") as f:
        stats = json.load(f)

    total_items = stats.get("total_items", 0)
    relation_stats = {}
    rare_relations = set()
    for rel_info in stats.get("relations", []):
        rel_name = rel_info["relation_name"]
        relation_stats[rel_name] = {
            "instance_count": rel_info.get("instance_count", 0),
            "frequency": rel_info.get("frequency", 0),
            "type": rel_info.get("type", "unknown"),
            "entity_patterns": rel_info.get("entity_patterns", []),
        }
        if relation_stats[rel_name]["type"] == "rare":
            rare_relations.add(rel_name)

    print(f"Loaded statistics: {total_items} items, {len(rare_relations)} rare relations")
    return {"total_items": total_items, "relation_stats": relation_stats,
            "rare_relations": rare_relations}


class GraphGenerator:
    """Generate items containing rare relations only, reusing a base item's entities."""

    def __init__(self, original_data, stats_data):
        self.original_data = original_data
        self.rare_relations = stats_data["rare_relations"]
        self.stats_data = stats_data
        self.relation_patterns = self._load_relation_patterns()
        self.generation_stats = {
            "total_attempts": 0, "success": 0, "under_generated": 0,
            "rare_rel_counts": defaultdict(int),
        }

    def _load_relation_patterns(self):
        """Map each relation to its top (head_type, tail_type) patterns."""
        patterns = {}
        for rel_name, rel_info in self.stats_data["relation_stats"].items():
            entity_patterns = rel_info.get("entity_patterns", [])
            if entity_patterns:
                patterns[rel_name] = [
                    (p["head_type"], p["tail_type"])
                    for p in entity_patterns[:TOP_PATTERNS_PER_RELATION]
                ]
            else:
                patterns[rel_name] = list(DEFAULT_ENTITY_PATTERNS)
        return patterns

    def _get_all_possible_pairs(self, entity_to_type, head_type, tail_type):
        """All (head, tail) entity pairs of the requested types, with head != tail."""
        heads = [e for e, t in entity_to_type.items() if t == head_type]
        tails = [e for e, t in entity_to_type.items() if t == tail_type]
        if not heads or not tails:
            return []
        return [(h, t) for h in heads for t in tails if h != t]

    def generate_new_item(self, base_item, priority_rare_rels=None):
        """Build one new item containing rare relations only (originals discarded)."""
        try:
            base_output = json.loads(base_item["output"])
        except Exception:
            return None

        new_entities = {k: v[:] for k, v in base_output.get("entities", {}).items()}
        entity_to_type = {}
        for ent_type, ent_list in new_entities.items():
            for ent in ent_list:
                entity_to_type[ent] = ent_type

        new_relations = []
        used_triples = set()

        candidate_rels = []
        if priority_rare_rels:
            for rel in priority_rare_rels:
                if rel in self.rare_relations:
                    candidate_rels.extend([rel] * PRIORITY_REL_WEIGHT)
        other_rares = [r for r in self.rare_relations if r not in (priority_rare_rels or [])]
        random.shuffle(other_rares)
        candidate_rels.extend(other_rares)
        if not candidate_rels:
            return None

        rel_type_pairs = {}
        for rel_type in set(candidate_rels):
            patterns = self.relation_patterns.get(rel_type, [DEFAULT_ENTITY_PATTERNS[0]])
            all_pairs = []
            for head_type, tail_type in patterns:
                pairs = self._get_all_possible_pairs(entity_to_type, head_type, tail_type)
                all_pairs.extend([(rel_type, h, t) for h, t in pairs])
            random.shuffle(all_pairs)
            rel_type_pairs[rel_type] = all_pairs

        generated_count = 0
        attempts = 0
        rel_idx = 0
        while generated_count < RARE_RELS_PER_ITEM and attempts < MAX_ATTEMPTS_PER_ITEM:
            attempts += 1
            current_rel = candidate_rels[rel_idx % len(candidate_rels)]
            rel_idx += 1
            for rel_type, head, tail in rel_type_pairs.get(current_rel, []):
                triple = (head, rel_type, tail)
                if triple not in used_triples:
                    new_relations.append([head, rel_type, tail])
                    used_triples.add(triple)
                    generated_count += 1
                    self.generation_stats["rare_rel_counts"][rel_type] += 1
                    break

        self.generation_stats["total_attempts"] += 1
        if generated_count < RARE_RELS_PER_ITEM:
            self.generation_stats["under_generated"] += 1
        if generated_count == 0:
            return None
        self.generation_stats["success"] += 1

        new_output = {"entities": new_entities, "relations": new_relations}
        return {
            "instruction": base_item.get("instruction", ""),
            "input": base_item.get("input", ""),
            "output": json.dumps(new_output, ensure_ascii=False),
        }

    def generate_dataset(self, num_new_items):
        """Generate `num_new_items` rare-relation-only items."""
        print(f"Generating {num_new_items} items "
              f"(target {RARE_RELS_PER_ITEM} rare relations each, rare relations only)...")

        rare_rel_targets = {}
        if self.rare_relations:
            base_target = num_new_items * RARE_RELS_PER_ITEM // len(self.rare_relations) + 1
            rare_rel_targets = {rel: base_target for rel in self.rare_relations}

        new_data = []
        max_total_attempts = num_new_items * MAX_TOTAL_ATTEMPTS_MULTIPLIER
        attempts = 0
        while len(new_data) < num_new_items and attempts < max_total_attempts:
            attempts += 1
            base_item = random.choice(self.original_data)

            target_rels = []
            if rare_rel_targets:
                sorted_rels = sorted(rare_rel_targets.items(), key=lambda x: x[1], reverse=True)
                for rel, remaining in sorted_rels[:RARE_RELS_PER_ITEM]:
                    if remaining > 0:
                        target_rels.append(rel)
                        rare_rel_targets[rel] = max(0, rare_rel_targets[rel] - 1)

            new_item = self.generate_new_item(base_item, target_rels or None)
            if new_item:
                new_data.append(new_item)

            if attempts % 500 == 0 or len(new_data) % 100 == 0:
                success_rate = (self.generation_stats["success"]
                                / max(self.generation_stats["total_attempts"], 1) * 100)
                print(f"  attempts: {attempts}, generated: {len(new_data)}/{num_new_items} "
                      f"(success rate {success_rate:.1f}%)")

        print("\nGeneration summary:")
        print(f"  total attempts: {attempts}")
        print(f"  generated: {len(new_data)}")
        print(f"  discarded: {self.generation_stats['total_attempts'] - self.generation_stats['success']}")
        print(f"  under {RARE_RELS_PER_ITEM} relations: {self.generation_stats['under_generated']}")
        print("  top rare relations generated:")
        for rel, count in sorted(self.generation_stats["rare_rel_counts"].items(),
                                 key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {rel:30s}: {count}")
        return new_data


def count_relations(items):
    """Count relation instances across a list of items."""
    counts = defaultdict(int)
    for item in items:
        try:
            output = json.loads(item["output"])
        except Exception:
            continue
        for rel in output.get("relations", []):
            if len(rel) == 3:
                counts[rel[1]] += 1
    return counts


def verify_distribution(original_data, new_data, stats_data, threshold):
    """Report the rare-relation frequency over original + generated combined."""
    rare_rels = stats_data["rare_relations"]
    new_rel_counts = count_relations(new_data)

    common_in_new = [rel for rel in new_rel_counts if rel not in rare_rels]
    print(f"\nGenerated relation instances: {sum(new_rel_counts.values())}")
    if common_in_new:
        print(f"  WARNING: common relations found in generated data: {common_in_new[:5]}")
    else:
        print("  OK: generated data contains rare relations only")

    combined_counts = count_relations(original_data + new_data)
    total = len(original_data) + len(new_data)
    target_freq = threshold * TARGET_FREQ_MULTIPLIER

    print(f"\nCombined distribution (original + generated, {total} items), "
          f"target rare frequency {target_freq:.3f}:")
    passed = below = total_rare = 0
    for rel in sorted(combined_counts, key=lambda x: combined_counts[x], reverse=True):
        count = combined_counts[rel]
        freq = count / total
        is_rare = rel in rare_rels
        if not (is_rare or freq > 0.05):
            continue
        if is_rare:
            total_rare += 1
            generated = new_rel_counts.get(rel, 0)
            source = f"generated {generated}" if generated > 0 else "original only"
            status = "ok" if freq >= target_freq else "low"
            passed += freq >= target_freq
            below += freq < target_freq
            print(f"  [rare]   {rel:27s} | {count:6d} | {freq:.3f} | {source:14s} | {status}")
        else:
            print(f"  [common] {rel:27s} | {count:6d} | {freq:.3f}")

    if total_rare > 0:
        print(f"\nRare relations meeting target: {passed}/{total_rare} "
              f"({passed / total_rare * 100:.1f}%), below target: {below}")


def run_step2():
    random.seed(RANDOM_SEED)
    stats_data = load_statistics()

    with open(ORIGINAL_DATA, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    print(f"Loaded {len(original_data)} items from {ORIGINAL_DATA}")

    num_new_items = int(len(original_data) * GENERATION_SCALE)
    print(f"Generating {num_new_items} items (scale={GENERATION_SCALE}); "
          f"only generated data is written to {GENERATED_FILE}")

    generator = GraphGenerator(original_data, stats_data)
    new_data = generator.generate_dataset(num_new_items)

    with open(GENERATED_FILE, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(new_data)} generated items to {GENERATED_FILE} "
          f"(original {len(original_data)} items not included)")

    verify_distribution(original_data, new_data, stats_data, GEN_VERIFY_THRESHOLD)


# =============================================================================
#  STEP 3  -  per-relation natural-language text generation (LLM)
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
    asl = num_words / num_sentences if num_sentences > 0 else 0
    asw = num_syllables / num_words if num_words > 0 else 0
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


def load_few_shot_examples(relation_type, max_examples=MAX_FEW_SHOT_EXAMPLES):
    """Load few-shot examples for a relation from the common then rare folders."""
    relation_file = os.path.join(COMMON_PRO_DIR, f"{relation_type}.json")
    if not os.path.exists(relation_file):
        relation_file = os.path.join(RARE_PRO_DIR, f"{relation_type}.json")
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
    """Generate a natural-language expression for one relation triple (with fallback)."""
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
        generated_texts, readability_results, relation_details = [], [], []
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

        if (idx + 1) % SAVE_INTERVAL == 0:
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDone. {len(results)} items saved to {output_file}")


def run_step3():
    random.seed(RANDOM_SEED)
    process_relations_with_llm(GENERATED_FILE, GENERATED_TEXT_FILE)


# =============================================================================
#  STEP 4  -  reference pool (SBERT) + event-text generation & evaluation (LLM)
# =============================================================================
def encode_texts(texts):
    """Embed texts with the configured sentence-transformer model."""
    from sentence_transformers import SentenceTransformer   # lazy: only when building the pool
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"Encoding {len(texts)} texts...")
    return model.encode(texts, show_progress_bar=True, convert_to_numpy=True)


def cluster_embeddings(embeddings, n_clusters):
    """Cluster embeddings with KMeans; return labels and cluster centres."""
    from sklearn.cluster import KMeans                       # lazy: only when building the pool
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    return labels, kmeans.cluster_centers_


def find_representative_samples(texts, embeddings, labels, cluster_centers, samples_per_cluster):
    """Pick the samples closest to each cluster centre."""
    from sklearn.metrics.pairwise import cosine_similarity   # lazy: only when building the pool
    representative_texts = []
    for cluster_id in range(len(cluster_centers)):
        cluster_indices = np.where(labels == cluster_id)[0]
        if len(cluster_indices) == 0:
            continue
        center = cluster_centers[cluster_id].reshape(1, -1)
        similarities = cosine_similarity(embeddings[cluster_indices], center).flatten()
        top = np.argsort(similarities)[-samples_per_cluster:][::-1]
        for idx in cluster_indices[top]:
            representative_texts.append(texts[idx])
    return representative_texts


def build_reference_pool():
    """Return the reference-example pool, building and caching it if necessary."""
    if os.path.exists(REFERENCE_FILE) and not REBUILD_REFERENCES:
        with open(REFERENCE_FILE, "r", encoding="utf-8") as f:
            pool = json.load(f)
        print(f"Loaded {len(pool)} reference samples from {REFERENCE_FILE}")
        return pool

    print(f"Building reference pool from {ORIGINAL_DATA}...")
    with open(ORIGINAL_DATA, "r", encoding="utf-8") as f:
        data = json.load(f)
    texts = [item.get("input", "") for item in data if item.get("input")]
    print(f"Extracted {len(texts)} texts")

    embeddings = encode_texts(texts)
    labels, centers = cluster_embeddings(embeddings, N_CLUSTERS)
    pool = find_representative_samples(texts, embeddings, labels, centers, SAMPLES_PER_CLUSTER)

    with open(REFERENCE_FILE, "w", encoding="utf-8") as f:
        json.dump(pool, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(pool)} reference samples to {REFERENCE_FILE}")
    return pool


def build_generation_prompt(ref_samples, entities, relations, generated_texts):
    """Assemble the prompt that asks the LLM to write one coherent article."""
    ref_section = "Reference Examples:\n"
    for i, ref in enumerate(ref_samples):
        ref_section += f"Example {i + 1}:\n{ref}\n\n"

    entity_section = "Entities:\n"
    for ent_type, ent_list in entities.items():
        entity_section += f"  {ent_type}: {', '.join(ent_list)}\n"

    relation_section = "Relations:\n"
    for rel in relations:
        if len(rel) == 3:
            relation_section += f"  - {rel[0]} --{rel[1]}--> {rel[2]}\n"

    text_section = ""
    if generated_texts:
        text_section = "\nRelation Expressions (for reference):\n"
        for i, text in enumerate(generated_texts):
            if text:
                text_section += f"  {i + 1}. {text}\n"

    return (
        f"{ref_section}"
        f"{GENERATION_TASK}"
        f"{entity_section}\n"
        f"{relation_section}"
        f"{text_section}\n"
        f"{GENERATION_REQUIREMENTS}"
    )


def parse_evaluation_result_detailed(eval_result, entities_list, relations_list):
    """Parse the LLM's "item: 0/1" lines into per-entity / per-relation scores."""
    eval_details = {
        "entities": {ent: 0 for ent in entities_list},
        "relations": {f"{h}|{r}|{t}": 0 for (h, r, t) in relations_list},
    }
    for line in eval_result.split("\n"):
        line = line.strip()
        if not line or ":" not in line:
            continue
        item_name, _, value_part = line.partition(":")
        item_name = item_name.strip()
        value_part = value_part.strip()
        if not value_part:
            continue
        try:
            score = int(value_part.split()[0])
        except (ValueError, IndexError):
            continue
        if score not in (0, 1):
            continue

        if item_name in eval_details["entities"]:
            eval_details["entities"][item_name] = score
        else:
            for h, r, t in relations_list:
                if item_name == f"{h} --{r}--> {t}":
                    eval_details["relations"][f"{h}|{r}|{t}"] = score
                    break
    return eval_details


def evaluate_event_text_detailed(event_text, entities, relations):
    """Ask the LLM which entities/relations appear; return (score, details)."""
    eval_prompt = EVALUATION_PROMPT_HEADER + f"Text: {event_text}\n\n"

    all_entities = []
    entity_section = "Entities:\n"
    for ent_type, ent_list in entities.items():
        for ent in ent_list:
            entity_section += f"{ent}: \n"
            all_entities.append(ent)

    all_relations = []
    relation_section = "Relations:\n"
    for rel in relations:
        if len(rel) == 3:
            relation_section += f"{rel[0]} --{rel[1]}--> {rel[2]}: \n"
            all_relations.append((rel[0], rel[1], rel[2]))

    eval_prompt += entity_section + "\n" + relation_section

    eval_result = call_llm(EVALUATION_SYSTEM_PROMPT, eval_prompt)
    eval_details = parse_evaluation_result_detailed(eval_result, all_entities, all_relations)

    all_scores = list(eval_details["entities"].values()) + list(eval_details["relations"].values())
    score = sum(all_scores) / len(all_scores) if all_scores else 0.0
    return score, eval_details


def build_filtered_output(entities, relations, eval_details):
    """Keep only the entities and relations the evaluator marked as present."""
    filtered_entities = {}
    for ent_type, ent_list in entities.items():
        kept = [ent for ent in ent_list if eval_details["entities"].get(ent, 0) == 1]
        if kept:
            filtered_entities[ent_type] = kept

    filtered_relations = []
    for rel in relations:
        if len(rel) == 3:
            head, rel_type, tail = rel
            if eval_details["relations"].get(f"{head}|{rel_type}|{tail}", 0) == 1:
                filtered_relations.append(rel)

    return {"entities": filtered_entities, "relations": filtered_relations}


def generate_event_dataset(ref_pool):
    """Generate and evaluate event text for every item, then write the results."""
    if isinstance(ref_pool, list) and len(ref_pool) > NUM_REFERENCE_SAMPLES:
        ref_samples = random.sample(ref_pool, NUM_REFERENCE_SAMPLES)
    else:
        ref_samples = ref_pool
    print(f"Using {len(ref_samples)} reference samples")

    print(f"Loading input data from {GENERATED_TEXT_FILE}...")
    with open(GENERATED_TEXT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Total items: {len(data)}")

    results = []
    skipped = 0
    for idx, item in enumerate(tqdm(data, desc="Generating events")):
        try:
            output_data = json.loads(item.get("output", "{}"))
        except Exception:
            skipped += 1
            continue

        entities = output_data.get("entities", {})
        relations = output_data.get("relations", [])
        generated_texts = item.get("generated_texts", [])

        prompt = build_generation_prompt(ref_samples, entities, relations, generated_texts)
        event_text = call_llm(GENERATION_SYSTEM_PROMPT, prompt)

        evaluation_score, eval_details, success = 0.0, None, False
        for _ in range(MAX_EVAL_RETRIES):
            try:
                evaluation_score, eval_details = evaluate_event_text_detailed(
                    event_text, entities, relations
                )
                if sum(eval_details["entities"].values()) > 0 or \
                   sum(eval_details["relations"].values()) > 0:
                    success = True
                    break
            except Exception:
                continue

        if not success:
            skipped += 1
            continue

        output_full = {"entities": entities, "relations": relations}
        output_filtered = build_filtered_output(entities, relations, eval_details)

        results.append({
            "instruction": item.get("instruction", DEFAULT_INSTRUCTION),
            "input": event_text,
            "output": json.dumps(output_full, ensure_ascii=False, separators=(",", ":")),
            "output_filtered": json.dumps(output_filtered, ensure_ascii=False, separators=(",", ":")),
            "evaluation": evaluation_score,
        })

        if (idx + 1) % SAVE_INTERVAL == 0:
            with open(EVENT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(EVENT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Processed: {len(results)}, skipped: {skipped}. Saved to {EVENT_FILE}")


def run_step4():
    random.seed(RANDOM_SEED)
    print("Step 4.1: build / load reference pool")
    ref_pool = build_reference_pool()
    print("\nStep 4.2: event-text generation & evaluation")
    generate_event_dataset(ref_pool)


# =============================================================================
#  STEP 5  -  schema-augmented instructions (final dataset)
# =============================================================================
def load_schema_from_statistics():
    """Build the full entity/relation schema from the statistics report."""
    print(f"Loading schema from {STATISTICS_REPORT}...")
    with open(STATISTICS_REPORT, "r", encoding="utf-8") as f:
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


def run_step5():
    random.seed(RANDOM_SEED)
    full_schema = load_schema_from_statistics()
    process_dataset(EVENT_FILE, FINAL_FILE, full_schema)
    verify_output(FINAL_FILE)
    print("\nAll done!")


# =============================================================================
#  PIPELINE RUNNER
# =============================================================================
def parse_args():
    """Command-line parameters. Each one defaults to the CONFIGURATION value
    above and overrides it when given. run.sh fills these in."""
    p = argparse.ArgumentParser(description="Run the data-augmentation pipeline (steps 1-5).")
    p.add_argument("--input", default=ORIGINAL_DATA, help="original dataset, a JSON list")
    p.add_argument("--llm-model", default=LLM_MODEL_NAME, help="LLM model name used by call_llm")
    p.add_argument("--embedding-model", default=EMBEDDING_MODEL,
                   help="sentence-transformer model for step 4 (use multilingual for non-English text)")
    p.add_argument("--seed", type=int, default=RANDOM_SEED, help="random seed")
    p.add_argument("--gen-scale", type=float, default=GENERATION_SCALE,
                   help="generated items = original count * this (step 2)")
    p.add_argument("--rare-threshold", type=float, default=RARE_FREQUENCY_THRESHOLD,
                   help="a relation is 'rare' if frequency < this (step 1)")
    p.add_argument("--target-common-ratio", type=float, default=TARGET_COMMON_RATIO,
                   help="down-sample common relations to ratio * items (step 1)")
    p.add_argument("--rare-rels-per-item", type=int, default=RARE_RELS_PER_ITEM,
                   help="target rare relations synthesised per item (step 2)")
    p.add_argument("--save-interval", type=int, default=SAVE_INTERVAL,
                   help="steps 3 & 4 write their output every N items")
    p.add_argument("--rebuild-references", action="store_true", default=REBUILD_REFERENCES,
                   help="rebuild the step-4 reference pool even if its cache exists")
    return p.parse_args()


def apply_args(args):
    """Override the module-level configuration with the parsed arguments."""
    global ORIGINAL_DATA, LLM_MODEL_NAME, EMBEDDING_MODEL, RANDOM_SEED, GENERATION_SCALE
    global RARE_FREQUENCY_THRESHOLD, TARGET_COMMON_RATIO, RARE_RELS_PER_ITEM
    global SAVE_INTERVAL, REBUILD_REFERENCES
    ORIGINAL_DATA            = args.input
    LLM_MODEL_NAME           = args.llm_model
    EMBEDDING_MODEL          = args.embedding_model
    RANDOM_SEED              = args.seed
    GENERATION_SCALE         = args.gen_scale
    RARE_FREQUENCY_THRESHOLD = args.rare_threshold
    TARGET_COMMON_RATIO      = args.target_common_ratio
    RARE_RELS_PER_ITEM       = args.rare_rels_per_item
    SAVE_INTERVAL            = args.save_interval
    REBUILD_REFERENCES       = args.rebuild_references


def main():
    apply_args(parse_args())
    print(f"Input: {ORIGINAL_DATA} | model: {LLM_MODEL_NAME or '(unset)'} | "
          f"embedding: {EMBEDDING_MODEL} | seed: {RANDOM_SEED}")
    for num, run in [(1, run_step1), (2, run_step2), (3, run_step3),
                     (4, run_step4), (5, run_step5)]:
        print(f"\n{'#' * 60}\n# STEP {num}\n{'#' * 60}")
        run()
    print("\nPipeline finished.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Pipeline error: {e}")
        traceback.print_exc()