
# =============================================================================
#  LLM CONFIGURATION  ->  *** FILL IN YOUR OWN MODEL / API SETTINGS BELOW ***
# =============================================================================
#  Kept here for project-wide consistency. This step generates data purely by
#  combinatorial pattern matching and does NOT call the LLM, so you may comment
#  out the `torch` / `transformers` imports if you run only this step.
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
import traceback
from collections import defaultdict


# =============================================================================
#  CONFIGURATION  -  dataset-specific parameters, tune everything here
# =============================================================================
INPUT_FILE      = "train.json"                  # source dataset (a JSON list)
OUTPUT_FILE     = "train_generated.json"        # output: ONLY the generated rare-relation items
STATISTICS_FILE = "relation_statistics.json"    # relation statistics produced by Step 1

# Verification target. Rare relations are read from STATISTICS_FILE (already
# classified in Step 1); this threshold is used only when checking the combined
# distribution, where target frequency = FREQUENCY_THRESHOLD * TARGET_FREQ_MULTIPLIER.
FREQUENCY_THRESHOLD    = 
TARGET_FREQ_MULTIPLIER = 

GENERATION_SCALE      =       # number of generated items = original count * this
RARE_RELS_PER_ITEM    =         # target number of rare relations to synthesise per item
MAX_ATTEMPTS_PER_ITEM = 500      # cap on attempts when synthesising one item

RANDOM_SEED = 42                 # seed for reproducible generation

TOP_PATTERNS_PER_RELATION     = 3    # number of top entity-type patterns reused per relation
PRIORITY_REL_WEIGHT           = 5    # how strongly priority rare relations are favoured
MAX_TOTAL_ATTEMPTS_MULTIPLIER = 5    # global attempt cap = target item count * this

# Fallback (head_type, tail_type) patterns for relations without a recorded pattern.
DEFAULT_ENTITY_PATTERNS = [("PER", "ORG"), ("PER", "LOC"), ("ORG", "LOC")]


# =============================================================================
#  STAGE 1  -  load relation statistics
# =============================================================================
def load_statistics():
    """Load the statistics report and return total item count, per-relation
    stats and the set of rare relations."""
    if not os.path.exists(STATISTICS_FILE):
        raise FileNotFoundError(f"Statistics file not found: {STATISTICS_FILE}")

    with open(STATISTICS_FILE, "r", encoding="utf-8") as f:
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
    return {
        "total_items": total_items,
        "relation_stats": relation_stats,
        "rare_relations": rare_relations,
    }


# =============================================================================
#  STAGE 2  -  rare-relation data generation
# =============================================================================
class GraphGenerator:
    """Generate items containing rare relations only, reusing a base item's
    entities and pairing them according to each relation's entity-type patterns."""

    def __init__(self, original_data, stats_data):
        self.original_data = original_data
        self.rare_relations = stats_data["rare_relations"]
        self.stats_data = stats_data
        self.relation_patterns = self._load_relation_patterns()
        self.generation_stats = {
            "total_attempts": 0,
            "success": 0,
            "under_generated": 0,
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

        # Reuse only the entities of the base item.
        new_entities = {k: v[:] for k, v in base_output.get("entities", {}).items()}
        entity_to_type = {}
        for ent_type, ent_list in new_entities.items():
            for ent in ent_list:
                entity_to_type[ent] = ent_type

        new_relations = []
        used_triples = set()

        # Build the candidate relation pool: priority relations weighted higher.
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

        # Pre-compute all candidate triples per relation type.
        rel_type_pairs = {}
        for rel_type in set(candidate_rels):
            patterns = self.relation_patterns.get(rel_type, [DEFAULT_ENTITY_PATTERNS[0]])
            all_pairs = []
            for head_type, tail_type in patterns:
                pairs = self._get_all_possible_pairs(entity_to_type, head_type, tail_type)
                all_pairs.extend([(rel_type, h, t) for h, t in pairs])
            random.shuffle(all_pairs)
            rel_type_pairs[rel_type] = all_pairs

        # Round-robin over candidate relations, adding the first unused triple found.
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

        # Spread generation roughly evenly across rare relations.
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

            # Pick the rare relations most in need of more instances.
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


# =============================================================================
#  STAGE 3  -  verification
# =============================================================================
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
    """Check that the generated data holds only rare relations and report the
    rare-relation frequency over original + generated combined."""
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


# =============================================================================
#  MAIN
# =============================================================================
def main():
    random.seed(RANDOM_SEED)
    try:
        stats_data = load_statistics()

        with open(INPUT_FILE, "r", encoding="utf-8") as f:
            original_data = json.load(f)
        print(f"Loaded {len(original_data)} items from {INPUT_FILE}")

        num_new_items = int(len(original_data) * GENERATION_SCALE)
        print(f"Generating {num_new_items} items (scale={GENERATION_SCALE}); "
              f"only generated data is written to {OUTPUT_FILE}")

        generator = GraphGenerator(original_data, stats_data)
        new_data = generator.generate_dataset(num_new_items)

        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(new_data, f, ensure_ascii=False, indent=2)
        print(f"\nSaved {len(new_data)} generated items to {OUTPUT_FILE} "
              f"(original {len(original_data)} items not included)")

        verify_distribution(original_data, new_data, stats_data, FREQUENCY_THRESHOLD)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    main()