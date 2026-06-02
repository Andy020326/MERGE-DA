
# =============================================================================
#  LLM CONFIGURATION  ->  *** FILL IN YOUR OWN MODEL / API SETTINGS BELOW ***
# =============================================================================
#  Phase 2 calls a large language model. Put your model name, key, endpoint and
#  the actual call logic here. `call_llm` is the single entry point used by the
#  rest of the project.
#
#  Only `torch` / `transformers` are needed for a LOCAL HF model; comment them
#  out if you call a hosted API instead. The reference pool (Phase 1) uses a
#  separate sentence-transformer model configured under CONFIGURATION.
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

import numpy as np

from tqdm import tqdm


# =============================================================================
#  CONFIGURATION  -  dataset-specific parameters, tune everything here
# =============================================================================
ORIGINAL_FILE   = "train.json"                  # original dataset (source of style references)
REFERENCE_FILE  = "reference_examples.json"     # cached reference pool (built in Phase 1)
INPUT_FILE      = "train_generated_text.json"   # items produced by Step 3
OUTPUT_FILE     = "train_event.json"            # final output: event text + filtered relations

# Reference pool (Phase 1).
EMBEDDING_MODEL     = "sbert_model"  # local path or HF name; use a multilingual model for non-English text
N_CLUSTERS          =          # number of clusters
SAMPLES_PER_CLUSTER =          # representative samples kept per cluster
REBUILD_REFERENCES  = False     # set True to rebuild the pool even if REFERENCE_FILE exists

# Generation + evaluation (Phase 2).
NUM_REFERENCE_SAMPLES = 3       # reference examples drawn from the pool per generation
MAX_EVAL_RETRIES      = 3       # evaluation retries before an item is dropped
SAVE_INTERVAL         = 10      # write the output file every N items

RANDOM_SEED = 42                # seed for KMeans and reference sampling


# =============================================================================
#  PROMPTS  -  edit to retune generation and evaluation
# =============================================================================
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

# Default 'instruction' field for output items that have none.
DEFAULT_INSTRUCTION = "Please analyze the entities and relations in the text."


# =============================================================================
#  PHASE 1  -  reference pool (sentence-transformer clustering)
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

    print(f"Building reference pool from {ORIGINAL_FILE}...")
    with open(ORIGINAL_FILE, "r", encoding="utf-8") as f:
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


# =============================================================================
#  PHASE 2  -  event-text generation and evaluation
# =============================================================================
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

    print(f"Loading input data from {INPUT_FILE}...")
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
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

        # Evaluate with retries; require a non-empty (not all-zero) evaluation.
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
            with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nDone. Processed: {len(results)}, skipped: {skipped}. Saved to {OUTPUT_FILE}")


# =============================================================================
#  MAIN
# =============================================================================
def main():
    random.seed(RANDOM_SEED)

    print("=== Phase 1: build / load reference pool ===")
    ref_pool = build_reference_pool()

    print("\n=== Phase 2: event-text generation & evaluation ===")
    generate_event_dataset(ref_pool)


if __name__ == "__main__":
    main()