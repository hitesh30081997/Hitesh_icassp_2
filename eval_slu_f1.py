"""
eval_slu_f1.py

Computes SLURP's official metrics:
  - Intent (scenario_action) accuracy
  - SLU-F1: span-level slot metric using word-level overlap between
    predicted and gold slot fillers (not exact string match), as defined
    in the SLURP paper (Bastianelli et al., 2020) and used by the
    official `slurp-eval` / NeMo / ESPnet-SLU scripts.

SLU-F1 definition (per example, then micro-averaged over the dataset):
  For each predicted (slot_type, filler) pair, compare against gold
  (slot_type, filler) pairs of the SAME type. A predicted slot counts
  as a true positive contributing partial credit based on the fraction
  of overlapping words between predicted filler and the best-matching
  gold filler of that slot type (word-level precision/recall), rather
  than requiring an exact string match. This rewards partially-correct
  transcriptions of slot values (e.g. ASR errors on a single word)
  instead of scoring the whole slot as wrong.

  precision = sum(overlap words) / sum(predicted words)
  recall    = sum(overlap words) / sum(gold words)
  F1        = 2PR / (P + R)

Usage:
  python eval_slu_f1.py --predictions preds.jsonl --references refs.jsonl

Where each line of predictions.jsonl / references.jsonl is:
  {"slurp_id": 123, "intent": "alarm_set", "slots": {"date": "friday", "time": "nine am"}}
"""

import json
import argparse
from collections import defaultdict


def tokenize(text):
    return text.lower().strip().split()


def word_overlap(pred_words, gold_words):
    """Multiset overlap count between two word lists (order-independent)."""
    gold_counts = defaultdict(int)
    for w in gold_words:
        gold_counts[w] += 1
    overlap = 0
    for w in pred_words:
        if gold_counts[w] > 0:
            overlap += 1
            gold_counts[w] -= 1
    return overlap


def best_match_overlap(pred_filler, gold_fillers):
    """
    pred_filler: string
    gold_fillers: list of gold strings with the SAME slot type
    Returns (overlap_words, pred_len, best_gold_len) for the best-matching
    gold filler (the one maximizing overlap), removing it from further
    matching (each gold filler can only be matched once).
    """
    pred_words = tokenize(pred_filler)
    best = (0, len(pred_words), 0, -1)  # overlap, pred_len, gold_len, idx
    for idx, g in enumerate(gold_fillers):
        if g is None:
            continue
        gold_words = tokenize(g)
        ov = word_overlap(pred_words, gold_words)
        if ov > best[0]:
            best = (ov, len(pred_words), len(gold_words), idx)
    return best


def compute_slu_f1(predictions, references):
    """
    predictions, references: dict[slurp_id] -> {"intent": str, "slots": {type: filler}}
    Returns dict with intent_accuracy, slot_precision, slot_recall, slot_f1
    """
    total = 0
    correct_intent = 0

    total_overlap = 0
    total_pred_words = 0
    total_gold_words = 0

    for sid, ref in references.items():
        pred = predictions.get(sid, {"intent": None, "slots": {}})
        total += 1
        if pred.get("intent") == ref.get("intent"):
            correct_intent += 1

        # group gold slots by type, allow multiple fillers of same type
        gold_by_type = defaultdict(list)
        for stype, filler in ref.get("slots", {}).items():
            gold_by_type[stype].append(filler)

        used_gold_idx = defaultdict(set)

        for stype, filler in pred.get("slots", {}).items():
            gold_fillers = gold_by_type.get(stype, [])
            # exclude already-matched gold fillers of this type
            available = [
                g if i not in used_gold_idx[stype] else None
                for i, g in enumerate(gold_fillers)
            ]
            overlap, pred_len, gold_len, idx = best_match_overlap(filler, available)
            if idx >= 0:
                used_gold_idx[stype].add(idx)
            total_overlap += overlap
            total_pred_words += pred_len

        for stype, fillers in gold_by_type.items():
            for f in fillers:
                total_gold_words += len(tokenize(f))

    intent_acc = correct_intent / total if total else 0.0
    precision = total_overlap / total_pred_words if total_pred_words else 0.0
    recall = total_overlap / total_gold_words if total_gold_words else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "n_examples": total,
        "intent_accuracy": round(intent_acc, 4),
        "slot_precision": round(precision, 4),
        "slot_recall": round(recall, 4),
        "slu_f1": round(f1, 4),
    }


def load_jsonl_by_id(path):
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            out[ex["slurp_id"]] = ex
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True, help="jsonl with slurp_id/intent/slots")
    ap.add_argument("--references", required=True, help="jsonl with slurp_id/intent/slots (gold)")
    args = ap.parse_args()

    preds = load_jsonl_by_id(args.predictions)
    refs = load_jsonl_by_id(args.references)

    metrics = compute_slu_f1(preds, refs)
    print(json.dumps(metrics, indent=2))
