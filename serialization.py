"""
Converts SLURP annotation records <-> a single serialized target string that
the decoder learns to generate autoregressively.

SLURP record (per utterance, from slurp/dataset/slurp/{train,devel,test}.jsonl)
looks roughly like:

{
  "slurp_id": 123,
  "sentence": "wake me up at nine am on friday",
  "scenario": "alarm",
  "action": "set",
  "entities": [
      {"type": "time", "filler": "nine am"},
      {"type": "date", "filler": "friday"}
  ],
  "recordings": [{"file": "audio-1488449...wav", ...}, ...]
}

Target string format (kept simple and unambiguous so it is easy to parse back
and reasonably short in tokens):

    intent : alarm_set | time = nine am ; date = friday

If there are no entities:

    intent : alarm_set | none
"""

import re
from typing import Dict, List, Tuple

INTENT_TAG = "intent :"
SLOT_SEP = "|"
PAIR_SEP = ";"
KV_SEP = "="
NO_SLOTS = "none"


def make_intent(scenario: str, action: str) -> str:
    return f"{scenario}_{action}"


def _clean_entities(entities: List[Dict]) -> List[Dict[str, str]]:
    """Drop/normalize malformed entity dicts (missing or non-string type/filler,
    which occurs for a handful of records in the SLURP release)."""
    cleaned = []
    for e in entities or []:
        etype = e.get("type")
        filler = e.get("filler")
        if etype is None or filler is None:
            continue
        etype, filler = str(etype).strip(), str(filler).strip()
        if etype and filler:
            cleaned.append({"type": etype, "filler": filler})
    return cleaned


def serialize_target(scenario: str, action: str, entities: List[Dict]) -> str:
    intent = make_intent(scenario, action)
    entities = _clean_entities(entities)
    if not entities:
        slots_str = NO_SLOTS
    else:
        parts = [f"{e['type']} {KV_SEP} {e['filler']}" for e in entities]
        slots_str = f" {PAIR_SEP} ".join(parts)
    return f"{INTENT_TAG} {intent} {SLOT_SEP} {slots_str}"


_TARGET_RE = re.compile(
    r"intent\s*:\s*(?P<intent>\S+)\s*\|\s*(?P<slots>.*)$", re.IGNORECASE
)


def parse_target(text: str) -> Tuple[str, List[Dict[str, str]]]:
    """Inverse of serialize_target. Robust to minor spacing/formatting noise
    from imperfect generations."""
    text = text.strip()
    m = _TARGET_RE.match(text)
    if not m:
        return "", []

    intent = m.group("intent").strip()
    slots_raw = m.group("slots").strip()

    entities = []
    if slots_raw and slots_raw.lower() != NO_SLOTS:
        for pair in slots_raw.split(PAIR_SEP):
            if KV_SEP not in pair:
                continue
            k, v = pair.split(KV_SEP, 1)
            k, v = k.strip(), v.strip()
            if k and v:
                entities.append({"type": k, "filler": v})
    return intent, entities


if __name__ == "__main__":
    tgt = serialize_target("alarm", "set", [
        {"type": "time", "filler": "nine am"},
        {"type": "date", "filler": "friday"},
    ])
    print("serialized:", tgt)
    print("parsed back:", parse_target(tgt))
    print("no-slot case:", serialize_target("general", "quirky", []))
