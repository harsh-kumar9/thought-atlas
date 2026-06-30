"""src/judge/schemas.py — guided-decoding JSON schemas for the judge (kills parse failures)."""

KIM_BEHAVIORS = ["Question_and_Answering", "Perspective_Shift",
                 "Conflict_of_Perspectives", "Reconciliation"]
GANDHI_BEHAVIORS = ["verification", "backtracking", "subgoal", "backward_chaining"]


def whole_count_schema(behaviors):
    return {"type": "object",
            "properties": {b: {"type": "integer", "minimum": 0} for b in behaviors},
            "required": behaviors, "additionalProperties": False}


def per_sentence_schema(behaviors):
    return {
        "type": "object",
        "properties": {"sentences": {"type": "array", "items": {
            "type": "object",
            "properties": {
                "index": {"type": "integer"},
                "behaviors": {"type": "array", "items": {"type": "string", "enum": behaviors}},
            },
            "required": ["index", "behaviors"], "additionalProperties": False,
        }}},
        "required": ["sentences"], "additionalProperties": False,
    }


KIM_WHOLE_SCHEMA = whole_count_schema(KIM_BEHAVIORS)
GANDHI_WHOLE_SCHEMA = whole_count_schema(GANDHI_BEHAVIORS)
KIM_SENT_SCHEMA = per_sentence_schema(KIM_BEHAVIORS)
GANDHI_SENT_SCHEMA = per_sentence_schema(GANDHI_BEHAVIORS)
