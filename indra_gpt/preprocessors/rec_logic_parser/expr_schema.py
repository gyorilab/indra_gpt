EPISTEMIC_MODE_SCHEMA = {
    "title": "EpistemicMode",
    "description": "A wrapper expressing the epistemic status of a normalized statement.",
    "type": "object",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "HYPOTHESIS": {
                    "type": "string",
                    "description": "A speculative or uncertain statement that expresses a possibility or hypothesis."
                }
            },
            "required": ["HYPOTHESIS"]
        },
        {
            "type": "object",
            "properties": {
                "DECLARATIVE": {
                    "type": "string",
                    "description": "A factual or asserted statement presented with confidence and without hedging."
                }
            },
            "required": ["DECLARATIVE"]
        }
    ]
}

LOGICAL_COORDINATION_SCHEMA= {
    "title": "GroupingType",
    "description": "Surface-level logical decomposition using AND, OR, or ATOM.",
    "type": "object",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "AND": {
                    "type": "array",
                    "minItems": 2,
                    "items": {"type": "string"},
                }
            },
            "required": ["AND"]
        },
        {
            "type": "object",
            "properties": {
                "OR": {
                    "type": "array",
                    "minItems": 2,
                    "items": {"type": "string"},
                }
            },
            "required": ["OR"]
        },
        {
            "type": "object",
            "properties": {
                "ATOM": {
                    "type": "string"
                }
            },
            "required": ["ATOM"]
        }
    ]
}

LOGICAL_UNARY_SCHEMA = {
    "title": "UnaryType",
    "description": "Surface-level logical decomposition using NOT or ATOMIC.",
    "type": "object",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "NOT": {
                    "type": "string"
                }
            },
            "required": ["NOT"]
        },
        {
            "type": "object",
            "properties": {
                "IDENTITY": {
                    "type": "string"
                }
            },
            "required": ["IDENTITY"]
        }
    ]
}
