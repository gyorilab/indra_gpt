RELATIONSHIP_SCHEMA = {
  "type": "object",
  "description": "A relationship is a statement which is constituted by: a relation and its arguments, where arguments can be terminal or recursively relational.",
  "properties": {
    "RELATIONSHIP": {
      "type": "object",
      "properties": {
        "RELATION": {
          "type": "string",
          "description": "The core verb or predicate expressing the relation."
        },
        "SUBJECT": {
          "type": "string",
          "description": "The subject of the relationship."
        },
        "OBJECT": {
          "type": "string",
          "description": "The object of the relationship, if present in the sentence."
        }
      },
      "required": ["RELATION", "SUBJECT"],
    }
  },
  "required": ["RELATIONSHIP"]
}

ARGUMENT_SCHEMA = {
    "type": "object",
    "description": "An argument is a grammatical phrase that either contains a relational verb or not.",
    "oneOf": [
        {
            "type": "object",
            "properties": {
                "NON_RELATIONAL_PHRASE": {
                    "type": "string",
                    "description": "A phrase that does not encode any relationship.",                
                }
            },
            "required": ["NON_RELATIONAL_PHRASE"]
        },
        {
            "type": "object",
            "properties": {
                "RELATIONAL_PHRASE": {
                    "type": "string",
                    "description": "A phrase that contains one or more encoded relationship."
                }
            },
            "required": ["RELATIONAL_PHRASE"]
        }
    ]
}
