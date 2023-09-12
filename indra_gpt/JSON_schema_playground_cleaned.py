"""This module contains functions for extracting statements from text
feeding the full json schema using ChatGPT."""

# import libaries

import argparse

import pandas as pd

from indra_gpt.api import run_openai_chat
from indra.sources import reach

from indra.statements.io import stmts_to_json_file

import json

# function to feed chatGPT a prompt including the full json schema and
# ask it to generate a json object for a sentence using information in
# the schema
def gpt_stmt_json(stmt_json_examples, evidence_text):

    ############################# PROMPT ENGINEERING ###########################

    JSON_Schema = [{
        "$schema": "http://json-schema.org/draft-06/schema#",
        "id": "http://www.indra.bio/schemas/statements.json",
        "definitions": {
            "ModCondition": {
                "type": "object",
                "description": "Mutation state of an amino acid position of an Agent.",
                "properties": {
                    "mod_type": {
                        "type": "string",
                        "description": "The type of post-translational modification, e.g., 'phosphorylation'. Valid modification types currently include: 'phosphorylation', 'ubiquitination', 'sumoylation', 'hydroxylation', and 'acetylation'. If an invalid modification type is passed an InvalidModTypeError is raised."
                    },
                    "residue": {
                        "type": "string",
                        "description": "String indicating the modified amino acid, e.g., 'Y' or 'tyrosine'. If None, indicates that the residue at the modification site is unknown or unspecified."
                    },
                    "position": {
                        "type": "string",
                        "description": "String indicating the position of the modified amino acid, e.g., '202'. If None, indicates that the position is unknown or unspecified."
                    },
                    "is_modified": {
                        "type": "boolean",
                        "description": "Specifies whether the modification is present or absent. Setting the flag specifies that the Agent with the ModCondition is unmodified at the site."
                    }
                },
                "required": ["mod_type", "is_modified"]
            },
            "MutCondition": {
                "type": "object",
                "description": "Mutation state of an amino acid position of an Agent.",
                "properties": {
                    "position": {
                        "type": ["string", "null"],
                        "description": "Residue position of the mutation in the protein sequence."
                    },
                    "residue_from": {
                        "type": ["string", "null"],
                        "description": "Wild-type (unmodified) amino acid residue at the given position."
                    },
                    "residue_to": {
                        "type": ["string", "null"],
                        "description": "Amino acid at the position resulting from the mutation."
                    }
                },
                "required": ["position", "residue_from", "residue_to"]
            },
            "ActivityCondition": {
                "type": "object",
                "description": "An active or inactive state of a protein.",
                "properties": {
                    "activity_type": {
                        "type": "string",
                        "description": "The type of activity, e.g. 'kinase'. The basic, unspecified molecular activity is represented as 'activity'. Examples of other activity types are 'kinase', 'phosphatase', 'catalytic', 'transcription', etc."
                    },
                    "is_active": {
                        "type": "boolean",
                        "description": "Specifies whether the given activity type is present or absent."
                    }
                },
                "required": ["activity_type", "is_active"]
            },
            "BoundCondition": {
                "type": "object",
                "description": "Identify Agents bound (or not bound) to a given Agent in a given context.",
                "properties": {
                    "agent": {
                        "$ref": "#/definitions/Agent"
                    },
                    "is_bound": {
                        "type": "boolean",
                        "description": "Specifies whether the given Agent is bound or unbound in the current context."
                    }
                },
                "required": ["agent", "is_bound"]
            },
            "Agent": {
                "type": "object",
                "description": "A molecular entity, e.g., a protein.",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the agent, preferably a canonicalized name such as an HGNC gene name."
                    },
                    "mods": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/ModCondition"},
                        "description": "Modification state of the agent."
                    },
                    "mutations": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/MutCondition"},
                        "description": "Amino acid mutations of the agent."
                    },
                    "bound_conditions": {
                        "type": "array",
                        "items": {"$ref": "#/definitions/BoundCondition"},
                        "description": "Other agents bound to the agent in this context."
                    },
                    "activity": {
                        "$ref": "#/definitions/ActivityCondition",
                        "description": "Activity of the agent."
                    },
                    "location": {
                        "type": "string",
                        "description": "Cellular location of the agent. Must be a valid name (e.g. 'nucleus') or identifier (e.g. 'GO:0005634')for a GO cellular compartment."
                    },
                    "db_refs": {
                        "type": "object",
                        "description": "Dictionary of database identifiers associated with this agent."
                    },
                    "sbo": {
                        "type": "string",
                        "description": "Role of this agent in the systems biology ontology"
                    }
                },
                "required": ["name", "db_refs"]
            },
            "Concept": {
                "type": "object",
                "description": "A concept/entity of interest that is the argument of a Statement",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name of the concept, possibly a canonicalized name."
                    },
                    "db_refs": {
                        "type": "object",
                        "description": "Dictionary of database identifiers associated with this concept."
                    }
                },
                "required": ["name", "db_refs"]
            },
            "Context": {
                "type": "object",
                "description": "The context in which a given Statement was reported.",
                "properties": {
                    "type": {
                        "type": "string",
                        "pattern": "^((bio)|(world))$",
                        "description": "Either 'world' or 'bio', depending on the type of context being repersented."
                    }
                }
            },
            "BioContext": {
                "type": "object",
                "description": "The biological context of a Statement.",
                "properties": {
                    "type": {
                        "type": "string",
                        "pattern": "^bio$",
                        "description": "The type of context, in this case 'bio'."
                    },
                    "location": {
                        "$ref": "#/definitions/RefContext",
                        "description": "Cellular location, typically a sub-cellular compartment."
                    },
                    "cell_line": {
                        "$ref": "#/definitions/RefContext",
                        "description": "Cell line context, e.g., a specific cell line, like BT20."
                    },
                    "cell_type": {
                        "$ref": "#/definitions/RefContext",
                        "description": "Cell type context, broader than a cell line, like macrophage."
                    },
                    "organ": {
                        "$ref": "#/definitions/RefContext",
                        "description": "Organ context."
                    },
                    "disease": {
                        "$ref": "#/definitions/RefContext",
                        "description": "Disease context."
                    },
                    "species": {
                        "$ref": "#/definitions/RefContext",
                        "description": "Species context."
                    }
                }
            },
            "WorldContext": {
                "type": "object",
                "description": "The temporal and spatial context of a Statement.",
                "properties": {
                    "type": {
                        "type": "string",
                        "pattern": "^world$",
                        "description": "The type of context, in this case 'world'."
                    },
                    "time": {
                        "$ref": "#/definitions/TimeContext",
                        "description": "The temporal context of a Statement."
                    },
                    "geo_location": {
                        "$ref": "#/definitions/RefContext",
                        "description": "The geographical context of a Statement."
                    }
                }
            },
            "TimeContext": {
                "type": "object",
                "description": "Represents a temporal context.",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The text associated with the temporal context."
                    },
                    "start": {
                        "type": "string",
                        "description": "The start time of the temporal context."
                    },
                    "end": {
                        "type": "string",
                        "description": "The end time of the temporal context."
                    },
                    "duration": {
                        "type": "string",
                        "description": "The duration of the temporal context."
                    }
                }
            },
            "RefContext": {
                "type": "object",
                "description": "Represents a context identified by name and grounding references.",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "The name associated with the context."
                    },
                    "db_refs": {
                        "type": "object",
                        "description": "Dictionary of database identifiers associated with this context."
                    }
                }
            },
            "Evidence": {
                "type": "object",
                "description": "Container for evidence supporting a given statement.",
                "properties": {
                    "source_api": {
                        "type": "string",
                        "description": "String identifying the INDRA API used to capture the statement, e.g., 'trips', 'biopax', 'bel'."
                    },
                    "pmid": {
                        "type": "string",
                        "description": "String indicating the Pubmed ID of the source of the statement."
                    },
                    "source_id": {
                        "type": "string",
                        "description": "For statements drawn from databases, ID of the database entity corresponding to the statement."
                    },
                    "text": {
                        "type": "string",
                        "description": "Natural language text supporting the statement."
                    },
                    "annotations": {
                        "type": "object",
                        "description": "Dictionary containing additional information on the context of the statement, e.g., species, cell line, tissue type, etc. The entries may vary depending on the source of the information."
                    },
                    "epistemics": {
                        "type": "object",
                        "description": "A dictionary describing various forms of epistemic certainty associated with the statement."
                    }
                }
            },
            "Statement": {
                "type": "object",
                "description": "All statement types, below, may have these fields and 'inherit' from this schema",
                "properties": {
                    "evidence": {
                        "type": "array",
                        "items": {
                            "$ref": "#/definitions/Evidence"
                        }
                    },
                    "id": {
                        "type": "string",
                        "description": "Statement UUID"
                    },
                    "supports": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    },
                    "supported_by": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        }
                    }
                },
                "required": ["id"]
            },
            "Modification": {
                "description": "Statement representing the modification of a protein.",
                "allOf": [
                    {
                        "$ref": "#/definitions/Statement"
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "pattern": "^((Phosphorylation)|(Dephosphorylation)|(Ubiquitination)|(Deubiquitination)|(Sumoylation)|(Desumoylation)|(Hydroxylation)|(Dehydroxylation)|(Acetylation)|(Deacetylation)|(Glycosylation)|(Deglycosylation)|(Farnesylation)|(Defarnesylation)|(Geranylgeranylation)|(Degeranylgeranylation)|(Palmitoylation)|(Depalmitoylation)|(Myristoylation)|(Demyristoylation)|(Ribosylation)|(Deribosylation)|(Methylation)|(Demethylation))$",
                                "description": "The type of the statement"
                            },
                            "enz": {
                                "$ref": "#/definitions/Agent",
                                "description": "The enzyme involved in the modification."
                            },
                            "sub": {
                                "$ref": "#/definitions/Agent",
                                "description": "The substrate of the modification."
                            },
                            "residue": {
                                "type": "string",
                                "description": "The amino acid residue being modified, or None if it is unknown or unspecified."
                            },
                            "position": {
                                "type": "string",
                                "description": "The position of the modified amino acid, or None if it is unknown or unspecified."
                            }
                        },
                        "required": ["type"]
                    }
                ]
            },
            "SelfModification": {
                "description": "Statement representing the self-modification of a protein.",
                "allOf": [
                    {
                        "$ref": "#/definitions/Statement"
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "pattern": "^((Autophosphorylation)|(Transphosphorylation))$",
                                "description": "The type of the statement"
                            },
                            "enz": {
                                "$ref": "#/definitions/Agent",
                                "description": "The enzyme involved in the modification."
                            },
                            "residue": {
                                "type": "string",
                                "description": "The amino acid residue being modified, or None if it is unknown or unspecified."
                            },
                            "position": {
                                "type": "string",
                                "description": "The position of the modified amino acid, or None if it is unknown or unspecified."
                            }
                        },
                        "required": ["type"]
                    }
                ]
            },
            "RegulateActivity": {
                "description": "Regulation of activity (such as activation and inhibition)",
                "allOf": [
                    {
                        "$ref": "#/definitions/Statement"
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "pattern": "^((Activation)|(Inhibition))$",
                                "description": "The type of the statement"
                            },
                            "subj": {
                                "$ref": "#/definitions/Agent",
                                "description": "The agent responsible for the change in activity, i.e., the 'upstream' node."
                            },
                            "obj": {
                                "$ref": "#/definitions/Agent",
                                "description": "The agent whose activity is influenced by the subject, i.e., the 'downstream' node."
                            },
                            "obj_activity": {
                                "type": "string",
                                "description": "The activity of the obj Agent that is affected, e.g., its 'kinase' activity."
                            }
                        },
                        "required": ["type"]
                    }
                ]
            },
            "ActiveForm": {
                "description": "Specifies conditions causing an Agent to be active or inactive. Types of conditions influencing a specific type of biochemical activity can include modifications, bound Agents, and mutations.",
                "allOf": [
                    {
                        "$ref": "#/definitions/Statement"
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "pattern": "^ActiveForm$",
                                "description": "The type of the statement"
                            },
                            "agent": {
                                "$ref": "#/definitions/Agent",
                                "description": "The Agent in a particular active or inactive state. The sets of ModConditions, BoundConditions, and MutConditions on the given Agent instance indicate the relevant conditions."
                            },
                            "activity": {
                                "type": "string",
                                "description": "The type of activity influenced by the given set of conditions, e.g., 'kinase'."
                            },
                            "is_active": {
                                "type": "boolean",
                                "description": "Whether the conditions are activating (True) or inactivating (False)."
                            }
                        },
                        "required": ["type", "agent", "activity"]
                    }
                ]
            },
            "Gef": {
                "description": "Exchange of GTP for GDP on a small GTPase protein mediated by a GEF. Represents the generic process by which a guanosine exchange factor (GEF) catalyzes nucleotide exchange on a GTPase protein.",
                "allOf": [
                    {
                        "$ref": "#/definitions/Statement"
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "pattern": "^Gef$",
                                "description": "The type of the statement"
                            },
                            "gef": {
                                "$ref": "#/definitions/Agent",
                                "description": "The guanosine exchange factor."
                            },
                            "ras": {
                                "$ref": "#/definitions/Agent",
                                "description": "The GTPase protein."
                            }
                        },
                        "required": ["type"]
                    }
                ]
            },
            "Gap": {
                "description": "Acceleration of a GTPase protein's GTP hydrolysis rate by a GAP. Represents the generic process by which a GTPase activating protein (GAP) catalyzes GTP hydrolysis by a particular small GTPase protein.",
                "allOf": [
                    {
                        "$ref": "#/definitions/Statement"
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "pattern": "^Gap$",
                                "description": "The type of the statement"
                            },
                            "gap": {
                                "$ref": "#/definitions/Agent",
                                "description": "The GTPase activating protein."
                            },
                            "ras": {
                                "$ref": "#/definitions/Agent",
                                "description": "The GTPase protein."
                            }
                        },
                        "required": ["type"]
                    }
                ]
            },
            "Complex": {
                "description": "A set of proteins observed to be in a complex.",
                "allOf": [
                    {
                        "$ref": "#/definitions/Statement"
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "pattern": "^((Complex)|(Association))$",
                                "description": "The type of the statement"
                            },
                            "members": {
                                "type": "array",
                                "items": {
                                    "$ref": "#/definitions/Agent"
                                }
                            }
                        },
                        "required": ["type"]
                    }
                ]
            },
            "Association": {
                "description": "A set of unordered concepts that are associated with each other.",
                "allOf": [
                    {
                        "$ref": "#/definitions/Complex"
                    }
                ]
            },
            "Translocation": {
                "description": "The translocation of a molecular agent from one location to another.",
                "allOf": [
                    {
                        "$ref": "#/definitions/Statement"
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "pattern": "^Translocation$",
                                "description": "The type of the statement"
                            },
                            "agent": {
                                "$ref": "#/definitions/Agent",
                                "description": "The agent which translocates."
                            },
                            "from_location": {
                                "type": "string",
                                "description": "The location from which the agent translocates. This must be a valid GO cellular component name (e.g. 'cytoplasm') or ID (e.g. 'GO:0005737')."
                            },
                            "to_location": {
                                "type": "string",
                                "description": "The location to which the agent translocates. This must be a valid GO cellular component name or ID."
                            }
                        },
                        "required": ["type", "agent"]
                    }
                ]
            },
            "RegulateAmount": {
                "description": "Represents directed, two-element interactions.",
                "allOf": [
                    {
                        "$ref": "#/definitions/Statement"
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "pattern": "^((IncreaseAmount)|(DecreaseAmount))$"
                            },
                            "subj": {
                                "$ref": "#/definitions/Agent",
                                "description": "The mediating protein"
                            },
                            "obj": {
                                "$ref": "#/definitions/Agent",
                                "description": "The affected protein"
                            }
                        },
                        "required": ["type"]
                    }
                ]
            },
            "Influence": {
                "description": "A causal influence between two events.",
                "allOf": [
                    {
                        "$ref": "#/definitions/Statement"
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "pattern": "^Influence$"
                            },
                            "subj": {
                                "$ref": "#/definitions/Event",
                                "description": "The event which acts as the influencer."
                            },
                            "obj": {
                                "$ref": "#/definitions/Event",
                                "description": "The event which acts as the influencee"
                            }
                        },
                        "required": ["type"]
                    }
                ]
            },
            "Event": {
                "description": "An event over a concept of interest.",
                "allOf": [
                    {
                        "$ref": "#/definitions/Statement"
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "pattern": "^Event$"
                            },
                            "concept": {
                                "$ref": "#/definitions/Concept",
                                "description": "The concept which the event happens to."
                            },
                            "delta": {
                                "type": ["object", "null"],
                                "description": "A dictionary specifying the polarity and adjactives of change in concept."
                            },
                            "context": {
                                "$ref": "#/definitions/Context",
                                "description": "The context associated with the event"
                            }
                        },
                        "required": ["type", "concept"]
                    }
                ]
            },
            "Conversion": {
                "description": "Conversion of molecular species mediated by a controller protein.",
                "allOf": [
                    {
                        "$ref": "#/definitions/Statement"
                    },
                    {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "pattern": "^Conversion$"
                            },
                            "subj": {
                                "$ref": "#/definitions/Agent",
                                "description": "The protein mediating the conversion."
                            },
                            "obj_from": {
                                "type": "array",
                                "items": {
                                    "$ref": "#/definitions/Agent"
                                },
                                "description": "The list of molecular species being consumed by the conversion."
                            },
                            "obj_to": {
                                "type": "array",
                                "items": {
                                    "$ref": "#/definitions/Agent"
                                },
                                "description": "The list of molecular species being created by the conversion."
                            }
                        },
                        "required": ["type"]
                    }
                ]
            }
        },

        "type": "array",
        "items": {
            "anyOf": [
                {"$ref": "#/definitions/RegulateActivity"},
                {"$ref": "#/definitions/Modification"},
                {"$ref": "#/definitions/SelfModification"},
                {"$ref": "#/definitions/ActiveForm"},
                {"$ref": "#/definitions/Gef"},
                {"$ref": "#/definitions/Gap"},
                {"$ref": "#/definitions/Complex"},
                {"$ref": "#/definitions/Association"},
                {"$ref": "#/definitions/Translocation"},
                {"$ref": "#/definitions/RegulateAmount"},
                {"$ref": "#/definitions/Influence"},
                {"$ref": "#/definitions/Conversion"},
                {"$ref": "#/definitions/Event"}
            ]
        }
    }] # variable storing full json schema

    json_schema_string = str(JSON_Schema) # converting json schema to a string
    # print(len(json))

    invalid_pieces = '"$ref": "#/definitions/Agent"' # store any invalid
    # syntax in the json schema in the variable invalid_pieces to ask
    # chatGPT to remove them from its generated json object

    # full prompt including schema
    PROMPT = "Read the following JSON schema for a statement " \
             "object:\n\n```json\n" + json_schema_string.replace('{',
                                                                 '{{').replace(
        '}',
        '}}') \
             + "\n```\n\nExtract the relation from the following sentence and put it in a JSON object matching the schema above. The JSON object needs to be able to pass a validation against the provided schema. Remove " + invalid_pieces + ". Only respond with " \
                                                                                                                                                                                                                                                "the JSON object.\n\nSentence: "

    # reduced prompt not including schema
    PROMPT_reduced = "Extract the relation from the following sentence and put " \
                     "it " \
                     "in a JSON object matching the schema above. The JSON object needs to be able to pass a validation against the provided schema. Remove " + invalid_pieces + ". Only respond with " \

    ############################## HISTORY ############################

    # variables to feed the chat history:
    stmt_json_1 = json_object_list[0] # first example json in the training
    # dataframe
    ev_text_1 = stmt_json_1['evidence'][0]['text'] # first example sentence
    # extracted from the example json
    stmt_json_2 = json_object_list[1] # second example json in the
    # training dataframe
    ev_text_2 = stmt_json_2['evidence'][0]['text'] # second example sentence
    # extracted from the example json


    # create the chat history, including the prompt with the schema,
    # the first example json to feed chatGPT as a sample, the reduced prompt
    # without schema, the second example json to feed chatGPT as a sample
    history = [
        {"role": "user",
         "content": PROMPT + ev_text_1},  # Prompt with schema
        {"role": "assistant",
         "content": stmt_json_1},  # first stmt json example
        {"role": "user",
         "content": PROMPT_reduced + ev_text_2},  # prompt without schema
        {"role": "assistant",
         "content": stmt_json_2}  # second stmt json example
    ]

    ################### RUN PROMPT TO ASK CHATGPT #####################

    prompt = (PROMPT_reduced + evidence_text).format(prompt=evidence_text)
    # format the main prompt to ask chatGPT without being fed sample
    # data to only include the reduced prompt without the json schema +
    # the main sentence inputted into the gpt_stmt_json function

    chat_gpt_english = run_openai_chat(prompt, model='gpt-3.5-turbo-16k',
                                       chat_history=history,
                                       max_tokens=16000, strip=False,
                                       debug=debug) # use run_openai_chat
    # function on prompt, specifying model and max_tokens parameters as
    # needed
    return chat_gpt_english # return chatGPT's response


# main function to run on the inputted traing dataframe of json objects
def main(training_df):
    json_object_list = [] # list of json objects (every item in the file)
    outputs = [] # list of every output by chatGPT
    sentences = [] # list of the sentences fed to the prompt

    statements = [] # list of json statements returned from outputted json
    # object by chatGPT (when applicable to a sentence and its outputted json
    # object)

    with open(training_df,
              'r') as f:
        contents = f.read # read file
        json_object_list.append(contents) # append entire file of sample json
        # object to contents

    for json_object in json_object_list:
        stmt_json_n = json_object # assign the json object
        # for the
        # current item in json_object_list to variable stmt_json_n
        ev_text_n = stmt_json_n['evidence'][json_object]['text'] # get the
        # sentence for the current item in json_object_list

        output_n = gpt_stmt_json(stmt_json_n,ev_text_n)  # run gpt_stmt_json
        # function on the current object and sentence to output a response
        # from chatGPT
        sliced_output_n = output_n['choices'][0]['message']['content'] # get
        # only the generated json object from chatGPT's outputted response

        outputs.append[sliced_output_n] # append to list of every output by
        # chatGPT for each fed sentence
        sentences.append[ev_text_n] # append to list of every sentence

        # Some generated json objects are able to take in the
        # json.loads and stmt_from_json functions to return a statement
        # json. Some are not. Use the try/except method to extract statement
        # jsons from the generated json objects for the ones that work.
        # Append them to the list of all statements extracted. If it doesn't
        # work on a generated json object, just append that there was an
        # error loading the statement.
        try:
            json_str_n = json.loads(sliced_output_n) #
            stmt_n = stmt_from_json(json_str_n)
            statements.append[stmt_n]
        except:
            statements.append["***error loading statement from generated " \
                              "json object***"]

    df = pd.DataFrame({ 'sentence'
                    ': ': [sentences],
                    'generated_json_object: ': [outputs],
                        'extracted_statement: ' : [statements]}) # put
    # sentences, outputs, and statements list into a pd dataframe

    df.columns = ['sentence: ', 'generated_json_object: ',
                  'extracted_statemenet: '] # headers

    df.to_csv('/Users/bihan/Work/indra_gpt/indra_gpt'
              '/JSON_schema_playground_cleaned.csv') # save file as csv (
    # change path as needed)

    print(outputs[:5]) # only print the first 5 results
    print("Done.") # print done to know when main function has finished

main('/Users/bihan/Downloads/indra_benchmark_corpus_sample_50.json') # run
# main function on path to training dataset of sample json objects (change
# path as needed)
