from indra_gpt.util.check_correctness import generate_example, generate_synonym_str, generate_prompt #, get_synonyms 


def test_generate_synonym_str():
    # Lists are tuples of synonyms (a, b) where a is the synonym used in the
    # sentence/evidence text and b is the synonym used in the statement text.
    synonymlist_long = [("a", "b"), ("c", "d")]
    synonymlist_short = [("a", "b")]

    def convert_to_agents_info(synonymlist):
        synonymlist_agents_info = {}
        for i, syn in enumerate(synonymlist):
            curie = f"syn_{i}"
            agent_info = {}
            agent_info['name'] = syn[0]
            agent_info['synonyms'] = [syn[0],syn[1]]
            agent_info['definition'] = None
            agent_info['syn_in_text'] = syn[0]
            agent_info['syn_in_stmt'] = syn[1]
            synonymlist_agents_info[curie] = agent_info
        return synonymlist_agents_info

    synonymlist_long = convert_to_agents_info(synonymlist_long)
    synonymlist_short = convert_to_agents_info(synonymlist_short)
    
    # Test long list without index
    syn_test = generate_synonym_str(synonymlist_long)
    assert '- "a" and "b"' in syn_test, syn_test
    assert '- "c" and "d"' in syn_test, syn_test

    # Test long list with index
    syn_test = generate_synonym_str(synonymlist_long, 1)
    assert '- "a" and "b"' in syn_test, syn_test
    assert '- "c" and "d"' in syn_test, syn_test
    assert (
        f"Assume the following list of pairs are synonyms in Sentence1 "
        f"and Statement1, respectively:" in syn_test
    ), syn_test

    # Test short list without index
    syn_test = generate_synonym_str(synonymlist_short)
    assert (
        f'Assume "a" in the sentence and "b" in the statement are '
        f"synonyms" in syn_test
    ), syn_test

    # Test short list with index
    syn_test = generate_synonym_str(synonymlist_short, 1)
    assert (
        f'Assume "a" in Sentence1 and "b" in Statement1 are ' f"synonyms" in syn_test
    ), syn_test


def test_generate_example():
    sentence = "a activates b in this text"
    statement = "A activates B"
    synonymlist = [("a", "A"), ("b", "B")]
    ex_str = generate_example(sentence, statement, synonymlist, 1)

    assert f'Sentence1: "{sentence}"' in ex_str, ex_str
    assert f'Statement1: "{statement}"' in ex_str, ex_str
    assert '- "a" and "A"' in ex_str, ex_str
    assert '- "b" and "B"' in ex_str, ex_str
    assert "synonyms in Sentence1 and Statement1, respectively" in ex_str, ex_str


def test_generate_example_no_synonyms():
    sentence = "A activates B in this text"
    statement = "A activates B"
    synonymlist = []
    ex_str = generate_example(sentence, statement, synonymlist, 1)

    assert f'Sentence1: "{sentence}"' in ex_str, ex_str
    assert f'Statement1: "{statement}"' in ex_str, ex_str
    assert "synonyms" not in ex_str, ex_str


# def test_get_synonyms():
#     # Test nonsensical case
#     ag_json_list = [
#         {"name": "a", "db_refs": {"TEXT": "a", "HGNC": "1234"}},
#         {"name": "b", "db_refs": {"TEXT": "b", "HGNC": "4321"}},
#     ]
#     syns = get_synonyms(ag_json_list)
#     assert any("a" in syn for syn in syns), syns
#     assert any("b" in syn for syn in syns), syns

def test_generate_prompt():
    """Quickly test the prompt generation by calling this function"""
    test_sentence1 = "a binds b and c in this text"
    test_stmt1 = "A binds B and C"
    test_synonyms1 = {
        "A": {
            "name": "a",
            "definition": "a is a protein",
            "synonyms": ["a", "A", "aa", "A-A"],
            "syn_in_text": "a",
            "syn_in_stmt": "A",
        },
        "B": {
            "name": "b",
            "definition": "b is a protein",
            "synonyms": ["b", "B"],
            "syn_in_text": "b",
            "syn_in_stmt": "B",
        },
        "C": {
            "name": "c",
            "definition": "c is a protein",
            "synonyms": ["c", "C"],
            "syn_in_text": "c",
            "syn_in_stmt": "C",
        },
    }

    test_sentence2 = "C phosphorylates D in this text"
    test_stmt2 = "C phosphorylates D"

    test_sentence3 = "E deactivates f in this text"
    test_stmt3 = "E activates F"
    test_synonyms3 = {
        "F": {
            "name": "F",
            "definition": "F is a small molecule",
            "synonyms": ["f", "F", "ff", "F3"],
            "syn_in_text": "f",
            "syn_in_stmt": "F",
        },
    }

    test_sentence4 = "X deactivates Y in this text"
    test_stmt4 = "x deactivates Y"
    test_synonyms4 = {
        "x": {
            "name": "X",
            "definition": "X is a protein",
            "synonyms": ["X", "x", "XX", "X1"],
            "syn_in_text": "X",
            "syn_in_stmt": "x",
        },
    }

    test_query_sentence = "a inhibits b in this text"
    test_query_stmt = "A inhibits B"
    test_query_synonyms = {
        "A": {
            "name": "A",
            "definition": "A is a protein",
            "synonyms": ["a", "A", "a1"],
            "syn_in_text": "a",
            "syn_in_stmt": "A",
        },
        "B": {
            "name": "B",
            "definition": "B is a protein",
            "synonyms": ["b", "B", "b1", "B1", "bb"],
            "syn_in_text": "b",
            "syn_in_stmt": "B",
        },
    }

    pos_examples = [
        (test_sentence1, test_stmt1, test_synonyms1),
        (test_sentence2, test_stmt2, None),
    ]
    neg_examples = [
        (test_sentence3, test_stmt3, test_synonyms3),
        (test_sentence4, test_stmt4, test_synonyms4),
    ]

    test_prompt = generate_prompt(
        query_sentence=test_query_sentence,
        query_stmt=test_query_stmt,
        pos_ex_list=pos_examples,
        neg_ex_list=neg_examples,
        query_agent_info=test_query_synonyms,
    )
    print(test_prompt)