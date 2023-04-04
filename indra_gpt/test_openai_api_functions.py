from .check_correctness import generate_synonym_str, generate_example, \
    get_synonyms


def test_generate_synonym_str():
    # Lists are tuples of synonyms (a, b) where a is the synonym used in the
    # sentence/evidence text and b is the synonym used in the statement text.
    synonymlist_long = [("a", "b"), ("c", "d")]
    synonymlist_short = [("a", "b")]

    # Test long list without index
    syn_test = generate_synonym_str(synonymlist_long)
    assert '- "a" and "b"' in syn_test, syn_test
    assert '- "c" and "d"' in syn_test, syn_test

    # Test long list with index
    syn_test = generate_synonym_str(synonymlist_long, 1)
    assert '- "a" and "b"' in syn_test, syn_test
    assert '- "c" and "d"' in syn_test, syn_test
    assert f"Assume the following list of pairs are synonyms in Sentence1 " \
           f"and Statement1, respectively:" in syn_test, syn_test

    # Test short list without index
    syn_test = generate_synonym_str(synonymlist_short)
    assert (f'Assume "a" in the sentence and "b" in the statement are '
            f'synonyms' in syn_test), syn_test

    # Test short list with index
    syn_test = generate_synonym_str(synonymlist_short, 1)
    assert (f'Assume "a" in Sentence1 and "b" in Statement1 are '
            f'synonyms' in syn_test), syn_test


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


def test_get_synonyms():
    # Test nonsensical case
    ag_json_list = [
        {'name': 'a', 'db_refs': {'TEXT': 'a', 'HGNC': '1234'}},
        {'name': 'b', 'db_refs': {'TEXT': 'b', 'HGNC': '4321'}},
    ]
    syns = get_synonyms(ag_json_list)
    assert any("a" in syn for syn in syns), syns
    assert any("b" in syn for syn in syns), syns

