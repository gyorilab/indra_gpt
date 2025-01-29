from indra.statements import get_all_descendants, Statement


def trim_stmt_json(stmt):
    """Function to get rid of irrelevant parts of the indra statement json

    Parameter
    ---------
    stmt : dict
        The indra statement json object

    Returns
    -------
    dict
        The indra statement json object with irrelevant parts removed
    """

    stmt["evidence"] = [{"text": stmt["evidence"][0].get("text")}]

    del stmt["id"]
    del stmt["matches_hash"]

    if 'supports' in stmt:
        del stmt['supports']
    if 'supported_by' in stmt:
        del stmt['supported_by']

    return stmt


def post_process_extracted_json(gpt_stmt_json):
    """Function to post process the extracted json from chatGPT

    Parameters
    ----------
    gpt_stmt_json : dict
        The extracted json from chatGPT

    Returns
    -------
    dict
        The post processed json or if there is a KeyError, the original json
    """
    try:
        stmt_type = gpt_stmt_json["type"]
        mapped_type = stmt_mapping.get(stmt_type.lower(), stmt_type)
        gpt_stmt_json["type"] = mapped_type
    except KeyError:
        pass

    return gpt_stmt_json


def _get_statement_mapping():
    stmt_classes = get_all_descendants(Statement)
    mapping = {stmt_class.__name__.lower(): stmt_class.__name__ for stmt_class in stmt_classes}
    return mapping


stmt_mapping = _get_statement_mapping()
