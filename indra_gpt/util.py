

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
