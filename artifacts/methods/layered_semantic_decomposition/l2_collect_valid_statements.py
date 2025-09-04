def valid_from_exp_tree(tree, not_count=0, has_hypothesis=False):
    identities = []

    if not isinstance(tree, dict):
        return identities

    for key, value in tree.items():
        if key == "HYPOTHESIS":
            # Skip hypothesis (mark it)
            identities += valid_from_exp_tree(value, not_count, True)

        elif key == "NOT":
            # Increment negation depth
            identities += valid_from_exp_tree(value, not_count + 1, has_hypothesis)

        elif key in ("AND", "OR"):
            # Recurse on each item in the list
            for item in value:
                identities += valid_from_exp_tree(item, not_count, has_hypothesis)

        elif key == "DECLARATIVE":
            identities += valid_from_exp_tree(value, not_count, has_hypothesis)

        elif key == "IDENTITY":
            if isinstance(value, dict):
                # Can be nested like {"ATOM": "..."} or {"AND": [...]}
                for subkey, subval in value.items():
                    if subkey == "ATOM":
                        if not has_hypothesis and not_count % 2 == 0:
                            identities.append(subval)
                    elif subkey in ("AND", "OR"):
                        for item in subval:
                            identities += valid_from_exp_tree(item, not_count, has_hypothesis)
            elif isinstance(value, str):
                # Sometimes identity may just map to a string
                if not has_hypothesis and not_count % 2 == 0:
                    identities.append(value)

    return identities
