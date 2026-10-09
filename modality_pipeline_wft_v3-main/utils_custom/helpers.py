def get_nested(d, *keys, default=False):
    for key in keys:
        if not isinstance(d, dict):
            return default
        d = d.get(key, default)
        if d is default:
            return default
    return d