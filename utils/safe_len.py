# Derived from https://stackoverflow.com/a/51352502
def safe_len(source) -> int:
    try:
        return len(source)
    except TypeError:
        return 0