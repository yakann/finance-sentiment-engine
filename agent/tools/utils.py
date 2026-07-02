def format_large_number(val: float | int | None) -> str | None:
    """Format a raw dollar value into a human-readable string with T/B/M suffix."""
    if val is None or val <= 0:
        return None
    val = float(val)
    if val >= 1_000_000_000_000:
        return f"${val / 1e12:.2f}T"
    if val >= 1_000_000_000:
        return f"${val / 1e9:.2f}B"
    return f"${val / 1e6:.2f}M"
