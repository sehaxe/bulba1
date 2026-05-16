"""
Triton-accelerated operations for Bulba1.
"""

def is_triton_available():
    """Check if Triton is available."""
    try:
        import triton
        return True
    except ImportError:
        return False