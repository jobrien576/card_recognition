import sys

if sys.version_info[:2] >= (3, 8):
    # Use importlib.metadata directly for Python >= 3.8
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    # For Python < 3.8, use importlib_metadata backport
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Set the package name; update this if your project name differs
    dist_name = "card_classifier"  # Replace with your actual package name if different
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError