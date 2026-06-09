"""SKVoice — Sovereign voice agent service."""

try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
    __version__ = _pkg_version("skvoice")
except (ImportError, PackageNotFoundError):
    try:
        from ._version import version as __version__
    except ImportError:
        __version__ = "0.0.0+unknown"
