"""Generic RAG API; dense dependencies are loaded only when requested."""
from .pipeline import Baseline, Generator, Retriever, PreviewGenerator, HTTPGenerator

__all__ = ["BGEEncoder", "EncoderConfig", "DenseRetriever", "Baseline", "Generator",
           "Retriever", "PreviewGenerator", "HTTPGenerator"]


def __getattr__(name):
    if name in ("BGEEncoder", "EncoderConfig", "DenseRetriever"):
        from . import core
        return getattr(core, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
