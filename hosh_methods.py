"""Backward-compatible import wrapper.

The revised manuscript uses the name MSH (Mesoscopic Structural Holes).
Some experiment scripts retain the legacy internal identifier HOSH to load
precomputed ranking caches generated during revision. This wrapper keeps those
scripts reproducible while all public documentation and output labels use MSH.
"""
from msh_methods import *  # noqa: F401,F403
