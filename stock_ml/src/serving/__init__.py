"""Serving / production deployment of leaderboard models.

This package packages a trained template into a self-contained *model bundle*
(models + resolved config + feature spec + manifest) so a separate production
service can load it and reproduce signals without the research DB or training
code. See memory note ``project_production_bundle_deploy``.
"""

from __future__ import annotations

# Bumped only when the on-disk bundle LAYOUT changes (not on every model export).
# load_bundle() refuses a bundle whose format_version major differs — fail loud
# rather than silently mis-reading an incompatible artifact.
BUNDLE_FORMAT_VERSION = "1.0"

__all__ = ["BUNDLE_FORMAT_VERSION"]
