"""Proje genelinde kullanilan stdlib logging yapilandirmasi."""

from __future__ import annotations

import logging


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Kok logger'i yapilandirir ve dondurur."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("humanai_detect")
