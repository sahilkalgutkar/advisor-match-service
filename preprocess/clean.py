"""Pandas/numpy cleaning pipeline for the raw advisor CSV export.

Real expert-network data is messy: inconsistent whitespace, tags typed with
stray spacing, years-of-experience entered as "12 yrs" or "N/A", and the
occasional advisor entered twice under a different id. This module turns
that into a clean, typed DataFrame ready to embed and index.
"""

import re

import numpy as np
import pandas as pd

_YEARS_PATTERN = re.compile(r"(\d+)")

EXPERIENCE_BUCKETS = ["junior", "mid", "senior", "principal"]
_EXPERIENCE_BUCKET_EDGES = [0, 5, 10, 20, np.inf]


def load_advisors(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def _clean_text(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()


def _parse_years_experience(value: object) -> float:
    text = _clean_text(value)
    match = _YEARS_PATTERN.search(text)
    return float(match.group(1)) if match else np.nan


def _parse_tags(value: object) -> list[str]:
    text = _clean_text(value)
    if not text:
        return []
    return [tag.strip().lower() for tag in text.split(";") if tag.strip()]


def experience_bucket(years: pd.Series) -> pd.Series:
    """Buckets years-of-experience into junior/mid/senior/principal bands."""
    indices = np.digitize(years.fillna(0), _EXPERIENCE_BUCKET_EDGES[1:-1])
    return pd.Series(
        [EXPERIENCE_BUCKETS[i] if pd.notna(y) else None for i, y in zip(indices, years)],
        index=years.index,
    )


def clean_advisors(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["name"] = cleaned["name"].map(_clean_text)
    cleaned["bio"] = cleaned["bio"].map(_clean_text)
    cleaned["tags"] = cleaned["tags"].map(_parse_tags)
    cleaned["years_experience"] = cleaned["years_experience"].map(_parse_years_experience)

    cleaned = cleaned[cleaned["bio"] != ""]
    cleaned = cleaned.drop_duplicates(subset=["name", "bio"], keep="first")
    cleaned["experience_bucket"] = experience_bucket(cleaned["years_experience"])

    return cleaned.reset_index(drop=True)
