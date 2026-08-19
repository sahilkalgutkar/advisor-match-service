import numpy as np
import pandas as pd

from preprocess.clean import clean_advisors, experience_bucket, load_advisors


def test_load_advisors_reads_the_seed_csv():
    df = load_advisors("data/advisors.csv")
    assert len(df) > 0
    assert {"id", "name", "bio", "tags", "years_experience"}.issubset(df.columns)


def test_clean_advisors_strips_whitespace_from_text_fields():
    df = pd.DataFrame(
        {
            "id": [1],
            "name": ["  Jane Doe  "],
            "bio": ["A  bio   with   extra   spaces.  "],
            "tags": ["Tag1; Tag2"],
            "years_experience": ["5"],
        }
    )

    cleaned = clean_advisors(df)

    assert cleaned.loc[0, "name"] == "Jane Doe"
    assert cleaned.loc[0, "bio"] == "A bio with extra spaces."


def test_clean_advisors_parses_tags_into_a_normalized_list():
    df = pd.DataFrame(
        {
            "id": [1],
            "name": ["Jane Doe"],
            "bio": ["Bio."],
            "tags": [" Healthcare ; M&A ;Due Diligence "],
            "years_experience": ["5"],
        }
    )

    cleaned = clean_advisors(df)

    assert cleaned.loc[0, "tags"] == ["healthcare", "m&a", "due diligence"]


def test_clean_advisors_parses_messy_years_experience():
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["A", "B", "C"],
            "bio": ["Bio A.", "Bio B.", "Bio C."],
            "tags": ["", "", ""],
            "years_experience": ["12 yrs", " 7 ", "N/A"],
        }
    )

    cleaned = clean_advisors(df)

    assert cleaned.loc[0, "years_experience"] == 12.0
    assert cleaned.loc[1, "years_experience"] == 7.0
    assert np.isnan(cleaned.loc[2, "years_experience"])


def test_clean_advisors_drops_rows_with_an_empty_bio():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["A", "B"],
            "bio": ["Real bio.", "   "],
            "tags": ["", ""],
            "years_experience": ["5", "5"],
        }
    )

    cleaned = clean_advisors(df)

    assert len(cleaned) == 1
    assert cleaned.loc[0, "name"] == "A"


def test_clean_advisors_deduplicates_the_same_advisor_entered_twice():
    df = pd.DataFrame(
        {
            "id": [1, 2],
            "name": ["Jane Doe", "Jane Doe"],
            "bio": ["Same bio.", "Same bio."],
            "tags": ["", ""],
            "years_experience": ["5", "5"],
        }
    )

    cleaned = clean_advisors(df)

    assert len(cleaned) == 1


def test_clean_advisors_end_to_end_on_the_real_seed_data():
    raw = load_advisors("data/advisors.csv")
    cleaned = clean_advisors(raw)

    # Row 5 (blank bio) is dropped, row 9 (duplicate of row 3) is deduped.
    assert len(cleaned) == len(raw) - 2
    assert cleaned["bio"].map(lambda b: b == "").sum() == 0
    assert cleaned["tags"].map(lambda t: isinstance(t, list)).all()


def test_experience_bucket_assigns_the_right_band():
    years = pd.Series([2.0, 7.0, 15.0, 25.0, np.nan])

    buckets = experience_bucket(years)

    assert list(buckets)[:4] == ["junior", "mid", "senior", "principal"]
    assert pd.isna(buckets.iloc[4])
