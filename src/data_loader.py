"""
src/data_loader.py
Student A — Data acquisition & loading layer.

Dataset: Steam 2024 — Top 1,500 Games by Revenue
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

RAW_FILENAME = "Steam_2024_bestRevenue_1500.csv"
RAW_PATH = RAW_DIR / RAW_FILENAME

# --- Schema contract -------------------------------------------------------
EXPECTED_COLUMNS: list[str] = [
    "name",
    "releaseDate",
    "copiesSold",
    "price",
    "revenue",
    "avgPlaytime",
    "reviewScore",
    "publisherClass",
    "publishers",
    "developers",
    "steamId",
]

DTYPE_MAP: dict[str, str] = {
    "name": "string",
    "copiesSold": "Int64",
    "price": "float64",
    "revenue": "float64",
    "avgPlaytime": "float64",
    "reviewScore": "Int64",
    "publisherClass": "category",
    "publishers": "string",
    "developers": "string",
    "steamId": "Int64",
}

DATE_COLUMN = "releaseDate"
DATE_FORMAT = "%d-%m-%Y"  # day-first! e.g. 07-03-2024 = 7 March 2024

VALID_PUBLISHER_CLASSES = {"Indie", "AA", "AAA", "Hobbyist"}
EXPECTED_ROW_COUNT = 1500

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def load_raw_data(path: Path | str = RAW_PATH, parse_dates: bool = True,
                  **read_kwargs) -> pd.DataFrame:
    """Load the Steam 2024 revenue CSV with correct dtypes and date parsing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Raw data not found at: {path}\n"
            f"Download the dataset from Kaggle and place the CSV in {RAW_DIR}/"
        )

    defaults = {"encoding": "utf-8", "dtype": DTYPE_MAP}
    defaults.update(read_kwargs)
    df = pd.read_csv(path, **defaults)

    if parse_dates and DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(
            df[DATE_COLUMN], format=DATE_FORMAT, errors="coerce"
        )
        n_bad = df[DATE_COLUMN].isna().sum()
        if n_bad:
            logger.warning("%d releaseDate value(s) failed to parse.", n_bad)

    logger.info("Loaded %s -> %d rows x %d columns", path.name, *df.shape)
    return df


def validate_schema(df: pd.DataFrame,
                    expected_columns: list[str] | None = None) -> bool:
    """Validate structure, column set, and domain rules for this dataset."""
    expected_columns = expected_columns or EXPECTED_COLUMNS

    if df.empty:
        raise ValueError("Loaded DataFrame is empty.")

    missing = [c for c in expected_columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing expected column(s): {missing}")

    extra = [c for c in df.columns if c not in expected_columns]
    if extra:
        logger.warning("Unexpected extra column(s): %s", extra)

    if len(df) != EXPECTED_ROW_COUNT:
        logger.warning("Expected %d rows, found %d.", EXPECTED_ROW_COUNT, len(df))

    # --- Domain checks ---
    if df["steamId"].duplicated().any():
        raise ValueError("Duplicate steamId values found — rows are not unique.")

    bad_class = set(df["publisherClass"].dropna().unique()) - VALID_PUBLISHER_CLASSES
    if bad_class:
        raise ValueError(f"Unknown publisherClass value(s): {bad_class}")

    if (df["price"] < 0).any() or (df["revenue"] < 0).any():
        raise ValueError("Negative price or revenue detected.")

    if not df["reviewScore"].dropna().between(0, 100).all():
        raise ValueError("reviewScore outside the 0-100 range.")

    logger.info("Schema validation passed.")
    return True


def inspect_data(df: pd.DataFrame) -> pd.DataFrame:
    """Per-column inspection table: dtype, nulls, unique counts."""
    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "n_missing": df.isna().sum(),
            "pct_missing": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(),
        }
    )
    return summary.reset_index(names="column")


def save_processed(df: pd.DataFrame, filename: str = "steam_2024_clean.csv") -> Path:
    """Persist a processed DataFrame to ``data/processed/``."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / filename
    df.to_csv(out_path, index=False)
    logger.info("Saved processed data -> %s", out_path)
    return out_path


def create_data_dictionary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a data dictionary: one row per column with its dtype, missing %,
    unique count, and a few example values. 'Description' is left blank
    for manual fill-in (e.g. in reports/data_dictionary.csv).
    """
    rows = []
    for col in df.columns:
        sample = df[col].dropna().head(3).tolist()
        rows.append(
            {
                "Column": col,
                "Type": str(df[col].dtype),
                "Description": "",
                "Missing (%)": round(df[col].isna().mean() * 100, 1),
                "Unique Values": int(df[col].nunique()),
                "Example Values": str(sample)[:50],
            }
        )
    return pd.DataFrame(rows)


class DataLoader:
    """
    Thin OO wrapper around the module-level functions above, so notebooks
    can do `from src.data_loader import DataLoader` and use a familiar
    class-based API. All the real logic still lives in the functions —
    this class just calls them and remembers the DataFrame in between steps.
    """

    def __init__(self, raw_path: Path | str = RAW_PATH,
                 processed_dir: Path | str = PROCESSED_DIR):
        self.raw_path = Path(raw_path)
        self.processed_dir = Path(processed_dir)
        self.df: pd.DataFrame | None = None

    def load_raw_data(self, path: Path | str | None = None,
                       parse_dates: bool = True, **read_kwargs) -> pd.DataFrame:
        """Load the raw CSV (delegates to load_raw_data) and cache it on self.df."""
        self.df = load_raw_data(
            path or self.raw_path, parse_dates=parse_dates, **read_kwargs
        )
        return self.df

    def validate_schema(self, expected_columns: list[str] | None = None) -> bool:
        """Validate the currently loaded DataFrame (delegates to validate_schema)."""
        if self.df is None:
            raise RuntimeError("No data loaded yet — call load_raw_data() first.")
        return validate_schema(self.df, expected_columns)

    def inspect_data(self) -> pd.DataFrame:
        """Inspect the currently loaded DataFrame (delegates to inspect_data)."""
        if self.df is None:
            raise RuntimeError("No data loaded yet — call load_raw_data() first.")
        return inspect_data(self.df)

    def save_processed(self, df: pd.DataFrame | None = None,
                        filename: str = "steam_2024_clean.csv") -> Path:
        """Save a processed DataFrame (defaults to self.df) to data/processed/."""
        target = df if df is not None else self.df
        if target is None:
            raise RuntimeError("No data to save — pass df= or call load_raw_data() first.")
        return save_processed(target, filename)

    def get_dataframe(self) -> pd.DataFrame:
        """Return the currently loaded/processed DataFrame."""
        if self.df is None:
            raise RuntimeError("No data loaded yet — call load_raw_data() first.")
        return self.df

    def create_data_dictionary(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Build a data dictionary for the given df (defaults to self.df)."""
        target = df if df is not None else self.df
        if target is None:
            raise RuntimeError("No data to document — pass df= or call load_raw_data() first.")
        return create_data_dictionary(target)


if __name__ == "__main__":
    data = load_raw_data()
    validate_schema(data)
    print(data.head())
    print(inspect_data(data).to_string(index=False))