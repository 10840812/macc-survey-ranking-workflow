"""Build a deterministic rank ordering of 2024 MAcc courses from survey data."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATA_PATH = Path("data/Grad Program Exit Survey Data 2024 (1).xlsx")
OUTPUT_CSV = Path("outputs/rank_order.csv")
OUTPUT_PNG = Path("outputs/rank_order.png")


def detect_header_row(excel_path: Path, max_rows_to_scan: int = 12) -> int:
    """Detect the real header row in case metadata rows are present.

    Qualtrics exports often contain:
    - row 0: short variable IDs (e.g., Q35_1)
    - row 1: full question text (best row for analysis)
    - row 2: import metadata JSON
    """
    preview = pd.read_excel(excel_path, header=None, nrows=max_rows_to_scan)

    best_row = 0
    best_score = -1

    for idx in range(len(preview)):
        row_text = " | ".join(str(v) for v in preview.iloc[idx].tolist())
        score = 0

        # Prefer rows that look like human-readable headers.
        if "Start Date" in row_text:
            score += 3
        if "Response ID" in row_text:
            score += 3
        if "rank order" in row_text.lower():
            score += 5

        # Penalize machine metadata rows.
        if "{\"ImportId\"" in row_text:
            score -= 5

        if score > best_score:
            best_score = score
            best_row = idx

    return best_row


def parse_course_name(column_name: str) -> str:
    """Extract a clean course label from a ranking column header."""
    match = re.search(r"(ACC\s*\d{4}.*)$", column_name)
    if match:
        return match.group(1).strip()
    return column_name.strip()


def identify_ranking_columns(df: pd.DataFrame) -> list[str]:
    """Identify 2024 course ranking columns from the survey export."""
    ranking_columns: list[str] = []

    for col in df.columns:
        name = str(col)
        is_rank_question = "rank order" in name.lower()
        has_course_code = re.search(r"ACC\s*\d{4}", name) is not None
        if is_rank_question and has_course_code:
            ranking_columns.append(name)

    if not ranking_columns:
        raise ValueError("No course ranking columns were found. Check header detection logic.")

    return ranking_columns


def build_rank_ordering(df: pd.DataFrame, ranking_columns: list[str]) -> pd.DataFrame:
    """Create overall course ordering using mean rank (lower = more beneficial)."""
    rankings = df[ranking_columns].copy()

    # Convert responses to numeric; non-numeric values become NaN (e.g., metadata rows).
    rankings = rankings.apply(pd.to_numeric, errors="coerce")

    summary = pd.DataFrame(
        {
            "course": [parse_course_name(c) for c in ranking_columns],
            "mean_rank": rankings.mean(axis=0, skipna=True).values,
            "num_responses": rankings.count(axis=0).values,
        }
    )

    summary = summary.sort_values(by=["mean_rank", "course"], ascending=[True, True]).reset_index(drop=True)
    summary["overall_rank"] = range(1, len(summary) + 1)

    # Keep columns in a friendly final order.
    return summary[["overall_rank", "course", "mean_rank", "num_responses"]]


def plot_rank_order(rank_df: pd.DataFrame, output_path: Path) -> None:
    """Save a simple horizontal bar chart for average course rank."""
    plt.figure(figsize=(11, 6))

    # Reverse for plotting so best-ranked course appears at the top.
    plot_df = rank_df.sort_values("mean_rank", ascending=False)

    bars = plt.barh(plot_df["course"], plot_df["mean_rank"], color="#3a7ca5")
    plt.xlabel("Average rank (lower is better)")
    plt.ylabel("Course")
    plt.title("MAcc 2024 Core Course Ranking (Exit Survey)")
    plt.grid(axis="x", linestyle="--", alpha=0.35)

    # Add exact values at the end of bars.
    for bar, val in zip(bars, plot_df["mean_rank"]):
        plt.text(bar.get_width() + 0.03, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=9)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    header_row = detect_header_row(DATA_PATH)
    df = pd.read_excel(DATA_PATH, header=header_row)

    ranking_columns = identify_ranking_columns(df)
    rank_df = build_rank_ordering(df, ranking_columns)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rank_df.to_csv(OUTPUT_CSV, index=False)

    plot_rank_order(rank_df, OUTPUT_PNG)

    print(f"Detected header row: {header_row}")
    print(f"Ranking columns found: {len(ranking_columns)}")
    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved figure: {OUTPUT_PNG}")


if __name__ == "__main__":
    main()
