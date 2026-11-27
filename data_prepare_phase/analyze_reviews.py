
import json
import argparse
import random
import re
import math

from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib.pyplot as plt


# ============================================================
# 1. CONFIGURATION
# ============================================================

RANDOM_SEED = 42
SAMPLE_SIZE = 100_000
TOP_N = 10

REQUIRED_FIELDS = [
    "review_text",
    "review_score",
    "review_category",
    "game_title",
    "game_platform",
    "game_genre",
    "reviewer_name",
    "review_date",
    "game_developer",
    "game_publisher"
]

plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3


# ============================================================
# 2. HELPER FUNCTIONS
# ============================================================

def natural_sort_key(path):
    """Sort review_data_part2 before review_data_part10."""
    match = re.search(r"part(\d+)", path.stem, re.IGNORECASE)
    return int(match.group(1)) if match else 0


def safe_string(value):
    """Convert values into safe strings for statistics."""
    if value is None:
        return "Unknown"

    value = str(value).strip()
    return value if value else "Unknown"


def parse_score(value):
    """Convert review score into a numeric value."""
    try:
        score = float(value)
        if not math.isfinite(score):
            return None
        return score
    except (TypeError, ValueError):
        return None


def parse_review_date(value):
    """Parse review dates into datetime objects."""
    if not value:
        return None

    value = str(value).strip()

    formats = [
        "%b %d, %Y",
        "%B %d, %Y",
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S"
    ]

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def add_reservoir_sample(samples, counts, key, value, capacity, rng):
    """
    Reservoir sampling:
    Keep a representative random sample without storing every value.
    """
    counts[key] += 1

    if len(samples[key]) < capacity:
        samples[key].append(value)
    else:
        index = rng.randrange(counts[key])
        if index < capacity:
            samples[key][index] = value


def get_median_from_counter(counter):
    """Calculate exact median from a numeric frequency counter."""
    total = sum(counter.values())

    if total == 0:
        return None

    positions = (
        (total - 1) // 2,
        total // 2
    )

    found = []
    cumulative = 0

    for value in sorted(counter):
        cumulative += counter[value]

        while len(found) < 2 and positions[len(found)] < cumulative:
            found.append(value)

    return sum(found) / len(found)


def count_to_percentage(count, total):
    return count / total * 100 if total else 0


# ============================================================
# 3. STREAMING DATA ANALYSIS
# ============================================================

def analyze_files(input_dir, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED):

    input_dir = Path(input_dir)

    files = sorted(
        input_dir.glob("review_data_part*.jsonl"),
        key=natural_sort_key
    )

    if not files:
        raise FileNotFoundError(
            f"No review_data_part*.jsonl files found in {input_dir}"
        )

    rng = random.Random(seed)

    # General counters
    stats = Counter()
    missing_fields = Counter()

    categories = Counter()
    scores = Counter()
    games = Counter()
    platforms = Counter()
    genres = Counter()
    reviewers = Counter()
    developers = Counter()
    publishers = Counter()
    monthly_reviews = Counter()

    # Duplicate detection
    seen_review_ids = set()

    # Review score statistics
    score_sum = 0
    score_squared_sum = 0
    score_count = 0

    # Review text statistics
    text_length_sum = 0
    word_count_sum = 0
    text_count = 0
    min_text_length = None
    max_text_length = 0

    # Reservoir samples for plots and examples
    length_samples = defaultdict(list)
    length_sample_counts = Counter()

    category_score_samples = defaultdict(list)
    category_score_counts = Counter()

    random_reviews = []
    random_review_count = 0

    # ========================================================
    # Process each JSONL file
    # ========================================================

    print("=" * 75)
    print("REVIEW DATA ANALYSIS")
    print("=" * 75)
    print(f"Input directory : {input_dir}")
    print(f"Input files     : {len(files)}")
    print()

    for file_index, file_path in enumerate(files, 1):

        print(
            f"[{file_index}/{len(files)}] "
            f"Processing: {file_path.name}"
        )

        file_records = 0

        with file_path.open("r", encoding="utf-8") as f:

            for line in f:

                if not line.strip():
                    stats["empty_lines"] += 1
                    continue

                stats["total_lines"] += 1

                try:
                    review = json.loads(line)
                except json.JSONDecodeError:
                    stats["invalid_json"] += 1
                    continue

                if not isinstance(review, dict):
                    stats["invalid_records"] += 1
                    continue

                stats["valid_records"] += 1
                file_records += 1

                # --------------------------------------------
                # Missing values
                # --------------------------------------------

                for field in REQUIRED_FIELDS:
                    value = review.get(field)

                    if value is None or (
                        isinstance(value, str) and not value.strip()
                    ):
                        missing_fields[field] += 1

                # --------------------------------------------
                # Duplicate review IDs
                # --------------------------------------------

                review_id = review.get("review_id")

                if review_id:
                    review_id = str(review_id)

                    if review_id in seen_review_ids:
                        stats["duplicate_review_ids"] += 1
                    else:
                        seen_review_ids.add(review_id)

                # --------------------------------------------
                # Category
                # --------------------------------------------

                category = safe_string(
                    review.get("review_category")
                ).lower()

                categories[category] += 1

                # --------------------------------------------
                # Score
                # --------------------------------------------

                score = parse_score(review.get("review_score"))

                if score is not None:
                    scores[score] += 1
                    score_sum += score
                    score_squared_sum += score ** 2
                    score_count += 1

                    add_reservoir_sample(
                        category_score_samples,
                        category_score_counts,
                        category,
                        score,
                        sample_size,
                        rng
                    )
                else:
                    stats["invalid_scores"] += 1

                # --------------------------------------------
                # Game / Platform / Genre
                # --------------------------------------------

                games[safe_string(review.get("game_title"))] += 1

                platforms[
                    safe_string(review.get("game_platform"))
                ] += 1

                genres[
                    safe_string(review.get("game_genre"))
                ] += 1

                reviewers[
                    safe_string(review.get("reviewer_name"))
                ] += 1

                developers[
                    safe_string(review.get("game_developer"))
                ] += 1

                publishers[
                    safe_string(review.get("game_publisher"))
                ] += 1

                # --------------------------------------------
                # Review text
                # --------------------------------------------

                review_text = review.get("review_text")

                if isinstance(review_text, str) and review_text.strip():

                    text_length = len(review_text)
                    word_count = len(review_text.split())

                    text_length_sum += text_length
                    word_count_sum += word_count
                    text_count += 1

                    min_text_length = (
                        text_length if min_text_length is None
                        else min(min_text_length, text_length)
                    )

                    max_text_length = max(
                        max_text_length,
                        text_length
                    )

                    add_reservoir_sample(
                        length_samples,
                        length_sample_counts,
                        "length",
                        word_count,
                        sample_size,
                        rng
                    )

                else:
                    stats["empty_review_text"] += 1

                # --------------------------------------------
                # Review date
                # --------------------------------------------

                review_date = parse_review_date(
                    review.get("review_date")
                )

                if review_date:
                    month = review_date.strftime("%Y-%m")
                    monthly_reviews[month] += 1
                else:
                    stats["invalid_dates"] += 1

                # --------------------------------------------
                # Random review samples
                # --------------------------------------------

                random_review_count += 1

                if len(random_reviews) < 5:
                    random_reviews.append(review)
                else:
                    index = rng.randrange(random_review_count)

                    if index < 5:
                        random_reviews[index] = review

        print(f"  Valid records: {file_records:,}")

    # ========================================================
    # 4. RETURN ANALYSIS RESULTS
    # ========================================================

    return {
        "stats": stats,
        "missing_fields": missing_fields,
        "categories": categories,
        "scores": scores,
        "games": games,
        "platforms": platforms,
        "genres": genres,
        "reviewers": reviewers,
        "developers": developers,
        "publishers": publishers,
        "monthly_reviews": monthly_reviews,
        "score_sum": score_sum,
        "score_squared_sum": score_squared_sum,
        "score_count": score_count,
        "text_length_sum": text_length_sum,
        "word_count_sum": word_count_sum,
        "text_count": text_count,
        "min_text_length": min_text_length,
        "max_text_length": max_text_length,
        "length_samples": length_samples,
        "category_score_samples": category_score_samples,
        "random_reviews": random_reviews
    }


# ============================================================
# 5. PRINT SUMMARY
# ============================================================

def print_summary(data):

    stats = data["stats"]
    total = stats["valid_records"]

    print("\n" + "=" * 75)
    print("REVIEW DATA ANALYSIS REPORT")
    print("=" * 75)

    print("\n[1] BASIC STATISTICS")
    print("-" * 75)

    print(f"Total lines          : {stats['total_lines']:,}")
    print(f"Valid records        : {total:,}")
    print(f"Invalid JSON         : {stats['invalid_json']:,}")
    print(f"Invalid records      : {stats['invalid_records']:,}")
    print(f"Empty lines          : {stats['empty_lines']:,}")
    print(f"Duplicate review IDs : {stats['duplicate_review_ids']:,}")

    print("\n[2] MISSING VALUES")
    print("-" * 75)

    for field, count in data["missing_fields"].most_common():
        print(
            f"{field:<25}: {count:,} "
            f"({count_to_percentage(count, total):.2f}%)"
        )

    print("\n[3] REVIEW CATEGORIES")
    print("-" * 75)

    for category, count in data["categories"].most_common():
        print(
            f"{category:<15}: {count:,} "
            f"({count_to_percentage(count, total):.2f}%)"
        )

    print("\n[4] SCORE STATISTICS")
    print("-" * 75)

    n = data["score_count"]

    if n:
        mean = data["score_sum"] / n
        variance = max(
            0,
            data["score_squared_sum"] / n - mean ** 2
        )

        print(f"Valid scores : {n:,}")
        print(f"Mean         : {mean:.2f}")
        print(f"Median       : {get_median_from_counter(data['scores']):.2f}")
        print(f"Min          : {min(data['scores'])}")
        print(f"Max          : {max(data['scores'])}")
        print(f"Std (pop.)   : {math.sqrt(variance):.2f}")

    else:
        print("No valid scores available.")

    print("\n[5] TOP GAMES")
    print("-" * 75)

    for i, (name, count) in enumerate(data["games"].most_common(TOP_N), 1):
        print(f"{i:2}. {name[:55]:<55} {count:,}")

    print("\n[6] PLATFORM DISTRIBUTION")
    print("-" * 75)

    for name, count in data["platforms"].most_common(TOP_N):
        print(f"{name[:40]:<40} {count:,}")

    print("\n[7] TOP GENRES")
    print("-" * 75)

    for name, count in data["genres"].most_common(TOP_N):
        print(f"{name[:40]:<40} {count:,}")

    print("\n[8] REVIEW TEXT STATISTICS")
    print("-" * 75)

    text_count = data["text_count"]

    if text_count:
        print(f"Valid review texts : {text_count:,}")
        print(
            f"Average characters: "
            f"{data['text_length_sum'] / text_count:.2f}"
        )
        print(
            f"Average words     : "
            f"{data['word_count_sum'] / text_count:.2f}"
        )
        print(f"Shortest review   : {data['min_text_length']}")
        print(f"Longest review    : {data['max_text_length']}")

    print("\n[9] TOP REVIEWERS")
    print("-" * 75)

    for name, count in data["reviewers"].most_common(TOP_N):
        print(f"{name[:35]:<35} {count:,}")

    print("\n[10] DEVELOPER / PUBLISHER")
    print("-" * 75)

    print("Top developers:")
    for name, count in data["developers"].most_common(5):
        print(f"  {name[:40]:<40} {count:,}")

    print("\nTop publishers:")
    for name, count in data["publishers"].most_common(5):
        print(f"  {name[:40]:<40} {count:,}")

    print("\n[11] DATE STATISTICS")
    print("-" * 75)

    dates = data["monthly_reviews"]

    if dates:
        print(f"Date range: {min(dates)} to {max(dates)}")
        print("Top 10 months:")

        for month, count in dates.most_common(10):
            print(f"  {month}: {count:,}")

    print("\n[12] ADDITIONAL INSIGHTS")
    print("-" * 75)

    print(f"Perfect scores (100): {data['scores'].get(100, 0):,}")
    print(f"Zero scores (0)     : {data['scores'].get(0, 0):,}")

    print("\n" + "=" * 75)


# ============================================================
# 6. RANDOM REVIEW PREVIEWS
# ============================================================

def show_random_reviews(reviews):

    print("\n" + "=" * 75)
    print("RANDOM REVIEW PREVIEWS")
    print("=" * 75)

    for i, review in enumerate(reviews, 1):

        print(f"\nReview #{i}")
        print("-" * 75)

        print(f"Game     : {review.get('game_title', 'Unknown')}")
        print(f"Platform : {review.get('game_platform', 'Unknown')}")
        print(f"Score    : {review.get('review_score', 'Unknown')}")
        print(f"Category : {review.get('review_category', 'Unknown')}")
        print(f"Reviewer : {review.get('reviewer_name', 'Unknown')}")
        print(f"Date     : {review.get('review_date', 'Unknown')}")

        text = review.get("review_text", "")
        print(f"Review   : {text[:250]}")


# ============================================================
# 7. VISUALIZATION
# ============================================================

def save_plots(data, output_dir):

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plots = []

    def save_plot(filename):
        path = output_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        plots.append(path)
        print(f"Saved: {path}")

    # --------------------------------------------------------
    # 1. Category distribution
    # --------------------------------------------------------

    categories = data["categories"]

    if categories:
        plt.figure()
        names = list(categories.keys())
        values = [categories[name] for name in names]

        plt.bar(names, values)
        plt.title("Review Category Distribution")
        plt.xlabel("Category")
        plt.ylabel("Number of Reviews")
        save_plot("category_distribution.png")

    # --------------------------------------------------------
    # 2. Score distribution
    # --------------------------------------------------------

    scores = data["scores"]

    if scores:
        plt.figure()

        score_values = sorted(scores)
        score_counts = [scores[x] for x in score_values]

        plt.bar(score_values, score_counts, width=0.8)
        plt.title("Review Score Distribution")
        plt.xlabel("Review Score")
        plt.ylabel("Number of Reviews")
        save_plot("score_distribution.png")

    # --------------------------------------------------------
    # 3. Top games
    # --------------------------------------------------------

    top_games = data["games"].most_common(TOP_N)

    if top_games:
        plt.figure()

        names = [x[0] for x in top_games][::-1]
        values = [x[1] for x in top_games][::-1]

        plt.barh(names, values)
        plt.title("Top 10 Games by Review Count")
        plt.xlabel("Number of Reviews")
        save_plot("top_games.png")

    # --------------------------------------------------------
    # 4. Platform distribution
    # --------------------------------------------------------

    top_platforms = data["platforms"].most_common(TOP_N)

    if top_platforms:
        plt.figure()

        names = [x[0] for x in top_platforms][::-1]
        values = [x[1] for x in top_platforms][::-1]

        plt.barh(names, values)
        plt.title("Top Platforms by Review Count")
        plt.xlabel("Number of Reviews")
        save_plot("platform_distribution.png")

    # --------------------------------------------------------
    # 5. Genre distribution
    # --------------------------------------------------------

    top_genres = data["genres"].most_common(TOP_N)

    if top_genres:
        plt.figure()

        names = [x[0] for x in top_genres][::-1]
        values = [x[1] for x in top_genres][::-1]

        plt.barh(names, values)
        plt.title("Top Genres by Review Count")
        plt.xlabel("Number of Reviews")
        save_plot("genre_distribution.png")

    # --------------------------------------------------------
    # 6. Review timeline
    # --------------------------------------------------------

    monthly = data["monthly_reviews"]

    if monthly:
        plt.figure()

        dates = sorted(monthly)
        values = [monthly[x] for x in dates]

        plt.plot(dates, values, marker="o")
        plt.title("Review Volume Over Time")
        plt.xlabel("Month")
        plt.ylabel("Number of Reviews")
        plt.xticks(rotation=45)
        save_plot("review_timeline.png")

    # --------------------------------------------------------
    # 7. Review length distribution
    # --------------------------------------------------------

    lengths = data["length_samples"].get("length", [])

    if lengths:
        plt.figure()

        plt.hist(lengths, bins=50)
        plt.title("Review Word Count Distribution")
        plt.xlabel("Number of Words")
        plt.ylabel("Frequency")
        save_plot("review_length_distribution.png")

    # --------------------------------------------------------
    # 8. Score by category
    # --------------------------------------------------------

    category_scores = data["category_score_samples"]

    valid_categories = [
        category
        for category, values in category_scores.items()
        if values
    ]

    if valid_categories:
        plt.figure()

        values = [
            category_scores[category]
            for category in valid_categories
        ]

        plt.boxplot(values, tick_labels=valid_categories)
        plt.title("Review Score by Category")
        plt.xlabel("Review Category")
        plt.ylabel("Review Score")
        save_plot("score_by_category.png")

    print(f"\nTotal plots generated: {len(plots)}")

    return plots


# ============================================================
# 8. EXPORT SUMMARY
# ============================================================

def export_summary(data, output_file):

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Capture console report into a text file.
    import contextlib

    with output_file.open("w", encoding="utf-8") as f:
        with contextlib.redirect_stdout(f):
            print_summary(data)

    print(f"\nSummary exported: {output_file}")


# ============================================================
# 9. MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Analyze game reviews from multiple JSONL files."
    )

    parser.add_argument(
        "--input-dir",
        default="./data/review_data",
        help="Directory containing review_data_part*.jsonl"
    )

    parser.add_argument(
        "--output-dir",
        default="./data/analysis_results",
        help="Directory for analysis outputs"
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=SAMPLE_SIZE,
        help="Maximum reservoir sample size for plots"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED
    )

    args = parser.parse_args()

    if args.sample_size <= 0:
        parser.error("--sample-size must be greater than zero")

    output_dir = Path(args.output_dir)
    plots_dir = output_dir / "plots"

    print("Starting review analysis...")

    data = analyze_files(
        input_dir=args.input_dir,
        sample_size=args.sample_size,
        seed=args.seed
    )

    print_summary(data)

    show_random_reviews(data["random_reviews"])

    export_summary(
        data,
        output_dir / "analysis_summary.txt"
    )

    save_plots(data, plots_dir)

    print("\n" + "=" * 75)
    print("ANALYSIS COMPLETED")
    print("=" * 75)


if __name__ == "__main__":
    main()
