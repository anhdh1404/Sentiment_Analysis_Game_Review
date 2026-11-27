
import json
import hashlib
import os
import shutil
import tempfile

from pathlib import Path
from collections import Counter

from langdetect import detect, LangDetectException, DetectorFactory
from tqdm import tqdm


# ============================================================
# 1. CONFIGURATION
# ============================================================

# Make language detection more reproducible
DetectorFactory.seed = 0

MAX_FILE_SIZE_MB = 95

# "text": deduplicate using review text only (old behavior)
# "game_text": deduplicate using game identity + review text
DEDUP_STRATEGY = "text"

# Keep the same output schema as the original pipeline
OUTPUT_FIELDS = [
    "review_text",
    "review_score",
    "review_category"
]


# ============================================================
# 2. TEXT PROCESSING
# ============================================================

def normalize_text(text):
    """
    Normalize review text for duplicate detection.

    Original review text is preserved in the output.
    Normalization is only used to identify duplicates.
    """
    return " ".join(text.strip().casefold().split())


def create_review_hash(review_text, game_identity=None):
    """
    Create a SHA-256 hash for duplicate detection.

    When game_identity is provided, the same text from
    different games is treated as different reviews.
    """
    normalized_text = normalize_text(review_text)

    if game_identity:
        normalized_text = (
            str(game_identity).strip().casefold()
            + "\x1f"
            + normalized_text
        )

    return hashlib.sha256(
        normalized_text.encode("utf-8")
    ).hexdigest()


def is_english(text):
    """
    Check whether a review is written in English.

    Returns:
        True  : English
        False : Non-English, empty, or undetectable
    """
    if not isinstance(text, str):
        return False

    text = text.strip()

    if len(text) < 3:
        return False

    try:
        return detect(text) == "en"

    except LangDetectException:
        return False


# ============================================================
# 3. DATA VALIDATION
# ============================================================

def validate_review(review):
    """
    Validate one raw review record.

    Returns:
        (is_valid, reason)

    This function does not enforce a specific score range
    or sentiment label mapping, because those depend on
    the original scraping dataset.
    """

    if not isinstance(review, dict):
        return False, "invalid_record_type"

    review_text = review.get("review_text")

    if not isinstance(review_text, str):
        return False, "invalid_review_text"

    if not review_text.strip():
        return False, "empty_review_text"

    return True, None


# ============================================================
# 4. OUTPUT FILE MANAGEMENT
# ============================================================

def get_output_filename(output_path, part_num):
    """
    Generate output filename.

    Part 1:
        aggregated_reviews_english.jsonl

    Part 2:
        aggregated_reviews_english_part2.jsonl
    """
    if part_num == 1:
        return output_path

    return output_path.parent / (
        f"{output_path.stem}_part{part_num}"
        f"{output_path.suffix}"
    )


def replace_output_files(staging_dir, output_path, output_files):
    """
    Replace old output files only after aggregation succeeds.

    This prevents incomplete processing from immediately
    overwriting the previous dataset.
    """

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    new_files = []

    for staged_file in output_files:
        final_file = output_path.parent / staged_file.name

        os.replace(staged_file, final_file)
        new_files.append(final_file)

    # Remove stale parts from previous runs
    expected_names = {
        file.name for file in new_files
    }

    pattern = f"{output_path.stem}_part*{output_path.suffix}"

    for old_file in output_path.parent.glob(pattern):
        if old_file.name not in expected_names:
            old_file.unlink()

    return new_files


# ============================================================
# 5. AGGREGATION
# ============================================================

def aggregate_reviews(
    input_dir,
    output_file,
    max_file_size_mb=MAX_FILE_SIZE_MB,
    dedup_strategy=DEDUP_STRATEGY
):
    """
    Aggregate raw JSONL review files.

    Processing:
        Raw JSONL
            -> Parse JSON
            -> Validate records
            -> English filtering
            -> Deduplication
            -> Extract required fields
            -> Split output JSONL files

    Args:
        input_dir: Directory containing raw JSONL files.
        output_file: First output JSONL filename.
        max_file_size_mb: Maximum size per output file.
        dedup_strategy: "text" or "game_text".

    Returns:
        List of generated output files.
    """

    input_path = Path(input_dir)
    output_path = Path(output_file)

    if max_file_size_mb <= 0:
        raise ValueError("max_file_size_mb must be greater than 0")

    if dedup_strategy not in {"text", "game_text"}:
        raise ValueError(
            "dedup_strategy must be 'text' or 'game_text'"
        )

    review_files = sorted(
        input_path.glob("review_data_part*.jsonl")
    )

    if not review_files:
        print(f"No review data files found in {input_path}")
        return []

    print("=" * 80)
    print("REVIEW DATA AGGREGATION")
    print("=" * 80)

    print(f"Input directory : {input_path}")
    print(f"Output file     : {output_path}")
    print(f"Input files     : {len(review_files)}")
    print(f"Max file size   : {max_file_size_mb} MB")
    print(f"Dedup strategy  : {dedup_strategy}")
    print()

    # Global statistics
    stats = Counter()

    # Store hashes instead of complete review texts
    seen_reviews = set()

    max_file_size_bytes = int(
        max_file_size_mb * 1024 * 1024
    )

    output_files = []

    current_part = 1
    current_size = 0
    current_output_file = None
    out_f = None

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Use a temporary directory so the previous output
    # remains intact if aggregation fails.
    with tempfile.TemporaryDirectory(
        prefix="aggregate_tmp_",
        dir=output_path.parent
    ) as temp_dir:

        staging_dir = Path(temp_dir)

        def open_new_output(part_num):
            nonlocal current_size, current_output_file, out_f

            if out_f is not None:
                out_f.close()

            current_output_file = get_output_filename(
                staging_dir / output_path.name,
                part_num
            )

            out_f = open(
                current_output_file,
                "w",
                encoding="utf-8",
                newline=""
            )

            output_files.append(current_output_file)
            current_size = 0

        open_new_output(current_part)

        try:
            for file_idx, file_path in enumerate(
                review_files, 1
            ):

                file_stats = Counter()

                print(
                    f"\n[{file_idx}/{len(review_files)}] "
                    f"Processing: {file_path.name}"
                )

                with open(
                    file_path,
                    "r",
                    encoding="utf-8"
                ) as in_f:

                    # Stream line by line instead of readlines()
                    for line in tqdm(
                        in_f,
                        desc="Filtering reviews",
                        unit="review",
                        leave=False
                    ):

                        if not line.strip():
                            continue

                        stats["total_input"] += 1
                        file_stats["total_input"] += 1

                        # ------------------------------------
                        # A. JSON parsing
                        # ------------------------------------

                        try:
                            review = json.loads(line)

                        except json.JSONDecodeError:
                            stats["invalid_json"] += 1
                            file_stats["invalid_json"] += 1
                            continue

                        # ------------------------------------
                        # B. Validation
                        # ------------------------------------

                        is_valid, reason = validate_review(
                            review
                        )

                        if not is_valid:
                            stats[reason] += 1
                            file_stats[reason] += 1
                            continue

                        review_text = review["review_text"]

                        # Track missing optional fields
                        if review.get("review_score") is None:
                            stats["missing_score"] += 1

                        if not review.get("review_category"):
                            stats["missing_category"] += 1

                        # ------------------------------------
                        # C. English filtering
                        # ------------------------------------

                        if not is_english(review_text):
                            stats["non_english_or_undetected"] += 1
                            file_stats["non_english_or_undetected"] += 1
                            continue

                        stats["english_before_dedup"] += 1
                        file_stats["english_before_dedup"] += 1

                        # ------------------------------------
                        # D. Deduplication
                        # ------------------------------------

                        game_identity = None

                        if dedup_strategy == "game_text":
                            game_identity = (
                                review.get("game_id")
                                or review.get("game_title")
                            )

                            if game_identity is None:
                                stats["missing_game_identity"] += 1

                        review_hash = create_review_hash(
                            review_text,
                            game_identity
                        )

                        if review_hash in seen_reviews:
                            stats["duplicates"] += 1
                            file_stats["duplicates"] += 1
                            continue

                        seen_reviews.add(review_hash)

                        # ------------------------------------
                        # E. Extract required fields
                        # ------------------------------------

                        filtered_review = {
                            "review_text": review_text,
                            "review_score": review.get(
                                "review_score"
                            ),
                            "review_category": review.get(
                                "review_category", ""
                            )
                        }

                        json_line = (
                            json.dumps(
                                filtered_review,
                                ensure_ascii=False
                            ) + "\n"
                        )

                        line_size = len(
                            json_line.encode("utf-8")
                        )

                        # ------------------------------------
                        # F. Split output files
                        # ------------------------------------

                        if (
                            current_size + line_size
                            > max_file_size_bytes
                            and current_size > 0
                        ):

                            current_part += 1

                            open_new_output(current_part)

                            print(
                                f"  -> New output: "
                                f"{current_output_file.name}"
                            )

                        out_f.write(json_line)

                        current_size += line_size

                        stats["unique_english"] += 1
                        file_stats["unique_english"] += 1

                # ----------------------------------------
                # G. Per-file statistics
                # ----------------------------------------

                file_total = file_stats["total_input"]

                english_percentage = (
                    file_stats["english_before_dedup"]
                    / file_total * 100
                    if file_total else 0
                )

                print(
                    f"  Input: {file_total:,} | "
                    f"English before dedup: "
                    f"{file_stats['english_before_dedup']:,} | "
                    f"Duplicates: {file_stats['duplicates']:,} | "
                    f"Unique output: "
                    f"{file_stats['unique_english']:,} | "
                    f"English rate: {english_percentage:.2f}%"
                )

        finally:
            if out_f is not None:
                out_f.close()

        # --------------------------------------------
        # H. Replace old output after successful run
        # --------------------------------------------

        final_files = replace_output_files(
            staging_dir,
            output_path,
            output_files
        )

    # ========================================================
    # 6. FINAL STATISTICS
    # ========================================================

    total_input = stats["total_input"]
    english_before_dedup = stats["english_before_dedup"]
    unique_english = stats["unique_english"]

    print("\n" + "=" * 80)
    print("AGGREGATION SUMMARY")
    print("=" * 80)

    print(f"Total input records       : {total_input:,}")
    print(f"English before dedup      : {english_before_dedup:,}")
    print(f"Duplicates removed        : {stats['duplicates']:,}")
    print(f"Unique English reviews    : {unique_english:,}")

    print("\nData quality:")
    print(f"Invalid JSON              : {stats['invalid_json']:,}")
    print(f"Invalid record type       : {stats['invalid_record_type']:,}")
    print(f"Invalid review text       : {stats['invalid_review_text']:,}")
    print(f"Empty review text         : {stats['empty_review_text']:,}")
    print(
        f"Non-English / undetected  : "
        f"{stats['non_english_or_undetected']:,}"
    )
    print(f"Missing score             : {stats['missing_score']:,}")
    print(f"Missing category          : {stats['missing_category']:,}")

    if dedup_strategy == "game_text":
        print(
            f"Missing game identity    : "
            f"{stats['missing_game_identity']:,}"
        )

    english_rate = (
        english_before_dedup / total_input * 100
        if total_input else 0
    )

    dedup_rate = (
        stats["duplicates"] / english_before_dedup * 100
        if english_before_dedup else 0
    )

    print("\nProcessing rates:")
    print(f"English rate              : {english_rate:.2f}%")
    print(f"Duplicate rate             : {dedup_rate:.2f}%")

    print(f"\nOutput files              : {len(final_files)}")

    for idx, file in enumerate(final_files, 1):
        file_size = file.stat().st_size / (1024 * 1024)

        print(
            f"  [{idx}] {file.name} "
            f"- {file_size:.2f} MB"
        )

    print("=" * 80)

    return final_files


# ============================================================
# 7. DATASET ANALYSIS
# ============================================================

def analyze_dataset(dataset_files):
    """
    Analyze the aggregated JSONL dataset.

    Uses streaming to avoid loading all records into memory.
    """

    if isinstance(dataset_files, (str, Path)):
        dataset_files = [dataset_files]

    existing_files = [
        Path(file)
        for file in dataset_files
        if Path(file).exists()
    ]

    if not existing_files:
        print("No dataset files found.")
        return

    total_count = 0

    category_counts = Counter()
    score_distribution = Counter()

    total_text_length = 0
    min_text_length = None
    max_text_length = 0

    invalid_lines = 0

    print("\n" + "=" * 80)
    print("DATASET ANALYSIS")
    print("=" * 80)

    for dataset_file in existing_files:

        print(f"Analyzing: {dataset_file.name}")

        with open(
            dataset_file,
            "r",
            encoding="utf-8"
        ) as f:

            for line in tqdm(
                f,
                desc="Reading dataset",
                unit="review",
                leave=False
            ):

                if not line.strip():
                    continue

                try:
                    review = json.loads(line)

                except json.JSONDecodeError:
                    invalid_lines += 1
                    continue

                total_count += 1

                category = review.get(
                    "review_category",
                    "unknown"
                )

                category_counts[str(category)] += 1

                score = review.get("review_score")

                if score is not None:
                    score_distribution[str(score)] += 1

                text = review.get("review_text", "")

                if isinstance(text, str):
                    text_length = len(text)

                    total_text_length += text_length

                    if min_text_length is None:
                        min_text_length = text_length

                    min_text_length = min(
                        min_text_length,
                        text_length
                    )

                    max_text_length = max(
                        max_text_length,
                        text_length
                    )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print("\nDataset summary:")
    print(f"Total data points : {total_count:,}")
    print(f"Invalid JSON lines: {invalid_lines:,}")

    if total_count > 0:
        print("\nReview text statistics:")

        print(
            f"Average length: "
            f"{total_text_length / total_count:.1f} characters"
        )

        print(f"Minimum length: {min_text_length:,}")
        print(f"Maximum length: {max_text_length:,}")

    print("\nCategory distribution:")

    for category, count in category_counts.most_common():

        percentage = (
            count / total_count * 100
            if total_count else 0
        )

        print(
            f"  {category:15} : "
            f"{count:>10,} ({percentage:5.2f}%)"
        )

    print("\nScore distribution:")

    for score, count in sorted(
        score_distribution.items()
    ):

        percentage = (
            count / total_count * 100
            if total_count else 0
        )

        print(
            f"  Score {score:10} : "
            f"{count:>10,} ({percentage:5.2f}%)"
        )

    total_file_size = sum(
        file.stat().st_size
        for file in existing_files
    )

    print("\nFile statistics:")

    print(
        f"Total size: "
        f"{total_file_size / (1024 * 1024):.2f} MB"
    )

    print(f"Number of files: {len(existing_files)}")

    for file in existing_files:
        size_mb = file.stat().st_size / (1024 * 1024)

        print(f"  {file.name}: {size_mb:.2f} MB")

    print("=" * 80)


# ============================================================
# 8. MAIN
# ============================================================

def main():

    project_root = Path(__file__).resolve().parent.parent

    input_dir = (
        project_root
        / "data"
        / "review_data"
    )

    output_file = (
        project_root
        / "data"
        / "aggregated_review_english"
        / "aggregated_reviews_english.jsonl"
    )

    print("Starting review aggregation...")
    print(f"Input directory: {input_dir}")
    print(f"Output file: {output_file}")

    result_files = aggregate_reviews(
        input_dir=input_dir,
        output_file=output_file,
        max_file_size_mb=MAX_FILE_SIZE_MB,
        dedup_strategy=DEDUP_STRATEGY
    )

    if result_files:
        analyze_dataset(result_files)


if __name__ == "__main__":
    main()
