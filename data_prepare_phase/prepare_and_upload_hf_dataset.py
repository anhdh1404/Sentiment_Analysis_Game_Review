"""
Simple script to process aggregated reviews, split into train/val/test, and upload to HuggingFace.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datasets import Dataset, DatasetDict, load_dataset
from huggingface_hub import HfApi
from dotenv import load_dotenv

# Get the project root directory (parent of data_prepare_phase)
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file in project root
load_dotenv(PROJECT_ROOT / ".env")

# Configuration - paths relative to project root
DATA_DIR = PROJECT_ROOT / "data" / "aggregated_review_english"
HF_DATASET_NAME = os.getenv("HF_DATASET_NAME", "your-username/game-reviews-sentiment")
TRAIN_RATIO = float(os.getenv("TRAIN_RATIO", "0.8"))
VAL_RATIO = float(os.getenv("VAL_RATIO", "0.1"))
TEST_RATIO = float(os.getenv("TEST_RATIO", "0.1"))


def get_hf_token():
    """Return a usable HF token, treating placeholders as missing."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token or token.startswith("your_"):
        return None
    return token


def load_all_jsonl_files(data_dir):
    """Load all JSONL files from the directory."""
    all_data = []
    
    # Get all .jsonl files in the directory
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    
    print(f"Found {len(jsonl_files)} JSONL files")
    
    for file_path in jsonl_files:
        print(f"Loading {file_path.name}...")
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    data = json.loads(line)
                    all_data.append(data)
    
    print(f"Loaded {len(all_data)} total reviews")
    return all_data


def split_data(data, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Split data into train/validation/test sets."""
    import random
    
    # Shuffle data for random split
    random.seed(42)  # For reproducibility
    shuffled_data = data.copy()
    random.shuffle(shuffled_data)
    
    total = len(shuffled_data)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_data = shuffled_data[:train_size]
    val_data = shuffled_data[train_size:train_size + val_size]
    test_data = shuffled_data[train_size + val_size:]
    
    print(f"\nDataset split:")
    print(f"  Train: {len(train_data)} samples ({len(train_data)/total*100:.1f}%)")
    print(f"  Validation: {len(val_data)} samples ({len(val_data)/total*100:.1f}%)")
    print(f"  Test: {len(test_data)} samples ({len(test_data)/total*100:.1f}%)")
    
    return train_data, val_data, test_data


def create_huggingface_dataset(train_data, val_data, test_data):
    """Create HuggingFace DatasetDict."""
    # Convert to HuggingFace Dataset format
    train_dataset = Dataset.from_list(train_data)
    val_dataset = Dataset.from_list(val_data)
    test_dataset = Dataset.from_list(test_data)
    
    # Create DatasetDict
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    })
    
    return dataset_dict


def load_and_split_dataset(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Load JSONL files with HuggingFace datasets and split them memory-efficiently."""
    jsonl_files = sorted(data_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")

    print(f"Found {len(jsonl_files)} JSONL files")
    dataset = load_dataset(
        "json",
        data_files=[str(path) for path in jsonl_files],
        split="train",
    )
    print(f"Loaded {len(dataset)} total reviews")

    val_test_ratio = val_ratio + test_ratio
    first_split = dataset.train_test_split(
        test_size=val_test_ratio,
        seed=42,
        shuffle=True,
    )
    train_dataset = first_split["train"]
    remaining = first_split["test"]

    test_fraction = test_ratio / val_test_ratio if val_test_ratio else 0.5
    second_split = remaining.train_test_split(
        test_size=test_fraction,
        seed=42,
        shuffle=True,
    )
    val_dataset = second_split["train"]
    test_dataset = second_split["test"]

    print("\nDataset split:")
    for name, split in (("Train", train_dataset), ("Validation", val_dataset), ("Test", test_dataset)):
        print(f"  {name}: {len(split)} samples ({len(split) / len(dataset) * 100:.1f}%)")

    return DatasetDict({
        "train": train_dataset,
        "validation": val_dataset,
        "test": test_dataset,
    })


def upload_to_huggingface(dataset_dict, dataset_name, token=None):
    """Upload dataset to HuggingFace Hub."""
    if token is None:
        token = get_hf_token()
        if token is None:
            raise ValueError(
                "HuggingFace token not found. Please set HF_TOKEN environment variable "
                "or pass token as parameter."
            )
    
    print(f"\nUploading dataset to HuggingFace Hub: {dataset_name}")
    
    # Push to hub
    dataset_dict.push_to_hub(
        dataset_name,
        token=token,
        private=False  # Set to True if you want a private dataset
    )
    
    print(f"✓ Dataset successfully uploaded to: https://huggingface.co/datasets/{dataset_name}")


def main(local_dir=None):
    """Main function to process the dataset, then save locally or upload."""
    print("=" * 60)
    print("Processing Game Reviews for HuggingFace Dataset")
    print("=" * 60)
    
    # 1-3. Load, split, and create the HuggingFace dataset efficiently
    print("\n[1/4] Loading data...")
    dataset_dict = load_and_split_dataset(
        DATA_DIR,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
    )

    print("\nDataset structure:")
    print(dataset_dict)
    print("\nSample from train set:")
    print(dataset_dict['train'][0])

    # 4a. Save locally (no token required)
    if local_dir:
        local_path = Path(local_dir)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\n[4/4] Saving dataset locally to: {local_path}")
        dataset_dict.save_to_disk(str(local_path))
        print("✓ Local dataset saved.")
        print(f"  Set HF_DATASET_NAME={local_path} in .env or pass --dataset {local_path}")
        return dataset_dict
    
    # 4b. Upload to HuggingFace
    print("\n[4/4] Uploading to HuggingFace...")
    
    # Get token from environment or prompt user
    token = get_hf_token()
    if token is None:
        print("\nHF_TOKEN environment variable not found or still a placeholder.")
        print("Please set it in .env, or use --local_dir to skip uploading.")
        token = input("\nEnter your HuggingFace token (or press Enter to skip upload): ").strip()
        
        if not token:
            print("\nSkipping upload. Dataset prepared but not uploaded.")
            return dataset_dict
    
    upload_to_huggingface(dataset_dict, HF_DATASET_NAME, token)
    
    print("\n" + "=" * 60)
    print("Process completed successfully!")
    print("=" * 60)
    
    return dataset_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare the train/validation/test dataset and save it locally or upload to HuggingFace."
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default=None,
        help="Save the dataset locally to this directory instead of uploading (e.g. data/hf_dataset)",
    )
    args = parser.parse_args()

    # Check if .env is configured when uploading
    if args.local_dir is None and HF_DATASET_NAME == "your-username/game-reviews-sentiment":
        print("⚠️  WARNING: Please configure your .env file first!")
        print("   Copy .env.example to .env and update with your settings.\n")
    
    print(f"Dataset name: {HF_DATASET_NAME}")
    print(f"Split ratios: Train={TRAIN_RATIO}, Val={VAL_RATIO}, Test={TEST_RATIO}\n")
    
    dataset = main(local_dir=args.local_dir)
