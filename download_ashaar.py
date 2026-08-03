"""Download arbml/Ashaar_dataset and save it as a Parquet file."""

from argparse import ArgumentParser
from pathlib import Path

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset


DATASET_ID = "arbml/Ashaar_dataset"


def download_ashaar(output: Path) -> None:
    dataset = load_dataset(DATASET_ID)

    if isinstance(dataset, DatasetDict):
        splits = [
            split.add_column("split", [name] * len(split))
            for name, split in dataset.items()
        ]
        dataset = concatenate_datasets(splits) if len(splits) > 1 else splits[0]

    if not isinstance(dataset, Dataset):
        raise TypeError(f"Expected a Dataset, got {type(dataset).__name__}")

    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(str(output))
    print(f"Saved {len(dataset):,} rows to {output}")


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/ashaar_dataset.parquet"),
        help="Destination file (default: data/ashaar_dataset.parquet)",
    )
    args = parser.parse_args()
    download_ashaar(args.output)


if __name__ == "__main__":
    main()
