import os
import sys

import torch
import pandas as pd
from transformers import pipeline
from datasets import load_dataset

def check_input_file(input_path):
    """Ensure the input CSV exists."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' not found.")


def init_classifier(model_name='facebook/bart-large-mnli', device=0):
    """Initialize the zero-shot classification pipeline on GPU."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available: ensure you have a GPU-enabled PyTorch build and proper drivers.")
    try:
        return pipeline(
            "zero-shot-classification",
            model=model_name,
            device=device,
        )
    except Exception as e:
        raise RuntimeError(f"Error initializing classifier on GPU: {e}")


def load_dataset_from_csv(input_path, column_name='clean_tweet'):
    """Load a Hugging Face dataset from a CSV file and check for required column."""
    try:
        ds = load_dataset('csv', data_files={'data': input_path}, split='data')
    except Exception as e:
        raise IOError(f"Error loading dataset from '{input_path}': {e}")

    if column_name not in ds.column_names:
        raise KeyError(f"Required column '{column_name}' not found in dataset.")
    return ds


def classify_dataset(ds, classifier, candidate_labels, batch_size=32):
    """Classify tweets in batches using the zero-shot pipeline."""
    def classify_batch(batch):
        texts = batch['clean_tweet']
        try:
            results = classifier(texts, candidate_labels, batch_size=batch_size)
        except Exception as e:
            raise RuntimeError(f"Batch classification failed: {e}")
        batch['predicted_label'] = [r['labels'][0] if r.get('labels') else None for r in results]
        batch['predicted_score'] = [r['scores'][0] if r.get('scores') else None for r in results]
        return batch

    # Perform batched mapping
    return ds.map(
        classify_batch,
        batched=True,
        batch_size=batch_size,
        remove_columns=[]  # keep all original columns
    )


def save_dataset_to_csv(ds, input_path):
    """Save the classified dataset to a new CSV file."""
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_classified{ext}"
    try:
        ds.to_csv(output_path, index=False)
        print(f"Results saved to '{output_path}'.")
    except Exception as e:
        raise IOError(f"Error saving dataset to '{output_path}': {e}")


def main():
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <input_csv>")
        sys.exit(1)
    input_path = sys.argv[1]

    # Candidate labels for zero-shot classification
    candidate_labels_dict = {
        'Voting Rights': ["expand voting rights", "restrict voting rights", "neutral"],
        'Taxes': ["support higher taxes", "support tax cuts", "neutral"],
        'Free Speech': ["support free speech protections", "support speech restrictions", "neutral"],
        'Healthcare': ["support universal healthcare", "oppose universal healthcare", "neutral"],
        'Energy Policy': ["favor renewable energy", "favor fossil fuels", "neutral"],
        'Foreign Policy': ["favor interventionism", "favor isolationism", "neutral"],
        'Criminal Justice': ["support criminal justice reform", "oppose criminal justice reform", "neutral"],
        'Police Reform': ["support police reform", "oppose police reform", "neutral"],
        'Inflation': ["support government action to curb inflation", "oppose government intervention on inflation", "neutral"],
        'Abortion': ["pro-choice", "pro-life", "neutral"],
        'Immigration': ["support immigration reform", "support stricter immigration controls", "neutral"],
        'Welfare': ["support welfare expansion", "oppose welfare expansion", "neutral"],
        'Climate Change': ["support climate action", "oppose climate action", "neutral"],
        'Gun Control': ["support gun control", "support gun rights", "neutral"],
        'Education': ["support increased education funding", "oppose increased education funding", "neutral"],
        'Unemployment': ["support expanded unemployment benefits", "oppose expanded unemployment benefits", "neutral"],
        'Social Security': ["support expanding Social Security", "oppose expanding Social Security", "neutral"],
        'LGBTQ Rights': ["support LGBTQ rights", "oppose LGBTQ rights", "neutral"],
        'Minimum Wage': ["support raising minimum wage", "oppose raising minimum wage", "neutral"],
        'Student Loans': ["support student loan forgiveness", "oppose student loan forgiveness", "neutral"],
    }

    candidate_labels = candidate_labels_dict["Energy Policy"]
    batch_size = 16
    try:
        # Check that the input file is valid
        check_input_file(input_path)
        # Load the dataset
        ds = load_dataset_from_csv(input_path)
        # Initialize the classifier
        classifier = init_classifier()

        # Classify
        ds_classified = classify_dataset(ds, classifier, candidate_labels, batch_size=batch_size)
        # Save
        save_dataset_to_csv(ds_classified, input_path)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
