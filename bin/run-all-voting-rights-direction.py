import os
import sys
import subprocess


def check_script(script_path):
    """Ensure the classification script exists."""
    if not os.path.isfile(script_path):
        raise FileNotFoundError(f"Classifier script '{script_path}' not found.")


def check_input_dir(input_dir):
    """Ensure the input directory exists and is a directory."""
    if not os.path.isdir(input_dir):
        raise NotADirectoryError(f"Input directory '{input_dir}' not found or not a directory.")


def find_csvs(input_dir, prefix):
    """
    Find all CSV files with 'prefix_STATEID' in the directory,
    excluding already classified files and those with state 'unknown'.
    """
    files = []
    for fname in os.listdir(input_dir):
        # Only consider CSV files
        if not fname.endswith('.csv'):
            continue
        # Skip files already classified
        if fname.endswith('_classified.csv'):
            continue

        # Split filename and extension
        root, _ = os.path.splitext(fname)
        # Must start with the correct prefix
        if not root.startswith(prefix):
            continue

        # Extract the state code after prefix
        state = root[len(prefix):]
        # Skip if missing or 'unknown'
        if not state or state.lower() == 'unknown':
            continue

        files.append(os.path.join(input_dir, fname))

    return sorted(files)


def run_classification(script_path, csv_path):
    """Invoke the classification script on a single CSV file."""
    try:
        result = subprocess.run(
            [sys.executable, script_path, csv_path],
            capture_output=True,
            text=True,
            check=False
        )
        if result.returncode != 0:
            print(f"Error processing '{csv_path}':\n{result.stderr}")
        else:
            print(f"Finished processing '{csv_path}'.")
    except Exception as e:
        print(f"Exception running classification on '{csv_path}': {e}")


def main():
    if len(sys.argv) != 3:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <classification_script> <input_directory>")
        sys.exit(1)

    script_path = sys.argv[1]
    input_dir = sys.argv[2]
    prefix = "Voting_Rights_"
    try:
        check_script(script_path)
        check_input_dir(input_dir)

        csv_files = find_csvs(input_dir, prefix)
        if not csv_files:
            print(f"No {prefix} CSV files found to process.")
            sys.exit(0)

        for csv_file in csv_files:
            run_classification(script_path, csv_file)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
