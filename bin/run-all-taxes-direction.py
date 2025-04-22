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


def find_taxes_csvs(input_dir):
    """
    Find all 'Taxes' CSV files (Topic_State.csv) in the directory,
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

        # Use os.path.splitext to separate filename and extension
        root, _ = os.path.splitext(fname)
        # Expect format 'Taxes_STATE'
        if not root.startswith('Taxes_'):
            continue

        parts = root.split('_')
        if len(parts) != 2:
            # More or fewer parts than expected
            continue
        _, state = parts
        # Skip unknown states
        if state.lower() == 'unknown':
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

    try:
        check_script(script_path)
        check_input_dir(input_dir)

        csv_files = find_taxes_csvs(input_dir)
        if not csv_files:
            print("No taxes CSV files found to process.")
            sys.exit(0)

        for csv_file in csv_files:
            run_classification(script_path, csv_file)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
