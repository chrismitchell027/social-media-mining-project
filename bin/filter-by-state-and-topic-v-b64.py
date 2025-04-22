import os
import sys
import pandas as pd

def check_input_file(input_path):
    """Ensure the input CSV exists."""
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"Input file '{input_path}' not found.")

def prepare_output_dir(output_dir):
    """Create the output directory if it doesn't exist."""
    if os.path.isdir(output_dir):
        print(f"Output directory '{output_dir}' already exists.")
    else:
        try:
            os.makedirs(output_dir)
            print(f"Created output directory '{output_dir}'.")
        except OSError as e:
            raise OSError(f"Could not create directory '{output_dir}': {e}")

def load_data(input_path):
    """Load the CSV into a DataFrame."""
    try:
        return pd.read_csv(input_path)
    except Exception as e:
        raise IOError(f"Error reading '{input_path}': {e}")

def filter_us_tweets(df, us_variants):
    """Filter DataFrame for rows where 'country' matches US variants."""
    us_df = df[df['country'].isin(us_variants)].copy()
    if us_df.empty:
        print("Warning: No rows found for United States variants.")
    return us_df

def write_label_state_csvs(us_df, political_labels, output_dir):
    """
    For each political label and state code, write a separate CSV.
    Returns a nested dict of DataFrames and the total file count.
    """
    nested_dfs = {}
    file_count = 0

    for label in political_labels:
        label_df = us_df[us_df['political_label'] == label].copy()
        if label_df.empty:
            continue

        label_df['state_key'] = label_df['state_code'].fillna('unknown')
        nested_dfs[label] = {}

        for state, group in label_df.groupby('state_key'):
            group = group.drop(columns='state_key')
            nested_dfs[label][state] = group

            safe_label = label.replace(" ", "_")
            safe_state = state
            filename = f"{safe_label}_{safe_state}.csv"
            out_path = os.path.join(output_dir, filename)

            try:
                group.to_csv(out_path, index=False)
                file_count += 1
            except Exception as e:
                print(f"Error writing '{out_path}': {e}")

    return nested_dfs, file_count

def main():
    input_path = "data/cleaned_tweets_with_labels_v_b64.csv"
    output_dir = "data/us_tweets_by_label_state/v_b64_nf"
    us_variants = {'United States of America', 'United States'}
    political_labels = {
        'Non-Political', 'Voting Rights', 'Taxes', 'Free Speech', 'Healthcare',
        'Energy Policy', 'Foreign Policy', 'Criminal Justice', 'Police Reform',
        'Inflation', 'Abortion', 'Immigration', 'Welfare', 'Climate Change',
        'Gun Control', 'Education', 'Unemployment', 'Social Security',
        'LGBTQ Rights', 'Minimum Wage', 'Student Loans'
    }

    try:
        check_input_file(input_path)
        prepare_output_dir(output_dir)

        df = load_data(input_path)
        us_df = filter_us_tweets(df, us_variants)
        if us_df.empty:
            sys.exit(0)

        _, file_count = write_label_state_csvs(us_df, political_labels, output_dir)
        print(f"Finished writing {file_count} CSV file(s) to '{output_dir}/'.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
