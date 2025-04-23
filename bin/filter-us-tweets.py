import pandas as pd

def filter_us_tweets(df, us_variants):
    '''Filter DataFrame for rows where 'country' matches US variants.'''
    us_df = df[df['country'].isin(us_variants)].copy()
    if us_df.empty:
        print('Warning: No rows found for United States variants.')
    return us_df

def main():
    # Specify the path of the input data
    input_path = 'data/cleaned_tweets_with_labels_v_b64.csv'
    # Specify the path where the output data should go
    output_path = 'data/us_big_cleaned_tweets_with_labels_v_b64.csv'
    # These are the ways the US is identified in the tweet locations
    us_variants = {'United States of America', 'United States'}
    # Load the input data
    df = pd.read_csv(input_path)

    # Filter out tweets not in the US
    us_df = filter_us_tweets(df, us_variants)

    # Save the us tweets
    us_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()