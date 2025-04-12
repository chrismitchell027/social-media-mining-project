import os
import zipfile
import contextlib
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
import geopandas as gpd
from shapely.geometry import Point
import sys


def check_and_download(download=True, verbose=False):
# ---------- Helper Functions ----------
    def print_message(message):
        if verbose:
            print(message)

    @contextlib.contextmanager
    def suppress_stdout():
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout

# ---------- Check if data exists ----------
    target_files = ['./data/hashtag_joebiden.csv', './data/hashtag_donaldtrump.csv']
    missing_files = [file for file in target_files if not os.path.isfile(file)]

    if not missing_files:
        print_message("Both CSV files are present in the current directory.")
        return True, target_files[0], target_files[1]
    
    
    print_message(f"Missing files detected: {missing_files}")

    if not download:
        print_message("Data files not detected.")
        return False, None, None
    
# ---------- Download files if missing ----------
    print_message("Data files not detected.")
    print_message("Initializing Kaggle API...")
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except OSError:
        print("Download Failed: Could not find kaggle.json. Make sure it's located in C:/Users/<username>/.kaggle or download it from Kaggle: Your Profile > Settings > API > Create New API Token.")
        return False, None, None

    dataset = 'manchunhui/us-election-2020-tweets'
    download_path = './data/'

    print_message(f"Downloading dataset '{dataset}'...")
    if not verbose:
        with suppress_stdout():
            api.dataset_download_files(dataset, path=download_path, unzip=False)
    else:
        api.dataset_download_files(dataset, path=download_path, unzip=False)
    print_message("Download complete.")

    zip_file_path = os.path.join(download_path, 'us-election-2020-tweets.zip')
    print_message(f"Extracting '{zip_file_path}'...")
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(download_path)
    print_message("Extraction complete.")

    os.remove(zip_file_path)
    print_message(f"Removed the zip file '{zip_file_path}'.")

    for file in missing_files:
        if os.path.isfile(file):
            print_message(f"'{file}' has been successfully downloaded and extracted.")
            return True, target_files[0], target_files[1]
        else:
            print_message(f"Warning: '{file}' is still missing after extraction.")
            return False, None, None
        

# ---------- Step 2: Load and combine CSVs ----------
def clean_and_combine_csvs(export=True):

    def clean_text(tweet, stop_words=False):

        if stop_words:
            nltk.download('stopwords')
            stop_words = set(stopwords.words('english'))
            negation_words = {"no", "nor", "not", "never", "none", "nobody", "nothing", "neither", "nowhere", "hardly", "barely", "scarcely", 
                            "isn't", "couldn't", "aren't", "isnt", "couldnt", "arent", "doesn't", "doesnt", "ain't", "aint", 
                            "shouldn't", "shouldnt", "wasn't", "wasnt", "weren't", "werent", "wont", "won't"}
            stop_words = stop_words - negation_words

        tweet = tweet.lower() # set all characters to lowercase
        tweet = re.sub(r'^rt\s+', '', tweet) # remove RT at start
        tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet) # remove @mentions
        tweet = re.sub(r'#', '', tweet) # remove hashtag symbols but keep words
        tweet = re.sub(r'https?:\/\/\S+', '', tweet) # remove hyperlinks
        tweet = re.sub(r'(.)\1{2,}', r'\1', tweet) # reduce character elongations (e.g., loool -> lol, pleeease -> please)
        tweet = re.sub(r'[^A-Za-z0-9\s]', '', tweet) # remove special characters
        tweet = tweet.encode('ascii', 'ignore').decode('ascii') # remove emojis
        # tweet = re.sub(r'\d+', '', tweet) # remove numbers
        tweet = re.sub(r'\s+', ' ', tweet).strip() # remove extra spaces
        
        # # remove stopwords (e.g., the, is, at, which, on, and, etc.)
        # tweet_tokens = tweet.split()
        # filtered_words = [word for word in tweet_tokens if word not in stop_words]
        # tweet = " ".join(filtered_words)
        return tweet

    check, biden_data, trump_data = check_and_download()
    if not check:
        print("Data files are not available. Exiting.")
        exit(1)

    df_don = pd.read_csv(trump_data, lineterminator='\n', parse_dates=True)
    df_joe = pd.read_csv(biden_data, lineterminator='\n', parse_dates=True)

    df_tweets = pd.concat([df_don, df_joe], ignore_index=True)
    df_tweets = df_tweets.drop_duplicates(subset=["tweet_id"])

    #### To download the geospatial data, you need to go to https://www2.census.gov/geo/tiger/GENZ2020/gdb/cb_2020_us_all_500k.zip (310MB zipped, 412MB unzipped) and unzip into ./data ####
    us_states = gpd.read_file("./data/cb_2020_us_all_500k.gdb", layer="cb_2020_us_state_500k")
    us_states = us_states.to_crs("EPSG:4326")
    df = df_tweets.copy()

    geo_df = df.dropna(subset=["lat", "long"]).copy()
    geo_df["geometry"] = [Point(xy) for xy in zip(geo_df["long"], geo_df["lat"])]
    gdf_tweets = gpd.GeoDataFrame(geo_df, geometry="geometry", crs="EPSG:4326")
    mask_geo = gdf_tweets.within(us_states.unary_union)
    geo_us = gdf_tweets[mask_geo]

    us_keywords = ["United States", "United States of America", "USA", "US"]
    mask_country = df_tweets["country"].str.contains("|".join(us_keywords), case=False, na=False)
    mask_user_location = df_tweets["user_location"].str.contains("|".join(us_keywords), case=False, na=False)
    mask = mask_country | mask_user_location
    meta_us = df_tweets[mask]

    df_us = pd.concat([geo_us, meta_us]).drop_duplicates(subset=["tweet_id"])

    df_cleaned = df_us.copy()
    df_cleaned["clean_tweet"] = df_cleaned["tweet"].apply(clean_text)

    columns_to_keep = ["tweet_id", "clean_tweet", "user_id", "lat", "long", "city", "country", "state", "state_code"]
    df_cleaned = df_cleaned[columns_to_keep]
    df_cleaned = df_cleaned.dropna(subset=["tweet_id", "clean_tweet"])

    if export:
        df_cleaned.to_csv("./data/cleaned_tweets.csv", index=False)
    return df_cleaned

if __name__ == "__main__":
    df = clean_and_combine_csvs(export=True)
    print("Data cleaning and combination complete.")
    print(f"Cleaned data saved to: {os.path.abspath('./data/cleaned_tweets.csv')}")