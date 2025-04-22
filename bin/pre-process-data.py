import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
negation_words = {"no", "nor", "not", "never", "none", "nobody", "nothing", "neither", "nowhere", "hardly", "barely", "scarcely", 
                  "isn't", "couldn't", "aren't", "isnt", "couldnt", "arent", "doesn't", "doesnt", "ain't", "aint", 
                  "shouldn't", "shouldnt", "wasn't", "wasnt", "weren't", "werent", "wont", "won't"}
stop_words = stop_words - negation_words

# --- Step 1: Load and combine CSV datasets ---
# Replace these file names with the actual paths to your CSV files
df1 = pd.read_csv("C:\\Users\\rachi\\Desktop\\Spring 2025\\CAP 4773\\Project\\social-media-mining-project\\data\\hashtag_donaldtrump.csv", lineterminator='\n', parse_dates=True)
df2 = pd.read_csv("C:\\Users\\rachi\\Desktop\\Spring 2025\\CAP 4773\\Project\\social-media-mining-project\\data\\hashtag_joebiden.csv", lineterminator='\n', parse_dates=True)

# Combine the two datasets into one DataFrame
df = pd.concat([df1, df2], ignore_index=True)

# --- Step 2: Create a new DataFrame with tweets located in the United States ---
us_keywords = ["United States", "United States of America", "USA", "US"]
mask_country = df["country"].str.contains("|".join(us_keywords), case=False, na=False)
mask_user_location = df["user_location"].str.contains("|".join(us_keywords), case=False, na=False)

mask = mask_country | mask_user_location

df_us = df[mask]

# List of columns to analyze
columns_to_check = ["state", "state_code", "country", "city", "lat", "long", "user_location"]

# Function to print out unique values and their counts for each column
def print_locations(df: pd.DataFrame):
    for col in columns_to_check:
        print(f"\nUnique values and counts for column: {col}")
        # Using value_counts to count each unique value (including NaN values if needed)
        unique_counts = df[col].value_counts(dropna=False)
        print(unique_counts)

# Function to clean the text of tweets
def clean_text(tweet):
    tweet = tweet.lower() # set all characters to lowercase
    tweet = re.sub(r'^rt\s+', '', tweet) # remove RT at start
    tweet = re.sub(r'@[A-Za-z0-9_]+', '', tweet) # remove @mentions
    tweet = re.sub(r'#', '', tweet) # remove hashtag symbols but keep words
    tweet = re.sub(r'https?:\/\/\S+', '', tweet) # remove hyperlinks
    tweet = re.sub(r'(.)\1{2,}', r'\1', tweet) # reduce character elongations (e.g., loool -> lol, pleeease -> please)
    tweet = re.sub(r'[^A-Za-z0-9\s]', '', tweet) # remove special characters
    tweet = tweet.encode('ascii', 'ignore').decode('ascii') # remove emojis
    tweet = re.sub(r'\d+', '', tweet) # remove numbers
    tweet = re.sub(r'\s+', ' ', tweet).strip() # remove extra spaces

    # remove stopwords (e.g., the, is, at, which, on, and, etc.)
    tweet_tokens = tweet.split()
    filtered_words = [word for word in tweet_tokens if word not in stop_words]
    tweet = " ".join(filtered_words)
    return tweet

# define a list of the columns to keep in the cleaned dataframe
columns_to_keep = ["tweet_id", "user_id", "clean_tweet", "lat", "long", "city", "country", "state", "state_code"]
# make a copy of the df_us 
df_cleaned = df_us.copy()
# clean the tweets and save into the clean_tweet column
df_cleaned["clean_tweet"] = df_cleaned["tweet"].apply(clean_text)
# keep only the columns in columns_to_keep
df_cleaned = df_cleaned[columns_to_keep]

# Save the cleaned dataset so we can load it later without pre-processing again
df_cleaned.to_csv("C:\\Users\\rachi\\Desktop\\Spring 2025\\CAP 4773\\Project\\social-media-mining-project\\data\\cleaned_tweets.csv", index=False)

