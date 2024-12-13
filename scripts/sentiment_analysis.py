import nltk  # type: ignore
from nltk.sentiment import SentimentIntensityAnalyzer  # type: ignore

# Function to download required NLTK resources
def initialize_sentiment_analyzer():
    nltk.download('vader_lexicon')
    return SentimentIntensityAnalyzer()

# Function to compute sentiment for a single text
def get_sentiment_vader(text, sia):
    score = sia.polarity_scores(text)
    return score['compound']

# Main function to perform sentiment analysis on a dataset
def analyze_sentiment(data):
    sia = initialize_sentiment_analyzer()  # Initialize the analyzer
    data['sentiment'] = data['headline'].apply(lambda text: get_sentiment_vader(text, sia))
    print(data['sentiment'].describe())
    return data