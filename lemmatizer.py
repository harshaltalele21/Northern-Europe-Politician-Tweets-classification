import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
import re

lemmatizer = WordNetLemmatizer()

##Tags the words in the tweets
def nltk_tag_to_wordnet_tag(nltk_tag):
    if nltk_tag.startswith('J'):
        return(wordnet.ADJ)
    elif nltk_tag.startswith('V'):
        return(wordnet.VERB)
    elif nltk_tag.startswith('N'):
        return(wordnet.NOUN)
    elif nltk_tag.startswith('R'):
        return(wordnet.ADV)
    else:          
        return(None)

##Lemmatizes the words in tweets and returns the cleaned and lemmatized tweet
def lemmatize_tweet(tweet):
    #tokenize the tweet and find the POS tag for each token
    tweet = tweet_cleaner(tweet) #tweet_cleaner() will be the function you will write
    nltk_tagged = nltk.pos_tag(nltk.word_tokenize(tweet))  
    #tuple of (token, wordnet_tag)
    wordnet_tagged = map(lambda x: (x[0], nltk_tag_to_wordnet_tag(x[1])), nltk_tagged)
    lemmatized_tweet = []
    for word, tag in wordnet_tagged:
        if tag is None:
            #if there is no available tag, append the token as is
            lemmatized_tweet.append(word)
        else:        
            #else use the tag to lemmatize the token
            lemmatized_tweet.append(lemmatizer.lemmatize(word, tag))
    return(" ".join(lemmatized_tweet))

def tweet_cleaner(tweet):
    # cleaned_tweet = tweet

    # # Removing links from the tweet
    # cleaned_tweet = re.sub('http[^\s]+','',tweet)

    # # Removing punctuations 
    # cleaned_tweet = re.sub(r'[^\w\s]', '', cleaned_tweet)

    #Removing links AND punctuations from the tweet 
    cleaned_tweet = re.sub(r'http[^\s]+|[^\w\s]', '', tweet)

    # Removing Emojis
    emojis = re.compile("["
        u"\U00002700-\U000027BF"  # Dingbats
        u"\U0001F600-\U0001F64F"  # Emoticons
        u"\U00002600-\U000026FF"  # Miscellaneous Symbols
        u"\U0001F300-\U0001F5FF"  # Miscellaneous Symbols And Pictographs
        u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        u"\U0001F680-\U0001F6FF"  # Transport and Map Symbols
                        "]+", re.UNICODE)
    cleaned_tweet = re.sub(emojis, '', cleaned_tweet)

    tweet_tokens = word_tokenize(cleaned_tweet)

    #Removing stop words AND words with length < 3
    stop_words = set(stopwords.words(['english','swedish','german','french','dutch']))
    cleaned_tweet = [word for word in tweet_tokens if len(word) >= 3 and word.lower() not in stop_words]

    cleaned_tweet = " ".join(str(item) for item in cleaned_tweet)
    # print(cleaned_tweet)

    return cleaned_tweet