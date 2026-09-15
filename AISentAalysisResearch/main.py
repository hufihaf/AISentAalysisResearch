import nltk
nltk.download('wordnet')
nltk.download('sentiwordnet')
from nltk.sentiment import SentimentIntensityAnalyzer
from pathlib import Path

# import txt file and initialize it as an array
file_path = Path(__file__).parent / "wordlist-english.txt"
english_words = file_path.read_text().splitlines()

from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn

def get_sentiment(word):
    synsets = wn.synsets(word)

    if not synsets:
        return None

    scores = []

    for synset in synsets:
        senti_synset = swn.senti_synset(synset.name())

        score = (
            senti_synset.pos_score()
            - senti_synset.neg_score()
        )

        scores.append(score)

    return sum(scores) / len(scores)

for word in english_words[0:100]:
    print(get_sentiment(word))
    
