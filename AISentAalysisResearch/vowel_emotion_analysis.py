"""
Compute average NRC-EmoLex emotion/sentiment scores per vowel phoneme,
restricted to monosyllabic English words.

Requires:
    pip install nltk pandas
    (first run will trigger nltk.download('cmudict') if not already present)

Input:
    NRC-Emotion-Lexicon-Wordlevel-v0.92.txt  (or whatever EmoLex file you have)
    Long format, tab-separated: word <TAB> emotion <TAB> 0/1
    One row per (word, emotion) pair, 10 emotion/sentiment categories per word.

Output:
    vowel_phoneme_emotion_averages.csv
    One row per vowel phoneme (ARPABET, stress stripped), with:
      - n_words: how many monosyllabic words carried that vowel
      - mean score for each of the 10 EmoLex categories
"""

import re
from collections import defaultdict
import nltk
nltk.download('cmudict')

import pandas as pd
import nltk

try:
    from nltk.corpus import cmudict
except LookupError:
    nltk.download("cmudict")
    from nltk.corpus import cmudict

PRON_DICT = cmudict.dict()

EMOLEX_PATH = "AISentAalysisResearch/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"  # <-- update to your file path
OUTPUT_CSV = "vowel_phoneme_emotion_averages.csv"
OUTPUT_WORDLIST_CSV = "vowel_phoneme_wordlists.csv"  # optional: which words fed each vowel

EMOTIONS = [
    "anger", "anticipation", "disgust", "fear", "joy",
    "negative", "positive", "sadness", "surprise", "trust",
]


def load_emolex(path):
    """Parse EmoLex long-format file into {word: {emotion: 0/1}}."""
    word_scores = defaultdict(dict)
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            word, emotion, score = parts
            if emotion not in EMOTIONS:
                continue
            word_scores[word][emotion] = int(score)
    return word_scores


def get_monosyllabic_vowel(word, use_first_pron_only=False):
    """
    Return the (stress-stripped) ARPABET vowel phoneme for a word if it is
    monosyllabic according to CMUdict, else None.

    CMUdict marks vowel phonemes with a trailing stress digit (0/1/2), so
    counting those digit-bearing phones is a free syllable count.
    Words with zero CMUdict entries, or with more/less than one vowel
    nucleus, are dropped.
    """
    prons = PRON_DICT.get(word.lower())
    if not prons:
        return None

    pron_list = prons[:1] if use_first_pron_only else prons
    results = set()
    for phones in pron_list:
        vowels = [p for p in phones if p[-1].isdigit()]
        if len(vowels) != 1:
            continue  # not monosyllabic under this pronunciation
        results.add(re.sub(r"\d", "", vowels[0]))

    if len(results) == 1:
        return results.pop()
    return None  # either no monosyllabic pronunciation, or disagreement across variants


def build_vowel_emotion_table(emolex_path):
    word_scores = load_emolex(emolex_path)

    vowel_sums = defaultdict(lambda: defaultdict(float))
    vowel_counts = defaultdict(int)
    vowel_words = defaultdict(list)

    skipped_not_monosyllabic = 0
    skipped_not_found = 0

    for word, scores in word_scores.items():
        vowel = get_monosyllabic_vowel(word)
        if vowel is None:
            if word.lower() in PRON_DICT:
                skipped_not_monosyllabic += 1
            else:
                skipped_not_found += 1
            continue

        vowel_counts[vowel] += 1
        vowel_words[vowel].append(word)
        for e in EMOTIONS:
            vowel_sums[vowel][e] += scores.get(e, 0)

    print(f"Words skipped (not in CMUdict): {skipped_not_found}")
    print(f"Words skipped (not monosyllabic / ambiguous): {skipped_not_monosyllabic}")
    print(f"Vowel phonemes recovered: {len(vowel_counts)}")

    rows = []
    for vowel, n in vowel_counts.items():
        row = {"vowel": vowel, "n_words": n}
        for e in EMOTIONS:
            row[e] = vowel_sums[vowel][e] / n
        rows.append(row)

    df = pd.DataFrame(rows).sort_values("vowel").reset_index(drop=True)
    return df, vowel_words


if __name__ == "__main__":
    df, vowel_words = build_vowel_emotion_table(EMOLEX_PATH)
    df.to_csv(OUTPUT_CSV, index=False)
    print(df.to_string(index=False))

    # Optional: save which words contributed to each vowel, for sanity-checking
    with open(OUTPUT_WORDLIST_CSV, "w", encoding="utf-8") as f:
        f.write("vowel,words\n")
        for vowel, words in sorted(vowel_words.items()):
            f.write(f'{vowel},"{" ".join(sorted(words))}"\n')
