import nltk
from nltk.corpus import cmudict
from collections import defaultdict
import re

d = cmudict.dict()

def load_nrc(path):
    lexicon = defaultdict(list)
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            word, emotion, value = line.strip().split("\t")
            if value == "1":
                lexicon[word].append(emotion)
    return lexicon

nrc_lexicon = load_nrc("NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")

def extract_words(target_phoneme):
    words = set()
    for word, pron_list in d.items():
        for pron in pron_list:
            if target_phoneme in pron:
                words.add(word)
                break
    return list(words)

gleam_words = extract_words("IY1")
glum_words = extract_words("AH1")

def clean(words):
    return [w for w in words if re.match("^[a-z]+$", w) and len(w) > 2]

gleam_words = clean(gleam_words)
glum_words = clean(glum_words)

gleam_set = set(gleam_words)
glum_set = set(glum_words)
gleam_words = list(gleam_set - glum_set)
glum_words = list(glum_set - gleam_set)

def is_monosyllabic(word, d):
    prons = d.get(word, [])
    if not prons:
        return False
    return sum(1 for p in prons[0] if p[-1] in "012") == 1

gleam_words = [w for w in gleam_words if is_monosyllabic(w, d)]
glum_words = [w for w in glum_words if is_monosyllabic(w, d)]

def get_emotions(words, lexicon):
    counts = defaultdict(int)
    for w in words:
        if w in lexicon:
            for emo in lexicon[w]:
                counts[emo] += 1
    return counts

gleam_emotions = get_emotions(gleam_words, nrc_lexicon)
glum_emotions = get_emotions(glum_words, nrc_lexicon)

def normalize(emotions):                          
    total = sum(emotions.values())
    return {k: v / total for k, v in emotions.items()} if total else {}

def print_profile(name, data):
    print(f"\n=== {name} EMOTION PROFILE ===")
    for k, v in sorted(data.items(), key=lambda x: -x[1]):
        print(f"{k:12s} {v:.4f}")

print("Gleam words:", len(gleam_words))
print("Glum words:", len(glum_words))

gleam_norm = normalize(gleam_emotions)
glum_norm = normalize(glum_emotions)

print_profile("GLEAM (IY1)", gleam_norm)
print_profile("GLUM (AH1)", glum_norm)