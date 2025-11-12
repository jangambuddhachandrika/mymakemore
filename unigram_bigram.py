from collections import Counter

# Sample corpus
sentences = [
    ["<s>", "I", "am", "Sam", "</s>"],
    ["<s>", "Sam", "I", "am", "</s>"],
    ["<s>", "I", "do", "not", "like", "green", "eggs", "and", "ham", "</s>"]
]

all_words = [w for sent in sentences for w in sent]
unigram_count = Counter(all_words)
total_tokens = sum(unigram_count.values())
unigram_probs = {w:c/total_tokens for w,c in unigram_count.items())
