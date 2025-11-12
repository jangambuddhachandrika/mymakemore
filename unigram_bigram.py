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
unigram_probs = {w:c/total_tokens for w,c in unigram_count.items()}
bigram_count = Counter()
for sent in sentences:
    for i in range(len(sent)-1):
        bigram = (sent[i],sent[i+1])
        bigram_count[bigram]+=1

bigram_probs = {}
for (w1,w2),count in bigram_count.items():
    bigram_probs[(w1,w2)] = count/unigram_count[w1]


# Display results
print("=== Unigram Probabilities ===")
for w, p in unigram_probs.items():
    print(f"P({w}) = {p:.4f}")

print("\n=== Bigram Probabilities ===")
for (w1, w2), p in bigram_probs.items():
    print(f"P({w2} | {w1}) = {p:.4f}")
    
        
