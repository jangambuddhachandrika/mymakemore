import random
from collections import Counter, defaultdict
import math


class NGramModel:
    def __init__(self, corpus):
        self.sentences = [[ "<s>" ] + s.split() + [ "</s>" ] for s in corpus]
        self._build_model()

    def _build_model(self):
        # Unigram counts
        words = [w for s in self.sentences for w in s]
        self.unigram_counts = Counter(words)
        self.total_tokens = sum(self.unigram_counts.values())

        # Bigram counts
        self.bigram_counts = Counter()
        for s in self.sentences:
            for i in range(len(s) - 1):
                self.bigram_counts[(s[i], s[i + 1])] += 1

        # Compute probabilities (unsmoothed)
        self.unigram_probs = {
            w: c / self.total_tokens for w, c in self.unigram_counts.items()
        }
        self.bigram_probs = {
            (w1, w2): c / self.unigram_counts[w1]
            for (w1, w2), c in self.bigram_counts.items()
        }

    # Generate a random sentence from bigrams
    def generate_sentence(self, max_len=15):
        word = "<s>"
        sentence = []
        for _ in range(max_len):
            next_words = [w2 for (w1, w2) in self.bigram_probs if w1 == word]
            if not next_words:
                break
            probs = [self.bigram_probs[(word, w2)] for w2 in next_words]
            word = random.choices(next_words, weights=probs)[0]
            if word == "</s>":
                break
            sentence.append(word)
        return " ".join(sentence)

    # Compute perplexity on a test corpus
    def perplexity(self, test_sentences):
        test_tokens = 0
        log_prob = 0
        for s in test_sentences:
            s = ["<s>"] + s.split() + ["</s>"]
            for i in range(1, len(s)):
                w1, w2 = s[i - 1], s[i]
                p = self.bigram_probs.get((w1, w2), 1e-10)  # tiny prob if unseen
                log_prob += math.log(p)
                test_tokens += 1
        return math.exp(-log_prob / test_tokens)
# Corpus 1: Informal email/chat text
email_corpus = [
    "hey how are you doing",
    "I am fine thanks",
    "let me know when you are free",
    "see you soon"
]

# Corpus 2: News article style
news_corpus = [
    "the government announced new policies today",
    "the prime minister met the president",
    "global markets see strong growth",
    "new technologies are transforming industries"
]

email_model = NGramModel(email_corpus)
news_model  = NGramModel(news_corpus)

print("=== Most common unigrams (Email) ===")
print(email_model.unigram_counts.most_common(5))

print("\n=== Most common unigrams (News) ===")
print(news_model.unigram_counts.most_common(5))

print("\n=== Example random sentences ===")
print("Email:", email_model.generate_sentence())
print("News:", news_model.generate_sentence())

# Perplexity comparison (using email sentences as test)
print("\n=== Perplexity (on email corpus) ===")
print("Email model:", email_model.perplexity(email_corpus))
print("News model:", news_model.perplexity(email_corpus))
