from collections import Counter
import nltk
import re

text = nltk.corpus.brown.words()

counts = {}
c = 0
for word in text:
  word = re.sub(r'\W+', '', word.lower())
  if word != '':
    if word in counts:
      counts[word] += 1
    else:
      counts[word] = 1

words = Counter(counts).most_common(5000)
with open('common_words.txt', 'w+') as f:
  f.write('\n'.join(map((lambda w: w[0]), words)))

