from collections import Counter
import pickle
import os.path
import re

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import linalg

import nltk
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist

text = list(map(lambda word: re.sub(r'\W+', '', word.lower()), nltk.corpus.brown.words()))

p_monogram = 'monograms.p'
p_bigram = 'bigrams.p'
p_ppmi = 'ppmi.p'

with open('common_words.txt') as f:
  tokens = list(map((lambda s: s.strip()), f.readlines()))
print("Loaded tokens of length", len(tokens))

if not os.path.isfile(p_bigram):
  bigram_object = ngrams(text, 2)
  bigram = dict(Counter(bigram_object))
  common_bigrams = {k:v for k,v in bigram.items() if (k[0] in tokens and k[1] in tokens)}
  pickle.dump(common_bigrams, open(p_bigram, 'wb'))
common_bigrams = pickle.load(open(p_bigram, 'rb'))
print("Loaded bigrams of length", len(common_bigrams))

if not os.path.isfile(p_monogram):
  monogram_object = ngrams(text, 1)
  monograms = dict(Counter(monogram_object))
  common_monograms = { word:monograms[(word,)] for word in tokens }
  pickle.dump(common_monograms, open(p_monogram, 'wb'))
common_monograms = pickle.load(open(p_monogram, 'rb'))
print("Loaded monograms of length", len(common_monograms))

if not os.path.isfile(p_ppmi):
  ppmi = {}
  finder = BigramCollocationFinder( FreqDist(common_monograms), FreqDist(common_bigrams) )
  bigram_measures = nltk.collocations.BigramAssocMeasures()

  for words, score in finder.score_ngrams(bigram_measures.pmi):
    ppmi[words] = max(score, 0)
  pickle.dump(ppmi, open(p_ppmi, 'wb'))
ppmi = pickle.load(open(p_ppmi, 'rb'))
print("Loaded ppmi of length", len(ppmi))


# common_bigrams = { (w1, w2): n }
M1 = lil_matrix((5000, 5000), dtype=np.uint8)
for w1, w2 in common_bigrams:
  i = tokens.index(w2)
  j = tokens.index(w1)
  c = common_bigrams[(w1, w2)]
  M1[i, j] = c # i=W, j=context

# ppmi = { (w1, w2): score }
M1_plus = lil_matrix((5000, 5000), dtype=np.float16)
for w1, w2 in ppmi:
  i = tokens.index(w2)
  j = tokens.index(w1)
  c = ppmi[(w1, w2)]
  M1_plus[i, j] = c

U, s, Vh = linalg.svds(M1_plus.asfptype(), k=10)
print(U.shape, s.shape, Vh.shape)
import pdb; pdb.set_trace()
