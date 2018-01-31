from collections import Counter
import pickle
import os.path
import re

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import linalg
from scipy.stats import pearsonr

import nltk
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist

from sklearn.metrics.pairwise import cosine_similarity

common_words_file = 'common_words.txt'
rg65_file = 'rg65_table1.txt'

p_text = 'text.p'
p_common_words = 'common.p'
p_monogram = 'monograms.p'
p_bigram = 'bigrams.p'
p_ppmi = 'ppmi.p'
p_M1 = 'M1.p'
p_M1_plus = 'M1+.p'
p_M2 = 'M2.p'

if not os.path.isfile(p_text):
  print("original length:", len(nltk.corpus.brown.words()))
  text = list(map(lambda word: re.sub(r'\W+', '', word.lower()), nltk.corpus.brown.words()))
  pickle.dump(text, open(p_text, 'wb'))
text = pickle.load(open(p_text, 'rb'))
print("Loaded text of length", len(text))

if not os.path.isfile(p_common_words):
  with open(common_words_file) as f:
    tokens = list(map((lambda s: s.strip()), f.readlines()))
  pickle.dump(tokens, open(p_common_words, 'wb'))
tokens = pickle.load(open(p_common_words, 'rb'))
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

if not os.path.isfile(p_M1):
  M1 = lil_matrix((5000, 5000), dtype=np.uint8)
  if w1 == 'coast' or w2 == 'coast':
    import pdb; pdb.set_trace()

  for w1, w2 in common_bigrams:
    i = tokens.index(w2)
    j = tokens.index(w1)
    c = common_bigrams[(w1, w2)]
    M1[i, j] = c # i=W, j=context
  pickle.dump(M1, open(p_M1, 'wb'))
M1 = pickle.load(open(p_M1, 'rb'))
print("Loaded M1 of shape", M1.shape)

if not os.path.isfile(p_M1_plus):
  M1_plus = lil_matrix((5000, 5000), dtype=np.float16)
  for w1, w2 in ppmi:
    i = tokens.index(w2)
    j = tokens.index(w1)
    c = ppmi[(w1, w2)]
    M1_plus[i, j] = c
  pickle.dump(M1_plus, open(p_M1_plus, 'wb'))
M1_plus = pickle.load(open(p_M1_plus, 'rb'))
print("Loaded M1+ of shape", M1_plus.shape)

if not os.path.isfile(p_M2):
  M2, s, Vh = linalg.svds(M1_plus.asfptype(), k=100)
  pickle.dump(M2, open(p_M2, 'wb'))
M2 = pickle.load(open(p_M2, 'rb'))

M2_10 = M2[:,:10]
M2_50 = M2[:,:50]
M2_100 = M2

P = []
S = []

with open(rg65_file) as f:
  for line in f.readlines():
    [w1, w2, score] = line.strip().split(' ')
    if w1 in tokens and w2 in tokens:
      print("Found pair", w1, w2)
      pair = (w1, w2)
      P.append(pair)
      S.append(float(score))

S_M1 = []
S_M1_plus = []
S_M2_10 = []
S_M2_50 = []
S_M2_100 = []

for pair in P:
  mat = M1
  pair_i = P.index(pair)
  i = tokens.index(pair[0])
  j = tokens.index(pair[1])
  S_M1.append(cosine_similarity(mat[i], mat[j])[0][0])

for pair in P:
  mat = M1_plus
  pair_i = P.index(pair)
  i = tokens.index(pair[0])
  j = tokens.index(pair[1])
  S_M1_plus.append(cosine_similarity(mat[i], mat[j])[0][0])

for pair in P:
  mat = M2_10
  pair_i = P.index(pair)
  i = tokens.index(pair[0])
  j = tokens.index(pair[1])
  S_M2_10.append(cosine_similarity(mat[i], mat[j])[0][0])

for pair in P:
  mat = M2_50
  pair_i = P.index(pair)
  i = tokens.index(pair[0])
  j = tokens.index(pair[1])
  S_M2_50.append(cosine_similarity(mat[i], mat[j])[0][0])

for pair in P:
  mat = M2_100
  pair_i = P.index(pair)
  i = tokens.index(pair[0])
  j = tokens.index(pair[1])
  S_M2_100.append(cosine_similarity(mat[i], mat[j])[0][0])

print("pearsonr[S, S]:", pearsonr(S, S)[0])
print("pearsonr[S, S_M1]:", pearsonr(S, S_M1)[0])
print("pearsonr[S, S_M1_plus]:", pearsonr(S, S_M1_plus)[0])
print("pearsonr[S, S_M2_10]:", pearsonr(S, S_M2_10)[0])
print("pearsonr[S, S_M2_50]:", pearsonr(S, S_M2_50)[0])
print("pearsonr[S, S_M2_100]:", pearsonr(S, S_M2_100)[0])
