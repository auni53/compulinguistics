from collections import Counter
import enum
import pickle
import os.path
from os.path import isfile
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

class Jar(enum.Enum):
  text      = 'jar/text.p'
  common    = 'jar/common.p'
  monograms = 'jar/monograms.p'
  bigrams   = 'jar/bigrams.p'
  ppmi      = 'jar/ppmi.p'
  M1        = 'jar/M1.p'
  M1_plus   = 'jar/M1_plus.p'
  M2        = 'jar/M2.p'


def _load(path):
  return pickle.load(open(path, 'rb'))


INDEX = None
def load_index(tokens):
  global INDEX
  INDEX = { x: i for i, x in enumerate(tokens) }
  return INDEX


def get_i(word):
  return (INDEX or load_index(load_common_words(Jar.common.value)))[word]


def load_text(path):
  if isfile(path):
    return _load(path)

  text = list(map(lambda word: re.sub(r'\W+', '', word.lower()), nltk.corpus.brown.words()))

  print("Caching full corpus of length", len(nltk.corpus.brown.words()))
  pickle.dump(text, open(path, 'wb'))
  return text


def load_common_words(path):
  if isfile(path):
    return _load(path)

  with open(common_words_file) as f:
    tokens = list(map((lambda s: s.strip()), f.readlines()))

  print("Caching common tokens of length", len(tokens))
  pickle.dump(tokens, open(path, 'wb'))
  return tokens


def load_common_bigrams(text, tokens, path=''):
  if isfile(path):
    return _load(path)

  bigram_object = ngrams(text, 2)
  bigram = dict(Counter(bigram_object))
  common_bigrams = {k:v for k,v in bigram.items() if (k[0] in tokens and k[1] in tokens)}

  print("Caching bigrams of length", len(common_bigrams))
  pickle.dump(common_bigrams, open(path, 'wb'))
  return common_bigrams


def load_common_monograms(text, tokens, path=''):
  if isfile(path):
    return _load(path)

  monogram_object = ngrams(text, 1)
  monograms = dict(Counter(monogram_object))
  common_monograms = { word:monograms[(word,)] for word in tokens }

  print("Caching monograms of length", len(common_monograms))
  pickle.dump(common_monograms, open(path, 'wb'))
  return common_monograms


def load_ppmi(monograms, bigrams, path=''):
  if isfile(path):
    return _load(path)

  ppmi = {}
  finder = BigramCollocationFinder( FreqDist(monograms), FreqDist(bigrams) )
  bigram_measures = nltk.collocations.BigramAssocMeasures()

  for words, score in finder.score_ngrams(bigram_measures.pmi):
    ppmi[words] = max(score, 0)

  print("Caching ppmi of length", len(ppmi))
  pickle.dump(ppmi, open(path, 'wb'))
  return ppmi


def load_M1(bigrams, path=''):
  if isfile(path):
    return _load(path)

  M1 = lil_matrix((5000, 5000), dtype=np.uint8)
  for w1, w2 in bigrams:
    i = get_i(w2)
    j = get_i(w1)
    c = bigrams[(w1, w2)]
    M1[i, j] = c # i=W, j=context

  print("Caching M1 of shape", M1.shape)
  pickle.dump(M1, open(Jar.M1.value, 'wb'))
  return M1


def load_M1_plus(ppmi, path=''):
  if isfile(path):
    return _load(path)

  M1_plus = lil_matrix((5000, 5000), dtype=np.float16)
  for w1, w2 in ppmi:
    i = get_i(w2)
    j = get_i(w1)
    c = ppmi[(w1, w2)]
    M1_plus[i, j] = c

  print("Caching M1+ of shape", M1_plus.shape)
  pickle.dump(M1_plus, open(Jar.M1_plus.value, 'wb'))
  return M1_plus


def load_M2(M1_plus, path=''):
  if isfile(path):
    return _load(path)

  M2, _s, _Vh = linalg.svds(M1_plus.asfptype(), k=100)

  print("Caching M1+ of shape", M1_plus.shape)
  pickle.dump(M1_plus, open(path, 'wb'))
  return M2


def main():
  text = load_text(Jar.text.value)
  tokens = load_common_words(Jar.common.value)

  common_bigrams = load_common_bigrams(text, tokens, Jar.bigrams.value)
  common_monograms = load_common_monograms(text, tokens, Jar.monograms.value)
  ppmi = load_ppmi(common_monograms, common_bigrams, Jar.ppmi.value)
  M1 = load_M1(common_bigrams, Jar.M1.value)
  M1_plus = load_M1_plus(ppmi, Jar.M1_plus.value)
  M2 = load_M2(M1_plus, Jar.M2.value)

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
    i = get_i(pair[0])
    j = get_i(pair[1])
    S_M1.append(cosine_similarity(mat[i], mat[j])[0][0])

  for pair in P:
    mat = M1_plus
    pair_i = P.index(pair)
    i = get_i(pair[0])
    j = get_i(pair[1])
    S_M1_plus.append(cosine_similarity(mat[i], mat[j])[0][0])

  # for pair in P:
  #   mat = M2_10
  #   pair_i = P.index(pair)
  #   i = tokens.index(pair[0])
  #   j = tokens.index(pair[1])
  #   S_M2_10.append(cosine_similarity(mat[i], mat[j])[0][0])

  # for pair in P:
  #   mat = M2_50
  #   pair_i = P.index(pair)
  #   i = tokens.index(pair[0])
  #   j = tokens.index(pair[1])
  #   S_M2_50.append(cosine_similarity(mat[i], mat[j])[0][0])

  # for pair in P:
  #   mat = M2_100
  #   pair_i = P.index(pair)
  #   i = tokens.index(pair[0])
  #   j = tokens.index(pair[1])
  #   S_M2_100.append(cosine_similarity(mat[i], mat[j])[0][0])

  print("pearsonr[S, S]:", pearsonr(S, S)[0])
  print("pearsonr[S, S_M1]:", pearsonr(S, S_M1)[0])
  print("pearsonr[S, S_M1_plus]:", pearsonr(S, S_M1_plus)[0])
  # print("pearsonr[S, S_M2_10]:", pearsonr(S, S_M2_10)[0])
  # print("pearsonr[S, S_M2_50]:", pearsonr(S, S_M2_50)[0])
  # print("pearsonr[S, S_M2_100]:", pearsonr(S, S_M2_100)[0])

if __name__ == '__main__':
  main()
