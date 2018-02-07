from collections import Counter
import enum
import pickle as pickle_module
import os.path
from os.path import isfile
import re

import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse import linalg
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk.util import ngrams
from nltk.collocations import BigramCollocationFinder
from nltk.probability import FreqDist

import pdb
stop = pdb.set_trace

common_words_file = 'common_words.txt'
rg65_file = 'rg65_table1.txt'
rg65_overlap_pairs = [('coast', 'forest'), ('coast', 'hill'), ('car', 'journey'), ('food', 'fruit'), ('hill', 'mount'), ('coast', 'shore'), ('automobile', 'car')]

ANALOGY_TEST_PATH = "./word2vec/word-test.v1.txt"

class Pickle(enum.Enum):
  text      = 'text.p'
  common    = 'common.p'
  monograms = 'monograms.p'
  bigrams   = 'bigrams.p'
  ppmi      = 'ppmi.p'
  M1        = 'M1.p'
  M1_plus   = 'M1_plus.p'
  M2        = 'M2.p'

  model     = 'google_model.p'
  analogies = 'analogies.p'


root = 'pickles/'
def get_path(pickle):
  if not pickle:
    return ''
  return root + pickle.value

def pickle_load(path):
  return pickle_module.load(open(path, 'rb'))

def pickle_dump(raw, path):
  return pickle_module.dump(raw, open(path, 'wb'))

INDEX = None
def load_index(tokens):
  global INDEX
  INDEX = { x: i for i, x in enumerate(tokens) }
  return INDEX


def get_i(word):
  return (INDEX or load_index(load_common_words(Pickle.common)))[word]


def query(matrix, w1, w2):
  if w2 is None:
    return matrix[:, get_i(w1)]

  return matrix[get_i(w1), get_i(w2)]

def load_text(pickle, cache=True):
  path = get_path(pickle)
  if cache and isfile(path):
    return pickle_load(path)

  text = list(map(lambda word: re.sub(r'\W+', '', word.lower()), nltk.corpus.brown.words()))

  print("Caching full corpus of length", len(nltk.corpus.brown.words()))
  pickle_dump(text, path)
  return text


def load_common_words(pickle, cache=True):
  path = get_path(pickle)
  if cache and isfile(path):
    return pickle_load(path)

  with open(common_words_file) as f:
    tokens = list(map((lambda s: s.strip()), f.readlines()))

  print("Caching common tokens of length", len(tokens))
  pickle_dump(tokens, path)
  return tokens


def load_rg65_words(existing):
  words = set()
  with open(rg65_file) as f:
    for line in f.readlines():
      [w1, w2, _score] = line.strip().split(' ')
      words.add(w1)
      words.add(w2)
  words -= set(existing)
  return list(words)


def load_common_bigrams(text, tokens, pickle=None, cache=True):
  path = get_path(pickle)
  if cache and isfile(path):
    return pickle_load(path)

  bigram_object = ngrams(text, 2)
  bigrams = dict(Counter(bigram_object))
  common_bigrams = {k:v for k,v in bigrams.items() if (k[0] in tokens and k[1] in tokens)}

  print("Caching bigrams of length", len(common_bigrams))
  pickle_dump(common_bigrams, path)
  return common_bigrams


def load_M1(bigrams, pickle=None, cache=True):
  path = get_path(pickle)
  if cache and isfile(path):
    return pickle_load(path)

  M1 = lil_matrix((5000, len(INDEX)), dtype=np.uint32)
  for w1, w2 in bigrams:
    i = get_i(w1)
    j = get_i(w2)
    if i < M1.shape[0] and j < M1.shape[1]:
      M1[i, j] = bigrams.get((w1, w2), 0)

  print("Caching M1 of shape", M1.shape)
  pickle_dump(M1, path)
  return M1


def load_common_monograms(text, tokens, pickle=None, cache=True):
  path = get_path(pickle)
  if cache and isfile(path):
    return pickle_load(path)

  monogram_object = ngrams(text, 1)
  monograms = dict(Counter(monogram_object))
  get = lambda word: monograms.get((word,), 0)
  common_monograms = { word:get(word) for word in tokens }

  print("Caching monograms of length", len(common_monograms))
  pickle_dump(common_monograms, path)
  return common_monograms


def load_ppmi(monograms, bigrams, pickle=None, cache=True):
  path = get_path(pickle)
  if cache and isfile(path):
    return pickle_load(path)

  ppmi = {}
  finder = BigramCollocationFinder( FreqDist(monograms), FreqDist(bigrams) )
  bigram_measures = nltk.collocations.BigramAssocMeasures()

  for words, score in finder.score_ngrams(bigram_measures.pmi):
    ppmi[words] = max(score, 0.0)

  print("Caching ppmi of length", len(ppmi))
  pickle_dump(ppmi, path)
  return ppmi


def load_M1_plus(ppmi, pickle=None, cache=True):
  path = get_path(pickle)
  if cache and isfile(path):
    return pickle_load(path)

  M1_plus = lil_matrix((5000, len(INDEX)), dtype=np.float16)
  for w1, w2 in ppmi:
    i = get_i(w1)
    j = get_i(w2)
    if i < M1_plus.shape[0] and j < M1_plus.shape[1]:
      M1_plus[i, j] = ppmi.get((w1, w2), 0.0)

  print("Caching M1+ of shape", M1_plus.shape)
  pickle_dump(M1_plus, path)
  return M1_plus


def load_M2(M1_plus, pickle=None, cache=True):
  path = get_path(pickle)
  if cache and isfile(path):
    return pickle_load(path)

  M2, _s, _Vh = linalg.svds(M1_plus.transpose().asfptype(), k=100)
  M2 = M2.transpose()

  print("Caching M2 of shape", M2.shape)
  pickle_dump(M2, path)
  return M2


def find_similarities(pairs, matrix):
  sims = []
  c = 0
  while c < len(pairs):
    i1 = get_i(pairs[c][0])
    i2 = get_i(pairs[c][1])
    row1 = matrix[:,i1].transpose()
    row2 = matrix[:,i2].transpose()
    if len(row1.shape) == 1 or len(row2.shape) == 1:
      row1 = row1.reshape(1, len(row1))
      row2 = row2.reshape(1, len(row2))
    pair_cosine = cosine_similarity(row1, row2)
    # print(row1.shape)
    # print(row2.shape)
    # print(pair_cosine)
    sims.append(pair_cosine[0][0])
    c += 1
  return sims


def load_analogies(pickle, cache=True):
  path = get_path(pickle)
  if cache and isfile(path):
    return pickle_load(path)
  analogies = {}

  test_type = None
  with open(ANALOGY_TEST_PATH) as f:
    for line in f:
      l = line.lower().strip().split(' ')
      if l[0] == ':':
        test_type = l[1]
        analogies[test_type] = []
      elif len(l) == 4:
        analogies[test_type].append(tuple(l))

  print("Caching analogies of length", len(analogies))
  pickle_dump(analogies, path)
  return analogies

def most_similar(matrix, tokens, p0, p1, n0):
  q = lambda word: query(matrix, word, None)
  vec1 = (q(p0) + q(p1) - q(n0)).reshape(1, 100)

  max_sim = 0
  word_c = None
  for c in range(len(tokens)):
    vec2 = matrix[:,c].reshape(1, 100)
    sim = cosine_similarity(vec1, vec2)[0][0]

    if sim > max_sim:
      max_sim = sim
      word_c = c

  return tokens[c]

def lsa_analogy_test(pairs, matrix, tokens):
  passed = 0
  failed = 0

  for pair in pairs:
    w1, w2, w3, w4 = pair
    try:
      guess = most_similar(matrix, tokens, w2, w3, w1)

      if guess == w4:
        passed += 1
      else:
        failed += 1
    except KeyError:
      pass

  return (passed, failed)

def main():
  text = load_text(Pickle.text)
  tokens = load_common_words(Pickle.common)
  tokens += load_rg65_words(tokens)
  load_index(tokens)

  common_bigrams = load_common_bigrams(text, tokens, Pickle.bigrams)
  common_monograms = load_common_monograms(text, tokens, Pickle.monograms)
  ppmi = load_ppmi(common_monograms, common_bigrams, Pickle.ppmi)

  M1 = load_M1(common_bigrams, Pickle.M1)
  M1_plus = load_M1_plus(ppmi, Pickle.M1_plus)
  M2 = load_M2(M1_plus, Pickle.M2)

  M2_10 = M2[:10]
  M2_50 = M2[:50]
  M2_100 = M2

  analogies = load_analogies(Pickle.analogies)
  for test_type in analogies:
    passed, failed = lsa_analogy_test(analogies[test_type], M2_100, tokens)
    print("test type {} passed {} failed {}".format(test_type, passed, failed))

  exit(0)

  P = []
  S = []

  with open(rg65_file) as f:
    for line in f.readlines():
      [w1, w2, score] = line.strip().split(' ')
      P.append((w1, w2))
      S.append(float(score))

  print("pearsonr[S, S]:", pearsonr(S, S)[0])

  S_M1 = find_similarities(P, M1)
  print("pearsonr[S, S_M1]:", pearsonr(S, S_M1)[0])

  S_M1_plus = find_similarities(P, M1_plus)
  print("pearsonr[S, S_M1_plus]:", pearsonr(S, S_M1_plus)[0])

  S_M2_10 = find_similarities(P, M2_10)
  print("pearsonr[S, S_M2_10]:", pearsonr(S, S_M2_10)[0])

  S_M2_50 = find_similarities(P, M2_50)
  print("pearsonr[S, S_M2_50]:", pearsonr(S, S_M2_50)[0])

  S_M2_100 = find_similarities(P, M2_100)
  print("pearsonr[S, S_M2_100]:", pearsonr(S, S_M2_100)[0])

if __name__ == '__main__':
  main()
