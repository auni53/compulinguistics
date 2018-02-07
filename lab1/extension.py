from lab1 import *
from gensim.models import KeyedVectors

import numpy as np
from scipy.stats import pearsonr

PRETRAINED_PATH = "./word2vec/GoogleNews-vectors-negative300.bin"
ANALOGY_TEST_PATH = "./word2vec/word-test.v1.txt"

from pdb import set_trace as stop

def get_path(pickle):
  if not pickle:
    return ''
  return root + pickle.value

def load_rg65_words():
  words = set()
  with open(rg65_file) as f:
    for line in f.readlines():
      [w1, w2, _score] = line.strip().split(' ')
      words.add(w1)
      words.add(w2)
  return list(words)

def load_rg65_scores():
  scores = {}
  with open(rg65_file) as f:
    for line in f.readlines():
      [w1, w2, score] = line.strip().split(' ')
      scores[(w1, w2)] = float(score)
  return scores


def get_model():
  return KeyedVectors.load_word2vec_format(PRETRAINED_PATH, binary=True)


def load_rg65_model_scores(pickle, cache=True):
  path = get_path(pickle)
  if cache and isfile(path):
    return pickle_load(path)

  model = get_model()
  scores = load_rg65_scores()
  for (w1, w2) in scores.keys():
    try:
      m1 = model[w1]
      m2 = model[w2]
      s = cosine_similarity(m1.reshape(1, 300), m2.reshape(1, 300))
      scores[(w1, w2)] = s[0][0]
    except KeyError:
      print("Could not find pair ({}, {})".format(w1, w2))

  print("Caching rg65 pair pretrained scores of length", len(scores))
  pickle_dump(scores, path)
  return scores


def load_analogies():
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
  return analogies


def run_analogy_test(pairs, model):
  passed = 0
  failed = 0

  for pair in pairs:
    w1, w2, w3, w4 = pair
    try:
      guess = model.most_similar(positive=[w2, w3], negative=[w1])
      if guess[0][0] == w4:
        passed += 1
      else:
        failed += 1
    except KeyError:
      pass

  return (passed, failed)


def main():
  words = load_rg65_words()

  rg65_human_scores = load_rg65_scores()
  rg65_model_scores = load_rg65_model_scores(Pickle.model)
  l1 = []
  l2 = []
  for key in rg65_model_scores:
    l1.append(rg65_human_scores[key])
    l2.append(rg65_model_scores[key])

  correlation = pearsonr(l1, l2)

  print(correlation)
  exit(0)

  # model = get_model()
  # wv = model.wv
  # wv.save('vectors.bin')

  wv = KeyedVectors.load('vectors.bin')

  analogies = load_analogies()
  stop()
  for test_type in analogies:
    passed, failed = run_analogy_test(analogies[test_type], wv)
    print("test type {} passed {} failed {}".format(test_type, passed, failed))


if __name__ == '__main__':
  main()
