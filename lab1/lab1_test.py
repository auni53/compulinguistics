from lab1 import *

from os.path import isfile

def test_get_path():
  load_index([1, 2, 3])
  assert get_path(Pickle.common) == 'pickles/common.p'
  assert get_path(Pickle.M1) == 'pickles/M1.p'

def test_indexing():
  assert get_i('the') == 0

def test_load_tokens():
  tokens = load_common_words(Pickle.common, cache=False)
  assert len(tokens) == 5000
  tokens += load_rg65_words(tokens)
  assert len(tokens) == 5031

def test_M1():
  text = load_text(Pickle.text)
  tokens = load_common_words(Pickle.common)
  load_index(tokens)

  common_bigrams = load_common_bigrams(text, tokens, Pickle.bigrams)
  M1 = load_M1(common_bigrams, Pickle.M1)

  assert M1.shape == (5000, 5000)
  assert query(M1, 'to', 'the') == common_bigrams.get(('to', 'the'), 0)
  assert query(M1, 'the', 'to') == common_bigrams.get(('the', 'to'), 0)
  assert query(M1, 'to', 'the') > query(M1, 'the', 'to')

def test_M1_similarity():
  text = load_text(Pickle.text)
  tokens = load_common_words(Pickle.common)
  load_index(tokens)

  common_bigrams = load_common_bigrams(text, tokens, Pickle.bigrams)
  M1 = load_M1(common_bigrams, Pickle.M1)

  sims = find_similarities(rg65_overlap_pairs, M1)
  assert len(sims) == len(rg65_overlap_pairs)

def test_M1_plus():
  text = load_text(Pickle.text)
  tokens = load_common_words(Pickle.common)
  load_index(tokens)

  common_bigrams = load_common_bigrams(text, tokens, Pickle.bigrams)
  M1 = load_M1(common_bigrams, Pickle.M1)

  common_monograms = load_common_monograms(text, tokens, Pickle.monograms)
  ppmi = load_ppmi(common_monograms, common_bigrams, Pickle.ppmi)
  assert ppmi.get(('to', 'the'), 0.0) > ppmi.get(('the', 'to'), 0.0)

  M1_plus = load_M1_plus(ppmi, Pickle.M1_plus)
  assert M1_plus.shape == (5000, 5000)
  assert abs(query(M1_plus, 'to', 'the') - ppmi.get(('to', 'the'), 0)) < 0.01
  assert abs(query(M1_plus, 'the', 'to') - ppmi.get(('the', 'to'), 0)) < 0.01
  assert query(M1_plus, 'to', 'the') > query(M1_plus, 'the', 'to')

def test_M2():
  text = load_text(Pickle.text)
  tokens = load_common_words(Pickle.common)
  load_index(tokens)

  common_bigrams = load_common_bigrams(text, tokens, Pickle.bigrams)
  M1 = load_M1(common_bigrams, Pickle.M1)
  common_monograms = load_common_monograms(text, tokens, Pickle.monograms)
  ppmi = load_ppmi(common_monograms, common_bigrams, Pickle.ppmi)
  M1_plus = load_M1_plus(ppmi, Pickle.M1_plus)
  M2 = load_M2(M1_plus, Pickle.M2)
  assert M2.shape == (5000, 100)

def test_rg65_matrices():
  text = load_text(Pickle.text)
  tokens = load_common_words(Pickle.common)
  tokens += load_rg65_words(tokens)
  load_index(tokens)

  common_bigrams = load_common_bigrams(text, tokens, Pickle.bigrams)
  M1 = load_M1(common_bigrams, Pickle.M1)

  assert M1.shape == (5000, 5031)
  assert query(M1, 'to', 'the') == common_bigrams.get(('to', 'the'), 0)
  assert query(M1, 'the', 'to') == common_bigrams.get(('the', 'to'), 0)
  assert query(M1, 'to', 'the') > query(M1, 'the', 'to')

  common_monograms = load_common_monograms(text, tokens, Pickle.monograms, cache=False)
  ppmi = load_ppmi(common_monograms, common_bigrams, Pickle.ppmi)
  assert ppmi.get(('to', 'the'), 0.0) > ppmi.get(('the', 'to'), 0.0)

  M1_plus = load_M1_plus(ppmi, Pickle.M1_plus)
  assert M1_plus.shape == (5000, 5031)
  assert abs(query(M1_plus, 'to', 'the') - ppmi.get(('to', 'the'), 0)) < 0.01
  assert abs(query(M1_plus, 'the', 'to') - ppmi.get(('the', 'to'), 0)) < 0.01
  assert query(M1_plus, 'to', 'the') > query(M1_plus, 'the', 'to')

  M2 = load_M2(M1_plus, Pickle.M2)
  assert M2.shape == (100, 5031)
