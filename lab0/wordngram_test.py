from wordngram import WordNGram

sample = {}
sample[0] = "The quick brown fox jumps over the lazy dog"
sample[1] = "the the the the auni auni the"

def test_clean():
  _clean = lambda text: WordNGram(0, text).text

  assert _clean('~~') == []
  assert _clean('abc~~') == ['abc']
  assert _clean('ABC~~') == ['abc']
  assert _clean('the quick brown fox ') == ['the', 'quick', 'brown', 'fox']
  assert _clean('the quick brown fox ~~@!') == ['the', 'quick', 'brown', 'fox']
  assert _clean('the quick@#% brown fox ') == ['the', 'quick', 'brown', 'fox']
  assert _clean('th@#%!e quick brown fox ') == ['the', 'quick', 'brown', 'fox']

def test_empty_count():
  ngram = WordNGram(0, '')
  result = ngram.empty_count()

  assert set(result.keys()) == set([c for c in ngram.cols()])
  assert len(set(map(lambda v: id(v), result.values()))) == len(ngram.cols())

def test_prev_n():
  _prev_n = lambda text,c,n: WordNGram(n, text).prev_n(c)

  assert _prev_n('abc', 3, 1) == list('')
  assert _prev_n('abc', 3, 3) == 'abc'.split(' ')
  assert _prev_n(sample[0], 3, 1) == 'brown'.split(' ')
  assert _prev_n(sample[0], 3, 2) == 'quick brown'.split(' ')
  x = 0
  assert _prev_n(sample[0], 8, 8-x) == sample[0].lower().split(' ')[x:-1]
  x = 1
  assert _prev_n(sample[0], 8, 8-x) == sample[0].lower().split(' ')[x:-1]
  x = 5
  assert _prev_n(sample[0], 8, 8-x) == sample[0].lower().split(' ')[x:-1]

def test_zero_gram_count():
  ngram = WordNGram(0, sample[0])
  expected_count = {
    'the': {'': 2},
    'quick': {'': 1},
    'brown': {'': 1},
    'fox': {'': 1},
    'jumps': {'': 1},
    'over': {'': 1},
    'lazy': {'': 1},
    'dog': {'': 1},
  }
  assert ngram.counts == expected_count

  ngram = WordNGram(0, sample[1])
  expected_count = {
    'auni': { '': 2 },
    'the':  { '': 5 },
  }
  assert ngram.counts == expected_count

def test_one_gram_count():
  ngram = WordNGram(1, sample[0])
  expected_count = {
    'the': {'over': 1},
    'quick': {'the': 1},
    'brown': {'quick': 1},
    'fox': {'brown': 1},
    'jumps': {'fox': 1},
    'over': {'jumps': 1},
    'lazy': {'the': 1},
    'dog': {'lazy': 1},
  }
  assert ngram.counts == expected_count

  ngram = WordNGram(1, sample[1])
  expected_count = {
    'auni': { 'auni': 1, 'the': 1 },
    'the':  { 'auni': 1, 'the': 3 },
  }
  assert ngram.counts == expected_count

def test_two_gram_count():
  ngram = WordNGram(2, sample[1])
  expected_count = {
    'auni': { 'the^the': 1, 'the^auni': 1 },
    'the':  { 'the^the': 2, 'auni^auni': 1 },
  }
  assert ngram.counts == expected_count

def test_three_gram_count():
  ngram = WordNGram(3, sample[1])
  expected_count = {
    'auni': { 'the^the^the': 1, 'the^the^auni': 1 },
    'the':  { 'the^the^the': 1, 'the^auni^auni': 1 },
  }
  assert ngram.counts == expected_count
