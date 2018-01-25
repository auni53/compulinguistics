from charngram import CharNGram

def test_clean():
  _clean = lambda text: CharNGram(0, text).text

  assert _clean('~~') == []
  assert _clean('abc~~') == list('abc')
  assert _clean('ABC~~') == list('abc')

def test_prev_n():
  _prev_n = lambda text,c,n: CharNGram(n, text).prev_n(c)

  assert _prev_n('abc', 3, 0) == list('')
  assert _prev_n('abc', 3, 1) == list('c')
  assert _prev_n('abc', 3, 2) == list('bc')
  assert _prev_n('abc', 3, 3) == list('abc')

def test_empty_count():
  ngram = CharNGram(0, '')
  result = ngram.empty_count()

  assert set(result.keys()).issubset(set([c for c in CharNGram.CHARS]))
  assert len(set(map(lambda v: id(v), result.values()))) == len(ngram.cols())

def test_count():
  ngram = CharNGram(1, 'foobar')
  expected_count = {
    'f': {},
    'o': { 'f': 1, 'o': 1 },
    'b': { 'o': 1 },
    'a': { 'b': 1 },
    'r': { 'a': 1 },
  }
  assert ngram.counts == expected_count

def test_zero_gram_count():
  _count = lambda text, x: CharNGram(0, text).get_count(x)

  assert _count('abc', 'a') == 1
  assert _count('abc', 'b') == 1
  assert _count('a', 'b') == 0
  assert _count('a', 'foobar') == 0

def test_one_gram_count():
  _count = lambda text, x, y: CharNGram(1, text).get_count(x, y)

  assert _count('abc', 'b', 'a') == 1
  assert _count('abc', 'c', 'a') == 0
  assert _count('abc', 'd', 'a') == 0
  assert _count('abc', 'c', 'b') == 1
  assert _count('abc', 'c', 'd') == 0

def test_two_gram_count():
  _count = lambda text, x, y: CharNGram(2, text).get_count(x, y)

  assert _count('abc', 'c', 'ab') == 1
  assert _count('abc', 'c', 'a') == 0
  assert _count('abc', 'c', 'foobar') == 0

  ngram = CharNGram(2, 'foobar')
  expected_count = {
    'f': {},
    'o': { 'f^o': 1 },
    'b': { 'o^o': 1 },
    'a': { 'o^b': 1 },
    'r': { 'b^a': 1 },
  }
  assert ngram.counts == expected_count

def test_bigram_get_count_smaller_n():
  text = 'abc'
  monogram = CharNGram(1, text)
  bigram = CharNGram(2, text)

  assert monogram.get_count('c', 'b') == bigram.get_count('c', 'b')
  assert monogram.get_count('c', 'a') == bigram.get_count('c', 'a')
  assert monogram.get_count('c', '') == bigram.get_count('c', '')
  assert monogram.get_count('c', '') == bigram.get_count('c', '')
  assert monogram.get_count('c', 'foobar') == bigram.get_count('c', 'a')

  ngram = CharNGram(2, 'foobar')
  expected_count = {
    'f': {},
    'o': { 'f^o': 1 },
    'b': { 'o^o': 1 },
    'a': { 'o^b': 1 },
    'r': { 'b^a': 1 },
  }
  assert ngram.counts == expected_count

def test_five_gram_count():
  ngram = CharNGram(5, 'foobar')
  assert ngram.get_count('r', 'fooba') == 1
  assert ngram.counts == dict(\
    {c: {} for c in set('fooba')}.items() +\
    {'r': { '^'.join('fooba'): 1 }}.items()\
  )
  
def test_distribution():
  ngram = CharNGram(1, 'foobar')
  expected = { 'b': 0.5, 'o': 0.5, 'f': 0.0, 'a': 0.0, 'r': 0.0 }
  result = ngram.distribution('o')
  print result
  assert sum(result[1]) == 1
  assert dict(zip(result[0], result[1])) == expected

def test_distribution_zerogram():
  ngram = CharNGram(0, 'abcdef')
  expected = { c: 1/6. for c in 'abcdef' }
  result = ngram.distribution('a')
  assert dict(zip(result[0], result[1])) == expected

def test_sample():
  ngram = CharNGram(1, 'foobar')
  for x in xrange(100):
    result = ngram.sample(['o'])
    assert result in 'ob'

  for x in xrange(100):
    result = ngram.sample(['b'])
    assert result == 'a'

def test_generate_sentence():
  ngram = CharNGram(1, 'abc')
  assert len(ngram.generate_sentence(5)) == 5
