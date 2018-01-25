from __future__ import division
import abc
import re
import numpy as np

class NGram(object):
  SEP = ''

  def __init__(self, n, text):
    self.n = n
    self.load_text(text)
    self.load_ngram()

  @abc.abstractmethod
  def load_text(self, text):
    pass

  def load_ngram(self):
    counts = self.empty_count()

    c = self.n
    while c < len(self.text):
      l = self.text[c]
      p = '^'.join(self.prev_n(c))

      if l:
        if p not in counts[l]:
          counts[l][p] = 1
        else:
          counts[l][p] += 1
      c += 1

    self.counts = counts

  def get_count(self, x, y=''):
    if len(y) > self.n:
      # raise RuntimeError('Invalid n-gram')
      return 0
    elif len(y) == self.n:
      p = '^'.join(y)
      if x in self.counts and p in self.counts[x]:
        return self.counts[x][p]
      else:
        return 0
    else:
      p = '^'.join(y)
      count = 0
      if x in self.counts:
        for x_prev in self.counts[x].keys():
          if x_prev[-len(p):] == p:
            count += self.counts[x][x_prev]
      return count

  def prev_n(self, i):
    return self.text[i - self.n: i]

  def empty_count(self):
    s = {}
    return { c: dict() for c in self.cols() }

  def generate_sentence(self, length):
    c = length
    s = []
    while c > 0:
      if len(s) < self.n:
        sampling = self.sample(s)
      else:
        sampling = self.sample(s[(len(s) - self.n):])
      s.append(sampling)
      c -= 1

    return self.SEP.join(s)

  def sample(self, previous):
    assert len(previous) <= self.n
    tokens, distribution = self.distribution('^'.join(previous))
    i = np.nonzero(np.random.multinomial(1, distribution))[0][0]
    return tokens[i]

  def distribution(self, previous):
    tokens = []
    counts = []
    for token in self.counts.keys():
      count = self.get_count(token, previous)
      tokens.append(token)
      counts.append(count)

    s = sum(counts)
    probability = s and (lambda c: c / s) or (lambda c: 1/len(counts))
    return (tokens, map(probability, counts))

  @abc.abstractmethod
  def cols(self):
    pass

  @staticmethod
  def clean(text):
    s = text.lower()
    s = re.sub(r'\n', ' ', s)
    s = re.sub(r'[^a-z ]+', ' ', s)
    return s
