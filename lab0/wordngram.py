from ngram import NGram
import re

class WordNGram(NGram):
  SEP = ' '

  def load_text(self, text):
    self.text = NGram.clean(text).split()

  def cols(self):
    return set(self.text)
