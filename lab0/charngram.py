from ngram import NGram
import string

class CharNGram(NGram):
  CHARS = ' ' + string.ascii_lowercase[:26]

  def load_text(self, text):
    self.text = list(NGram.clean(text))

  def cols(self):
    return set(self.text)
