import re
from charngram import CharNGram
from wordngram import WordNGram

FILENAME='finn.txt'

def main():
  with open(FILENAME) as f:
    text = f.read()

  ngram = WordNGram(0, text)
  print "Zero-order word approximation"
  print ngram.generate_sentence(10)
  print

  ngram = WordNGram(1, text)
  print "First-order word approximation"
  print ngram.generate_sentence(10)
  print

  ngram = WordNGram(2, text)
  print "Second-order word approximation"
  print ngram.generate_sentence(10)
  print

if __name__ == '__main__':
  main()
