import re
from charngram import CharNGram
from wordngram import WordNGram

FILENAME='finn.txt'

def main():
  with open(FILENAME) as f:
    text = f.read()

  print
  ngram = WordNGram(0, text)
  print "sentences from monogram"
  print ngram.generate_sentence(6)
  print ngram.generate_sentence(7)
  print ngram.generate_sentence(8)
  print ngram.generate_sentence(9)
  print ngram.generate_sentence(10)
  print

  ngram = WordNGram(1, text)
  print "sentences from bigrams"
  print ngram.generate_sentence(6)
  print ngram.generate_sentence(7)
  print ngram.generate_sentence(8)
  print ngram.generate_sentence(9)
  print ngram.generate_sentence(10)
  print

  ngram = WordNGram(2, text)
  print "sentences from trigram"
  print ngram.generate_sentence(6)
  print ngram.generate_sentence(7)
  print ngram.generate_sentence(8)
  print ngram.generate_sentence(9)
  print ngram.generate_sentence(10)
  print


if __name__ == '__main__':
  main()
