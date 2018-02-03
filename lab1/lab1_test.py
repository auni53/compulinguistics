from lab1 import *
from os.path import isfile

def test_files_cached():
  for name in Jar.__members__:
    assert isfile(Jar[name].value)

def test_indexing():
  assert get_i('the') == 0
