from extension import *

def test_scores():
  scores = load_rg65_scores()
  assert len(scores) == 65

def test_load_analogies():
  analogies = load_analogies()
  assert len(analogies) == 14
  assert analogies['capital-common-countries'][0] == ('athens', 'greece', 'baghdad', 'iraq')
  print(analogies)
