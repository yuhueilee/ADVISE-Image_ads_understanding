
import functools
import os
import sys
import json
import string
import argparse

sys.path.append('/content/ADVISE-Image_ads_understanding')

import nltk
from readers.utils import load_action_reason_annots
from readers.utils import load_symbol_cluster
from readers.utils import load_symbol_raw_annots


def main(args):
  """Main."""
  word_to_id, id_to_symbol = load_symbol_cluster(args.symbol_cluster_path)
  print('Load %i pairs of mapping.' % (len(word_to_id)), file=sys.stderr)
  print('Symbol list: \n%s' % (json.dumps(id_to_symbol, indent=2)), file=sys.stderr)

  id_to_symbol = sorted(id_to_symbol.items(), key=functools.cmp_to_key(cmp))
  with open(args.output_vocab_path, 'w') as fp:
    for symbol_id, symbol in id_to_symbol:
      if symbol_id != 0:
        fp.write('%s\t%i\n' % (symbol, 999))

  print('Done', file=sys.stderr)

def cmp(x, y):
  if x[0] < y[0]:
    return -1
  elif x[0] > y[0]:
    return 1
  return 0

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--symbol_cluster_path', type=str,
      default='data/additional/clustered_symbol_list.json', 
      help='Path to the symbol clustering json file.')
  parser.add_argument(
      '--output_vocab_path', type=str,
      default='output/symbol_vocab.txt', 
      help='Path to the output vocab file.')

  args = parser.parse_args()
  assert os.path.isfile(args.symbol_cluster_path)

  print('parsed input parameters:', file=sys.stderr)
  print(json.dumps(vars(args), indent=2), file=sys.stderr)

  main(args)

  exit(0)
