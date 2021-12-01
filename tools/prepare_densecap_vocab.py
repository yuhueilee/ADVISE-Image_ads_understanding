
import os
import sys
import json
import string
import argparse

sys.path.append('/content/ADVISE-Image_ads_understanding')

import nltk
from readers.utils import tokenize
import functools

def _create_vocab(sentences, min_count=2):
  """Computes the vocab given the corpus.

  Args:
    sentences: a list of strings.
    min_count: words appear less than min_count would be pruned.

  Returns:
    vocab: a dict mapping from word to frequency.
  """
  vocab = {}
  for sentence in sentences:
    for word in tokenize(sentence):
      vocab[word] = vocab.get(word, 0) + 1

  print('Number of words: %i.' %(len(vocab)), file=sys.stderr)
  for k in vocab.keys():
    if vocab[k] < min_count:
      del vocab[k]
  print('Number of words after pruning: %i.' % (len(vocab)), file=sys.stderr)
  return vocab


def _check_coverage(vocab, sentences):
  """Checks the coverage of the vocabulary.

  Args:
    vocab: a dict mapping from word to anything.
    sentences: a list of sentences.
  """
  uncover = 0
  for sentence in sentences:
    for word in tokenize(sentence):
      if not word in vocab:
        uncover += 1
        break
  print('Vocab coverage: %.4lf' % (1.0 - 1.0 * uncover / len(sentences)), file=sys.stderr)


def _save_and_index_vocab(vocab, output_vocab_path):
  """Saves and creates index for the vocabulary.

  Args:
    vocab: a dict mapping from word to frequency.
    output_vocab_path: output path.

  Returns:
    index: a mapping from word to id.
  """
  vocab = sorted(vocab.items(), key=functools.cmp_to_key(cmp), reverse=True)
  index = {}
  with open(output_vocab_path, 'w') as fp:
    for idx, (k, v) in enumerate(vocab):
      fp.write('%s\t%i\n' % (k, v))
      index[k] = idx + 1
  return index


def main(args):
  """Main."""
  # Load dataset.
  with open(args.densecap_annot_path, 'r') as fp:
    annots = json.loads(fp.read())
  print('Load annotations for %i images.' % (len(annots)), file=sys.stderr)

  # Create corpus.
  sentences = []
  count = {}
  for _, annot in annots.items():
    for region in annot['regions']:
      sentences.append(region['name'])
  print('Build a corpus with %i sentences.' % (len(sentences)), file=sys.stderr)

  # Generate vocabulary.
  word_and_freq = _create_vocab(sentences, args.min_count)

  _check_coverage(word_and_freq, sentences)
  _save_and_index_vocab(word_and_freq, args.output_vocab_path)
  print('Vocab saved to %s' % (args.output_vocab_path), file=sys.stderr)

  print('Done', file=sys.stderr)

def cmp(x, y):
  if x[1] < y[1]:
    return -1
  elif x[1] > y[1]:
    return 1
  return 0

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--densecap_annot_path', type=str,
      default='output/densecap_train.json', 
      help='Path to the densecap annotation file.')
  parser.add_argument('--min_count', type=int, default=1, 
      help='Words appear less then min_count would be pruned.')
  parser.add_argument(
      '--output_vocab_path', type=str,
      default='output/densecap_vocab.txt', 
      help='Path to the output densecap vocab file.')

  args = parser.parse_args()
  assert os.path.isfile(args.densecap_annot_path)

  print('parsed input parameters:', file=sys.stderr)
  print(json.dumps(vars(args), indent=2), file=sys.stderr)

  main(args)

  exit(0)
