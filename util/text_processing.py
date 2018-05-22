from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import sys
import io

def load_vocab_dict_from_file(dict_file, pad_at_first=True):
    if (sys.version_info > (3, 0)):
        with open(dict_file, encoding='utf-8') as f:
            words = [w.strip() for w in f.readlines()]
    else:
        with io.open(dict_file, encoding='utf-8') as f:
            words = [w.strip() for w in f.readlines()]
    if pad_at_first and words[0] != '<pad>':
        raise Exception("The first word needs to be <pad> in the word list.")
    vocab_dict = {words[n]:n for n in range(len(words))}
    return vocab_dict

UNK_IDENTIFIER = '<unk>' # <unk> is the word used to identify unknown words
SENTENCE_SPLIT_REGEX = re.compile(r'(\W+)')
def sentence2vocab_indices(sentence, vocab_dict):
    if isinstance(sentence, bytes):
        sentence = sentence.decode()
    words = SENTENCE_SPLIT_REGEX.split(sentence.strip())
    words = [w.lower().replace(" ","") for w in words if len(w.strip()) > 0]
    # remove .
    if len(words) > 0 and (words[-1] == '.' or words[-1] == '?'):
        words = words[:-1]
    vocab_indices = [(vocab_dict[w] if w in vocab_dict else vocab_dict[UNK_IDENTIFIER])
        for w in words]
    return vocab_indices

PAD_IDENTIFIER = '<pad>'
def preprocess_vocab_indices(vocab_indices, vocab_dict, T, mode='zseq'):
    # Truncate long sentences
    assert mode in ['zseq', 'seqz'], "preprocess_vocab_indices mode should be zseq/seqz"
    if len(vocab_indices) > T:
        vocab_indices = vocab_indices[:T]
    if len(vocab_indices) < T:
        if mode == 'zseq':
        # Pad short sentences at the beginning with the special symbol '<pad>'
            vocab_indices = [vocab_dict[PAD_IDENTIFIER]] * (T - len(vocab_indices)) + vocab_indices
        else:
        # Pad short sentences at the end with the special symbol '<pad>'
            vocab_indices = vocab_indices + [vocab_dict[PAD_IDENTIFIER]] * (T - len(vocab_indices))
    return vocab_indices

def preprocess_sentence(sentence, vocab_dict, T, mode='zseq'):
    vocab_indices = sentence2vocab_indices(sentence, vocab_dict)
    return preprocess_vocab_indices(vocab_indices, vocab_dict, T, mode)
