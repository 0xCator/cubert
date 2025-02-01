
import itertools
from typing import Any, Dict, Iterator, List, Sequence

from tensor2tensor.data_generators import text_encoder

from cubert import code_to_subtokenized_sentences
from cubert import unified_tokenizer

from bert import tokenization

HOLE_NAME = "__HOLE__"
UNKNOWN_TOKEN = "unknown_token_default"

class FullCuBertTokenizer():
  """Wraps the CuBERT tokenizers to behave like BERT's tokenization API."""

  def __init__(self, code_tokenizer_class, vocab_file):
    # "Tokenizer" going from code to subtokenized sentences:
    self.code_tokenizer = code_tokenizer_class()
    # CuBERT skips Comments/Whitespace in finetuned tasks.
    self.code_tokenizer.replace_reserved_keywords((
        HOLE_NAME,
        # Although we don't produce the unknown token when generating VarMisuse
        # examples, we put the unknown token into the common initialization for
        # the tokenizer so that, when the model asks for the tokenization of
        # that special token, it gets a consistent result.
        UNKNOWN_TOKEN))
    self.code_tokenizer.update_types_to_skip((
        unified_tokenizer.TokenKind.COMMENT,
        unified_tokenizer.TokenKind.WHITESPACE,
    ))
    self.subwork_tokenizer = text_encoder.SubwordTextEncoder(vocab_file)

  def tokenize(self, text):
    subtokenized_sentences = (
        code_to_subtokenized_sentences.code_to_cubert_sentences(
            code=text,
            initial_tokenizer=self.code_tokenizer,
            subword_tokenizer=self.subwork_tokenizer))
    return list(itertools.chain(*subtokenized_sentences))

  def convert_tokens_to_ids(self, tokens):
    return tokenization.convert_by_vocab(
        self.subwork_tokenizer._subtoken_string_to_id,  # pylint: disable = protected-access
        tokens)