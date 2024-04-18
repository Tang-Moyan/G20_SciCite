import re

from util.stemmer import stem
from itertools import chain
from unidecode import unidecode

ACCEPTED_PUNCTUATIONS = ["+", "-"]


def tokenize_and_reduce(free_text):
    """
    Tokenizes and reduces a free text query to the root form of each word, after removing
    punctuation and changing all words to lowercase.

    Reduction is done by stemming and lemmatizing the words.

    The only punctuations that will be preserved are the plus and minus signs.

    :param str free_text: the free text query
    :return: the text tokenized and stemmed, with no punctuation and in all lowercase
    :rtype: list[str]
    """
    return [stem(token) for token in tokenize(free_text)]


def tokenize(free_text):
    """
    Extracts tokens from a free text, removing punctuation and changing all
    words to lowercase.

    Note that a token is not necessarily a word. For example, "Article 13(b)" is
    two words but three tokens, namely "article", "13", and "b".

    The only punctuations that will be preserved are the plus and minus signs.

    :param str free_text: the free text query
    :return: the text tokenized, with no punctuation and in all lowercase
    :rtype: list[str]
    """
    # between every non-ascii and ascii character, add a space
    ascii_blocks = [unidecode(block) for block in re.findall(
        r"([\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+|[^\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+)",
        free_text)]

    # word tokenize each block
    return [word.lower() for word in chain(*[capture_groups(block) for block in ascii_blocks])]


def capture_groups(input_string):
    """
    Captures tokens in an input string.

    A token is captured if it is either:
    1. An alphabetical word with each letter separated by a period, e.g. "U.S.A."
    2. A pure alphabetical word, e.g. "hello"
    3. A pure numeric word, e.g. "123"
    4. A punctuation that is either + or - only if it is not preceded by an alphanumeric character
         and is followed by an numeric character, e.g. "+40", "-40", "++40"

    :param str input_string: the input string to capture tokens from
    :return: a list of captured groups
    """
    return [match.group() for match in
            re.finditer(r"[a-z](?:\.[a-z])+|[a-z]+|[0-9]+|((?<![a-z0-9])|(?<=[+\-]))[+-](?=[\d+\-])",
                        input_string,
                        flags=re.IGNORECASE)]


def count_tokens(tokens):
    """
    Maps a token list to the number of tokens in the list.

    :param list[str] tokens: the tokens to count
    :return: a mapping of each unique token to the number of times it appears in the list
    :rtype: dict[str, int]
    """
    mapping = {}
    for token in tokens:
        if token in mapping:
            mapping[token] += 1
        else:
            mapping[token] = 1

    return mapping


def parse(free_text):
    """
    Maps a free text query to a dictionary of terms and their frequencies.
    Will first perform tokenization and stemming.

    :param str | list[str] free_text: the free text query, either as a string or a list of tokens.
     If given as a string, will be tokenized and stemmed.
    :return: a dictionary mapping terms to their frequencies
    :rtype: dict[str, int]
    """
    if isinstance(free_text, str):
        free_text = tokenize_and_reduce(free_text)

    return count_tokens(free_text)
