from nltk.stem.snowball import SnowballStemmer


stemmer = SnowballStemmer(language='english')


def stem(word):
    """
    Stems the word.

    Note: The stemmer only works on ascii characters. If the word contains non-ascii characters,
    the word is returned as is.

    Please use the tokenizer to tokenize the word before passing it to this function.

    :param str word: the word to stem
    :return: the stemmed word
    :rtype: str
    """
    return stemmer.stem(word)
