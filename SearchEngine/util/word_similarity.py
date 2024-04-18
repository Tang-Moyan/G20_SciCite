from nltk.corpus import wordnet


def is_synonym(word, other_word):
    """
    Checks if the two words are synonyms.

    :param str word: the first word
    :param str other_word: the second word
    :return: True if the words are synonyms, False otherwise
    :rtype: bool
    """
    word_synonyms = wordnet.synsets(word)
    other_word_synonyms = wordnet.synsets(other_word)

    for word_synonym in word_synonyms:
        for other_word_synonym in other_word_synonyms:
            if word_synonym.path_similarity(other_word_synonym) > 0.5:
                return True

    return False
