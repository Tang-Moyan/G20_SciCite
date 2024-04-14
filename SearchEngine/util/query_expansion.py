import itertools
from nltk.corpus import wordnet as wn
from collections import ChainMap
from nltk.corpus import stopwords


stopwords = set(stopwords.words('english'))

# Synonym Lookup
good = {"good", "better", "great", "best"}
rape = {"forced_sex", "rape", "forced_intercourse"}
offence = {"controversy", "scandal", "misconduct", "offence", "wrongdoing"}

# Every synonym in each group must map to the group it belongs to
synonyms = ChainMap({synonym: good for synonym in good},
                    {synonym: rape for synonym in rape},
                    {synonym: offence for synonym in offence})


def generate_synonyms(term):
    """
    Generates and returns a list of synonyms of the given term.
    The synonyms are generated using wordnet from the nltk.corpus
    library.

    If the term is a phrase, the function will combine the synonyms
    of each word in the phrase, and return a list of all possible
    combinations of the synonyms.

    In certain scenarios, the synonym generated for a term might be
    a combination of two words. These synonyms will contain a "_"
    character between them. In such cases, the synonym is converted
    into a phrase by enclosing them between "" symbols, and replacing
    the "_" character with actual spaces (" ").
    """

    if term in stopwords:
        return set()

    term = term.replace(" ", "_")

    native_synonyms = {synonym.replace("_", " ") for synonym in synonyms.get(term, default=[term])}

    default_synonyms = set(
        str(lemma.name()).replace("_", " ").lower()  # convert the synonym into a phrase
        for s in wn.synsets(term)  # get the synsets for the term
        for lemma in itertools.chain(s.lemmas(),  # chain the derived forms (eg, silence -> silent)
                                     *map(lambda l: l.derivationally_related_forms(), s.lemmas()),
                                     *map(lambda l: l.also_sees(), s.lemmas()),
                                     *map(lambda l: l.similar_tos(), s.lemmas()))
    ) | native_synonyms

    if "_" in term:
        # if term is a phrase, permute the synonyms of each word in the phrase
        return set(
            " ".join(perm)  # convert the permutation into a phrase
            for perm in itertools.product(*[synonyms.get(word, default=[word])
                                            for word in term.split("_")])
        ) | default_synonyms
    else:
        return default_synonyms
