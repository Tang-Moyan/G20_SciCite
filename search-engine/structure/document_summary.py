from math import log
from typing import Iterable
from datetime import datetime

from util.stemmer import stem


class DocumentSummary:
    """
    Summary of a document.

    Contains the magnitude of the document vector (computed by 1 + log(tf)) and the document ID,
    as well as the raw term frequencies of the terms in the document.
    """

    def __init__(self, document_id, court, date):
        """
        Creates a new document summary.

        :param int document_id: the document ID
        :param str court: the court
        :param datetime date: the date
        """
        self._square_magnitude = 0
        self._document_id = document_id
        self._court = court
        self._date = date
        self._term_frequencies = {}

    def add_term(self, term, term_frequency):
        """
        Adds a term to the document summary.
        :param str term: the term
        :param int term_frequency: the term frequency
        :rtype: None
        """
        self._term_frequencies[term] = term_frequency
        self._square_magnitude += (1 + log(term_frequency, 10)) ** 2

    def get_document_id(self):
        """
        Returns the document ID of this document summary.
        :rtype: int
        """
        return self._document_id

    def get_court(self):
        """
        Returns the court of this document summary.
        :rtype: str
        """
        return self._court

    def get_date(self):
        """
        Returns the date of this document summary.
        :rtype: datetime
        """
        return self._date

    def get_magnitude(self):
        """
        Returns the magnitude of this document summary.
        :rtype: float
        """
        return self._square_magnitude ** 0.5

    def get_term_frequency(self, term, stem_term=True):
        """
        Returns the term frequency of a term in this document summary.
        :param str term: the term
        :param bool stem_term: whether to stem the term
        :rtype: int
        """
        if stem_term:
            return self._term_frequencies.get(stem(term), 0)
        else:
            return self._term_frequencies.get(term, 0)

    def __contains__(self, item):
        """
        Checks if this document summary contains a term.

        :param str item: the term
        :rtype: bool
        """
        return stem(item) in self._term_frequencies

    def get_unique_terms(self):
        """
        Returns all the unique terms in this document summary.

        :rtype: Iterable[str]
        """
        return self._term_frequencies.keys()

    def get_top_terms(self, n, min_length=6):
        """
        Returns the top n terms in this document summary.

        :param int n: the number of terms to return
        :param int min_length: the minimum length of the terms to return
        :rtype: Iterable[str]
        """
        terms = [term for term in self._term_frequencies if term.isalpha() and len(term) >= min_length]
        return sorted(terms, key=self._term_frequencies.get, reverse=True)[:n]

    def get_rarest_terms(self, n, min_length=6):
        """
        Returns the rarest n terms in this document summary.
        :param int n: the number of terms to return
        :param int min_length: the minimum length of the terms to return
        :rtype: Iterable[str]
        """
        terms = [term for term in self._term_frequencies if term.isalpha() and len(term) >= min_length]
        return sorted(terms, key=self._term_frequencies.get, reverse=False)[:n]

