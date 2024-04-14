class TermRecord:
    """
    A record of a term in the dictionary, containing the actual word recorded,
    the document frequency and the pointer to its respective list in the posting
    list.

    Supported native operations: `str()`
    """

    def __init__(self, term, doc_freq, seek_position):
        """
        Creates a new term record

        :param str term: the actual word to be stored
        :param int doc_freq: the frequency of occurrence in the posting list
        :param int seek_position: the position in the posting list file where
            the posting list object appears
        """
        self._term = term
        self._doc_freq = doc_freq
        self._seek_position = seek_position

    def get_term(self):
        return self._term

    def get_doc_freq(self):
        return self._doc_freq

    def get_seek_position(self):
        return self._seek_position

    def __str__(self):
        return f"(Term: '{self._term}', " \
               f"Document Frequency: {self._doc_freq}, " \
               f"Seek Position: {self._seek_position})"
