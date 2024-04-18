class DocumentPosting:
    def __init__(self, document_id, positional_indices=None):
        """
        Creates a new document posting.
        :param int document_id: the document ID
        :param list[int] positional_indices: the positional indices of the term in the document
        """
        self._document_id = document_id
        self._positional_indices = positional_indices or []

    def add_position(self, position):
        self._positional_indices.append(position)

    def get_document_id(self):
        """
        Returns the document ID of this document posting.
        :rtype: int
        """
        return self._document_id

    def get_term_frequency(self):
        """
        Returns the term frequency of this document posting, i.e the number of times the term
        appears in the document.
        :rtype: int
        """
        return len(self._positional_indices)

    def merge(self, other, sort=False):
        """
        Merges this document posting with another document posting.

        :param DocumentPosting other: the other document posting
        :param bool sort: whether to sort the positional indices after merging
        :rtype: None
        """
        self._positional_indices.extend(other._positional_indices)
        if sort:
            self.sort()

    def sort(self):
        """
        Sorts the positional indices of this document posting.
        :rtype: None
        """
        self._positional_indices.sort()

    def get_positional_indices(self):
        """
        Returns the positional indices of this document posting.
        :rtype: list[int]
        """
        return list(self._positional_indices)

    def __str__(self):
        return f"({self._document_id}, {self._positional_indices})"

    def __eq__(self, other):
        """
        Checks if this document posting is equal to another document posting.
        :param DocumentPosting other: the other document posting
        :return: whether this document posting is equal to the other document posting
        """
        return self._document_id == other._document_id \
            and self._positional_indices == other._positional_indices

    def __hash__(self):
        return hash((self._document_id, self._positional_indices))

    def __repr__(self):
        return self.__str__()
