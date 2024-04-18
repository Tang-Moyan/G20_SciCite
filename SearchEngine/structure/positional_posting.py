from structure.document_posting import DocumentPosting


class PositionalPosting:
    """
    Represents a positional posting, which is a list that merges the posting lists of different
    terms in a phrase query, keeping track of the positions in the document where each term appears
    and also the order (priority) of the term in the phrase query.

    A positional posting collects the positions in order with their priority for the purpose of
    checking contiguity of the positions and phrase terms.

    For example, in document 1 containing the phrase "rising interest rates" from the 5th word,
    where the priorities for each of the term in the phrase are 1, 2 and 3 respectively,
    the positional posting be:

    `1 -> (position=5, priority=1), (position=6, priority=2), (position=7, priority=3)`
    """

    def __init__(self, doc_id, positions):
        """
        Creates a new positional posting with the given document ID and positions.

        :param int doc_id: the document ID
        :param list[PhrasalQueryScorer.PositionalPosting.TermPosition] positions:
         the positions of the tokens in the document
        """
        self._doc_id: int = doc_id
        self._positions: list[PositionalPosting.TermPosition] = positions

    class TermPosition:
        """
        Represents a position in a document for the Positional Posting for the purpose of
        phrasal queries. Each position is also assigned a priority, which is the order
        of the token in the phrase.

        For example, if the word "interest" appears in document 1 at position 6 and the
        phrasal query is "rising interest rates", then the positional posting for "interest"
        would be TermPosition(position=6, priority=1).

        If the same document also contains the word "rising" at position 5, then the
        positional posting for "rising" would be TermPosition(position=5, priority=0).
        """

        def __init__(self, position, priority):
            """
            Creates a new position with the given position and priority.

            :param int position: the position of the token in the document
            :param int priority: the priority (position) of the token in the phrase
            """
            self._position = position
            self._priority = priority

        def get_position(self):
            return self._position

        def get_priority(self):
            return self._priority

        def __lt__(self, other):
            """
            Compares this position with another position. The position with the lower
            :param PhrasalQueryScorer.PositionalPosting.TermPosition other: the other position to compare with
            :return: whether this position is less than the other position
            """
            if self._position == other.get_position():
                return self._priority < other.get_priority()
            return self._position < other.get_position()

        def __eq__(self, other):
            """
            Compares this position with another position. The term positions are
            the same only if they have the same position and priority.

            :param PositionalPosting.TermPosition other: the other position to compare with
            :return: whether this position is equal to the other position
            """
            if not isinstance(other, PositionalPosting.TermPosition):
                return False
            return self._position == other.get_position() and self._priority == other.get_priority()

        def __repr__(self):
            return f"TermPosition(position={self._position}, priority={self._priority})"

        def __str__(self):
            return repr(self)

        def __hash__(self):
            return hash((self._position, self._priority))

    def get_document_id(self):
        return self._doc_id

    def get_positions(self):
        return list(self._positions)

    def extend_positions(self, positions):
        """
        Extends the positions of the positional posting with the given positions.
        :param list[PhrasalQueryScorer.PositionalPosting.TermPosition] positions: the positions to extend with
        :rtype: None
        """
        self._positions.extend(positions)

    def sort(self):
        """
        Sorts the positions of the positional posting.
        """
        self._positions.sort()

    def __repr__(self):
        return f"PositionalPosting(doc_id={self._doc_id}, positions={self._positions})"

    def __str__(self):
        return repr(self)

    def __hash__(self):
        return hash((self._doc_id, tuple(self._positions)))

    def __eq__(self, other):
        if not isinstance(other, PositionalPosting):
            return False
        return self._doc_id == other.get_document_id() and self._positions == other.get_positions()

    @classmethod
    def merge_posting_lists(cls, posting_lists_iterators):
        """
        Merges the posting lists together for each identical document ID,
        recording the (doc positional index, token order) for each token.

        :param list[PostingList.PostingListIterator] posting_lists_iterators:
         the posting lists to merge. They are given in the order of the tokens.
        :return: a list of positional postings
        :rtype: list[PhrasalQueryScorer.PositionalPosting]
        """
        # the current document postings is in order of the tokens
        current_doc_postings: list[DocumentPosting | None]
        current_doc_postings = [next(iterator) if not iterator.is_exhausted() else None
                                for iterator in posting_lists_iterators]

        docs_with_positions: list[PositionalPosting] = []

        # union algorithm
        while any(current_doc_postings):
            # find the minimum document ID
            min_doc_id = min(filter(lambda x: x is not None, current_doc_postings),
                             key=lambda x: x.get_document_id()).get_document_id()

            # run through the current postings, if the document ID is the minimum, merge it
            # with the positional posting.
            for i in range(len(current_doc_postings)):
                # i is the order (priority) of the token in the phrase
                if current_doc_postings[i] is None:
                    # if the iterator for this term is exhausted, break
                    continue

                if current_doc_postings[i].get_document_id() == min_doc_id:
                    # create a list of positions for the current token
                    list_of_positions = [PositionalPosting
                                         .TermPosition(pos,  # the position of the token in the document
                                                       i)  # the priority (order) of the token in the phrase
                                         for pos in current_doc_postings[i].get_positional_indices()]

                    # if nothing in the union list or the last element in the union list
                    # has a different document ID, add a new element to the union list
                    if len(docs_with_positions) == 0 or docs_with_positions[-1].get_document_id() < min_doc_id:
                        if len(docs_with_positions) > 0:
                            # sort the positions in the last element in the union list
                            docs_with_positions[-1].sort()

                        docs_with_positions.append(PositionalPosting(min_doc_id, list_of_positions))
                    else:
                        # otherwise, add the positions to the last element in the union list
                        docs_with_positions[-1].extend_positions(list_of_positions)

                    # advance the iterator
                    if not posting_lists_iterators[i].is_exhausted():
                        current_doc_postings[i] = next(posting_lists_iterators[i])
                    else:
                        current_doc_postings[i] = None

        # sort the positions in the last element in the union list
        if len(docs_with_positions) > 0:
            docs_with_positions[-1].sort()

        return docs_with_positions

