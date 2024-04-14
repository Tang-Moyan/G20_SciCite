import os
import pickle

from structure.query import Query
from util.free_text_parser import tokenize_and_reduce


class RelevanceHistory:
    """
    Looks up the search history of the relevant documents given a query.
    In addition, also memorizes the current relevant documents for future use.

    However, not all relevant documents for past queries are relevant for future queries,
    the documents returned are merely a suggestion and possible a subset of the relevant documents.
    """

    def __init__(self, memory_file_path):
        """
        Initializes the memory recall object.

        :param str memory_file_path: the path to the memory file
        """
        self._history_file_path = memory_file_path

        self._history: dict[Query, set[int]] = {}

        # if the memory file exists, load it
        if os.path.exists(memory_file_path):
            self._history = pickle.load(open(memory_file_path, "rb"))
        else:
            self._history = {}

    class LabelledSet(set):
        """
        A set that comes with a label of whether it is exact or not.
        """
        def __init__(self, is_exact, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.is_exact = is_exact

    def save_relevant_docs_with_query(self, query, relevant_docs):
        """
        Saves the relevant documents given a query.

        This will copy the query to prevent the query from being modified.

        :param Query query: the query
        :param set[int] relevant_docs: the relevant documents
        :return: None
        """
        self._history[query.copy()] = relevant_docs

    def get_relevant_docs(self, current_query):
        """
        Returns the relevant documents given a query.

        For each saved query, check if the current query is more general than the saved query.

        A query is tighter if it is covered by the entire scope of the saved query.

        For example, if the saved query is 'fertility treatment damages', 'fertility treatment'
        or 'damages' can use the relevant documents of the saved query (more general information).

        But if the saved query was '"fertility treatment damages"' (phrasal), then neither
        'fertility treatment damages' nor '"maternal fertility treatment" AND damages' can use the
        relevant documents of the saved query since the current query is more specific.

        Similarly, relevant documents of '"fertility treatment" AND damages' can be used by 'fertility'.

        If an exact match is found, the relevant documents of the exact match is returned.

        :param Query current_query: the query
        :return: the relevant documents
        :rtype: RelevanceHistory.LabelledSet[int]
        """
        relevant_docs = RelevanceHistory.LabelledSet(False)

        for saved_query, saved_relevant_docs in self._history.items():
            # check if the current query is an exact match and return the relevant documents
            if current_query == saved_query:
                relevant_docs = RelevanceHistory.LabelledSet(True, saved_relevant_docs)
                break

            # check if the tokens in the current query are a subset of the tokens in the saved query
            if set(current_query.get_tokens()).issubset(saved_query.get_tokens()):
                # if the saved query is entirely free text, we can use its relevant documents
                if all(not group.is_phrase() for group in saved_query.get_token_groups()):
                    relevant_docs.update(saved_relevant_docs)
                else:
                    # but if the saved query has a phrase, we need to check if the current query
                    # contains subsets of the saved query's phrase
                    # sorry for ugly code
                    all_covers = True  # assume all phrases cover the current query
                    for saved_group in saved_query.get_token_groups():
                        if saved_group.is_phrase():
                            covers = False  # assume saved phrase does not cover current query
                            for current_group in current_query.get_token_groups():
                                # only if both are phrases
                                if current_group.is_phrase():
                                    # in each group, there is only one member
                                    saved_member: set[str] = set(tokenize_and_reduce(next(iter(saved_group.members))))
                                    current_member: set[str] = set(tokenize_and_reduce(next(iter(current_group.members))))
                                    if current_member.issubset(saved_member):
                                        covers = True
                                        # this phrase is covered, so we can break
                                        break
                            if not covers:
                                # we cannot use the saved query's relevant documents
                                all_covers = False
                                break

                    if all_covers:
                        relevant_docs.update(saved_relevant_docs)

        return relevant_docs

    def flush_to_file(self):
        """
        Flushes the memory to the memory file.

        :return: None
        """
        pickle.dump(self._history, open(self._history_file_path, "wb"))

