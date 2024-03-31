from heapq import nlargest
from math import log

from structure.document_summary import DocumentSummary
from structure.query import Query
from util.logger import logger
from structure.document_posting import DocumentPosting
from structure.term_dictionary import TermDictionary

import itertools


def closest_cluster(*sets):
    """
    Find the closest cluster of points in a set of sets.

    :param sets: the sets of points
    :return: the closest cluster of points
    """
    min_distance = float('inf')
    closest_group = None
    for group in itertools.product(*sets):
        distance = sum((max(group) - min(group)) for group in zip(*sets))
        if distance < min_distance:
            min_distance = distance
            closest_group = group
    return closest_group


def standard_deviation(group):
    """
    Calculates the standard deviation of a group of numbers.

    :param group: the group of numbers
    :return: the standard deviation
    """
    mean = sum(group) / len(group)
    return (sum((x - mean) ** 2 for x in group) / len(group)) ** 0.5


def get_court_ranking(court):
    top = {"SG Court of Appeal", "SG Privy Council", "UK House of Lords",
           "UK Supreme Court", "High Court of Australia", "CA Supreme Court"}
    mid = {"SG High Court", "Singapore International Commercial Court", "HK High Court",
           "HK Court of First Instance", "UK Crown Court",
           "UK Court of Appeal", "UK High Court", "Federal Court of Australia", "NSW Court of Appeal",
           "NSW Court of Criminal Appeal", "NSW Supreme Court"}

    if court in top:
        return 3
    elif court in mid:
        return 2
    else:
        return 1


class DocumentScorer:
    """
    A class that computes the cosine score of a pool of documents given a query.

    The cosine score is computed as follows:
    score = tf of term in document * (tf of term in query) * idf of term in corpus
    where tf is the term frequency
    and idf is the inverse document frequency of the term in the corpus

    The cosine score is then normalized by the magnitude of the document vector
    multiplied by the magnitude of the query vector.
    """

    def __init__(self, term_dictionary, magnitudes_dictionary):
        """
        Creates a new cosine score calculator.

        :param TermDictionary term_dictionary: the term dictionary
        :param dict[int, Document] magnitudes_dictionary: the mapping of document IDs to documents
        """
        self._term_dictionary: TermDictionary = term_dictionary
        self._magnitudes: dict[int, float] = magnitudes_dictionary

    class DocumentScore:
        """
        A class that represents a document score. This is used to encapsulate the document ID and the score
        for easy assignment and sorting.
        """

        def __init__(self, document_id, score):
            self.document_id = document_id
            self.score = score
            self.term_positions: list[list[int]] = []

        def add_positional_posting(self, positional_posting):
            """
            Adds a list of positions to the recorded positions where
            any of the terms in the query appear in the document.

            We do not record the term itself, because its knowledge is
            not required to determine closeness of the bag of words.

            :param list[int] positional_posting: the list of positions to add
            :return: None
            """
            self.term_positions.append(positional_posting)

        def __lt__(self, other):
            return self.score < other.score

    def rank_top_k(self, query: Query, document_pool=None, k_count=None, with_scores=False):
        """
        :param Query query: the query to rank the documents by
        :param set[int] document_pool: the set of documents to rank. If None, all documents are ranked
        :param int k_count: the number of top results to return, defaulted to return all relevant documents
        :param bool with_scores: whether to return the scores along with the document IDs
        :return: the list of documents ranked by descending cosine score
        """
        # scores is a dictionary of document IDs to rank, mapped to document scores
        # if document_set is None, then all documents are ranked
        #logger.debug("Bag of words:\n\t" +
                     #"\n\t".join(f"{token} : {query.get_token_weight(token)}" for token in query.get_tokens()))

        universal_posting_list = self._term_dictionary.get_universal_posting_list()

        total_document_count = len(universal_posting_list)
        assert total_document_count == 17137, f"Total document count is {total_document_count}"

        if document_pool is None:
            document_pool = set(posting.get_document_id()
                                for posting in universal_posting_list)
        # document_pool is the set of documents to rank <= full set of documents

        scores: dict[int, DocumentScorer.DocumentScore] = {}
        # scores contains all the documents with at least one term in the query <= document pool size

        for term in query.get_tokens():
            frequency = query.get_token_weight(term)

            document_postings = self._term_dictionary.get_term_posting(term)
            # document_postings contains all the documents that contain the term

            document_frequency = len(document_postings)
            # document frequency is the number of documents that contain the term

            if document_frequency == 0:
                # if the query term does not appear in any document, then it is not relevant
                continue

            weighted_query_term_frequency = (1 + log(frequency, 10)) * log(total_document_count
                                                                           / document_frequency, 10)

            document_postings_iterator = document_postings.to_iterator()

            while not document_postings_iterator.is_exhausted():
                doc_posting: DocumentPosting = next(document_postings_iterator)

                doc_id = doc_posting.get_document_id()
                if doc_id not in document_pool:
                    continue

                weighted_document_term_frequency = 1 + log(doc_posting.get_term_frequency(), 10)

                additional_score = weighted_document_term_frequency * weighted_query_term_frequency

                scores.setdefault(doc_id, DocumentScorer.DocumentScore(doc_id, 0))
                scores[doc_id].add_positional_posting(doc_posting.get_positional_indices())
                scores[doc_id].score += additional_score

        for document_id, score in scores.items():
            magnitude = self._magnitudes[document_id]
            score.score = 0 if magnitude == 0 else score.score / magnitude

        if k_count is not None:
            # initialize a max heap
            heap = [(score.score, score.document_id) for score in scores.values()]

            # heapify first k elements, then compare the rest of the elements to the smallest element
            # if the element is larger, replace the smallest element with the larger element
            # after processing all the elements, the heap is sorted
            k_largest = nlargest(k_count, heap)
        else:
            k_largest = sorted([(score.score, score.document_id) for score in scores.values()],
                               reverse=True)

        if with_scores:
            return list(x for x in k_largest if x[0] != 0)
        else:
            return list(x[1] for x in k_largest if x[0] != 0)

    @staticmethod
    def rank_with_court_priority(ranked_documents, document_summary_dictionary, court_weight):
        """
        Sorts the documents by court priority, and then by the number of documents in the group.

        :param list[int] ranked_documents: the list of documents to rank
        :param dict[int, DocumentSummary] document_summary_dictionary: the dictionary of
         document IDs to document summaries
        :param float court_weight: the weight to give to the court priority
        """
        def score_with_court_priority(score, document_id):
            document_summary: DocumentSummary = document_summary_dictionary[document_id]
            return score * (1 - court_weight) + court_weight * get_court_ranking(document_summary.get_court())

        # rerank by with court weight
        return [document_id for score, document_id in
                sorted(ranked_documents, key=lambda x: score_with_court_priority(*x), reverse=True)]


