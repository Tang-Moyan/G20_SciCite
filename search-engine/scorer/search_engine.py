import itertools

from structure.relevance_history import RelevanceHistory
from structure.query import Query
from structure.document_summary import DocumentSummary
from structure.term_dictionary import TermDictionary
from scorer.document_scorer import DocumentScorer
from structure.document_summary_dictionary import DocumentSummaryDictionary
from util.free_text_parser import tokenize_and_reduce
from util.logger import logger
from util.query_expansion import generate_synonyms


SEARCH_HISTORY_FILENAME = './search_history.txt'


def promote_relevant_docs(relevant_docs, scored_documents):
    """
    Intercept the results and put the (initial) relevant docs at the top

    :param set[int] relevant_docs: the set of documents deemed relevant.
    :param list[int] scored_documents: the list of documents ranked by descending cosine score
    :return: the list of documents ranked by descending cosine score with relevant docs at the
     top in the order of appearance in the original ranked list
    """
    if not relevant_docs:
        # if there are no relevant docs, we return the original list
        return scored_documents

    extracted_relevant_docs = list()
    for doc_id in scored_documents:
        if doc_id in relevant_docs:
            extracted_relevant_docs.append(doc_id)
    scored_documents = extracted_relevant_docs + [doc_id for doc_id in scored_documents if
                                                  doc_id not in relevant_docs]
    return scored_documents


class SearchEngine:
    """
    Processes a compound query.
    """

    def __init__(self, term_dictionary, document_summary_dictionary):
        """
        Creates a new compound query processor with the given term dictionary and cosine score calculator.

        :param TermDictionary term_dictionary: the term dictionary
        :param DocumentSummaryDictionary document_summary_dictionary: mapping of document IDs to their summary
         containing the magnitude of the document vector
        """
        self._scorer = DocumentScorer(term_dictionary, document_summary_dictionary.map_id_to_magnitude())
        self._term_dictionary = term_dictionary
        self._document_summary_dictionary = document_summary_dictionary
        self._history = RelevanceHistory(SEARCH_HISTORY_FILENAME)

    def submit_query(self, query, relevant_docs=None, k_count=None,
                     query_expansion=False, pseudo_relevant_feedback=False):
        """
        Process a query string and return a list of document IDs that matches the query in the given order.

        :param Query query: the query to rank the documents by
        :param set[int] relevant_docs: the list of documents deemed relevant. If None, all documents are considered
        :param int k_count: the number of top results to return, defaulted to return all relevant documents
        :param bool query_expansion: whether to expand the query with synonyms
        :param bool pseudo_relevant_feedback: whether to perform pseudo relevant feedback
        :return: the list of documents ranked by descending cosine score
        """
        perceived_relevant_docs = None  # weaker than relevant_docs

        if relevant_docs:
            # if there are relevant docs, we save them to the history recall
            self._history.save_relevant_docs_with_query(query, relevant_docs)
            self._history.flush_to_file()
        else:
            # if there are no relevant docs, we try to find some in the history
            past_relevance_docs = self._history.get_relevant_docs(query)
            if past_relevance_docs.is_exact:
                # if we have exact matches in the history, we use them as if
                # relevant docs were provided
                relevant_docs = set(past_relevance_docs)
                logger.debug(f"Exact relevance found in search history: {relevant_docs}")
            else:
                # otherwise, we use the perceived relevant docs
                perceived_relevant_docs = set(past_relevance_docs)
                logger.debug(f"Fuzzy relevance from search history: {perceived_relevant_docs}")

        logger.info(f"Query expansion: \t{'enabled' if query_expansion else 'disabled'}")
        logger.info(f"Relevant feedback: \t{'enabled' if pseudo_relevant_feedback else 'disabled'}")

        if query_expansion:
            logger.debug(f"Original query vector size: {len(query.get_tokens())}")
            query = self._expand_query(query, relevant_docs)
            logger.debug(f"Query vector size after synonym expansion: {len(query.get_tokens())}")

        eligible_docs = self.get_eligible_docs(query)

        if pseudo_relevant_feedback:
            if not relevant_docs and not perceived_relevant_docs:
                # if we still don't have any perceived relevant docs from history
                # we interpret some ourselves
                perceived_relevant_docs = self._rank(query, eligible_docs, k_count=10)
                logger.debug(f"Perceived relevant docs: {perceived_relevant_docs}")

            query = self._rocchio_expand(query,
                                         relevant_docs if relevant_docs else perceived_relevant_docs,
                                         alpha=0.8)
            logger.debug(f"Query vector size after Rocchio expansion: {len(query.get_tokens())}")

        scored_documents = self._rank(query, eligible_docs, k_count=k_count)

        # intercept the results and put the (initial) relevant docs at the top
        return promote_relevant_docs(relevant_docs, scored_documents)

    def get_eligible_docs(self, query):
        """
        Get the list of documents that are eligible for the given query.

        :param Query query: the query to rank the documents by
        :return: the list of documents ranked by descending cosine score
        """
        working_set = set()
        # guaranteed to return something

        # for groups containing a phrase, the scope is iteratively reduced to
        # documents that contain the phrase only
        first_pass = True

        for group_index, group in enumerate(query.get_token_groups()):
            logger.debug(f"Group {group_index}: {group}")

            doc_with_any_member = set(itertools.chain.from_iterable(
                self.get_docs_containing_member(member) for member in group.members))
            if first_pass:
                working_set = doc_with_any_member
                first_pass = False
            else:
                working_set &= doc_with_any_member

        logger.debug(f"Working set size: {len(working_set)}")

        return working_set

    def _rank(self, query: Query, eligible_documents, k_count=None):
        """
        Find all eligible documents and rank them by cosine score, then by court priority
        for close matches.

        :param Query query: the query to rank the documents by
        :param set[int] eligible_documents: the list of documents that are eligible for the given query
        :param int k_count: the number of top results to return, defaulted to 10
        :return: the list of k documents ranked by descending cosine score
        """
        scored_documents = self._scorer.rank_top_k(query=query,
                                                   document_pool=eligible_documents,
                                                   k_count=k_count,
                                                   with_scores=True)

        return DocumentScorer.rank_with_court_priority(scored_documents,
                                                       self._document_summary_dictionary,
                                                       court_weight=0.3)

    def get_docs_containing_member(self, member):
        """
        Returns a list of document IDs that contains the given term in the given order completely.

        If the term is a phrase, the function will return a list of document IDs that contains the
        phrase exactly.

        Otherwise, the term must be a single token, and the function will return a list of
        document IDs that contains the token.

        :param str member: the member of a query group to search for
        :return: the list of document IDs that contains the given term
        """
        tokens = tokenize_and_reduce(member)
        if len(tokens) > 1:
            # if term is a phrase, get the list of documents that contains the phrase
            matched_docs = self._term_dictionary.get_documents_with_phrase(member)
        else:
            # if term is a single token, get the list of documents that contains the token
            matched_docs = [doc.get_document_id()
                            for doc in self._term_dictionary.get_term_posting(tokens[0])]
        return set(matched_docs)

    def _expand_query(self, query: Query, relevant_docs=None):
        """
        Generates and returns an expanded query from the original query.

        The function first splits the query into its constituent terms,
        and for each term it checks whether it is either a phrase or a
        boolean operator. If it is any of these, the term is ignored, else,
        a list of synonyms for the word is generated using wordnet from
        the nltk.corpus library. This list of synonyms is appended to the
        end of the original query, and a set datastructure is used to
        ensure the synonyms appended are unique and not repeated.

        :param Query query: the original query from the user to expand
        :param set[int] relevant_docs: the list of documents deemed relevant. If None, all documents are considered
        :return Query: the expanded query, a new query string with the synonyms of any free text words
         appended to the end
        """
        expanded_query: Query = query.copy()
        for group_ind, group in enumerate(query.get_token_groups()):
            group: Query.Group
            for member in group.members:
                synonyms: set[str] = generate_synonyms(member)

                for synonym in synonyms:
                    if expanded_query.contains_token_or_phrase(synonym):
                        continue

                    # only if all relevant documents contain the synonym, add it to the query
                    if relevant_docs is None \
                            or len(relevant_docs) == 0 \
                            or all(synonym in self._document_summary_dictionary[doc_id] for doc_id in relevant_docs):
                        # if no relevant documents are specified, add the synonym to the query
                        # OR if all relevant documents contain the synonym, add it to the query
                        expanded_query.add_member_to_group(synonym, group_ind)
                    # otherwise, ignore the synonym

        logger.debug(f"Expanded query: {expanded_query}")
        return expanded_query

    def _rocchio_expand(self, query: Query, relevant_documents, alpha=0.5):
        """
        Expand the initial query using the rocchio formula.

        :param Query query: the query object
        :param set[int] relevant_documents: the list of document IDs that is deemed relevant
        :param float alpha: weight to give the initial query, must be less than 1
        :return:
        """
        centroid = {}
        for doc_id in relevant_documents:
            if doc_id not in self._document_summary_dictionary:
                logger.error(f"Document ID {doc_id} not found in document summary dictionary")
                continue

            summary: DocumentSummary = self._document_summary_dictionary[doc_id]

            for term in summary.get_top_terms(50):
                centroid.setdefault(term, 0)
                centroid[term] += summary.get_term_frequency(term, stem_term=False)

        token_weight = {}

        for term in query.get_tokens():
            token_weight.setdefault(term, 0)
            token_weight[term] += alpha * query.get_token_weight(term)

        for term in centroid:
            token_weight.setdefault(term, 0)
            token_weight[term] += (1 - alpha) * (centroid[term] / len(relevant_documents))

        return Query(token_weight, query.get_token_groups())
