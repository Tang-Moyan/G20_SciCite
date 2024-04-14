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
from util.allennlp_supplier import retrieve_allennlp_predictions
from util.stemmer import stem
import pandas as pd
import spacy
from nltk.corpus import wordnet as wn

nlp = spacy.load("en_core_web_sm")

SEARCH_HISTORY_FILENAME = './search_history.txt'

# Function to get the synsets of a word
def get_synsets(word):
    '''
    Returns the synsets of a word
    '''
    return wn.synsets(word)

def calculate_similarity(word1, word2):
    '''
    Returns the similarity between two words

    :param str word1: the first word
    :param str word2: the second word
    '''
    synsets1 = get_synsets(word1)
    synsets2 = get_synsets(word2)
    
    max_similarity = 0.0
    
    for synset1 in synsets1:
        for synset2 in synsets2:
            similarity = synset1.path_similarity(synset2)
            if similarity is not None and similarity > max_similarity:
                max_similarity = similarity
    
    return max_similarity

def get_centroid(documents, document_summary_dictionary):
    """
    Get the centroid of the documents

    :param set[int] documents: the set of documents
    :param DocumentSummaryDictionary document_summary_dictionary: the document summary dictionary
    :return: the centroid of the documents
    """
    centroid = {}
    for doc_id in documents:
        if doc_id not in document_summary_dictionary:
            logger.error(f"Document ID {doc_id} not found in document summary dictionary")
            continue

        summary: DocumentSummary = document_summary_dictionary[doc_id]

        for term in summary.get_top_terms(100, min_length=0):
            centroid.setdefault(term, 0)
            centroid[term] += summary.get_term_frequency(term, stem_term=False)

    return centroid

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

    def __init__(self, term_dictionary, document_summary_dictionary, corpus_file_jsonl):
        """
        Creates a new compound query processor with the given term dictionary and cosine score calculator.

        :param TermDictionary term_dictionary: the term dictionary
        :param DocumentSummaryDictionary document_summary_dictionary: mapping of document IDs to their summary
         containing the magnitude of the document vector
        :param str corpus_file_jsonl: the path to the corpus file in jsonl format
        """
        self._scorer = DocumentScorer(term_dictionary, document_summary_dictionary.map_id_to_magnitude())
        self._term_dictionary = term_dictionary
        self._document_summary_dictionary = document_summary_dictionary

        # This is hack, load the corpus (jsonl) into memory (because I need to find the original document string)
        self._corpus = pd.read_json(corpus_file_jsonl, lines=True, dtype={'unique_id': str})


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
        logger.info(f"Query expansion: \t{'enabled' if query_expansion else 'disabled'}")
        logger.info(f"Relevance feedback: \t{'enabled' if pseudo_relevant_feedback else 'disabled'}")

        if query_expansion:
            logger.debug(f"Original query vector size: {len(query.get_tokens())}")
            query = self._expand_query(query, relevant_docs)
            logger.debug(f"Query vector size after synonym expansion: {len(query.get_tokens())}")

        eligible_docs = self.get_eligible_docs(query)

        # Rank the eligible documents
        initial_scored_documents = self._rank(query, eligible_docs, k_count=k_count)

        # Print the cosine scores of the initial results
        print("Initial Results")
        for doc_id, score in initial_scored_documents:
            print(f"{doc_id} {score}")

        if pseudo_relevant_feedback:
            query = self._rocchio_expand(query, eligible_docs, alpha=0.5)
            logger.debug(f"Query vector size after Rocchio expansion: {len(query.get_tokens())}")

        refined_eligible_docs = self.get_eligible_docs(query)
        scored_documents = self._rank(query, refined_eligible_docs, k_count=k_count, use_tfidf=False)

        return scored_documents

    def get_eligible_docs(self, query):
        """
        Get the list of documents that are eligible for the given query.

        :param Query query: the query to rank the documents by
        :return: an unordered set of document IDs that are eligible for the given query
        """
        working_set = set()
        # guaranteed to return something

        # for groups containing a phrase, the scope is iteratively reduced to
        # documents that contain the phrase only
        #first_pass = True

        # for group_index, group in enumerate(query.get_token_groups()):
        #     logger.debug(f"Group {group_index}: {group}")

        #     doc_with_any_member = set(itertools.chain.from_iterable(
        #         self.get_docs_containing_member(member) for member in group.members))
        #     if first_pass:
        #         working_set = doc_with_any_member
        #         first_pass = False
        #     else:
        #         working_set &= doc_with_any_member

        # logger.debug(f"Working set size: {len(working_set)}")

        # For each token in the members of the query, as long as the document contains any of the tokens,
        # it is considered eligible
        for token in query.get_tokens():
            working_set |= {doc.get_document_id() for doc in self._term_dictionary.get_term_posting(token)}

        return working_set

    def _rank(self, query: Query, eligible_documents, k_count=None, use_tfidf=True):
        """
        Find all eligible documents and rank them by cosine score

        :param Query query: the query to rank the documents by
        :param set[int] eligible_documents: the list of documents that are eligible for the given query
        :param int k_count: the number of top results to return, defaulted to 10
        :return: the list of k documents ranked by descending cosine score
        """
        scored_documents = self._scorer.rank_top_k(query=query,
                                                   document_pool=eligible_documents,
                                                   k_count=k_count,
                                                   with_scores=True,
                                                    use_tfidf=use_tfidf)
        return scored_documents

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

    def _rocchio_expand(self, query: Query, initial_documents, alpha=0.5, beta=0.3):
        """
        Expand the initial query using the rocchio formula using the set of initial documents.

        If find the labels of the initial documents and the label of the query.

        If the label of a document is the same as the label of the query, the document is deemed relevant.

        :param Query query: the query object
        :param set[str] initial_documents: the set of initial documents
        :param float alpha: weight to give the initial query, must be less than 1
        :param float beta: weight to give the relevant documents, must be less than 1
        :return:
        """
        initial_documents = [str(doc_id) for doc_id in initial_documents]
        # The weight given to the irrelevant documents
        gamma = 1 - alpha - beta

        documents = []
        # Format the query and documents into (id, string)
        query_string = query.get_query_string()
        documents.append(('query', query_string))

        for doc_id in initial_documents:
            documents.append((doc_id, self._corpus[self._corpus['unique_id'] == doc_id]['string'].values[0]))
        
        print(documents)

        id_label_map = {doc_id: label for doc_id, label in retrieve_allennlp_predictions(documents)}

        print(id_label_map)

        # Get the labels of the documents
        query_label = id_label_map['query']
        print(f"Query label: {query_label}")

        relevant_documents = {doc_id for doc_id, label in id_label_map.items() if label == query_label and doc_id != 'query'}

        irrelevant_documents = {doc_id for doc_id, label in id_label_map.items() if label != query_label and doc_id != 'query'}
        
        for doc_id in relevant_documents:
            print(f"Relevant doc: {doc_id}, Label: {id_label_map[doc_id]}")

        print("-----------")

        for doc_id in irrelevant_documents:
            print(f"Irrelevant doc: {doc_id}, Label: {id_label_map[doc_id]}")

        relevant_centroid = get_centroid(relevant_documents, self._document_summary_dictionary)

        irrelevant_centroid = get_centroid(irrelevant_documents, self._document_summary_dictionary)

        token_weight = {}

        for term in query.get_tokens():
            token_weight.setdefault(term, 0)
            token_weight[term] += alpha * query.get_token_weight(term)

        for term in relevant_centroid:
            token_weight.setdefault(term, 0)
            token_weight[term] += beta * (relevant_centroid[term] / len(relevant_documents))

        # Get the nouns/proper nouns from the query string
        query_nlp = nlp(query_string)
        nouns = [word.text for word in query_nlp if word.pos_ in ['NOUN', 'PROPN']]

        selected_relevant_doc_nouns = set()
        # For each irrelevant document
        for doc_id in irrelevant_documents:
            # Get the document string
            doc_string = self._corpus[self._corpus['unique_id'] == doc_id]['string'].values[0]
            # Get the nouns/proper nouns from the document string
            doc_nlp = nlp(doc_string)
            doc_nouns = [word.text for word in doc_nlp if word.pos_ in ['NOUN', 'PROPN']]

            for noun in nouns:
                for doc_noun in doc_nouns:
                    sim = calculate_similarity(noun, doc_noun)
                    if sim > 0.3:
                        selected_relevant_doc_nouns.add(stem(doc_noun))

        for term in irrelevant_centroid:
            token_weight.setdefault(term, 0)
            if term in selected_relevant_doc_nouns:
                token_weight[term] += gamma * (irrelevant_centroid[term] / len(irrelevant_documents))
            else:
                token_weight[term] -= gamma * (irrelevant_centroid[term] / len(irrelevant_documents))

        return Query(query_string, token_weight, query.get_token_groups())