#!/usr/bin/python3

import time
import sys
import getopt

import nltk

from scorer.search_engine import SearchEngine
from scorer.document_scorer import DocumentScorer
from structure.relevance_history import RelevanceHistory
from structure.query import Query
from structure.document_summary_dictionary import DocumentSummaryDictionary
from structure.term_dictionary import TermDictionary
from index import DOCUMENT_SUMMARY_FILENAME
from util.logger import logger


HISTORY_RECALL_FILENAME = 'search_history.txt'


def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")


def run_search(dict_file, postings_file, query_string):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    logger.info('Running search on the queries...')
    term_dictionary = TermDictionary(dict_file, postings_file)
    document_summary_dictionary = DocumentSummaryDictionary(DOCUMENT_SUMMARY_FILENAME)

    start_time = time.time()
    search_engine = SearchEngine(term_dictionary, document_summary_dictionary)

    query = Query.parse(query_string)

    # logger.info(f"Relevant docs received: {relevant_docs}")
    try:
        # before submission, set k_count to None
        top_list = search_engine.submit_query(query=query,
                                            k_count=10,
                                            relevant_docs=None,
                                            query_expansion=True,
                                            pseudo_relevant_feedback=False)

        for doc_id, score in top_list:
            print(f"{doc_id} {score}")
        logger.debug(f"Length of results: {len(top_list)}")

    except ValueError as e:
        print(f"Error: {e}")

    logger.info(f"Query processed in {time.time() - start_time:.5} seconds")

    term_dictionary.close()


if __name__ == "__main__":
    nltk.download('wordnet')
    nltk.download('stopwords')

    dictionary_file = postings_file = file_of_queries = file_of_output = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
    except getopt.GetoptError:
        usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-q':
            query = a
        else:
            assert False, "unhandled option"

    if query is None:
        usage()
        sys.exit(2)

    run_search("dictionary", "postings", query)
