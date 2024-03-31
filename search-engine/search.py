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


def run_search(dict_file, postings_file, queries_file, results_file):
    """
    using the given dictionary file and postings file,
    perform searching on the given queries file and output the results to a file
    """
    logger.info('Running search on the queries...')
    term_dictionary = TermDictionary(dict_file, postings_file)
    document_summary_dictionary = DocumentSummaryDictionary(DOCUMENT_SUMMARY_FILENAME)
    write_file = open(results_file, 'w')

    start_time = time.time()
    search_engine = SearchEngine(term_dictionary, document_summary_dictionary)

    with open(queries_file, 'r') as queries_file:
        query_str = queries_file.readline().strip()
        query = Query.parse(query_str)

        relevant_docs = set(int(doc_id) for doc_id in queries_file)

        logger.info(f"Relevant docs received: {relevant_docs}")
        try:
            # before submission, set k_count to None
            top_list = search_engine.submit_query(query=query,
                                                  relevant_docs=relevant_docs,
                                                  query_expansion=True,
                                                  pseudo_relevant_feedback=True)

            write_file.write(' '.join(str(e) for e in top_list))
            logger.debug(f"Length of results: {len(top_list)}")

        except ValueError as e:
            print(f"Error: {e}")

    logger.info(f"Query processed in {time.time() - start_time:.5} seconds")

    write_file.close()

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
        if o == '-d':
            dictionary_file = a
        elif o == '-p':
            postings_file = a
        elif o == '-q':
            file_of_queries = a
        elif o == '-o':
            file_of_output = a
        else:
            assert False, "unhandled option"

    if dictionary_file is None or postings_file is None or file_of_queries is None or file_of_output is None:
        usage()
        sys.exit(2)

    run_search(dictionary_file, postings_file, file_of_queries, file_of_output)
