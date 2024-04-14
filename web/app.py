from flask import Flask, render_template, url_for, jsonify, request
from flask_fontawesome import FontAwesome
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SearchEngine'))


from SearchEngine.scorer.search_engine import SearchEngine
from SearchEngine.structure.query import Query
from SearchEngine.structure.document_summary_dictionary import DocumentSummaryDictionary
from SearchEngine.structure.term_dictionary import TermDictionary
from SearchEngine.index import DOCUMENT_SUMMARY_FILENAME

app = Flask(__name__)
fa = FontAwesome(app)

dict_file = f'{os.path.dirname(__file__)}/../SearchEngine/dictionary'
postings_file = f'{os.path.dirname(__file__)}/../SearchEngine/postings'
corpus_file_jsonl = f'{os.path.dirname(__file__)}/../data/corpus/experimental-corpus.jsonl'

term_dictionary = TermDictionary(dict_file, postings_file)
document_summary_dictionary = DocumentSummaryDictionary(f'{os.path.dirname(__file__)}/../SearchEngine/{DOCUMENT_SUMMARY_FILENAME}')

# Verify that the files exist
assert os.path.exists(dict_file)
assert os.path.exists(postings_file)
assert os.path.exists(corpus_file_jsonl)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/search', methods=['POST', 'GET'])
def get_search_results():
    query = request.form['query']

    search_engine = SearchEngine(term_dictionary, document_summary_dictionary, corpus_file_jsonl)

    query = Query.parse(query)

    results = []

    # logger.info(f"Relevant docs received: {relevant_docs}")
    try:
        # before submission, set k_count to None
        top_list = search_engine.submit_query(query=query,
                                            k_count=10,
                                            relevant_docs=None,
                                            query_expansion=False,
                                            pseudo_relevant_feedback=True)

        for doc_id, score in top_list:
            results.append({'id': doc_id, 'citation': score})

    except ValueError as e:
        print(f"Error: {e}")

    return jsonify({'message': results})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True) 