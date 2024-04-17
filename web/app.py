from flask import Flask, render_template, url_for, jsonify, request
from flask_fontawesome import FontAwesome
import os
import sys
import json

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'SearchEngine'))

dict_file = f'{os.path.dirname(__file__)}/../SearchEngine/dictionary'
postings_file = f'{os.path.dirname(__file__)}/../SearchEngine/postings'
corpus_file_jsonl = f'{os.path.dirname(__file__)}/../data/corpus/allennlp-data.jsonl'
word2vec_file_location = f'{os.path.dirname(__file__)}/../data/word2vec.model'

assert os.path.exists(dict_file)
assert os.path.exists(postings_file)
assert os.path.exists(corpus_file_jsonl)
assert os.path.exists(word2vec_file_location)


from SearchEngine.scorer.search_engine import SearchEngine
from SearchEngine.structure.query import Query
from SearchEngine.structure.document_summary_dictionary import DocumentSummaryDictionary
from SearchEngine.structure.term_dictionary import TermDictionary
from SearchEngine.index import DOCUMENT_SUMMARY_FILENAME
from SearchEngine.util.allennlp_supplier import retrieve_allennlp_prediction, retrieve_allennlp_predictions

app = Flask(__name__)
fa = FontAwesome(app)

term_dictionary = TermDictionary(dict_file, postings_file)
document_summary_dictionary = DocumentSummaryDictionary(f'{os.path.dirname(__file__)}/../SearchEngine/{DOCUMENT_SUMMARY_FILENAME}')

search_engine = SearchEngine(term_dictionary, document_summary_dictionary, corpus_file_jsonl, word2vec_file_location)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def get_search_results():
    query = request.form['query']

    queryLabel = retrieve_allennlp_prediction(query)

    query = Query.parse(query)

    results = []

    # logger.info(f"Relevant docs received: {relevant_docs}")
    try:
        # before submission, set k_count to None
        top_list = search_engine.submit_query(query=query, k_count=20)

        for doc_id, score in top_list:
            citation = search_engine.get_text(doc_id)
            results.append({'id': doc_id, 'citation': citation, 'score': score})

    except ValueError as e:
        print(f"Error: {e}")

    return jsonify({'message': results, 'queryLabel': queryLabel})

@app.route('/predict', methods=['POST'])
def get_prediction():
    # List of documents
    doc_ids = json.loads(request.form['eligibleDocs'])
    # Get the text of each document
    docs = [(doc_id, search_engine.get_text(doc_id)) for doc_id in doc_ids]

    doc_labels = {doc_id: label for doc_id, label in retrieve_allennlp_predictions(docs)}

    return jsonify({'message': doc_labels})

@app.route('/rocchio', methods=['POST'])
def get_rocchio_results():
    query = request.form['query']
    query = Query.parse(query)

    eligible_docs = json.loads(request.form['eligibleDocs'])

    # logger.info(f"Relevant docs received: {relevant_docs}")
    try:
        query, predicted_labels = search_engine.rocchio_expand(query, eligible_docs, alpha=1.0, beta=0.75, gamma=0.15, return_labels=True)

        # Filter out the query id
        predicted_labels = {doc_id: label for doc_id, label in predicted_labels.items() if doc_id != 'query'}

        refined_results = search_engine.get_eligible_docs(query)
        scored_documents = search_engine._rank(query, refined_results, k_count=15, use_tfidf=False)

        results = []
        for doc_id, score in scored_documents:
            citation = search_engine.get_text(doc_id)
            label = retrieve_allennlp_prediction(citation)
            results.append({'id': doc_id, 'citation': citation, 'score': score, 'label': label})

    except ValueError as e:
        print(f"Error: {e}")

    return jsonify({'message': results, 'previous_predicted_labels': predicted_labels})


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080)