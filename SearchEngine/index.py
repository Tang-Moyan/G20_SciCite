#!/usr/bin/python3

import getopt
import time
import pickle
import os
import sys

from itertools import chain, groupby
from math import ceil
from uuid import uuid4
from multiprocessing import cpu_count
from hashlib import md5
from queue import Queue

from portalocker import Lock as FileLock, AlreadyLocked

from structure.document_summary import DocumentSummary
from util.map_reduce import MapReduce
from util import compressor
from util.document_harvester import DocumentHarvester
from util.free_text_parser import tokenize_and_reduce
from util.timer import format_seconds, measure
from util.logger import logger
from structure.document_posting import DocumentPosting
from structure.posting_list import PostingList

INDEX_PATH = "/tmp/index_output/"
'''
Index folder containing intermediate indexes, during the merge process, 
and the final index, after the merge process.
'''
DOCUMENT_EXISTENCE_MARKER = ""
DOCUMENT_SUMMARY_FILENAME = "document_summaries.txt"
SUBPROCESS_COUNT = cpu_count()
BATCH_COUNT = 4


def print_usage():
    logger.info("usage: " + sys.argv[0] + " -i (data jsonl file)")

def build_index(in_dataset):
    """
    build index from documents stored in the input directory,
    then output the dictionary file and postings file
    """
    out_dict = "dictionary"
    out_postings = "postings"

    os.makedirs(INDEX_PATH, exist_ok=True)

    if os.listdir(INDEX_PATH):
        logger.error("ERROR: An existing index folder was detected to be non-empty. "
                     f"All files will now be deleted from '{INDEX_PATH}' before starting a new indexing process.")
        cleanup_index_folder()
        return

    start_time = time.time()

    logger.info("Starting indexing process.\n"
                f"Indexing will be performed at '{INDEX_PATH}'. This process may take a while.\n"
                f"Number of subprocesses: {SUBPROCESS_COUNT} | Number of batches: {BATCH_COUNT}")

    document_harvester = DocumentHarvester(in_dataset)
    num_documents = document_harvester.get_unique_document_count_from_csv()
    document_harvester.entries_to_yield = ceil(num_documents / (SUBPROCESS_COUNT * BATCH_COUNT))

    logger.info(f"{num_documents} documents to be processed.")

    partial_index_filenames = construct(document_generator=document_harvester,
                                        process_count=SUBPROCESS_COUNT,
                                        batches=BATCH_COUNT)

    construction_time = time.time() - start_time

    logger.info(f"Index construction complete, elapsed time: {format_seconds(construction_time)}.\n"
                f"\tSize: {sum(os.path.getsize(f) for f in os.listdir('.') if os.path.isfile(f))} bytes.\n")

    logger.info("Starting bookkeeping operation.")

    bookkeep(partial_index_filenames, document_harvester, out_dict, out_postings)

    bookkeeping_time = time.time() - start_time - construction_time

    logger.info(f"Bookkeeping operation complete, elapsed time: {format_seconds(bookkeeping_time)}.\n"
                f"\tDictionary file of size {os.stat(out_dict).st_size} bytes.\n"
                f"\tPosting list file of size {os.stat(out_postings).st_size} bytes.\n"
                f"\tDocument file of size {os.stat(DOCUMENT_SUMMARY_FILENAME).st_size} bytes.")

    # Clean up the index folder
    cleanup_index_folder()

    logger.info("All operations successfully completed!")
    logger.info(f"Total elapsed time: {format_seconds(time.time() - start_time)}.")


def cleanup_index_folder():
    """
    Deletes all files in the index folder, and then deletes the folder itself.
    """
    for filename in os.listdir(INDEX_PATH):
        os.remove(os.path.join(INDEX_PATH, filename))

    os.rmdir(INDEX_PATH)


def construct(document_generator, process_count, batches=1):
    """
    Constructs the full index in a pseudo-MapReduce fashion.

    It takes in the dataset CSV file, and chunks the dataset into smaller chunks of
    documents. Each chunk is sent to a subprocess to have its (term, document ID, position)
    tuples generated, which are returned to the main process.

    The main process then groups these tuples by term in a dictionary, sorts within each group
    by (document ID, position).

    The dictionary is then chunked into smaller chunks, each chunk containing a partition of
    keys and their values (a list of (document ID, position) tuples). Each chunk is sent to a
    subprocess to have its postings lists constructed, which are written into the index folder.

    :param DocumentHarvester document_generator: the generator that returns documents by chunks
    :param int process_count: the number of processes to use for the construction
    :param int batches: the number of batches each subprocess may make over the dataset.
     The dataset is split into process_count * batches chunks, and each subprocess is given
     one chunk to process at a time.
    :return: None
    """

    # Map: generate (term, document ID, position) tuples
    partition_filenames = set(filename for created in
                              measure(MapReduce.map)(map_function=process_documents_to_term_tuples,
                                                     process_count=process_count,
                                                     number_of_partitions=process_count * batches,
                                                     document_generator=document_generator)
                              for filename in created)

    # Reduce: construct postings lists
    logger.info("Partitioning sorted tuples...")
    partial_index_filenames = measure(MapReduce.reduce)(reduce_function=reduce_partition_to_postings,
                                                        process_count=process_count,
                                                        partition_filenames=partition_filenames)

    return partial_index_filenames


def process_documents_to_term_tuples(documents, number_of_partitions, index_path=INDEX_PATH):
    """
    Processes a chunk of documents, generating the (term, document ID, position) tuples,
    and writes them to a file in the index folder.

    Used by the subprocesses in the MapReduce construction process.

    :param list documents: the list of documents to process
    :param int number_of_partitions: the number of partitions to split the terms into
    :param str index_path: the path to the index folder
    :return: a list of filenames that were created by this function
    """
    tuple_entries = sorted((term, unique_id, position)
                           for unique_id, string in documents
                           for position, term in enumerate(chain([DOCUMENT_EXISTENCE_MARKER],
                                                                 tokenize_and_reduce(string)
                                                                 )))
    # group the term tuples by partition (partition number, compressed term tuples)
    # the partition number is determined by the term's hash
    # compressing the term tuples reduces write time to disk
    partition_groups = [(partition_number, compressor.dumps(list(group)))
                        for partition_number, group in
                        groupby(tuple_entries,
                                key=lambda x: int(md5(x[0].encode()).hexdigest(), 16) % number_of_partitions)]

    groups_queue = Queue()
    for group in partition_groups:
        groups_queue.put(group)

    logger.debug(f"\tProcess {os.getpid()} writing partitions to disk...")

    # for partition, write it to the partition file after a successful lock
    created_partitions = set()

    while not groups_queue.empty():
        partition_number, compressed_entries = groups_queue.get()
        if try_make_partition_file(partition_number, index_path):
            created_partitions.add(partition_number)
        try:
            with FileLock(filename=os.path.join(index_path, f"partition-{partition_number}.pkl"),
                          mode="ab",
                          fail_when_locked=True) as partition_file:
                pickle.dump(compressed_entries, partition_file)
        except AlreadyLocked:
            # if the partition file is locked, put the group back in the queue (FIFO)
            groups_queue.put((partition_number, compressed_entries))

    logger.debug(f"\tProcess {os.getpid()} finished processing {len(documents)} documents.")

    return [f"partition-{partition_number}.pkl" for partition_number in created_partitions]


def try_make_partition_file(partition_number, index_path):
    """
    Attempts to create a partition file with the given partition number.

    :param int partition_number: the partition number
    :param str index_path: the path to the index folder
    :return: True if the file was created, False otherwise
    """
    try:
        open(os.path.join(index_path, f"partition-{partition_number}.pkl"), "xb").close()
        return True
    except FileExistsError:
        return False


def reduce_partition_to_postings(partition_filename):
    """
    Processes a chunk of entries in the partition, which are a sequence of
    (term, list of (term, document ID, position) tuples)
    and constructs the postings lists for each term.

    The posting lists are written to a file in the index folder.

    Used by the subprocesses in the MapReduce construction process.

    :param str partition_filename: the name of the file containing the partition
    :return: the name of the file containing the postings list
    """
    filename = "index-" + uuid4().hex
    with open(os.path.join(INDEX_PATH, filename), "wb") as index_file, \
            open(os.path.join(INDEX_PATH, partition_filename), "rb") as partition_file:
        # read the (term, document ID, position) tuples from the partition file
        tuple_entries = []
        while True:
            try:
                # the pickled file contains a list of (term, document ID, position) tuples
                tuple_entries.extend(compressor.loads(pickle.load(partition_file)))
            except EOFError:
                break
        tuple_entries.sort()  # sort by term, then by document ID, then by position

        # group, first by term, then by document ID
        entries = [(term, list((doc_id, list(group_doc))
                               for doc_id, group_doc in groupby(list(group_term), key=lambda x: x[1])))
                   for term, group_term in groupby(tuple_entries, key=lambda x: x[0])]

        for term, doc_id_groups in entries:
            posting_list = PostingList([DocumentPosting(document_id, [position for _, _, position in data_list])
                                        for document_id, data_list in doc_id_groups])
            # compress the posting list and write it to the index file
            term_entry = (term, compressor.dumps(posting_list.to_serializable_posting_list()))
            pickle.dump(term_entry, index_file)

    logger.debug(f"\tProcess {os.getpid()} finished processing {partition_filename}.")

    return filename


def bookkeep(partial_index_filenames, document_generator, dictionary_filename, posting_filename):
    """
    Writes the term records, including the full term, the document frequency and the
    pointer to the posting list, to the dictionary file and the posting lists of each
    term into the posting file.

    In addition, for each document, the magnitude of the document vector is also
    calculated and written to the document magnitude file.

    :param list[str] partial_index_filenames: the list of filenames containing the partial indices
    :param DocumentHarvester document_generator: the generator of documents
    :param str dictionary_filename: the dictionary filename
    :param str posting_filename: the posting filename

    :rtype: None
    """
    posting_file = open(posting_filename, "wb")  # the final posting file to be used by the search engine
    dictionary_entries = []
    summaries: dict[str, DocumentSummary] = {}  # document ID -> document summary

    # iterate through each mini index file
    for filename in partial_index_filenames:
        with open(os.path.join(INDEX_PATH, filename), "rb") as mini_index_file:
            # every mini index file contains a list of (term, posting list) tuples
            # the terms will never overlap between files
            while True:
                try:
                    term, compressed_posting_list = pickle.load(mini_index_file)
                except EOFError:
                    # nothing left to unpickle
                    break

                # transfer the posting list to the final posting file
                seek_position = posting_file.tell()  # record destination of the pickling operation
                pickle.dump(compressed_posting_list, posting_file)

                serialized_posting_list = compressor.loads(compressed_posting_list)
                dictionary_entries.append(f"{term} {len(serialized_posting_list)} {seek_position}")

                if term == DOCUMENT_EXISTENCE_MARKER:
                    # don't include the document existence marker in the magnitude calculation
                    continue

                # for each term in the entry, compute the weighted frequency of that term in the document
                posting_list: PostingList = PostingList.deserialize(serialized_posting_list)
                posting_list_iter: PostingList.PostingListIterator = posting_list.to_iterator()

                for posting in posting_list_iter:
                    posting: DocumentPosting
                    doc_id = str(posting.get_document_id())

                    if doc_id not in summaries:
                        # the document has not been seen before
                        summaries[doc_id] = DocumentSummary(doc_id)

                    summaries[doc_id].add_term(term, posting.get_term_frequency())

    posting_file.close()

    dictionary_entries.sort()  # sorts by term
    with open(dictionary_filename, "w", encoding="utf-8") as dictionary_file:
        dictionary_file.write("\n".join(dictionary_entries))

    # write the document summaries to document summary file
    with open(DOCUMENT_SUMMARY_FILENAME, "wb") as document_file:
        compressor.dump(summaries, document_file)


if __name__ == "__main__":
    input_corpus_jsonl = output_file_dictionary = output_file_postings = None

    try:
        opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
    except getopt.GetoptError:
        print_usage()
        sys.exit(2)

    for o, a in opts:
        if o == '-i':  # input directory
            input_corpus_jsonl = a
        else:
            assert False, "unhandled option"

    if input_corpus_jsonl is None:
        print_usage()
        sys.exit(2)

    build_index(input_corpus_jsonl)
