from structure.bidirectional_dictionary import BidirectionalDictionary
from structure.positional_posting import PositionalPosting
from structure.posting_list import PostingList
from structure.term_record import TermRecord
from util import compressor
from util.free_text_parser import tokenize_and_reduce


class TermDictionary:
    """
    The TermDictionary stores all the term records in memory and retrieves the term's posting
    on demand by reading from the posting list file.

    The posting file is immediately opened on creation of the TermDictionary.

    Use `get_term_record(term)` to retrieve the term record, containing the actual term word, the
    document frequency and the byte position of the posting file where the term's posting list
    occurs.

    Use `get_term_posting(term)` to fetch the posting list of the term, performed by seeking the
    position of the term's posting list occurrence in the posting list file and deserializing the
    `PostingList` object at that location.

    Use `close()` to close the posting list file. No more calls to `get_term_posting()` may be
    invoked after this operation.
    """

    def __init__(self, dictionary_filename, posting_list_filename):
        """
        Loads the entire term dictionary into memory, and only loads
        the required posting list on demand

        :param str dictionary_filename: the file name of the dictionary to load
        :param str posting_list_filename: the posting list file object
        """
        self._posting_list_file = open(posting_list_filename, "rb")

        self._term_mapping: BidirectionalDictionary[str, TermRecord] = BidirectionalDictionary()

        with open(dictionary_filename, "r", encoding='utf-8') as file:
            while True:
                try:
                    line_record = next(file)
                    term, freq, seek_position = line_record.split(" ")
                    self._term_mapping[term] = TermRecord(term, int(freq), int(seek_position))
                    # maps the word to the record
                    # cocoa --> (term: "cocoa", freq: 3, seek_position: 0)
                except StopIteration:
                    break

    def get_term_record(self, term):
        """
        Returns the record for the term.

        :param term: the word to find the record of
        :rtype: TermRecord
        """
        return self._term_mapping[term]

    def __contains__(self, item):
        return item in self._term_mapping

    def get_term_posting(self, term):
        """
        Retrieves the posting list by seek position specified by the term record in the dictionary.

        An error will be raised if the term does not exist in the dictionary.

        :param term: the term whose posting list is desired
        :return: the entire posting list for the term
        :rtype: PostingList
        """
        try:
            term_record = self.get_term_record(term.lower())
        except KeyError:
            return PostingList()

        self._posting_list_file.seek(term_record.get_seek_position())
        serialized_posting_list = compressor.load(self._posting_list_file)
        return serialized_posting_list.to_posting_list()

    def get_universal_posting_list(self):
        """
        Retrieves the posting list of all documents in the corpus.

        :rtype: PostingList
        """
        return self.get_term_posting("")

    def get_documents_with_phrase(self, phrase):
        """
        Retrieves all the documents that contain the given phrase.

        :rtype: Iterable[int]
        """
        tokens = tokenize_and_reduce(phrase)

        posting_lists_iterators: list[PostingList.PostingListIterator]
        posting_lists_iterators = [self.get_term_posting(token).to_iterator() for token in tokens]

        positional_postings = PositionalPosting.merge_posting_lists(posting_lists_iterators)

        documents_with_phrase = []

        if len(tokens) == 1:
            # if the phrase is only one word, then just return the document IDs
            return [doc.get_document_id() for doc in positional_postings]

        for doc in positional_postings:
            positions = doc.get_positions()
            # sliding window
            consecutive_positions = 1
            previous = positions[0]
            for i in range(1, len(positions)):
                current = positions[i]
                # if the positions are not consecutive. or the priorities are not consecutive
                # then reset the window
                if current.get_position() != previous.get_position() + 1 \
                        or current.get_priority() != previous.get_priority() + 1:
                    consecutive_positions = 1
                else:
                    consecutive_positions += 1
                    if consecutive_positions == len(tokens):
                        # if window length == phrase length, the strictest match achieved
                        documents_with_phrase.append(doc.get_document_id())
                        break

                previous = current

        # calculate the cosine score for each document
        return documents_with_phrase

    def __iter__(self):
        return iter(self._term_mapping)  # iterates through the keys

    def __len__(self):
        return len(self._term_mapping)

    def close(self):
        """
        Closes the posting list file.

        No more calls to `get_term_posting()` may be invoked after this operation.

        :rtype: None
        """

        self._posting_list_file.close()
