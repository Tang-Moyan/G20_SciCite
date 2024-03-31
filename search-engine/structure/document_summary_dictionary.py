from util import compressor
from index import DOCUMENT_SUMMARY_FILENAME


class DocumentSummaryDictionary(dict):
    """
    This class is used to represent a document summary dictionary.

    This object can be used like a normal dictionary, but it will have to
    load the document summary from the given file on creation.
    """

    def __init__(self, summary_filename=DOCUMENT_SUMMARY_FILENAME):
        """
        Creates a new document summary dictionary by loading from the given file.

        The document summary can be obtained by this dictionary by using the document id as the key.

        :param str summary_filename: the file name of the summary file
        """
        self._summary_filename = summary_filename
        with open(summary_filename, "rb") as file:
            super().__init__(compressor.load(file))

    def get_number_of_documents(self):
        """
        Returns the number of documents in the collection.

        :rtype: int
        """
        return len(self)

    def map_id_to_magnitude(self):
        """
        Returns a dictionary that maps each document id to the magnitude of the document.

        :rtype: dict[int, float]
        """
        return {doc_id: self[doc_id].get_magnitude() for doc_id in self}
