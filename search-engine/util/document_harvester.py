import pandas as pd
from datetime import datetime


class DocumentHarvester:
    """
    This class is used to yield documents from a given jsonl file.
    
    At any point during the yielding, this class provides statistics about the
    unique number of documents yielded.

    It is not reusable, once all documents have been yielded, it will raise a
    StopIteration exception.
    """

    def __init__(self, jsonl_file_path, entries_to_yield=1):
        """
        :param str jsonl_file_path: the path to the jsonl file to be harvested
        """
        self._jsonl_file_path = jsonl_file_path
        self.entries_to_yield = entries_to_yield

        # a set of unique document IDs
        self.all_unique_ids = set()

        self._number_of_entries_collected = 0

    def get_unique_document_count_from_csv(self):
        """
        :return: the number of unique CSV entries
        :rtype: int
        """
        reader = pd.read_json(self._jsonl_file_path, lines=True)

        return len(set(reader["unique_id"]))

    def __iter__(self):
        """
        Yields documents equals to the number of entries to yield.

        Specified by the entries_to_yield parameter in the constructor, the
        generator will yield a list of documents equal to the number of entries.

        If there are less than the number of entries to yield, the generator
        will yield the remaining documents.

        :return: a generator that yields documents from the CSV file, which when iterated
            over, yields a list of documents equal to the number of entries to yield
        """

        df = pd.read_json(self._jsonl_file_path, lines=True)

        entries = []
        for unique_id, section_name, content, source in df[["unique_id", "sectionName", "string", "source"]].values:
            self._number_of_entries_collected += 1

            self.all_unique_ids.add(unique_id)

            entries.append((unique_id,
                            section_name,
                            content,
                            source))

            if len(entries) == self.entries_to_yield:
                yield entries
                entries = []

        if entries:
            yield entries

    def get_unique_document_ids(self):
        """
        :return: a set of unique document IDs
        :rtype: set[str]
        """
        return self.all_unique_ids

    def get_number_of_entries_collected(self):
        """
        :return: the number of entries processed
        :rtype: int
        """
        return self._number_of_entries_collected
