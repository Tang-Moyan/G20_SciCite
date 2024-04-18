import os
import pickle
from concurrent.futures.process import ProcessPoolExecutor

from util.document_harvester import DocumentHarvester


class MapReduce:

    @staticmethod
    def map(map_function, process_count, document_generator, number_of_partitions):
        """
        Map the documents to (term, document ID, position) tuples.

        :param function map_function: the function to use to map the documents to (term, document ID, position) tuples
        :param int process_count: the number of processes to use
        :param DocumentHarvester document_generator: the generator that yields the documents to be processed
        :param int number_of_partitions: the number of partitions to create
        :return: a list of the results of the map function by each subprocess
        """
        futures = []
        with ProcessPoolExecutor(max_workers=process_count) as executor:
            # for each subprocess, send a chunk of the dataset to be processed
            # into (term, document ID, position) tuples
            for documents in document_generator:
                futures.append(executor.submit(map_function,
                                               documents,
                                               number_of_partitions))

        return [future.result() for future in futures]

    @staticmethod
    def reduce(reduce_function, process_count, partition_filenames):
        """
        Reduce the (term, document ID, position) tuples in each partition to a posting list.
        :param function reduce_function: the function to use to reduce the (term, document ID, position) tuples
            in each partition to a posting list
        :param int process_count: the number of processes to use
        :param list[str] partition_filenames: the names of the partitions to be reduced
        :return: a list of the results of the reduce function by each subprocess
        """
        futures = []

        with ProcessPoolExecutor(max_workers=process_count) as executor:
            # for each subprocess, send a partition to be processed to a posting list
            for partition_name in partition_filenames:
                futures.append(executor.submit(reduce_function, partition_name))

        return [future.result() for future in futures]


