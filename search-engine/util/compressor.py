import pickle
import bz2
import random


def loads(bytestream):
    """
    Loads from the bytestream a compressed chunk of bytestream data, then decompresses
    and reconstitutes the Python object.

    :param bytestream: the bytestream to read the pickled object from
    :return: the reconstituted Python object
    """
    return pickle.loads(bz2.decompress(bytestream))


def load(file):
    """
    Loads from the file a compressed chunk of bytestream data, then decompresses
    and reconstitutes the Python object.

    :param file: the file to read the pickled object from
    :return: the reconstituted Python object
    """
    return pickle.loads(bz2.decompress(pickle.load(file)))


def dumps(obj):
    """
    Dumps a pickleable Python object into a bytestream, then compresses it.

    :param obj: the object to be pickled
    :return: the compressed bytestream
    """
    return bz2.compress(pickle.dumps(obj))


def dump(obj, file):
    """
    Dumps a pickleable Python object into the specified file and transforms it into a bytestream,
    writing it into the specified file.

    :param obj: the object to be pickled
    :param file: the file to write the pickled object into
    :return:
    """
    pickle.dump(bz2.compress(pickle.dumps(obj), compresslevel=9), file)
