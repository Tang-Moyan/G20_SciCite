class BidirectionalDictionary(dict):
    """
    A bijective mapping that maps each key to a unique value.

    Native supported operations: `[], len`
    """
    def __init__(self, iterable=None):
        if iterable is None:
            super().__init__()
            self._reverse = {}
        else:
            super().__init__(iterable)
            keys, values = zip(*iterable)
            if len(set(keys)) != len(set(values)):
                raise ValueError("The mapping from keys to values is not bijective")
            self._reverse = dict((value, key) for key, value in iterable)

    def __setitem__(self, key, value):
        if value in self._reverse:
            raise ValueError(f"Value {value} already exists in the bijective mapping")
        if key in self:
            raise ValueError(f"Key {key} already exists in the bijective mapping")

        self._reverse[value] = key
        return super().__setitem__(key, value)

    def __getitem__(self, item):
        assert self._reverse[super().__getitem__(item)] == item
        return super().__getitem__(item)

    def pop(self, __key, default=None):
        if __key not in self:
            return default

        value = super().pop(__key)
        del self._reverse[value]
        return value

    def popitem(self):
        popped = super().popitem()
        del self._reverse[popped[1]]
        return popped

    def get_key(self, value):
        """
        Gets the primary key of the mapped value in the forward dictionary
        :param value: the mapped value belonging a key in the forward dictionary
        :return: the key of the mapped value
        """

        assert self[self._reverse[value]] == value
        return self._reverse[value]

    def contains_value(self, value):
        """
        Confirms whether the value exists in the bijective mapping.

        :param value: the mapped value to determine existence of
        :return: whether the value exists in the bijective mapping

        :rtype: bool
        """

        return value in self._reverse.keys()
