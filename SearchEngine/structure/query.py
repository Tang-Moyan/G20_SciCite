import itertools

from util.stemmer import stem
from util.free_text_parser import tokenize, tokenize_and_reduce


class Query:
    """
    A query object that stores the token frequencies and token groups of a query.
    """

    class Group:
        """
        A group of terms that are joined by disjunction (OR).
        """

        def __init__(self, is_phrase, members):
            """
            Creates a new group of terms.

            :param bool is_phrase: whether the group is a phrase
            :param set[str] members: the members of the group
            """
            self._is_phrase = is_phrase
            self.members = members

        def is_phrase(self):
            """
            Returns whether the group is a phrase.
            :rtype: bool
            """
            return self._is_phrase

        def __str__(self):
            return f"{'Phrase' if self._is_phrase else 'Free Text'} | Members: {self.members}"

        def __repr__(self):
            return str(self)

        def copy(self):
            """
            Returns a copy of the group.
            :rtype: Group
            """
            return Query.Group(self._is_phrase, self.members.copy())

        def __hash__(self):
            return hash((self._is_phrase, frozenset(self.members)))

        def __eq__(self, other):
            if isinstance(other, Query.Group):
                return self._is_phrase == other._is_phrase and self.members == other.members
            return False

    def __init__(self, query_string, tokens: dict = None, literal_groups: list = None):
        """
        Creates a new query object.

        A query object stores the token frequencies and literal groups of a query.

        The terms in each literal group are not tokenized. This is because phrases must be
        searched as a whole, and not tokenized.

        :param dict[str, int] tokens: mapping of terms to its term frequency. The term keys are stemmed.
        :param list[Group] literal_groups: groups of terms that are joined by conjunction (AND).
         Each group is a tuple of a boolean indicating whether the group is a phrase, and a list of terms.
        """
        self._query_string = query_string
        self._token_weight = tokens if tokens else {}
        self._token_groups = literal_groups if literal_groups else []

    def __str__(self):
        return f"\n\tToken Weight: {self._token_weight} " \
               f"\n\tGroups: {self._token_groups}"

    @staticmethod
    def parse(query: str):
        """
        Takes a query string and returns a query object.

        :param query: the query string
        :return: the query object
        """
        # first, split the query into term groups (separated by AND)
        segments = query.split("AND")
        token_frequency = {}
        token_groups: list[Query.Group] = []

        for segment in segments:
            segment = segment.strip()
            tokens: list[str]

            if segment[0] == "\"" and segment[-1] == "\"":
                # if the segment start and end with quotes, it is a phrase
                segment = segment.strip("\"")

                # the group is added as a single untouched phrase
                token_groups.append(Query.Group(True, {segment}))

            else:  # if the term is a free text
                # each individual token is added to the group
                # we cannot split by whitespace because one word may be split into multiple tokens,
                # or may contain punctuation
                token_groups.append(Query.Group(False, set(tokenize(segment))))

            tokens = tokenize_and_reduce(segment)
            for token in tokens:
                token_frequency[token] = token_frequency.get(token, 0) + 1

        # create a new query object using the parsed tokens and token groups
        return Query(query, token_frequency, token_groups)

    def copy(self):
        """
        Returns a copy of the query object.

        :rtype: Query
        """
        return Query(self._query_string, self._token_weight.copy(), [group.copy() for group in self._token_groups])

    def get_tokens(self):
        """
        Returns a list of all the terms in the query.

        :rtype: list[str]
        """
        print("Token weights", self._token_weight)
        return list(self._token_weight.keys())

    def get_token_groups(self):
        """
        Returns a list of all the token groups in the query.

        :rtype: list[Group]
        """
        return self._token_groups

    def get_token_weight(self, term):
        """
        Returns the weight of the given term.

        :param str term: the term to get the term frequency of
        :rtype: int
        """
        if term in self._token_weight:
            return self._token_weight[term]
        return 0
    
    def get_query_string(self):
        """
        Returns the query string.

        :rtype: str
        """
        return self._query_string

    def contains_token_or_phrase(self, term):
        """
        Checks if the query contains the given token or phrase.

        :param str term: the term to check
        :rtype: bool
        """
        return any(term in group.members for group in self._token_groups) or stem(term) in self._token_weight

    def increment_token_frequency(self, token, increment=1):
        """
        Increments the given term frequency by a default value of 1.

        :param token: term to add
        :param int increment: amount to increment the term frequency by
        :return: None
        """
        self._token_weight.setdefault(token, 0)
        self._token_weight[token] += increment

    def add_tokenized_group(self, group):
        """
        Tokenizes each term in the group and adds the given group of terms to the term groups list.
        Tokenizes the terms before adding them.

        :param list[str] group: list of terms to add
        :return:
        """
        self._token_groups.append(set(itertools.chain(*[stem(term) for term in group])))

    def add_member_to_group(self, member, group_index):
        """
        Adds the given term to the given term group. Cleans the term before
        adding it.

        :param member: term to add
        :param group_index: index of the term group to add the term to
        :return: None
        """
        if group_index >= len(self._token_groups):
            raise ValueError("Invalid group index")

        self._token_groups[group_index].members.add(member)
        member = " ".join(tokenize(member.strip().strip("\"").lower()))

        for token in tokenize_and_reduce(member):
            self.increment_token_frequency(token)

    def __eq__(self, other):
        """
        Checks if the given query object is equal to this query object.

        :param Query other: the other query object
        :return: True if the two query objects are equal, False otherwise
        """
        return self._token_weight == other._token_weight and self._token_groups == other._token_groups

    def __hash__(self):
        return hash((frozenset(self._token_weight.items()), frozenset(self._token_groups)))
