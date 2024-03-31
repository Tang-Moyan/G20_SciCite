from collections import Counter

def frequency_distribution(tokens):
    token_counts = Counter(tokens)
    frequency_dist = Counter()
    for count in token_counts.values():
        frequency_dist[10 ** (count // 10)] += 1
    return frequency_dist
