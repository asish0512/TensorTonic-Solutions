def compute_pair_count(corpus: list[list[int]]):
    pair_counts = {}
    for seq in corpus:
        for i in range(len(seq) - 1):
            pair = (seq[i], seq[i+1])
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    return pair_counts


def merge_in_sequence(seq, pair, new_id):
    new_seq = []
    i = 0
    while i < len(seq):
        if i < len(seq) - 1 and (seq[i], seq[i+1]) == pair:
            new_seq.append(new_id)
            i += 2          # skip both merged elements
        else:
            new_seq.append(seq[i])
            i += 1
    return new_seq


def expand(token_id, vocab_map):
    """ Fully expand a token id into its base byte sequence."""
    if token_id < 256:
        return [token_id]
    return vocab_map[token_id]


def merge_most_frequent(pair: tuple, byte_corpus: list[list[int]], vocab: list, merges: list, vocab_map: dict):
    a, b = pair
    new_id = 256 + len(merges)

    # 1. replace the pair in every sequence
    byte_corpus[:] = [merge_in_sequence(seq, pair, new_id) for seq in byte_corpus]

    # 2. record the merge rule
    merges.append([a, b, new_id])

    # 3. record the vocab entry as a FULLY EXPANDED byte sequence
    expanded = expand(a, vocab_map) + expand(b, vocab_map)
    vocab.append([new_id, expanded])
    vocab_map[new_id] = expanded

    return new_id


def train_bpe(corpus: list[str], vocab_size: int) -> dict:
    """
    Returns a dictionary of learned vocab entries and ordered merges.
    """
    vocab, merges = [], []
    vocab_map = {}  # id -> expanded byte sequence, used for building new vocab entries
    byte_corpus = [list(s.encode()) for s in corpus]

    while len(vocab) + 256 < vocab_size:
        pair_counts = compute_pair_count(byte_corpus)
        if not pair_counts:          # <-- the actual "no pairs left" check
            break
        best_pair = max(pair_counts, key=lambda p: (pair_counts[p], expand(p[0],vocab_map), expand(p[1], vocab_map)))
        merge_most_frequent(best_pair, byte_corpus, vocab, merges, vocab_map)

    return {"vocab": vocab, "merges": merges}