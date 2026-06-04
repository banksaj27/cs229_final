import random

ALPHABET = list("ABCD")
ALPHABET_SIZE = len(ALPHABET)

# =========================
# generation
# =========================

def hop(seq, p):
    n = len(seq)
    ref_pos = n - 1
    target = seq[ref_pos]

    for poo in range(p):
        found = -1
        for j in range(ref_pos - 1, -1, -1):
            if seq[j] == target:
                found = j
                break
        if found == -1:
            return None
        if found + 1 >= n:
            return None
        target = seq[found + 1]
        ref_pos = found + 1

    return target


def generate_one_example(p=32, n=256):
    # 1. chain with no consecutive duplicates
    chain = [random.randint(0, ALPHABET_SIZE - 1)]
    for _ in range(p):
        c = random.randint(0, ALPHABET_SIZE - 2)
        if c >= chain[-1]:
            c += 1
        chain.append(c)

    # 2. random left padding + random gaps (so answer position varies)
    min_gap = 3
    slack = n - 1 - min_gap * p
    left_pad = random.randint(0, slack)
    remaining = slack - left_pad
    gaps = [min_gap] * p
    for _ in range(remaining):
        gaps[random.randint(0, p - 1)] += 1

    # 3. place pairs right-to-left
    seq = [None] * n
    seq[n - 1] = chain[0]
    ref = n - 1
    pair_positions = []
    for k in range(p):
        j = ref - gaps[k]
        seq[j] = chain[k]
        seq[j + 1] = chain[k + 1]
        pair_positions.append(j)
        ref = j + 1

    # 4. forbidden zones (sets)
    forbidden = {}
    ref = n - 1
    for k in range(p):
        j = pair_positions[k]
        for pos in range(j + 2, ref):
            forbidden.setdefault(pos, set()).add(chain[k])
        ref = j + 1

    # 5. fill remaining
    for i in range(n):
        if seq[i] is None:
            f = forbidden.get(i, set())
            allowed = [c for c in range(ALPHABET_SIZE) if c not in f]
            seq[i] = random.choice(allowed if allowed else list(range(ALPHABET_SIZE)))

    return "".join(ALPHABET[c] for c in seq), ALPHABET[chain[p]]
    
# =========================
# tokenization
# =========================

VOCABULARY = list("ABCD>") + ["<PAD>"]
VOCAB_SIZE = len(VOCABULARY)
id_to_char = {i: c for i, c in enumerate(VOCABULARY)}
char_to_id = {c: i for i, c in enumerate(VOCABULARY)}
PAD_ID = char_to_id["<PAD>"]
SEP_ID = char_to_id[">"]


def encode(s):
    return [char_to_id[c] for c in s]


def decode(ids):
    return "".join(id_to_char[i] for i in ids if i != PAD_ID)
