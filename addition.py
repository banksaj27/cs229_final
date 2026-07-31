"""n-ary addition task, following Saunshi et al. 2025 (arXiv:2502.17416) §2.1.

Sum n operands of 3 digits each, sampled uniformly in [0, 999] and zero-padded
to 3 characters. Example from the paper (n=4):

    Input:  "315 + 120 + 045 + 824 ="      Output: "1304"

We train on a uniform mixture over N_TRAIN operand counts and evaluate per-n,
exactly mirroring how the p-hop task mixes p values. One deviation for
batching convenience: the answer is always zero-padded to ANS_LEN = 5 digits
(max possible sum 32*999 = 31968), giving a fixed-length answer region.

Vocabulary: digits 0-9, '+', '=', <PAD>  (size 13)
"""
import random

N_TRAIN = (2, 4, 8, 16, 32)          # training mixture (paper's)
N_EVAL = (2, 4, 8, 16, 24, 32)       # per-n eval; 24 is unseen at train time

# operand width: paper uses 3 digits; 2 digits halves carry depth per operand,
# a difficulty knob for smaller models. set_op_digits() reconfigures the module.
OP_DIGITS = 3
ANS_LEN = 5


def set_op_digits(d):
    global OP_DIGITS, ANS_LEN
    OP_DIGITS = d
    ANS_LEN = len(str(max(N_EVAL) * (10 ** d - 1)))

DIGITS = [str(i) for i in range(10)]
VOCABULARY = DIGITS + ["+", "=", "<PAD>"]
VOCAB_SIZE = len(VOCABULARY)
char_to_id = {c: i for i, c in enumerate(VOCABULARY)}
id_to_char = {i: c for c, i in char_to_id.items()}
PAD_ID = char_to_id["<PAD>"]


def generate_example(n_operands):
    """returns (prompt_ids, answer_ids); answer is always ANS_LEN tokens"""
    hi = 10 ** OP_DIGITS - 1
    ops = [random.randint(0, hi) for _ in range(n_operands)]
    s = str(sum(ops)).zfill(ANS_LEN)
    prompt = []
    for i, v in enumerate(ops):
        if i:
            prompt.append("+")
        prompt.extend(str(v).zfill(OP_DIGITS))
    prompt.append("=")
    return [char_to_id[c] for c in prompt], [char_to_id[c] for c in s]


def seq_len_for(n_operands):
    # operands: OP_DIGITS*n chars, separators: n-1, '=': 1, answer: ANS_LEN
    return (OP_DIGITS + 1) * n_operands + ANS_LEN
