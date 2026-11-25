"""
Tiny English → French parallel corpus — handcrafted for teaching.

Enough to learn a non-trivial mapping without downloading WMT.
"""

from __future__ import annotations

# (english, french)
PAIRS: list[tuple[str, str]] = [
    ("hello", "bonjour"),
    ("hi", "salut"),
    ("good morning", "bonjour"),
    ("good evening", "bonsoir"),
    ("good night", "bonne nuit"),
    ("how are you", "comment allez vous"),
    ("i am fine", "je vais bien"),
    ("thank you", "merci"),
    ("thanks a lot", "merci beaucoup"),
    ("you are welcome", "de rien"),
    ("please", "sil vous plait"),
    ("yes", "oui"),
    ("no", "non"),
    ("goodbye", "au revoir"),
    ("see you later", "a plus tard"),
    ("what is your name", "comment vous appelez vous"),
    ("my name is alice", "je m appelle alice"),
    ("my name is bob", "je m appelle bob"),
    ("where is the station", "ou est la gare"),
    ("where is the hotel", "ou est l hotel"),
    ("i love machine learning", "j aime l apprentissage automatique"),
    ("i love deep learning", "j aime l apprentissage profond"),
    ("this is a cat", "c est un chat"),
    ("this is a dog", "c est un chien"),
    ("the cat is black", "le chat est noir"),
    ("the dog is white", "le chien est blanc"),
    ("i am a student", "je suis etudiant"),
    ("i am an engineer", "je suis ingenieur"),
    ("he is a teacher", "il est professeur"),
    ("she is a doctor", "elle est medecin"),
    ("we are friends", "nous sommes amis"),
    ("they are here", "ils sont ici"),
    ("open the door", "ouvre la porte"),
    ("close the window", "ferme la fenetre"),
    ("i want water", "je veux de l eau"),
    ("i want coffee", "je veux du cafe"),
    ("i want tea", "je veux du the"),
    ("the book is on the table", "le livre est sur la table"),
    ("the car is red", "la voiture est rouge"),
    ("the sky is blue", "le ciel est bleu"),
    ("today is monday", "aujourd hui c est lundi"),
    ("tomorrow is tuesday", "demain c est mardi"),
    ("i speak english", "je parle anglais"),
    ("i speak french", "je parle francais"),
    ("do you speak french", "parlez vous francais"),
    ("how much is it", "combien ca coute"),
    ("where do you live", "ou habitez vous"),
    ("i live in paris", "j habite a paris"),
    ("i live in lille", "j habite a lille"),
    ("nice to meet you", "enchante"),
    ("have a nice day", "bonne journee"),
    ("i am hungry", "j ai faim"),
    ("i am thirsty", "j ai soif"),
    ("the weather is nice", "il fait beau"),
    ("it is raining", "il pleut"),
    ("let us go", "allons y"),
    ("come here", "venez ici"),
    ("help me", "aidez moi"),
    ("i understand", "je comprends"),
    ("i do not understand", "je ne comprends pas"),
    ("can you help me", "pouvez vous m aider"),
    ("what time is it", "quelle heure est il"),
    ("i work in artificial intelligence", "je travaille en intelligence artificielle"),
    ("neural networks are powerful", "les reseaux de neurones sont puissants"),
    ("transformers changed nlp", "les transformers ont change le nlp"),
]


# Templated pairs: the same few patterns with many word combinations, so that a held-out split can test whether the
# model *composes* (translates a combination it has never seen) rather than memorises. French agreement makes this
# non-trivial: the article and the adjective follow the noun's gender, the verb follows the subject.
NOUNS = [  # english, french, gender
    ("cat", "chat", "m"), ("dog", "chien", "m"), ("book", "livre", "m"), ("garden", "jardin", "m"),
    ("bike", "velo", "m"), ("car", "voiture", "f"), ("house", "maison", "f"), ("table", "table", "f"),
    ("door", "porte", "f"), ("window", "fenetre", "f"),
]
ADJECTIVES = [  # english, masculine, feminine
    ("black", "noir", "noire"), ("white", "blanc", "blanche"), ("red", "rouge", "rouge"), ("blue", "bleu", "bleue"),
    ("green", "vert", "verte"), ("small", "petit", "petite"), ("big", "grand", "grande"), ("old", "vieux", "vieille"),
]
SUBJECTS = [("i", "je"), ("he", "il"), ("she", "elle"), ("we", "nous"), ("they", "ils")]
VERBS = {  # english base, english 3rd person, french conjugation per subject
    "see": ("sees", {"je": "je vois", "il": "il voit", "elle": "elle voit", "nous": "nous voyons", "ils": "ils voient"}),
    "like": ("likes", {"je": "j aime", "il": "il aime", "elle": "elle aime", "nous": "nous aimons", "ils": "ils aiment"}),
    "open": ("opens", {"je": "j ouvre", "il": "il ouvre", "elle": "elle ouvre", "nous": "nous ouvrons",
                       "ils": "ils ouvrent"}),
}
OPENABLE = {"book", "car", "house", "door", "window"}


def template_pairs() -> list[tuple[str, str]]:
    """'the X is ADJ' (gender agreement) and 'SUBJ VERB the X' (conjugation, elision): 205 pairs."""
    pairs = []
    for en_n, fr_n, g in NOUNS:
        art = "le" if g == "m" else "la"
        for en_a, fr_m, fr_f in ADJECTIVES:
            pairs.append((f"the {en_n} is {en_a}", f"{art} {fr_n} est {fr_m if g == 'm' else fr_f}"))
    for en_s, fr_s in SUBJECTS:
        for verb, (third, conj) in VERBS.items():
            en_v = third if en_s in ("he", "she") else verb
            for en_n, fr_n, g in NOUNS:
                if verb == "open" and en_n not in OPENABLE:
                    continue
                pairs.append((f"{en_s} {en_v} the {en_n}", f"{conj[fr_s]} {'le' if g == 'm' else 'la'} {fr_n}"))
    return pairs


def train_test_split(
    test_fraction: float = 0.2, seed: int = 0, hold_out=None
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Hand-written PAIRS always train; templated pairs are split.

    By default a random `test_fraction` of the templated pairs is held out. With `hold_out` (a predicate on the
    (english, french) pair), exactly the matching pairs are held out instead: a systematic split, e.g. every
    "the window is ..." sentence.

    A held-out pair is moved back to training if one of its words (English or French) would otherwise never be seen in
    training: the test measures composition of known words, not guessing unknown ones.
    """
    import random

    templated = [p for p in template_pairs() if p[0] not in {e for e, _ in PAIRS}]
    if hold_out is None:
        rng = random.Random(seed)
        rng.shuffle(templated)
        n_test = int(round(test_fraction * len(templated)))
        test, rest = templated[:n_test], templated[n_test:]
    else:
        test = [p for p in templated if hold_out(p)]
        rest = [p for p in templated if not hold_out(p)]
    train = list(PAIRS) + rest
    seen = {w for e, f in train for w in (e + " " + f).split()}
    kept = []
    for e, f in test:
        if set((e + " " + f).split()) <= seen:
            kept.append((e, f))
        else:
            train.append((e, f))
            seen |= set((e + " " + f).split())
    return train, kept


PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIAL = [PAD, BOS, EOS, UNK]


def build_vocab(sentences: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    words = set()
    for s in sentences:
        words.update(s.split())
    itos = SPECIAL + sorted(words)
    stoi = {w: i for i, w in enumerate(itos)}
    return stoi, {i: w for w, i in stoi.items()}


def encode(sentence: str, stoi: dict[str, int], add_bos_eos: bool = True) -> list[int]:
    ids = [stoi.get(w, stoi[UNK]) for w in sentence.split()]
    if add_bos_eos:
        return [stoi[BOS]] + ids + [stoi[EOS]]
    return ids


def decode(ids: list[int], itos: dict[int, str]) -> str:
    words = []
    for i in ids:
        w = itos[i]
        if w in (BOS, PAD):
            continue
        if w == EOS:
            break
        words.append(w)
    return " ".join(words)
