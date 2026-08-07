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
