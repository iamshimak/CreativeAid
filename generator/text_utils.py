from string import punctuation
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.parts_of_speech import *


def is_clean(word):
    """
    Check word proper input for NLP process
    https://spacy.io/api/annotation#pos-tagging
    :param word: spacy token
    :return: Boolean
    """
    return word.is_stop or word.is_space or word.pos in [NUM, SYM, PUNCT, DET, CCONJ, CONJ, SCONJ, X]


def clean_sentence(sentence):
    sentence = sentence.lower()
    sentence = ''.join([i for i in sentence if not i.isdigit()])
    sentence = ' '.join([word for word in sentence.split(' ') if word not in STOP_WORDS])
    sentence = sentence.translate(str.maketrans('', '', punctuation))
    return sentence.strip()


def is_qualified(sentence):
    return len(sentence.split(' ')) > 3
