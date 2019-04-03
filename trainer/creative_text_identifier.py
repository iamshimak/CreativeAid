import pickle
import time
import numpy
import string
import logging
import en_core_web_lg
import os
from spacy.parts_of_speech import *
from models.models import Word, WordPair
from trainer.count_utils import get_sa
from corpus_reader import CorpusReader
from models.models import Corpus
from gensim.models import KeyedVectors
from generator.text_utils import is_clean, clean_sentence, is_qualified, is_stop


class CreativeTextIdentifier(object):
    # TODO make this class as controller and create creative text identifier class and add to spaCy as extension
    #  https://spacy.io/usage/processing-pipelines#custom-components-attributes

    kmeans_path = 'model/mini_batch_kmeans'
    word2vec_path = 'model/glove.840B.300d.bin'

    def __init__(self, word2vec_coverage=0.5):
        logging.info(f'directory: {os.path.dirname(os.path.realpath(__file__))}')
        # TODO word2vec coverage in percentage 0-1
        # TODO word_pair_freq [increase accuracy]
        self.nlp = en_core_web_lg.load()
        # self.mini_batch_kmeans = pickle.load(open(kmeans_path, 'rb'))
        # self.word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

        self.word_pair_freq = pickle.load(open('verb_noun_freq_2019-03-27_23-31-01', 'rb'))
        self.nlp.remove_pipe('ner')

    def identify_with_corpus(self, corpus_reader):
        sentences = []
        for file in corpus_reader.corpus():
            sentences += self._identify(file.contents_lines())
        return sentences

    def identify_with_sentences(self, sentences):
        return self._identify(sentences)

    def _identify(self, sentences):
        creative_sentences = []
        # TODO clean text
        # check for values its different with versions
        nsubj = 429
        dobj = 416

        sentences = [(line.strip(), idx) for idx, line in enumerate(sentences) if is_qualified(line.strip())]
        for sentence, _ in self.nlp.pipe(sentences, as_tuples=True):
            for chunk in sentence.noun_chunks:
                # TODO get average SA score for sentence and decide sentence is literal
                # TODO verb-verb and noun noun relation ?
                if not chunk.root.head.pos == VERB or not (chunk.root.dep == nsubj or chunk.root.dep == dobj):
                    continue

                logging.debug("===========================================================")
                logging.debug("Chunk: {}".format(chunk.doc.text))
                # ==================================================================================
                noun_norm = chunk.root.text if chunk.root.pos == PRON else chunk.root.lemma_
                noun = Word(noun_norm, chunk.root)
                verb = Word(chunk.root.head.lemma_, chunk.root.head)
                word_pair = WordPair(verb, noun)
                logging.debug("Words: verb:{} noun:{}".format(chunk.root.head.lemma_, noun_norm))
                # ==================================================================================
                process_begin_time_0 = time.process_time()
                # word_pair = self.w2v(word_pair)
                logging.debug(
                    "Word vectorized in {} minutes".format((time.process_time() - process_begin_time_0) / 60))

                if not word_pair.has_vector():
                    logging.debug("Vector not found for verb:{} noun:{}".format(
                        word_pair.verb.has_vector(), word_pair.noun.has_vector()))
                    continue
                # ==================================================================================
                process_begin_time_0 = time.process_time()
                # word_pair = self.v2c(word_pair)
                logging.debug("Words Clusters: verb:{} noun:{} time:{}".format(
                    word_pair.verb.cluster, word_pair.noun.cluster,
                    (time.process_time() - process_begin_time_0) / 60))
                # ==================================================================================
                process_begin_time_0 = time.process_time()
                word_pair.sa = get_sa(self.word_pair_freq, word_pair.verb.cluster, word_pair.noun.cluster)
                if not word_pair.is_literal():
                    creative_sentences.append(sentence)

                logging.debug(
                    "SA of words: {} time:{}".format(word_pair.sa,
                                                     (time.process_time() - process_begin_time_0) / 60))
                logging.debug("Word Literal: {}".format(word_pair.is_literal()))

        return creative_sentences


def clean_text(self, text):
    text = text.lower()
    text = ''.join([i for i in text if not i.isdigit()])
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()


def w2v(self, word_pair):
    word_pair.verb.vector = self._w2v(word_pair.verb)
    word_pair.noun.vector = self._w2v(word_pair.noun)
    return word_pair


def _w2v(self, word):
    """
    Identify vector or most similar vector for given word
    :param word: word
    :return: vector
    """
    try:
        vector = self.word2vec.wv.get_vector(word.text_)
    except KeyError:
        try:
            similar = self.word2vec.most_similar(word.text_, topn=1)
            vector = self.word2vec.wv.get_vector(similar[0][0])
        except KeyError:
            vector = None
    return vector


def v2c(self, word_pair):
    word_pair.verb.cluster = self._v2c(word_pair.verb)
    word_pair.noun.cluster = self._v2c(word_pair.noun)
    return word_pair


def _v2c(self, word):
    """
    Predict cluster for given word
    :param word: word
    :return: cluster
    """
    return self.mini_batch_kmeans.predict(numpy.array([word.vector]))[0]


if __name__ == '__main__':
    cr = CorpusReader("C:/Users/ShimaK/PycharmProjects/CreativeAid!/test_corpus/test_generate_corpus/cliche", "")
    ti = CreativeTextIdentifier()
    ti.identify_with_corpus(cr)
