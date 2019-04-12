import pickle
import time
import numpy
import string
import logging
import os
from spacy.parts_of_speech import *
from creativeaid.nlp.nlp import NLP
from creativeaid.models.models import Token, WordPair
from creativeaid.identifier.count_utils import get_sa, get_sps
from creativeaid.corpus_reader import CorpusReader
from creativeaid.generator.text_utils import is_qualified


class CreativeTextIdentifier(object):
    # TODO make this class as controller and create creative text identifier class and add to spaCy as extension
    #  https://spacy.io/usage/processing-pipelines#custom-components-attributes

    def __init__(self, kmeans_path='model/mini_batch_kmeans'):
        logging.info(f'directory: {os.path.dirname(os.path.realpath(__file__))}')
        # TODO word2vec coverage in percentage 0-1
        # TODO word_pair_freq [increase accuracy]
        self.nlp = NLP()
        # self.mini_batch_kmeans = pickle.load(open(kmeans_path, 'rb'))
        self.word_pair_freq = pickle.load(open('verb_noun_freq_2019-03-27_23-31-01', 'rb'))

    def identify_with_corpus(self, corpus_reader):
        sentences = []
        for file in corpus_reader.corpus():
            sentences += self._identify(file.contents_lines())
        return sentences

    def identify_with_sentences(self, sentences):
        return self._identify(sentences)

    def _identify(self, sentences):
        creative_sentences = []
        # check for values its different with versions
        NSUBJ = 429
        DOBJ = 416

        sentences = [(line.strip(), idx) for idx, line in enumerate(sentences) if is_qualified(line.strip())]
        for sentence, _ in self.nlp.pars_document(sentences, as_tuples=True):
            for chunk in sentence.noun_chunks:
                if not chunk.root.head.pos == VERB or not (chunk.root.dep == NSUBJ or chunk.root.dep == DOBJ):
                    continue

                logging.debug("===========================================================")
                logging.debug("Chunk: {}".format(chunk.doc.text))
                # ==================================================================================
                noun_norm = chunk.root.text if chunk.root.pos == PRON else chunk.root.lemma_
                noun = Token(noun_norm, chunk.root)
                verb = Token(chunk.root.head.lemma_, chunk.root.head)
                word_pair = WordPair(verb, noun)
                logging.debug("Words: verb:{} noun:{}".format(chunk.root.head.lemma_, noun_norm))
                # ==================================================================================
                process_begin_time_0 = time.process_time()
                # word pair vectorized
                # word_pair = self.w2v(word_pair)
                logging.debug(
                    "Word vectorized in {} minutes".format((time.process_time() - process_begin_time_0) / 60))

                if not word_pair.has_vector():
                    logging.debug("Vector not found for verb:{} noun:{}".format(
                        word_pair.verb.has_vector(), word_pair.noun.has_vector()))
                    continue
                # ==================================================================================
                process_begin_time_0 = time.process_time()
                # word pair clustered
                # word_pair = self.v2c(word_pair)
                logging.debug("Words Clusters: verb:{} noun:{} time:{}".format(
                    word_pair.verb.cluster, word_pair.noun.cluster,
                    (time.process_time() - process_begin_time_0) / 60))
                # ==================================================================================
                process_begin_time_0 = time.process_time()
                # SPS identification
                # word_pair.sps = get_sps(self.word_pair_freq, word_pair.verb.cluster)

                # SA identification
                word_pair.sa = get_sa(self.word_pair_freq, word_pair.verb.cluster, word_pair.noun.cluster)
                if not word_pair.is_literal():
                    creative_sentences.append(sentence)

                logging.debug(
                    "SA of words: {} time:{}".format(word_pair.sa,
                                                     (time.process_time() - process_begin_time_0) / 60))
                logging.debug("Word Literal: {}".format(word_pair.is_literal()))

        return creative_sentences

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
    cr = CorpusReader(
        "C:/Users/ShimaK/PycharmProjects/CreativeAid!/creativeaid/test_corpus/test_generate_corpus/cliche", "")
    ti = CreativeTextIdentifier()
    ti.identify_with_corpus(cr)
