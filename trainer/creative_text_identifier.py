import spacy
import pickle
import time
import numpy
import string
from trainer.count_utils import *
from corpus_reader import CorpusReader
from corpus_reader import File
from gensim.models import KeyedVectors


class CreativeTextIdentifier:

    def __init__(self,
                 corpus_reader,
                 batch_size=10000,
                 kmeans_path='model/mini_batch_kmeans',
                 word2vec_path='model/glove.840B.300d.bin',
                 word2vec_coverage=0.5):
        # TODO word2vec coverage in percentage 0-1
        self.corpus_reader = corpus_reader
        self.batch_size = batch_size
        self.nlp = spacy.load('en_core_web_sm')
        self.mini_batch_kmeans = pickle.load(open(kmeans_path, 'rb'))
        self.word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=100000)

        self.nlp.remove_pipe('ner')

    def identify(self):
        """
        Begin creative text identification
        :return:
        """
        for file in self.files_for_identification(self.corpus_reader.files()):
            self.creative_text_for_file(file)

    def files_for_identification(self, files):
        """
        Override the method exclude include files
        :param files: [File]
        :return: [File]
        """
        return files

    def creative_text_for_file(self, file):
        # TODO clean text
        # check for values its different with versions
        nsubj = 429
        dobj = 416
        verb = 100
        pron = 95

        clean_sentences = [self.clean_text(s) for s in file.contents]
        for doc in self.nlp.pipe(clean_sentences, batch_size=self.batch_size):
            for chunk in doc.noun_chunks:
                if chunk.root.head.pos == verb or (chunk.root.dep == nsubj or chunk.root.dep == dobj):
                    logging.debug("===========================================================")
                    logging.debug("Chunk: {}".format(chunk.doc.text))
                    # ==================================================================================
                    word = chunk.root.text if chunk.root.pos == pron else chunk.root.lemma_
                    logging.debug("Words: verb:{} noun:{}".format(chunk.root.head.lemma_, word))

                    noun = Word("noun", word, chunk.root.dep_)
                    verb = Word("verb", chunk.root.head.lemma_)
                    word_pair = WordPair(verb, noun)
                    # ==================================================================================
                    process_begin_time_0 = time.process_time()
                    word_pair = self.w2v(word_pair)
                    logging.debug(
                        "Word vectorized in {} minutes".format((time.process_time() - process_begin_time_0) / 60))
                    if not word_pair.is_vectorized():
                        logging.debug("Vector not found for verb:{} noun:{}".format(
                            word_pair.verb.is_vectorized(), word_pair.noun.is_vectorized()))
                        continue
                    # ==================================================================================
                    process_begin_time_0 = time.process_time()
                    word_pair = self.v2c(word_pair)
                    logging.debug("Words Clusters: verb:{} noun:{} time:{}".format(
                        word_pair.verb.cluster, word_pair.noun.cluster,
                        (time.process_time() - process_begin_time_0) / 60))
                    # ==================================================================================
                    # process_begin_time_0 = time.process_time()
                    # sa = get_sa(verb_noun_freq, verb_cluster, noun_cluster)
                    #
                    # logging.debug(
                    #     "SA of words: {} time:{}".format(sa, (time.process_time() - process_begin_time_0) / 60))
                    # logging.debug("Word Literal: {}".format(sa > 1.32 if sa is not None else "None"))

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
            vector = self.word2vec.wv.get_vector(word.text)
        except KeyError:
            try:
                similar = self.word2vec.most_similar(word.text, topn=1)
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


class WordPair:
    def __init__(self, verb, noun):
        self.verb = verb
        self.noun = noun

    def is_vectorized(self):
        return self.verb.is_vectorized() and self.noun.is_vectorized()


class Word:
    def __init__(self, type, text, relation=None):
        self.type = type
        self.text = text
        self.relation = relation
        self.vector = None
        self.cluster = None

    def is_vectorized(self):
        return self.vector is not None


if __name__ == '__main__':
    cr = CorpusReader("C:/Users/ShimaK/PycharmProjects/Test", "")
    ti = CreativeTextIdentifier(cr)
    ti.identify()
