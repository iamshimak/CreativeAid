import time
import logging
import en_core_web_lg
from gensim.models import KeyedVectors
from creativeaid.nlp.lexsub import LexSub


class NLP:

    def __init__(self,
                 word2vec_path='../identifier/model/glove.840B.300d.bin',
                 word2vec_limit=500000,
                 word2vec_coverage=0.5, requires_word2vec=False):
        self.nlp = en_core_web_lg.load()
        if requires_word2vec:
            time_0 = time.time()
            self.word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=word2vec_limit)
            logging.info(f'Took {time.time() - time_0} seconds to load word2vec-{word2vec_limit}')
            self.lexsub = LexSub(self.word2vec, candidate_generator='word2vec')

    @property
    def vocab(self):
        return self.nlp.vocab

    def pars_document(self, sentences, as_tuples=False):
        return self.nlp.pipe(sentences, as_tuples=as_tuples)

    def pars_sentence(self, sentence):
        return self.nlp(sentence)

    def similar_word(self, word, topn=10):
        return self.word2vec.most_similar(word, topn=topn)

    def similar_word_for_sentence(self, word, sentence):
        result = self.lexsub.lex_sub(f"{word}.n", sentence)
        return result

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
