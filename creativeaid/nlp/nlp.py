import en_core_web_lg
from gensim.models import KeyedVectors


class NLP:

    def __init__(self,
                 word2vec_path='model/glove.840B.300d.bin',
                 word2vec_coverage=0.5):
        self.nlp = en_core_web_lg.load()
        # self.word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True)

    def pars_document(self, sentences, as_tuples=False):
        return self.nlp.pipe(sentences, as_tuples=as_tuples)

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
