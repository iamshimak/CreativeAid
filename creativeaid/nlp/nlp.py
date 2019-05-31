import time
import logging
import en_core_web_lg
import en_core_web_sm
from gensim.models import KeyedVectors
from creativeaid.nlp.lexsub import LexSub
from string import punctuation
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.parts_of_speech import *


class NLP:

    def __init__(self,
                 word2vec_path='../identifier/model/glove.840B.300d.bin',
                 word2vec_limit=500000,
                 word2vec_coverage=0.5,
                 requires_word2vec=False,
                 is_nlp_sm=False):
        if is_nlp_sm:
            self.nlp = en_core_web_sm.load()
        else:
            self.nlp = en_core_web_lg.load()
        if requires_word2vec:
            time_0 = time.time()
            self.word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=word2vec_limit)
            logging.info(f'Took {time.time() - time_0} seconds to load word2vec-{word2vec_limit}')
            self.lexsub = LexSub(self.word2vec, candidate_generator='word2vec')

    @property
    def vocab(self):
        return self.nlp.vocab

    def pars_document(self, sentences, as_tuples=False, pipe=None, name=None):
        if pipe and pipe.name not in self.nlp.pipe_names:
            self.nlp.add_pipe(pipe, name=name)
        return self.nlp.pipe(sentences, as_tuples=as_tuples)

    def pars_sentence(self, sentence):
        return self.nlp(sentence)

    def similar_word(self, word, topn=10):
        return self.word2vec.most_similar(word, topn=topn)

    def similar_word_for_sentence(self, word, sentence):
        result = self.lexsub.lex_sub(f"{word}.n", sentence)
        return result

    def w2v(self, word_pair):
        word_pair.verb.vector = self.w2v_word(word_pair.verb)
        word_pair.noun.vector = self.w2v_word(word_pair.noun)
        return word_pair

    def w2v_word(self, word):
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
                logging.error(f"word_vec & similar not found for {word}")

        return vector

    def is_valid(word):
        """
        Check word proper input for NLP process
        https://spacy.io/api/annotation#pos-tagging
        :param word: spacy token
        :return: Boolean
        """
        return not (word.is_stop or word.is_space or word.pos in [NUM, SYM, PUNCT, DET, CCONJ, CONJ, SCONJ, X])

    def is_stop(self, word):
        return word in STOP_WORDS

    def clean_sentence(self, sentence):
        sentence = sentence.lower()
        sentence = ''.join([i for i in sentence if not i.isdigit()])
        sentence = ' '.join([word for word in sentence.split(' ') if word not in STOP_WORDS])
        sentence = sentence.translate(str.maketrans('', '', punctuation))
        return sentence.strip()

    def is_qualified(self, sentence):
        return len(sentence.split(' ')) > 3

    def unique(self, iter):
        "removes duplicates from iterable preserving order"
        result = list()
        seen = set()
        for x in iter:
            if x not in seen:
                seen.add(x)
                result.append(x)
        return result

    def process_candidates(self, candidates, target):
        """
        words to lower case, replace underscores, remove duplicated words,
        filter out target word and stop words
        """
        filterwords = STOP_WORDS + [target]
        return self.unique(filter(lambda x: x not in filterwords,
                                  map(lambda s: s.lower().replace('_', ' '), candidates)))
