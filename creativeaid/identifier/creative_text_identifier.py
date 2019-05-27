import pickle
import time
import numpy
import logging
import os
from spacy.parts_of_speech import *
from spacy.tokens import Doc, Span, Token

from creativeaid.identifier.word_frequency import WordFrequency
from creativeaid.nlp import NLP
from creativeaid.models import Token, WordPair, CreativeSentence
from creativeaid.corpus_reader import CorpusReader
from creativeaid.nlp.text_utils import is_qualified

logging.basicConfig(format=u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level=logging.NOTSET)

kmeans_path = 'model/kmeans_glove840B300d_c200000_all'
word_freq_path = 'verb_noun_freq_2019-03-27_23-31-01'


class CreativeTextIdentifier(object):

    def __init__(self, nlp, word_freq=None, kmeans=None):
        self.nlp = nlp

        if kmeans is None:
            self.mini_batch_kmeans = pickle.load(open(kmeans_path, 'rb'))
        else:
            self.mini_batch_kmeans = kmeans

        if word_freq is None:
            self.word_freq = WordFrequency(pickle.load(open(word_freq_path, 'rb')))
        else:
            self.word_freq = WordFrequency(word_freq)

        # Register attribute on the Span. We'll be overwriting this on __call__
        Span.set_extension("word_pairs", default=None)

    def identify_with_corpus(self, corpus_reader):
        sentences = []
        for file in corpus_reader.corpus():
            sentences += self._identify(file.contents_lines())
        return sentences

    def identify_with_sentences(self, sentences):
        return self._identify(sentences)

    def _identify(self, sentences):
        pipe = CreativeTextIdentifierPipe(self.word_freq, self.mini_batch_kmeans, self.nlp)

        sentences = [(line.strip(), idx) for idx, line in enumerate(sentences) if is_qualified(line.strip())]
        creative_sentences = []
        for doc, _ in self.nlp.pars_document(sentences,
                                             as_tuples=True,
                                             pipe=pipe,
                                             name="creative_text_identifier"):
            for sent in doc.sents:
                word_pairs = sent._.word_pairs
                if word_pairs is not None:
                    for word_pair in word_pairs:
                        if word_pair is not None and word_pair.is_creative:
                            creative_sentence = CreativeSentence(sent.text, sent, word_pair)
                            creative_sentences.append(creative_sentence)

        return creative_sentences


class CreativeTextIdentifierPipe(object):
    name = "creative_text_identifier"  # component name, will show up in the pipeline

    def __init__(self, word_freq, mini_batch_kmeans, nlp):
        self.word_freq = word_freq
        self.mini_batch_kmeans = mini_batch_kmeans
        self.nlp = nlp

    def __call__(self, doc):
        NSUBJ = 429
        DOBJ = 416

        # doc._.set("word_pair", None)
        for sentence in doc.sents:
            word_pairs = []

            for chunk in sentence.noun_chunks:
                if not chunk.root.head.pos == VERB or not (chunk.root.dep == NSUBJ or chunk.root.dep == DOBJ):
                    continue

                noun_norm = chunk.root.text if chunk.root.pos == PRON else chunk.root.lemma_
                noun = Token(noun_norm, chunk.root)
                verb = Token(chunk.root.head.lemma_, chunk.root.head)
                word_pair = WordPair(verb, noun)
                word_pair.noun_chunk = chunk

                # word pair vectorized
                word_pair = self.nlp.w2v(word_pair)

                if not word_pair.has_vector:
                    continue

                # word pair clustered
                word_pair = self.v2c(word_pair)

                # SPS identification
                word_pair.sps = self.word_freq.sps(word_pair.verb.cluster)

                # SA identification
                word_pair.sa = self.word_freq.sa(word_pair.verb.cluster, word_pair.noun.cluster)
                word_pairs.append(word_pair)

            sentence._.set("word_pairs", word_pairs)

        return doc

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
