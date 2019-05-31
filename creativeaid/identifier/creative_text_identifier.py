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
from creativeaid.nlp.text_utils import is_qualified, is_valid

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

                if not (chunk.root.head.is_stop and chunk.root.is_stop):
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
                word_pair.sa = 0
                for word in chunk:
                    word_pair.sa += self.word_freq.sa(word_pair.verb.cluster, word.cluster)

                word_pair.sa = word_pair.sa / len(chunk)

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


if __name__ == '__main__':
    import string


    def clean_text(text):
        text = text.lower()
        text = ''.join([i for i in text if not i.isdigit()])
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.strip()


    sent = ""

    corpus_reader = CorpusReader(
        "/root/PycharmProjects/CreativeAid!/creativeaid/test_corpus/test_title", "")

    text_identifier = CreativeTextIdentifier(NLP(word2vec_limit=500000, requires_word2vec=False, is_nlp_sm=False))
    s = text_identifier.identify_with_corpus(corpus_reader)
    print(s)

    # corpus_reader = CorpusReader(
    #     "/root/PycharmProjects/CreativeAid!/creativeaid/test_corpus/test_generate_corpus/cliche", "")
    # text_identifier = CreativeTextIdentifier(NLP())
    # text = text_identifier.identify_with_corpus(corpus_reader)
    #
    # from nltk.corpus.reader.bnc import BNCCorpusReader
    # import string
    #
    # # corpus_reader = CorpusReader(
    # #     "C:/Users/ShimaK/PycharmProjects/CreativeAid!/creativeaid/test_corpus/test_generate_corpus/cliche", "")
    # process_begin_time_0 = time.process_time()
    #
    # # corpus_reader = CorpusReader("./creativeaid/test_corpus/test_generate_corpus/cliche", "")
    # text_identifier = CreativeTextIdentifier(NLP())
    #
    # logging.debug(f"Loading Time {(time.process_time() - process_begin_time_0) / 60}")
    # logging.debug(f"================================================================")
    #
    #
    # path = "C:/Users/ShimaK/PycharmProjects/download/Texts"
    #
    # bnc_reader = BNCCorpusReader(root=path, fileids=r'[A-K]/\w*/\w*\.xml')
    #
    # test_sentences = []
    # for root, dirs, files in os.walk(path):
    #     for name in files:
    #         process_begin_time = time.process_time()
    #
    #         file = os.path.join(root, name).replace(path, "")
    #         sentences = bnc_reader.sents(fileids=file[1:])
    #
    #         clean_sentences = [clean_text(" ".join(s)) for s in sentences]
    #         if len(test_sentences) <= 1600:
    #             test_sentences += clean_sentences
    #         else:
    #             break
    #     else:
    #         continue  # only executed if the inner loop did NOT break
    #     break  #
    #
    # for length in [100, 200, 400, 800, 1600]:
    #     process_begin_time_0 = time.process_time()
    #     text_identifier.identify_with_sentences(test_sentences[:length])
    #     logging.debug(f"Identification time of length {length} {(time.process_time() - process_begin_time_0) / 60.0}")
    #     logging.debug(f"================================================================")
    # text_identifier.identify_with_corpus(corpus_reader)

    # pickle.dump(text, open('text', 'wb'))
