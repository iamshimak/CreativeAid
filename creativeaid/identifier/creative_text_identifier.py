import pickle
import time
import numpy
import logging
import os
from spacy.parts_of_speech import *
from spacy.tokens import Doc, Span, Token

from creativeaid.identifier.word_frequency import WordFrequency
from creativeaid.nlp import NLP
from creativeaid.models import Token, WordPair
from creativeaid.corpus_reader import CorpusReader
from creativeaid.nlp.text_utils import is_qualified

logging.basicConfig(format=u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level=logging.NOTSET)

kmeans_path = 'model/kmeans_glove840B300d_c200000_all'
word_freq_path = 'verb_noun_freq_2019-03-27_23-31-01'


class CreativeTextIdentifier(object):
    # TODO make this class as controller and create creative text identifier class and add to spaCy as extension
    #  https://spacy.io/usage/processing-pipelines#custom-components-attributes

    def __init__(self, nlp, word_freq=None):
        logging.info(f'directory: {os.path.dirname(os.path.realpath(__file__))}')
        # TODO word2vec coverage in percentage 0-1
        # TODO word_pair_freq [increase accuracy]
        self.nlp = nlp
        self.mini_batch_kmeans = pickle.load(open(kmeans_path, 'rb'))
        if word_freq is None:
            self.word_freq = WordFrequency(pickle.load(open(word_freq_path, 'rb')))
        else:
            self.word_freq = WordFrequency(word_freq)

    def identify_with_corpus(self, corpus_reader):
        sentences = []
        for file in corpus_reader.corpus():
            sentences += self._identify(file.contents_lines())
        return sentences

    def identify_with_sentences(self, sentences):
        return self._identify(sentences)

    def _identify(self, sentences):
        pipe = CreativeTextIdentifierPipe()

        sentences = [(line.strip(), idx) for idx, line in enumerate(sentences) if is_qualified(line.strip())]
        creative_titles = []
        for sentence, _ in self.nlp.pars_document(sentences,
                                                  as_tuples=True,
                                                  pipe=pipe,
                                                  name="creative_text_identifier"):
            word_pair = sentence._.word_pair
            if not word_pair.is_literal:
                creative_titles.append(sentence)

        creative_sentences = []
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


class CreativeTextIdentifierPipe(object):
    name = "creative_text_identifier"  # component name, will show up in the pipeline

    def __init__(self):
        # self.nlp = nlp
        self.word_freq = WordFrequency(pickle.load(open(word_freq_path, 'rb')))
        # Register attribute on the Span. We'll be overwriting this on __call__
        Doc.set_extension("word_pair", default=None)

    def __call__(self, doc):
        NSUBJ = 429
        DOBJ = 416

        for chunk in doc.noun_chunks:
            if not chunk.root.head.pos == VERB or not (chunk.root.dep == NSUBJ or chunk.root.dep == DOBJ):
                continue

            noun_norm = chunk.root.text if chunk.root.pos == PRON else chunk.root.lemma_
            noun = Token(noun_norm, chunk.root)
            verb = Token(chunk.root.head.lemma_, chunk.root.head)
            word_pair = WordPair(verb, noun)

            # word pair vectorized
            # word_pair = self.nlp.w2v(word_pair)

            if not word_pair.has_vector:
                continue
            # word pair clustered
            # word_pair = self.v2c(word_pair)

            # SPS identification
            word_pair.sps = self.word_freq.sps(word_pair.verb.cluster)

            # SA identification
            word_pair.sa = self.word_freq.sa(word_pair.verb.cluster, word_pair.noun.cluster)
            doc._.set("word_pair", word_pair)
        return doc


if __name__ == '__main__':

    corpus_reader = CorpusReader(
        "C:/Users/ShimaK/PycharmProjects/CreativeAid!/creativeaid/test_corpus/test_generate_corpus/cliche", "")
    text_identifier = CreativeTextIdentifier(NLP())
    text = text_identifier.identify_with_corpus(corpus_reader)

    from nltk.corpus.reader.bnc import BNCCorpusReader
    import string

    # corpus_reader = CorpusReader(
    #     "C:/Users/ShimaK/PycharmProjects/CreativeAid!/creativeaid/test_corpus/test_generate_corpus/cliche", "")
    process_begin_time_0 = time.process_time()

    # corpus_reader = CorpusReader("./creativeaid/test_corpus/test_generate_corpus/cliche", "")
    text_identifier = CreativeTextIdentifier(NLP(requires_word2vec=True))

    logging.debug(f"Loading Time {(time.process_time() - process_begin_time_0) / 60}")
    logging.debug(f"================================================================")


    def clean_text(text):
        text = text.lower()
        text = ''.join([i for i in text if not i.isdigit()])
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.strip()


    path = "C:/Users/ShimaK/PycharmProjects/download/Texts"

    bnc_reader = BNCCorpusReader(root=path, fileids=r'[A-K]/\w*/\w*\.xml')

    test_sentences = []
    for root, dirs, files in os.walk(path):
        for name in files:
            process_begin_time = time.process_time()

            file = os.path.join(root, name).replace(path, "")
            sentences = bnc_reader.sents(fileids=file[1:])

            clean_sentences = [clean_text(" ".join(s)) for s in sentences]
            if len(test_sentences) <= 1600:
                test_sentences += clean_sentences
            else:
                break
        else:
            continue  # only executed if the inner loop did NOT break
        break  #

    for length in [100, 200, 400, 800, 1600]:
        process_begin_time_0 = time.process_time()
        text_identifier.identify_with_sentences(test_sentences[:length])
        logging.debug(f"Identification time of length {length} {(time.process_time() - process_begin_time_0) / 60.0}")
        logging.debug(f"================================================================")
    # text_identifier.identify_with_corpus(corpus_reader)

    # pickle.dump(text, open('text', 'wb'))
