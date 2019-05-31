class Token:
    def __init__(self, text_, doc):
        self.text_ = text_
        self.text = doc.text
        self.doc = doc
        self.lemma = doc.lemma
        self.pos = doc.pos
        self.pos_ = doc.pos_
        self.dep = doc.dep
        self.dep_ = doc.dep_
        self.vector = doc.vector
        # self.cluster = doc.cluster

    @property
    def cluster(self):
        return self.doc.cluster

    @cluster.setter
    def cluster(self, value):
        pass

    def has_vector(self):
        return self.vector is not None

    def __repr__(self):
        return self.text


class WordPair:
    _optimal_sa_score = 0.000426201810129815

    def __init__(self, verb, noun):
        self.verb = verb
        self.noun = noun
        self.sps = None
        self.sa = None
        self.noun_chunk = None

    @property
    def has_vector(self):
        return self.verb.has_vector() and self.noun.has_vector()

    @property
    def is_literal(self):
        return self.sa > self._optimal_sa_score if self.sa is not None else None

    @property
    def is_creative(self):
        return self.sa < self._optimal_sa_score if self.sa is not None else None

    @staticmethod
    def set_optimal_sa_score(score):
        _optimal_sa_score = score

    def __repr__(self):
        return f"verb [{self.verb}] | noun [{self.noun}]"


class Corpus:
    def __init__(self, name, path, encoding=None):
        self.name = name
        self.path = path
        self.contents = ""

    def get_contents(self):
        # TODO check the error with read()
        return open(self.path, encoding='utf-8').read()
        # return open(file.path, encoding=self.encoding).readlines()
        # return self.contents

    def contents_lines(self):
        return self.contents.split('\n')


class Sentence(object):
    def __init__(self, text):
        self.text = text
        self.preprocessed = None
        self.tokens = None
        self.keywords = None

    # @property
    # def has_vector(self):
    #     return self.vector is not None

    def __repr__(self):
        return self.text


class Title(Sentence):
    def __init__(self, text):
        super().__init__(text)
        self.text = text
        self.doc = None
        self.doc_ = None
        self.important_keyword_indexes = None
        self.important_keyword_similar_words = {}

    def __iter__(self):
        return self.doc.__iter__()

    def __getitem__(self, item):
        return self.doc.__getitem__(item)

    def __len__(self):
        return len(self.text)


class Template(Sentence):
    def __init__(self, doc, text):
        super().__init__(text)
        self.doc = doc
        self.nlp_text = None
        self.text = doc.text
        self.description = None

    def __repr__(self):
        return self.text


class CreativeTitle(Sentence):
    def __init__(self, title, template, text):
        super().__init__(text)
        self.title = title
        self.template = template
        self.text = text


class CreativeSentence(Sentence):
    def __init__(self, text, doc, word_pair):
        super().__init__(text)
        self.text = text
        self.doc = doc
        self.word_pair = word_pair

    def __repr__(self):
        return self.text
