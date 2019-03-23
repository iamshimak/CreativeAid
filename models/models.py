class Word:
    def __init__(self, text_, doc):
        self.text_ = text_
        self.text = doc.text
        self.lemma = doc.lemma
        self.pos = doc.pos
        self.pos_ = doc.pos_
        self.dep = doc.dep
        self.dep_ = doc.dep_
        self.vector = None
        self.cluster = None

    def has_vector(self):
        return self.vector is not None


class WordPair:
    _optimal_sa_score = 0.000426201810129815

    def __init__(self, verb, noun):
        self.verb = verb
        self.noun = noun
        self.sa = None

    def has_vector(self):
        return self.verb.has_vector() and self.noun.has_vector()

    def is_literal(self):
        return self.sa > self._optimal_sa_score if self.sa is not None else None

    @staticmethod
    def set_optimal_sa_score(score):
        _optimal_sa_score = score


class Corpus:
    def __init__(self, name, path):
        self.name = name
        self.path = path
        self.contents = ""

    def get_contents(self):
        # TODO check the error with read()
        return open(self.path, encoding="utf-8").read()
        # return open(file.path, encoding=self.encoding).readlines()
