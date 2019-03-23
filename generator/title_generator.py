from corpus_reader import CorpusReader
from gensim.summarization.summarizer import summarize


class TitleGenerator:

    def __init__(self, corpus_reader, title_ratio=0.1):
        self.corpus_reader = corpus_reader
        self.ratio = title_ratio

    def get_title(self):
        for file in self.corpus_reader.files():
            print(summarize(file.contents, ratio=self.ratio, word_count=10))


if __name__ == '__main__':
    cr = CorpusReader("../test_corpus/test_title", "")
    title_gen = TitleGenerator(cr)
    title_gen.get_title()
