from creativeaid.corpus_reader import CorpusReader
from gensim.summarization.summarizer import summarize


class TitleGenerator:

    def __init__(self):
        pass

    def generate_title_from_corpus(self, corpus_reader, ratio=0.2, word_count=None):
        titles = []
        for file in corpus_reader.corpus():
            titles.append(self._generate_title(file.contents, ratio, word_count))
        return titles

    def generate_title_from_sentences(self, sentences, ratio=0.2, word_count=None):
        return self._generate_title(sentences, ratio, word_count)

    def _generate_title(self, text, ratio, word_count):
        """Get a summarized version of the given text.
            Parameters
            ----------
            text : str
                Given text.
            ratio : float, optional
                Number between 0 and 1 that determines the proportion of the number of
                sentences of the original text to be chosen for the summary.
            word_count : int or None, optional
                Determines how many words will the output contain.
                If both parameters are provided, the ratio will be ignored.
            """
        return summarize(text, ratio=ratio, word_count=word_count)


if __name__ == '__main__':
    cr = CorpusReader("../test_corpus/test_title", "")
    title_gen = TitleGenerator()
    print(title_gen.generate_title_from_corpus(cr))
