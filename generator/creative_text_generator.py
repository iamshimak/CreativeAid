import spacy
import logging
import numpy as np
import en_core_web_lg
import en_core_web_sm
from random import randint, choice
from numpy.linalg import norm
from sklearn.metrics.pairwise import cosine_similarity

"""
---------
Insertion
---------
For the insertions, an adjective (or adverb) k is inserted before the noun (verb) w that appears most often 
in an appropriate dependency relation with it (i.e. “amod” and “advmod” respectively).

-------------------------------------------
check cosine similarity of two vector array
-------------------------------------------
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity([1, 0, -1], [-1,-1, 0])
array([[-0.5]])
"""


class CreativeTextGenerator:
    def __init__(self, title):
        self.title = title
        self.nlp = en_core_web_sm.load()

    def generate(self):
        title = self.nlp(self.title)
        creative_sentences = self.search_candidate_creative_sentences(title)
        for creative_sentence in creative_sentences:
            v = self.generate_creative_sentence(creative_sentence, title)
            print(f"cs:{v}")

    def search_candidate_creative_sentences(self, title, additional_sentences=None):
        lines = self.load_creative_sentences()
        lines = [(line.strip(), idx) for idx, line in enumerate(lines)]

        candidate_lines = []
        for doc, _ in self.nlp.pipe(lines, as_tuples=True, batch_size=10000):
            if self.check_similarity_for_sentence(title, doc):
                candidate_lines.append(doc)

        return candidate_lines

    def check_similarity_for_sentence(self, title, sentence):
        # TODO improve
        title_vector = self.clean_words_with_vector(title)
        sentence_vector = self.clean_words_with_vector(sentence)
        score = self._cosine_score(title_vector, sentence_vector)
        return score > 0.66

    def clean_words_with_vector(self, token):
        words = []
        for token in token:
            if not token.is_stop or not token.is_space:
                words.append(token.text)
        # TODO improve
        doc = self.nlp(' '.join(word for word in words))
        return doc

    def _cosine_score(self, a, b):
        return a.similarity(b)

    def generate_creative_sentence(self, temple_sentence, title):
        important_keywords = self.important_keywords_indexes(title)
        sent = [i for i in temple_sentence]
        for index in important_keywords:
            title_token = title[index]
            if title_token.is_stop or title_token.is_space:
                continue

            i = 0
            for token in temple_sentence:
                if token.is_stop or token.is_space:
                    continue
                if token.pos is title_token.pos:
                    if token.tag_ == title_token.tag_:
                        sent[i] = title_token
                    # else:
                    #     pass
                i += 1

        return " ".join(i.text for i in sent)

    # def remove_unwanted_words(self, tokens):
    #     words = []
    #     for token in tokens:
    #         if not token.is_stop or not token.is_space:
    #             words.append(token)

    def important_keywords_indexes(self, tokens):
        keyword_index = []
        random_counts = randint(1, len(tokens) - 1)
        idx = list(range(0, random_counts))
        for i in range(0, random_counts):
            val = choice(idx)
            idx.remove(val)
            keyword_index.append(val)

        return keyword_index

    def load_creative_sentences(self):
        return open("../test_corpus/test_generate_corpus/cliches.txt", encoding="utf-8").readlines()


if __name__ == '__main__':
    ctg = CreativeTextGenerator("The Obama administration is planning to issue a final rule designed to enhance the "
                                "safety of offshore oil drilling equipment")
    ctg.generate()
