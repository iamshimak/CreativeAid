import spacy
import logging
import numpy as np
import en_core_web_lg
import en_core_web_sm
import pickle
import time
import logging
import csv
from corpus_reader import CorpusReader
from random import randint, choice
from numpy.linalg import norm
from spacy.parts_of_speech import *
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import KeyedVectors
from generator.text_utils import is_clean, clean_sentence, is_qualified

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

logging.basicConfig(format=u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level=logging.NOTSET)


class CreativeTextGenerator:
    def __init__(self,
                 titles,
                 corpus_reader,
                 word2vec_path='../trainer/model/glove.840B.300d.bin',
                 word2vec_coverage=0.5,
                 word2vec_limit=500000):
        self.titles = titles
        self.corpus_reader = corpus_reader
        self.nlp = en_core_web_lg.load()
        time_0 = time.time()
        self.word2vec = KeyedVectors.load_word2vec_format(word2vec_path, binary=True, limit=word2vec_limit)
        logging.info(f'Took {time.time() - time_0} seconds to load word2vec-{word2vec_limit}')

    def enhance_title_info(self, title):
        title.doc = self.nlp(title.text)
        title.important_keyword_indexes = self.important_keywords_indexes(title)
        title.important_keyword_similar_words = self.similar_keywords_for_indexes(title)

        sent = []
        for idx in title.important_keyword_indexes:
            sent.append(title[idx].text)
            for word in title.important_keyword_similar_words[idx]:
                sent.append(word)

        title.doc_ = self.nlp(" ".join(sent))

    def generate(self):
        for title in self.titles:
            print('==============================================================================')
            print(f'Title: {title.text}\n')
            self.enhance_title_info(title)
            creative_sentences = self.search_candidate_creative_sentences(title)
            for creative_sentence in creative_sentences:
                v, replaced, inserted = self.generate_creative_sentence(creative_sentence, title)
                if replaced or inserted:
                    print(
                        f"template:{creative_sentence.text} | modified:{v} | replaced:{replaced} | inserted:{inserted}")

    def search_candidate_creative_sentences(self, title, additional_sentences=None):
        lines = []
        for file in self.corpus_reader.files():
            lines += file.contents_lines()
        lines = [(line.strip(), idx) for idx, line in enumerate(lines) if is_qualified(line.strip())]

        candidate_lines = []
        for sentence, _ in self.nlp.pipe(lines, as_tuples=True, batch_size=10000):
            sentence = CreativeSentence(sentence)
            sentence.nlp_text = self.nlp(clean_sentence(sentence.doc.text))

            sentence_similarity_score = self.doc_score(title, sentence)
            keyword_score = self.keyword_score(title, sentence)

            if sentence_similarity_score >= 0.5 and keyword_score >= 0.5:
                candidate_lines.append(((sentence_similarity_score + keyword_score) / 2, sentence))

        candidate_lines = sorted(candidate_lines, key=lambda tup: tup[0], reverse=True)
        candidate_lines = [candidate for _, candidate in candidate_lines]
        return candidate_lines

    def doc_score(self, title, sentence):
        sentence_similarity_score = 0
        if title.doc.has_vector and sentence.nlp_text.has_vector:
            sentence_similarity_score = title.doc.similarity(sentence.nlp_text)
        return sentence_similarity_score

    def keyword_score(self, title, sentence):
        # TODO look again
        i = 0
        token_score = 0
        for index, similar_words in title.important_keyword_similar_words.items():
            for similar_word in similar_words:
                for token in sentence.nlp_text:
                    if is_clean(token):
                        continue

                    try:
                        cat_id = self.nlp.vocab.strings[similar_word]
                        cat_vector = self.nlp.vocab.vectors[cat_id]
                        s = cosine_similarity([token.vector], [cat_vector])[0][0]
                        if s > 0.5:
                            token_score += s
                            i += 1
                    except KeyError:
                        pass

        return token_score / i if token_score != 0 else 0

    def generate_creative_sentence(self, temple_sentence, title):
        important_keywords = title.important_keyword_indexes
        replace_words = {}
        insertion_words = {}

        for index in important_keywords:
            title_token = title[index]
            if is_clean(title_token):
                continue
            # TODO check a word is already replaced
            i = 0
            for token in temple_sentence.doc:
                if is_clean(token):
                    i += 1
                    # TODO check has vector
                    continue
                if token.pos is title_token.pos:
                    # if token.tag_ == title_token.tag_:
                    score = token.similarity(title_token)
                    if score > 0.5 and score != 1:
                        replace_words[i] = title_token
                elif i > 0 and token.pos == NOUN and title_token.pos == ADJ:
                    score = token.similarity(title_token)
                    if score > 0.5 and score != 1:
                        insertion_words[i] = title_token
                    # elif i > 0 and token.pos == VERB and title_token.pos == ADV:
                    #     score = token.similarity(title_token)
                    #     if score > 0.5 and score != 1:
                    #         sent[i - 1] = title_token
                    #         c += 1
                i += 1

        sent = [i for i in temple_sentence.doc]
        for index, val in replace_words.items():
            sent[index] = val

        for index, val in insertion_words.items():
            sent[index - 1] = val

        return " ".join(sent.text for sent in sent), len(replace_words) > 0, len(insertion_words) > 0

    # def remove_unwanted_words(self, tokens):
    #     words = []
    #     for token in tokens:
    #         if not token.is_stop or not token.is_space:
    #             words.append(token)

    def important_keywords_indexes(self, tokens):
        #  TODO implement token.ent_type_ NER
        keyword_index = []
        random_counts = len(tokens) - 1
        idx = list(range(0, random_counts))
        for i in range(0, random_counts):
            val = choice(idx)
            idx.remove(val)
            if not tokens.doc[val].is_stop or not tokens.doc[val].is_space:
                keyword_index.append(val)

        return keyword_index
        # return [17, 6, 10]

    def similar_keywords_for_indexes(self, title):
        similar_words_for_indexes = {}
        for index in title.important_keyword_indexes:
            similar_words_for_indexes[index] = {}
            word = title[index].text if title[index].pos in [PROPN, ADJ] else title[index].lemma_
            try:
                similar_words = self.word2vec.most_similar(word, topn=10)
                # TODO remove same words
                similar_words = [token.lower() for token, _ in similar_words
                                 if token.lower() != word and word not in token.lower()]
                similar_words = similar_words[:2]
                similar_words_for_indexes[index] = similar_words
            except KeyError:
                pass
        return similar_words_for_indexes


class Title(object):
    def __init__(self, text):
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
        return self.doc.__len__()


class CreativeSentence(object):
    def __init__(self, doc):
        self.doc = doc
        self.nlp_text = None
        self.text = doc.text
        self.description = None


if __name__ == '__main__':
    # titles = []
    # with open('C:/Users/ShimaK/PycharmProjects/CreativeAid!/test_corpus/test_title/titles/abcnews-date-text.csv') \
    #         as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     line_count = 0
    #     for row in csv_reader:
    #         titles.append(Title(row[1]))
    #         line_count += 1
    #     print(f'Processed {line_count} lines.')
    # titles = titles[1:]

    title = Title('The Obama administration is planning to issue a final rule designed to enhance the safety of '
                  'offshore oil drilling equipment')
    cr = CorpusReader("C:/Users/ShimaK/PycharmProjects/CreativeAid!/test_corpus/test_generate_corpus/cliche", "")
    ctg = CreativeTextGenerator([title], cr)
    ctg.generate()
