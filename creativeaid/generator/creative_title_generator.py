import logging
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from spacy.parts_of_speech import *
from creativeaid.corpus_reader import CorpusReader
from creativeaid.nlp.text_utils import is_valid, clean_sentence, is_qualified
from creativeaid.models import Template, Title, CreativeTitle
from creativeaid.nlp import NLP

"""
---------
Insertion
---------
For the insertions, an adjective (or adverb) k is inserted before the noun (verb) w that appears most often 
in an appropriate dependency relation with it (i.e. â€œamodâ€� and â€œadvmodâ€� respectively).

------------------from generator.text_utils import is_clean, clean_sentence, is_qualified, is_stop-------------------------
check cosine similarity of two vector array
-------------------------------------------
from sklearn.metrics.pairwise import cosine_similarity
cosine_similarity([1, 0, -1], [-1,-1, 0])
array([[-0.5]])
"""

logging.basicConfig(format=u'[LINE:%(lineno)d]# %(levelname)-8s [%(asctime)s]  %(message)s', level=logging.NOTSET)

keywords_path = 'model/keywords'
headline_count = 1493775
keyword_threshold = 0.02


class CreativeTitleGenerator:
    def __init__(self, nlp, keyword_coverage=None):
        self.nlp = nlp
        if keyword_coverage is None:
            self.word_coverage = pickle.load(open(keywords_path, 'rb'))
        else:
            self.word_coverage = keyword_coverage

    def _clean(self, templates):
        return [(line.strip(), idx) for idx, line in enumerate(templates) if is_qualified(line.strip())]

    def generate_with_corpus(self, titles, corpus_reader, clean=True, minimum_candidates=10):
        """
        Generate creative text for titles from corpus reader
        :param titles: Title object
        :param corpus_reader: CorpusReader object
        :param clean: should clean sentences
        :param minimum_candidates: minimum candidates amount
        :return: max scored candidates
        """

        templates = []
        for file in corpus_reader.corpus():
            templates += file.contents_lines()

        if type(titles[0]) is not Title:
            titles = [Title(title) for title in titles]

        if clean:
            templates = self._clean(templates)
        return self.generate(titles, templates)

    def generate_with_templates(self, titles, templates, clean=True, minimum_candidates=10):
        """
        Generate creative text for titles from given templates
        :param titles: Title object
        :param clean: should clean sentences
        :param minimum_candidates: minimum candidates amount
        :return: max scored candidates
        """
        if type(titles[0]) is not Title:
            titles = [Title(title) for title in titles]

        if clean:
            templates = self._clean(templates)
        return self.generate(titles, templates)

    def enhance_title_info(self, title):
        title.doc = self.nlp.pars_sentence(title.text)
        title.important_keyword_indexes = self.important_keywords_indexes(title)
        title.important_keyword_similar_words = self.similar_keywords_for_indexes(title)

        sent = []
        for idx in title.important_keyword_indexes:
            sent.append(title[idx].text)
            for word in title.important_keyword_similar_words[idx]:
                sent.append(word)

        title.doc_ = self.nlp.pars_sentence(" ".join(sent))
        return title

    def generate(self, titles, templates):
        candidates = []
        for title in titles:
            logging.debug('==============================================================================')
            logging.debug(f'Title: {title.text}\n')
            title = self.enhance_title_info(title)
            creative_sentences = self.search_candidates_for_creative_sentences(title, templates)
            for creative_sentence in creative_sentences:
                v, replaced, inserted = self.substitute_words(creative_sentence, title)
                if replaced or inserted:
                    creative_title = CreativeTitle(title, creative_sentence, v)
                    candidates.append(creative_title)
                    logging.debug(
                        f"template:{creative_sentence.text} | modified:{v} | replaced:{replaced} | inserted:{inserted}")
        return candidates

    def search_candidates_for_creative_sentences(self, title, templates):
        candidate_templates = []
        for template, _ in self.nlp.pars_document(templates, as_tuples=True):
            template = Template(template, template.text)
            template.nlp_text = self.nlp.pars_sentence(clean_sentence(template.doc.text))

            sentence_similarity_score = self.title_sentence_similarity(title, template)
            keyword_score = self.keyword_score(title, template)

            if sentence_similarity_score >= 0.5 and keyword_score >= 0.5:
                candidate_templates.append(((sentence_similarity_score + keyword_score) / 2, template))

        candidate_templates = sorted(candidate_templates, key=lambda tup: tup[0], reverse=True)
        candidate_templates = [candidate for _, candidate in candidate_templates]
        return candidate_templates

    def title_sentence_similarity(self, title, sentence):
        similarity_score = 0
        if title.doc.has_vector and sentence.nlp_text.has_vector:
            similarity_score = title.doc.similarity(sentence.nlp_text)
        return similarity_score

    def keyword_score(self, title, sentence):
        # TODO look again
        i = 0
        token_score = 0
        for index, similar_words in title.important_keyword_similar_words.items():
            for similar_word in similar_words:
                for token in sentence.nlp_text:
                    if not is_valid(token):
                        continue

                    try:
                        voacab_id = self.nlp.vocab.strings[similar_word]
                        vocab_vector = self.nlp.vocab.vectors[voacab_id]
                        s = cosine_similarity([token.vector], [vocab_vector])[0][0]
                        if s > 0.5:
                            token_score += s
                            i += 1
                    except KeyError:
                        pass

        return token_score / i if token_score != 0 else 0

    def substitute_words(self, temple_sentence, title):
        important_keywords = title.important_keyword_indexes
        replace_words = {}
        insertion_words = {}

        for index in important_keywords:
            title_token = title[index]
            if not is_valid(title_token):
                continue
            i = 0
            for token in temple_sentence.doc:
                if not is_valid(token):
                    i += 1
                    continue
                # Replace word
                if token.pos is title_token.pos:
                    score = token.similarity(title_token)
                    if 0.5 < score < 0.9:
                        replace_words[i] = (title_token, token, score)
                # Insert word as adjective
                elif i > 0 and ((token.pos == NOUN and title_token.pos == ADJ) or
                                (token.pos == ADJ and title_token.pos == NOUN)):
                    score = token.similarity(title_token)
                    if 0.5 < score < 0.9:
                        insertion_words[i] = (title_token, token, score)
                i += 1

        sent = [(i, None, -1) for i in temple_sentence.doc]
        for index, val in replace_words.items():
            sent[index] = val

        # TODO - Insert at index
        for index, val in insertion_words.items():
            if sent[index][2] < val[2]:
                sent.insert(index, val)
                # sent[index - 1] = val

        return " ".join(sent[0].text for sent in sent), len(replace_words) > 0, len(insertion_words) > 0

    def important_keywords_indexes(self, tokens):
        i = -1
        keyword_index = []
        for token in tokens:
            i += 1
            try:
                if not is_valid(token):
                    continue
                # Checking named entity
                if token.ent_type != 0:
                    keyword_index.append(i)
                    continue

                # Checking word is an important word for words collected from corpus
                word = token.text if token.pos in [PRON, ADJ] else token.lemma_
                val = self.word_coverage[word.lower()]
                score = val / headline_count
                if score > keyword_threshold:
                    keyword_index.append(i)
            except KeyError:
                pass

        return keyword_index

    def similar_keywords_for_indexes(self, title):
        similar_words_for_indexes = {}
        for index in title.important_keyword_indexes:
            similar_words_for_indexes[index] = {title[index].text}
            # IF entity type
            if title[index].ent_type != 0:
                continue

            word = title[index].text if title[index].pos in [PROPN, ADJ] else title[index].lemma_
            try:
                similar_words = self.nlp.similar_word_for_sentence(word, title.text)
                similar_words = [token.lower() for token in similar_words
                                 if token.lower() != word and word not in token.lower()]
                similar_words = similar_words[:2]
                similar_words_for_indexes[index] = similar_words
            except KeyError:
                pass
        return similar_words_for_indexes


if __name__ == '__main__':
    # import csv, random, time
    #
    # titles = []
    # with open('C:/Users/ShimaK/Downloads/Compressed/million-headlines/abcnews-date-text.csv') as csv_file:
    #     csv_reader = csv.reader(csv_file, delimiter=',')
    #     line_count = 0
    #     for row in csv_reader:
    #         titles.append(row[1])
    #         line_count += 1
    #     print('Processed {} lines.'.format(line_count))
    # titles = titles[1:100]
    # titles = [Title(title) for title in titles]
    # randomy = random.choices(titles, k=10)

    titles = [
        # "The Obama administration is planning to issue a final rule designed to enhance the safety of offshore oil "
        # "drilling equipment",
        # "Russia’s defense ministry has rejected complaints by U.S. officials who claimed Russian attack planes buzzed "
        # "dangerously close to a U.S. Navy destroyer in the Baltic Sea earlier this week.",
        # "Time for Wales to step up",
        "Pyongyang drivers are feeling some pain at the pump as rising gas prices put a pinch on what has been major traffic growth"
    ]

    # process_begin_time_0 = time.process_time()
    cr = CorpusReader(
        "C:/Users/ShimaK/PycharmProjects/CreativeAid!/creativeaid/test_corpus/test_generate_corpus/cliche", "")
    ctg = CreativeTitleGenerator(NLP(word2vec_limit=500000, requires_word2vec=True))
    # logging.debug(f"Loading time {(time.process_time() - process_begin_time_0) / 60.0}")

    # process_begin_time_0 = time.process_time()
    s = ctg.generate_with_corpus(titles, cr)
    print(s)
    # logging.debug(f"Generate time {(time.process_time() - process_begin_time_0) / 60.0}")
