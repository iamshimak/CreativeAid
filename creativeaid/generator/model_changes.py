import pickle

o = "/root/PycharmProjects/CreativeAid!/creativeaid/identifier/verb_noun_freq_2019-03-27_23-31-01"
keywords_path = '/root/PycharmProjects/CreativeLine/app/models/keywords'

if __name__ == '__main__':
    keywords = pickle.load(open(keywords_path, 'rb'))
    keywords['want'] = 1502
    # val = keywords['popularity'] / 1493775
    pickle.dump(keywords, open(keywords_path, 'wb'))
