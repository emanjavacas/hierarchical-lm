
<<<<<<< HEAD
import numpy as np
import pandas as pd
import RAKE
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
import spacy

nlp = spacy.load('en_core_web_sm', pipeline=['tagger', 'lemmatizer', 'tokenizer'])
tokenizer = spacy.lang.en.English().Defaults.create_tokenizer(nlp)


def preprocess(doc):
    output = []
    for sent in doc:
        output += [w.lemma_ for w in nlp(sent, disable=['parser'])
                   if not (w.text in spacy.lang.en.STOP_WORDS or not w.text.isalpha())]
    return output


def export_lims(output='limericks.txt'):
    with open(output, 'w+') as f:
        for lim in load_limmericks():
            f.write(' '.join(lim) + '\n')


def get_rake_keywords(doc):
    return RAKE.Rake(list(spacy.lang.en.stop_words.STOP_WORDS)).run(doc)


def load_limmericks(path='./limericks.csv'):
    for _, lim in pd.read_csv('./limericks.csv', header=None).iterrows():
        item = []
        for w in preprocess(lim.values[0].split('\r\n')):
            if "?" in w:
                if w == '?':
                    item.append(w)
                else:
                    for subw in w.split("?"):
                        item.append(subw)
            else:
                item.append(w)
        yield item


def load_sonnets(path='./sonnets-gutenberg/sonnet_train.txt'):
    with open(path) as f:
        for line in f:
            yield preprocess(line.split('<eos>'))


def find_topic(texts, topic_model, n_topics, vec_model="tf", **kwargs):
    """Return a list of topics from texts by topic models
    texts: array-like strings
    topic_model: {"nmf", "svd", "lda"} for LSA_NMF, LSA_SVD, LDA
    n_topics
    vec_model: {"tf", "tfidf"} for term_freq, term_freq_inverse_doc_freq
    thr: threshold for finding keywords in a topic model
    """
    vectorizer = CountVectorizer if vec_model == "tf" else TfidfVectorizer
    vectorizer = vectorizer(lowercase=True, min_df=2)
    texts = vectorizer.fit_transform(texts)
    topic_model = {
        "nmf": NMF,
        "svd": TruncatedSVD,
        "lda": LatentDirichletAllocation
    }[topic_model](n_topics, **kwargs).fit(texts)

    return topic_model, vectorizer


def topic_keywords(model, vectorizer, n_words=25, thr=1e-2):
    keywords = np.array(vectorizer.get_feature_names())
    topics = []
    for weight in model.components_:
        locs = (-weight[weight > thr]).argsort()[:n_words]
        topics.append(keywords[locs])
    return topics


# if __name__ == '__main__':
if True:
    # texts = list(load_limmericks())
    # texts = load_sonnets()
    topic_model, vectorizer = find_topic(
        [' '.join(text) for text in texts],
        # "lda", 50, max_iter=10, batch_size=128, learning_method='online'
        "nmf", 50
    )
    for topic in topic_keywords(topic_model, vectorizer): print('|'.join(topic)); print()
=======

import json
from gensim.summarization import keywords


def load_verses(path='data/ohhla-new.jsonl'):
    with open(path, errors='ignore') as f:
        for idx, line in enumerate(f):
            try:
                for verse in json.loads(line)['text']:
                    yield [w['token'] for line in verse for w in line if w['token']]
            except json.decoder.JSONDecodeError:
                print("Couldn't read song #{}".format(idx+1))


verses = load_verses()

for _ in range(100):
    try:
        text = ' '.join(next(verses))
        ks = keywords(text, ratio=0.25).split('\n')
        print(text)
        print()
        print("Keywords: ", '-'.join(ks))
        print()
        print('-' * 10)
        print()
    except:
        pass
>>>>>>> 45dceb498ec7878b762f7fec89455f21d6c2dfcb
