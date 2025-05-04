from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

def compute_coherence(model, vec, docs, kind="c_v", topn=10):
    dictionary = Dictionary(docs)
    corpus = [dictionary.doc2bow(doc) for doc in docs]

    # extract top‑terms per topic
    terms = vec.get_feature_names_out()
    topic_terms = [
        [terms[i] for i in comp.argsort()[-topn:]]
        for comp in model.components_
    ]

    cm = CoherenceModel(
        topics=topic_terms,
        texts=docs,
        dictionary=dictionary,
        coherence=kind
    )
    return cm.get_coherence()