from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation

def featurize_corpus(corpus, ngram_range=(1,2), max_df=0.9, min_df=5, random_state=42):
    """
    Vectorize text corpus into TF-IDF and Count matrices, and optionally apply TruncatedSVD.

    Returns:
      tfidf_vectorizer, count_vectorizer, svd_model (or None),
      X_tfidf, X_counts, X_tfidf_reduced
    """
    # initialize vectorizers
    tfidf = TfidfVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df)
    count = CountVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df)

    # fit & transform
    X_tfidf = tfidf.fit_transform(corpus)
    X_counts = count.fit_transform(corpus)

    # dimensionality reduction
    svd_model = None
    svd_components=100
    X_tfidf_reduced = None
    if svd_components is not None:
        svd_model = TruncatedSVD(n_components=svd_components, random_state=random_state)
        X_tfidf_reduced = svd_model.fit_transform(X_tfidf)

    return tfidf, count, svd_model, X_tfidf, X_counts, X_tfidf_reduced

def extraction_pipeline(
    df,
    text_col='clean_text',
    n_topics=10,
    ngram_range=(1,2),
    max_df=0.9,
    min_df=5,
    random_state=42
):
    """
    Full pipeline: featurize corpus, optionally reduce dims, extract topics via NMF and LDA.

    Returns:
      df with 'topic_id_nmf' and 'topic_id_lda' columns
    """
    # featurize
    tfidf_vec, count_vec, svd_model, X_tfidf, X_counts, X_reduced = featurize_corpus(
        df[text_col], 
        ngram_range, 
        max_df, 
        min_df
    )

    # matrix for topic models
    nmf_input = X_tfidf

    # fit NMF
    nmf = NMF(n_components=n_topics, random_state=random_state)
    W = nmf.fit_transform(nmf_input)
    df['topic_id_nmf'] = W.argmax(axis=1)

    # top terms per topic
    terms = tfidf_vec.get_feature_names_out()
    print("NMF Topics:")
    for i, comp in enumerate(nmf.components_):
        top = [terms[j] for j in comp.argsort()[-10:][::-1]]
        print(f" Topic {i}: {', '.join(top)}")

    # LDA
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=random_state, learning_method='batch')
    min_val = X_reduced.min()
    X_nlda = X_reduced - min_val
    W_lda = lda.fit_transform(X_nlda)
    df['topic_id_lda'] = W_lda.argmax(axis=1)
    terms_lda = count_vec.get_feature_names_out()
    print("LDA Topics:")
    
    for i, comp in enumerate(lda.components_):
        top = [terms_lda[j] for j in comp.argsort()[-10:][::-1]]
        print(f" Topic {i}: {', '.join(top)}")

    return df, nmf, lda, tfidf_vec, count_vec