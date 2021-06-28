from typing import List, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle


class Vectorizer:
    """
    This class contains utility method to fit tfidf vectorizer on noun chunks and return important
    noun chunks based on tfidf scores

    """

    def __init__(self, tfidfvector_path='models/tfidf_vectorizer.pkl'):
        self.tfidfvector_path: str = tfidfvector_path

    def transform_and_pickle_vectorizer(self, noun_chunk_sentence_list: List) -> None:
        """
        This function takes list of noun chuncks extracted from text and fits tfidf vectorizer
        """
        vectorizer: TfidfVectorizer = TfidfVectorizer()
        vectorizer.fit(noun_chunk_sentence_list)

        # save Tfidf vectorized
        with open(self.tfidfvector_path, 'wb') as file:
            pickle.dump(vectorizer, file)

    def get_important_noun_chunck(self, vectorizer: any, noun_phrases: Set) -> List:
        """
        Computes noun chuncks importance based on formula sum(tfidf_vector)/len(tfidf vocab)
        and returns top 20 noun chuncks
        """

        noun_phrases: List = list(noun_phrases)

        # sum of idf scores of each word in a sentence
        scores: List = []
        for noun_phrase in noun_phrases:
            sentence_vec = vectorizer.transform([noun_phrase])
            scores.append(np.sum(sentence_vec))

        # calculating 90 percentile
        ninty: float = np.percentile(scores, q=[90])

        # find index whose score is less than  90 percentile of scores
        args_index: List = np.argwhere(scores < ninty).flatten()

        noun_phrases: List = np.array(noun_phrases)
        scores: List = np.array(scores)

        noun_phrases: List = noun_phrases[args_index]
        scores: List = scores[args_index]

        # find top 10 scores in decreasing order
        top_10_scores_args: List = np.argsort(-scores)[:10]

        return [noun_phrases[index] for index in top_10_scores_args]

    def load_tfidf_vectorizer(self) -> TfidfVectorizer:
        """
        Load pickled tfidf vectorizer
        """

        with open(self.tfidfvector_path, 'rb') as file:
            tfidf_vectorizer = pickle.load(file)

        return tfidf_vectorizer
