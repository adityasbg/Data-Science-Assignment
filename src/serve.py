from flask import Flask, request, jsonify
from typing import Dict, List ,Set
from tqdm import tqdm

from nc_extraction import Noun_Phrase_extraction
from tfidf_vectorize import Vectorizer

# Create the application instance
app = Flask(__name__)


# Create a URL route in our application for "/"
@app.route('/nc_keyword_extraction', methods=['POST'])
def top_noun_phrase():
    """
    Request: {"data": ["News Article 1", "News Article 2", "News Article 3" ... "News Article 20"]}

    return: Response: {"noun_chunks": {"nc": "Trump Administration", "COVID Vaccine", ... "Crypto trading"}}
    """

    # get request json
    news_text_dict: Dict = request.get_json()
    news_text_list: List[str] = news_text_dict.get('data')

    top_noun_phrase: List = [str]
    # if data is not empty
    if news_text_list is not None:

        extractor = Noun_Phrase_extraction()
        batch_noun_phrases: Set = set()

        # iterate through all the text and extract noun phrases
        for news_text in tqdm(news_text_list):
            preprocessed_news_string = extractor.prepocess_document(news_text)
            document_noun_phrase = extractor.get_noun_phrases(preprocessed_news_string)
            batch_noun_phrases.update(document_noun_phrase)

        # load tfidf vectorizer
        vectorizer = Vectorizer()
        tfidf = vectorizer.load_tfidf_vectorizer()
        # get top noun phrases
        top_noun_phrase = vectorizer.get_important_noun_chunck(tfidf, batch_noun_phrases)

    return jsonify({"noun_chunck":{'nc': top_noun_phrase}})

if __name__ == '__main__':
    app.run(debug=True)
