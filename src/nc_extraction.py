from spacy.matcher import Matcher
from typing import List, Set
import spacy

class Noun_Phrase_extraction:
    """
    This class contains methods to clean text and extract noun chunks from it
    """

    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def prepocess_document(self, news_text: str) -> str:
        """
        Remove stop words and puntuations and return filtered news
        """
        doc = self.nlp(news_text)
        filtered_sentence: List[str] = []
        for token in doc:
            if (token.is_stop is False) and (token.is_punct is False):
                filtered_sentence.append(token.text)
        return " ".join(filtered_sentence)

    def get_noun_phrases(self, news_text: str) -> Set:
        """
        Capture tokens that match with pattern of noun phrase 
        returns Set containing noun chuncks
        """

        doc: any = self.nlp(self.prepocess_document(news_text))

        matcher: any = Matcher(self.nlp.vocab)
        noun_phrases: Set = set()

        pattern_all_noun: List = [{"POS": "NOUN"}, {
            "POS": "NOUN"}, {'POS': 'NOUN', 'op': '?'}]
        pattern_all_propernoun: List = [{"POS": "PROPN"}, {
            "POS": "PROPN"}, {'POS': 'PROPN', 'op': '?'}]
        pattern_email = [{'LIKE_EMAIL': False}, {'LIKE_URL': False}]

        matcher.add("allnoun", [pattern_all_noun])
        matcher.add("allpropernoun", [pattern_all_propernoun])
        matcher.add('email', [pattern_email])

        # span is noun phrase if it matches defined patterns
        matches = matcher(doc)
        for match_id, start, end in matches:
            noun_phrases.add(doc[start:end].text)

        return noun_phrases
