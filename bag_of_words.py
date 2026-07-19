import re

class BagOfWords:
    def __init__(self):
        self.vocabulary = {}
        self.feature_names = []

    def preprocess(self, text):
        text = text.lower()
        text = re.sub(r'[^a-z0-9\s]', '', text)
        return text.split()

    def fit(self, documents):
        unique_words = set()

        for doc in documents:
            words = self.preprocess(doc)
            unique_words.update(words)

        self.feature_names = sorted(unique_words)
        self.vocabulary = {
            word: index
            for index, word in enumerate(self.feature_names)
        }

        return self

    def transform(self, documents):
        matrix = []

        for doc in documents:
            words = self.preprocess(doc)

            vector = [0] * len(self.feature_names)

            for word in words:
                if word in self.vocabulary:
                    index = self.vocabulary[word]
                    vector[index] += 1

            matrix.append(vector)

        return matrix

    def fit_transform(self, documents):
        self.fit(documents)
        return self.transform(documents)

    def get_feature_names(self):
        return self.feature_names