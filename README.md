# Bag of Words from Scratch

A pure Python implementation of the Bag of Words (BoW) algorithm without using scikit-learn.

## Features

- Text preprocessing
- Vocabulary creation
- Count-based vectorization
- fit()
- transform()
- fit_transform()
- get_feature_names()

## Usage

```python
from bag_of_words import BagOfWords

docs = [
    "I love NLP",
    "NLP is fun"
]

bow = BagOfWords()

X = bow.fit_transform(docs)

print(X)
```

## Future Improvements

- Stopword removal
- TF-IDF
- N-grams
- Binary vectors
- max_features