from bag_of_words import BagOfWords

documents = [
    "I love machine learning",
    "Machine learning is amazing",
    "I love Python programming",
    "Python is used for machine learning"
]

bow = BagOfWords()

X = bow.fit_transform(documents)

print("Vocabulary")
print(bow.vocabulary)

print("\nFeature Names")
print(bow.get_feature_names())

print("\nBag of Words Matrix")

for row in X:
    print(row)