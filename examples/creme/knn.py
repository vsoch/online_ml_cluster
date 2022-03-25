from online_ml_custom.creme.knn import KNeighbors
from creme import feature_extraction

sentences = [
    "The quick brown cat ran over the lazy dog",
    "The quick brown fox ran over the lazy dog",
    "The quick brown dog ran over the lazy cat",
    "I think therefore I am",
    "You think therefore you are",
    "We think therefore we are not.",
]


model = feature_extraction.TFIDF() | KNeighbors(n_neighbors=2)

# You can provide an identifier (uid) for later association (here we use the forloop index)

for uid, sentence in enumerate(sentences):
    model.fit_one(x=sentence, uid=uid)

# Adding to window 0: the quick brown cat ran over lazy dog
# Adding to window 1: the quick brown fox ran over lazy dog
# Adding to window 3: think therefore am
# Adding to window 4: you think therefore are
# Adding to window 5: we think therefore are not

# Or not! Note that if you run this directly after, we will be under the minimum difference
# threshold and you won't see these points added (they are redundant).

for sentence in sentences:
    model.fit_one(x=sentence)

# And then we can get predictions! But you'll get back the full list of neighbors for you
# to decide how to act upon.

for sentence in sentences:
    res = model.predict_one(x=sentence)
    print(res)
