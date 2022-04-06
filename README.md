# Online ML Custom Models

These are custom models intended for [river](https://github.com/online-ml/river) or [creme](https://gituhb.com/MaxHalford/creme).
The intention is to be able to use them in multiple places (and test out new ideas) without relying on official river releases.

## Quick Start

```bash
$ pip install -e .
```

or from GitHub

```bash
$ pip install git+https://github.com/vsoch/online_ml_custom.git@main
```

## Models

### KNeighbors

I wanted a custom KNN classifier that would:

 - add new data to the window only given it is more different than some minimum similarity threshold (default `min_distance_keep` 0.05)
 - allow storing an identifier with the datum to be able to associate later
 - return neighbors instead of a prediction (for me to act on)
 
Complete details are in [this GitHub issue comment](https://github.com/online-ml/river/issues/891#issuecomment-1078633483)
and you can see the example in [examples/creme/knn.py](examples/creme/knn.py)

Here is example usage:

```python
from online_ml_custom.creme.knn import KNeighbors
from creme import feature_extraction

sentences = ["The quick brown cat ran over the lazy dog", 
             "The quick brown fox ran over the lazy dog",
             "The quick brown dog ran over the lazy cat",
             "I think therefore I am",
             "You think therefore you are",
             "We think therefore we are not."]


model = feature_extraction.TFIDF() | KNeighbors(n_neighbors=2)
```

You can provide an identifier (uid) for later association (here we use the forloop index)

```python
for uid, sentence in enumerate(sentences):
    model.fit_one(x=sentence, uid=uid)

# Adding to window 0: the quick brown cat ran over lazy dog
# Adding to window 1: the quick brown fox ran over lazy dog
# Adding to window 3: think therefore am
# Adding to window 4: you think therefore are
# Adding to window 5: we think therefore are not
```

Or not! Note that if you run this directly after, we will be under the minimum difference
threshold and you won't see these points added (they are redundant).

```python
for sentence in sentences:
    model.fit_one(x=sentence)
```

And then we can get predictions! But you'll get back the full list of neighbors for you
to decide how to act upon.

```python
for sentence in sentences:
    res = model.predict_one(x=sentence)
    print(res)
```
```python
[({'we': 0.7071067811865475,
   'think': 0.35355339059327373,
   'therefore': 0.35355339059327373,
   'are': 0.35355339059327373,
   'not': 0.35355339059327373},
  None,
  5,
  0.02163152256588398),
 ({'you': 0.7559289460184544,
   'think': 0.3779644730092272,
   'therefore': 0.3779644730092272,
   'are': 0.3779644730092272},
  None,
  4,
  1.3528150063811903)]
```

The above is one result object, and note since k=2 we are getting two back. Each item
in the above list is a tuple of neighbors with an added similarity score. The first
item in the tuple are the features (x), followed by an actual value Y (we didn't provide so it's empty)
and then the identifier and similarity score. For the above, the first match is the same data point,
and the reason the score isn't exactly 1 is because of the TFIDF - the frequency of a particular feature
across our samples changes as we learn more, so the value you derive for this new item is slightly different
than the old one.

## License

This code is licensed under the MPL 2.0 [LICENSE](LICENSE).
