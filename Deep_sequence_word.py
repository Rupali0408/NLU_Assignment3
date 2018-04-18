import codecs
import numpy as np
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

docs1 = []
docs2 = []
docs = []
with open('ner.txt', errors='replace') as fp:
    docs1 = fp.read().splitlines()
for item in docs1:
    t = tuple(item.split())
    docs2.append(t)
doc = []
for t in docs2:
    if t != ():
        doc.append(t)

    else:
        docs.append(doc)
        doc = []

data = []
w = []
l = []
for i, doc in enumerate(docs):

    # Obtain the list of tokens in the document
    tokens = [t for t, label in doc]
    labels = [label for t, label in doc]
    # Perform POS tagging
    tagged = nltk.pos_tag(tokens)

    # Take the word, POS tag, and its label
    data.append([(w, pos, label) for (w, label), (word, pos) in zip(doc, tagged)])
    for word in tokens:
        w.append(word)
    for label in labels:
        l.append(label)

sentences = data

words = list(set(w))
n_words = len(words)

tags = list(set(l))
n_tags = len(tags)

max_len = 75
word2idx = {w: i + 1 for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

from keras.preprocessing.sequence import pad_sequences
X = [[word2idx[w[0]] for w in s] for s in sentences]
X = pad_sequences(maxlen=max_len, sequences=X, padding="post", value=0)
y = [[tag2idx[w[2]] for w in s] for s in sentences]
y = pad_sequences(maxlen=max_len, sequences=y, padding="post", value=tag2idx["O"])
from keras.utils import to_categorical
y = [to_categorical(i, num_classes=n_tags) for i in y]
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.1)

from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from keras_contrib.layers import CRF

input = Input(shape=(max_len,))
model = Embedding(input_dim=n_words + 1, output_dim=20,
                  input_length=max_len, mask_zero=True)(input)  # 20-dim embedding
model = Bidirectional(LSTM(units=50, return_sequences=True,
                           recurrent_dropout=0.1))(model)  # variational biLSTM
model = TimeDistributed(Dense(50, activation="relu"))(model)  # a dense layer as suggested by neuralNer
crf = CRF(n_tags)  # CRF layer
out = crf(model)  # output

model = Model(input, out)

model.compile(optimizer="rmsprop", loss=crf.loss_function, metrics=[crf.accuracy])

truths = []
predictions = []
for i in range(len(y_te)):
    p = model.predict(np.array([X_te[i]]))
    p = np.argmax(p, axis=-1)
    true = np.argmax(y_te[i], -1)
    for w, t, pred in zip(X_te[i], true, p[0]):
        if w != 0:
            truths.append(t)
            predictions.append(pred)
#print(predictions)
accuracy = accuracy_score(truths, predictions)
print(accuracy)

print(classification_report(
    truths, predictions,
target_names=["D", "O","T"]))
