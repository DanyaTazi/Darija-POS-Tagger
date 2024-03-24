from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer
from training import sent2features, vectorizer, y_train, X_train_vectorized, sent2features
from sklearn.linear_model import SGDClassifier

# Load the trained model and vectorizer
classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=None, max_iter=5, tol=None)
classifier.fit(X_train_vectorized, [item for sublist in y_train for item in sublist])

def read_custom_sentences(input_sentences):
    sentences = []
    for sent_id, sentence in enumerate(input_sentences, start=1):
        sent_with_null_tags = [(word, None) for word in sentence]
        sentences.append(sent_with_null_tags)
    return sentences

# Prepare test sentences
input_sentences = [
    ["7atoh", "f", "sachie", "o", "b3et", "tayech", "l", "al", "be7ar"],
    ["walah", "yrid", "messi", "rah", "yal3ab", "m3a", "ziani"],
    # Add more sentences as needed
]

sentences = read_custom_sentences(input_sentences)

# Extract features for test sentences
X_test_features = [sent2features(sentence) for sentence in sentences]
print(X_test_features)
# Predict POS tags for test sentences
X_test_vectorized = vectorizer.transform([item for sublist in X_test_features for item in sublist])
print(X_test_vectorized)
y_pred = classifier.predict(X_test_vectorized)
print(y_pred)

# Assuming you have the actual POS tags for comparison
actual_tags = [["VERB", "ADP", "NOUN", "CCONJ", "VERB", "ADJ", "ADP", "DET", "NOUN"], ["INTJ", "VERB", "NPP", "VERB", "VERB", "ADP", "NPP"]]


for i in range(len(sentences)):
    print("Sentence:", sentences[i])
    print("Actual Tags:", actual_tags[i])
    print("Predicted Tags:", y_pred[:len(actual_tags[i])])    
    print()

# Compute accuracy
accuracy = accuracy_score([item for sublist in actual_tags for item in sublist], y_pred)
print("Accuracy:", accuracy)
