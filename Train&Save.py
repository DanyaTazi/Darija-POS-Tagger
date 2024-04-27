import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from conllu import parse_incr
import joblib

data_dir = 'data'

# Step 1: Data Preprocessing
def read_conllu_file(file_path):
    file_path = os.path.join(data_dir, file_path)
    sentences = []
    with open(file_path, 'r', encoding='latin-1') as file:
        current_sentence = []
        for line in file:
            if line.startswith('# sent_id'):
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            elif not line.startswith('#'):
                columns = line.strip().split('\t')
                if len(columns) == 10:  # Assuming standard CoNLL-U format with 10 columns
                    current_sentence.append((columns[1], columns[3]))  # Form and UPOS
    if current_sentence:  # Append the last sentence
        sentences.append(current_sentence)
    return sentences


def read_text_file(file_path):
    file_path = os.path.join(data_dir, file_path)
    sentences = []
    with open(file_path, 'r', encoding='latin-1') as file:
        current_sentence = []
        for line in file:
            line = line.strip()
            if line.startswith('# sent_id'):
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
            elif not line.startswith('#'):
                columns = line.split('\t')
                if len(columns) >= 4:  # At least four columns should be present
                    darija_word = columns[1]
                    pos_tag = columns[3]
                    current_sentence.append((darija_word, pos_tag))  # Darija word and POS tag
    if current_sentence:  # Append the last sentence
        sentences.append(current_sentence)
    return sentences


train_data = read_text_file('AllTopicsTags.txt')
test_data = read_text_file('testMarche.txt')

# Step 2: Feature Extraction
def extract_features(sentence, index):
    word = sentence[index][0]
    features = {
        'word': word,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'prev_word': '' if index == 0 else sentence[index - 1][0],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1][0],
    }
    return features

def sent2features(sentence):
    return [extract_features(sentence, i) for i in range(len(sentence))]

def sent2labels(sentence):
    return [label for _, label in sentence]

X_train = [sent2features(sentence) for sentence in train_data]
y_train = [sent2labels(sentence) for sentence in train_data]

X_test = [sent2features(sentence) for sentence in test_data]
y_test = [sent2labels(sentence) for sentence in test_data]

# Step 3: Model Training or Loading
model_filename = 'pos_tagger_model.joblib'
vectorizer_filename = 'vectorizer.joblib'

try:
    classifier = joblib.load(model_filename)
    vectorizer = joblib.load(vectorizer_filename)
except FileNotFoundError:
    vectorizer = DictVectorizer(sparse=False)
    X_train_vectorized = vectorizer.fit_transform([item for sublist in X_train for item in sublist])
    
    classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=None, max_iter=5, tol=None)
    classifier.fit(X_train_vectorized, [item for sublist in y_train for item in sublist])
    
    # Save the trained model and vectorizer
    joblib.dump(classifier, model_filename)
    joblib.dump(vectorizer, vectorizer_filename)

# Step 4: Evaluation
X_test_vectorized = vectorizer.transform([item for sublist in X_test for item in sublist])

y_pred = classifier.predict(X_test_vectorized)
accuracy = accuracy_score([item for sublist in y_test for item in sublist], y_pred)
print("Accuracy:", accuracy)

