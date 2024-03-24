from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics
from preconllu import parse_corpus

# Load preprocessed data
corpus_file = 'TrainArabizi.conllu'
X, y = parse_corpus(corpus_file)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

# Initialize and train CRF
model = CRF()
model.fit([X_train], [y_train])  # Note: Wrap X_train and y_train with lists to form a sequence

# Predict POS tags for the test data
predicted_tags = model.predict([X_test])  # Note: Wrap X_test with a list to form a sequence

# Evaluate the performance of the model
print(metrics.flat_accuracy_score([y_test], predicted_tags))  # Note: Wrap y_test with a list to form a sequence
