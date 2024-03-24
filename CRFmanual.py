#CRF with manual word testing inputs

from sklearn_crfsuite import CRF
from preconllu import parse_corpus

# Example usage:
corpus_file = 'TrainArabizi.conllu'  # Specify the path to your file containing your data
X, y = parse_corpus(corpus_file)  # Assuming parse_corpus function correctly parses your XML file

# Initialize and train CRF
model = CRF()
model.fit([X], [y])  # Fit the model on the entire dataset

# Define test words
test_words = ['b7ar', 'kelb', 'khda', 'mchina']

# Predict POS tags for the test words
predicted_tags = model.predict([test_words])

# Print the predicted tags for the test words
for word, tags in zip(test_words, predicted_tags[0]):
    print(f'Word: {word}, Predicted POS tags: {tags}')
