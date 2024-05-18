import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from conllu import parse_incr

data_dir = 'data'
retrain_file_path = os.path.join(data_dir, 'retrain.txt')

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
                if len(columns) == 10:
                    current_sentence.append((columns[1], columns[3]))  # form and UPOS
    if current_sentence:  
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
                if len(columns) >= 4: 
                    darija_word = columns[1]
                    pos_tag = columns[3]
                    current_sentence.append((darija_word, pos_tag)) 
    if current_sentence:  
        sentences.append(current_sentence)
    return sentences


# train_data = read_text_file('Train3ALG.txt')      # algerian corpus with just NOUN VERB DET, tested on algerian
# test_data = read_text_file('Test3ALG.txt')        # .8 accuracy

# train_data = read_text_file('AllTopicsTags.txt')    #our corpus and our tags
# test_data = read_text_file('testMarche.txt')        # .68 accuracy


# train_data = read_conllu_file('train.conllu')          # alegiran corpus with all tags on algerian
# test_data = read_conllu_file('test.conllu')            # .74 accuracy

# train_data = read_conllu_file('train.conllu')          # alegiran corpus with all tags tested on our corpus
# test_data = read_text_file('AllTopicsTags.txt')        # .45 accuracy

train_data = read_text_file('Train3ALG.txt')          # alegiran corpus with 3 tags tested on our corpus with 3 tags
test_data = read_text_file('OurTags3.txt')            #  .65 accuracy



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

#for initial training
X_train1 = [sent2features(sentence) for sentence in train_data]
y_train1 = [sent2labels(sentence) for sentence in train_data]


#testing
X_test = [sent2features(sentence) for sentence in test_data]
y_test = [sent2labels(sentence) for sentence in test_data]

# Step 3: Model Training
vectorizer = DictVectorizer()

#for inital training
X_train_vectorized1 = vectorizer.fit_transform([item for sublist in X_train1 for item in sublist])
X_test_vectorized = vectorizer.transform([item for sublist in X_test for item in sublist])

#setting up the classifier
classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=None, max_iter=5, tol=None)

#initial training
classifier.fit(X_train_vectorized1, [item for sublist in y_train1 for item in sublist])

#for retraining
if os.path.exists(retrain_file_path):
    retrain_data = read_text_file('retrain.txt')

    X_train2 = [sent2features(sentence) for sentence in retrain_data]
    y_train2 = [sent2labels(sentence) for sentence in retrain_data]

    X_Retrain_vectorized = vectorizer.transform([item for sublist in X_train2 for item in sublist])

    classifier.partial_fit(X_Retrain_vectorized, [item for sublist in y_train2 for item in sublist], classes=classifier.classes_)
    print("Retraining done!")

# Step 4: Evaluation
y_pred = classifier.predict(X_test_vectorized)


accuracy = accuracy_score([item for sublist in y_test for item in sublist], y_pred)
print("Accuracy:", accuracy)


# Step 5: Saving Misclassified Sentences
def save_misclassified_sentences(test_data, y_test, y_pred, file_name):
    file_path=os.path.join(data_dir, file_name)

    with open(file_path, 'a', encoding='latin-1') as file:
        for sentence, true_labels, predicted_labels in zip(test_data, y_test, y_pred):
            for word, true_label, predicted_label in zip(sentence, true_labels, predicted_labels):
                if true_label != predicted_labels:
                    # file.write(f"{word[0]}\t_\t{true_label}\t_\t{predicted_labels}\n")
                    file.write(f"{word[0]}\t_\t{true_label}\t_\n")
            file.write('\n')

save_misclassified_sentences(test_data, y_test, y_pred, 'retrain.txt')




# trying to get it to work with context

# def save_misclassified_sentences(test_data, y_test, y_pred, file_path='retrain.txt'):
#     with open(file_path, 'a', encoding='latin-1') as file:
#         for sentence, true_labels, predicted_labels in zip(test_data, y_test, y_pred):
#             misclassified_sentence = []
#             original_sentence = ""
#             for word, true_label, predicted_label in zip(sentence, true_labels, y_pred):
#                 if true_label != predicted_label:
#                     file.write(f"{word[0]}\t_\t_\t{true_label}\n")
#                     misclassified_sentence.append(word[0])  # Append the misclassified word
#                 else:
#                     misclassified_sentence.append(word[0])  # Append the correctly classified word
#                 original_sentence = word[0] if original_sentence == "" else original_sentence + " " + word[0]
#             # Write the original sentence line
#             file.write(f"# text = {original_sentence}\n")
#             file.write('\n')