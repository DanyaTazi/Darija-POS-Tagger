import os
from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_accuracy_score
from conllu import parse_incr

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
                if len(columns) == 10:  
                    current_sentence.append((columns[1], columns[3])) 
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
# test_data = read_text_file('Test3ALG.txt')        # .82 accuracy (better)

# train_data = read_text_file('AllTopicsTags.txt')    #our corpus and our tags
# test_data = read_text_file('testMarche.txt')        # .64 accuracy (worse)

# train_data = read_conllu_file('train.conllu')          # alegiran corpus with all tags on algerian
# test_data = read_conllu_file('test.conllu')            # .74 accuracy (same)

# train_data = read_conllu_file('train.conllu')          # alegiran corpus with all tags tested on our corpus
# test_data = read_text_file('AllTopicsTags.txt')        # .43 accuracy (worse)

# train_data = read_text_file('Train3ALG.txt')          # alegiran corpus with 3 tags tested on our corpus with 3 tags
# test_data = read_text_file('OurTags3.txt')            #  .66 accuracy (bit better)

train_data = read_text_file('NEWours3.txt')             #our corpus new 3 tags on our marche 3 new
test_data = read_text_file('NEWmarche3.txt')            # .76 accuracy (bit worse than stochastic)

# train_data = read_text_file('NEWtrainALG3.txt')             #algerian new 3 tags on algerian 3 new
# test_data = read_text_file('NEWtestALG3.txt')             # .79 accuracy (bit worse also than noun det verb)

# train_data = read_text_file('NEWtrainALG3.txt')             #algerian new 3 tags on ours 3 new
# test_data = read_text_file('NEWours3.txt')                  # .6 accuracy 

#train_data = read_text_file('AlgerianAUG.txt')             #algerian new 3 tags on ours 3 new
#test_data = read_text_file('NEWours3.txt')                  # .6 accuracy

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

# Step 3: Model Training
crf = CRF(algorithm='lbfgs', max_iterations=100, all_possible_transitions=True)
crf.fit(X_train, y_train)


# Step 4: Evaluation
y_pred = crf.predict(X_test)

# Flatten the true labels and predicted labels
flat_y_test = [label for sentence_labels in y_test for label in sentence_labels]
flat_y_pred = [label for sentence_labels in y_pred for label in sentence_labels]

# making sure that the number of samples in y_test and y_pred matches
if len(flat_y_test) != len(flat_y_pred):
    print("Number of samples in y_test and y_pred does not match")
    
else:
    # Calculate accuracy
    correct_predictions = sum(1 for true_label, pred_label in zip(flat_y_test, flat_y_pred) if true_label == pred_label)
    accuracy = correct_predictions / len(flat_y_test)
    print("Accuracy:", accuracy)


# Extra: Saving Misclassified Sentences
def save_misclassified_sentences(test_data, y_test, y_pred, file_name):
    file_path=os.path.join(data_dir, file_name)

    with open(file_path, 'a', encoding='latin-1') as file:
        for sentence, true_labels, predicted_labels in zip(test_data, y_test, y_pred):
            for word, true_label, predicted_label in zip(sentence, true_labels, predicted_labels):
                if true_label != predicted_labels:
                    # file.write(f"{word[0]}\t_\t{true_label}\t_\t{predicted_labels}\n")
                    file.write(f"{word[0]}\t_\t{true_label}\t_\n")
            file.write('\n')

save_misclassified_sentences(test_data, y_test, y_pred, 'AlgerianAUG.txt')

#IS IT NOT CHANGING WITH THE AUG FILE