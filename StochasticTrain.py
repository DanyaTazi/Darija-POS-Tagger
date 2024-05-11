import os
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from conllu import parse_incr

data_dir = 'data'

# Step 1: Data Preprocessing -----------------------------------------------------------
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
                    current_sentence.append((columns[1], columns[3]))  # Form and UPOS
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

# train_data = read_text_file('Train3ALG.txt')          # alegiran corpus with 3 tags tested on our corpus with 3 tags
# test_data = read_text_file('OurTags3.txt')            #  .65 accuracy

train_data = read_text_file('NEWours3.txt')             #our corpus new 3 tags on our marche 3 new
test_data = read_text_file('NEWmarche3.txt')            # .78 accuracy

# train_data = read_text_file('NEWtrainALG3.txt')             #algerian new 3 tags on algerian 3 new
# test_data = read_text_file('NEWtestALG3.txt')             # .82 accuracy 

# train_data = read_text_file('NEWtrainALG3.txt')             #algerian new 3 tags on ours 3 new
# test_data = read_text_file('NEWours3.txt')                  # .62 accuracy (bit worse also than noun det verb)


# Step 2: Feature Extraction -------------------------------------------------
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

# Step 3: Model Training ---------------------------------------
vectorizer = DictVectorizer()
X_train_vectorized = vectorizer.fit_transform([item for sublist in X_train for item in sublist])
X_test_vectorized = vectorizer.transform([item for sublist in X_test for item in sublist])

classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-4, random_state=None, eta0=0.1) #new improved params
classifier.fit(X_train_vectorized, [item for sublist in y_train for item in sublist])

# Step 4: Evaluation ------------------------------------------
y_pred = classifier.predict(X_test_vectorized)
accuracy = accuracy_score([item for sublist in y_test for item in sublist], y_pred)
print("Accuracy:", accuracy)
