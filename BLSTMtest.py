import os
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, LSTM, Bidirectional, Embedding
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

data_dir = 'data'

# Step 1: Data Parsing ----------------------------------
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

# Step 2: Load text data ----------------------------------------------------------------


# train_data = read_text_file('NEWtrainALG3.txt')  #.96
# test_data = read_text_file('NEWtestALG3.txt')

# train_data = read_text_file('Train3ALG.txt')          # alegiran corpus with 3 tags tested on our corpus with 3 tags OLD
# test_data = read_text_file('OurTags3.txt')            #  .97 accuracy

# train_data = read_conllu_file('train.conllu')          # alegiran corpus with all tags tested on our corpus
# test_data = read_text_file('AllTopicsTags.txt')        # . accuracy

train_data = read_text_file('NEWtrainALG3.txt')             #algerian new 3 tags on ours 3 new
test_data = read_text_file('NEWours3.txt')                  #  accuracy 

# Step 3: Word Indexing ----------------------------------------------------------------------------
word2idx = {}
for sentence in train_data + test_data:
    for word, _ in sentence:
        if word not in word2idx:
            word2idx[word] = len(word2idx) + 1

# Step 4: Padding ---------------------------------------------------------------------------------
if len(train_data) > 0 and len(test_data) > 0:
    max_length = max(len(sentence) for sentence in train_data + test_data)
else:
    max_length = 0

X_train = [[word2idx[word] for word, _ in sentence] for sentence in train_data]
X_test = [[word2idx[word] for word, _ in sentence] for sentence in test_data]

if max_length > 0:
    X_train_padded = pad_sequences(X_train, maxlen=max_length, padding='post')
    X_test_padded = pad_sequences(X_test, maxlen=max_length, padding='post')
else:
    print("No sentences found in training or testing data.")

# Defining POS tags 
tag2idx = {'NOFLEX': 0, 'NOUN': 1, 'VERB': 2 }
default_index = 0

# Padding POS tag sequences to the same length as input sequences
y_train_padded = pad_sequences([[tag2idx.get(tag, default_index) for _, tag in sentence] for sentence in train_data],
                               maxlen=max_length, padding='post')
y_test_padded = pad_sequences([[tag2idx.get(tag, default_index) for _, tag in sentence] for sentence in test_data],
                              maxlen=max_length, padding='post')

# Step 5: Model Training -----------------------------------------------------------------------------------------
if max_length > 0:
    embedding_dim = 100
    num_classes = len(tag2idx)
    model = Sequential()
    model.add(Embedding(input_dim=len(word2idx) + 1, output_dim=embedding_dim))
    model.add(Bidirectional(LSTM(units=64, return_sequences=True)))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train_padded, y_train_padded, validation_data=(X_test_padded, y_test_padded), epochs=70, batch_size=32) #epochs and batch size modifiable 


    # Evaluation-----------------
    accuracy = model.evaluate(X_test_padded, y_test_padded)[1]
    print("Accuracy:", accuracy)

    #Printing info chart--------------
    model.summary()


    #Extra stuff for classification report-------------------
    # Predicting on test data
    predictions = model.predict(X_test_padded)
    predicted_classes = predictions.argmax(axis=-1)

    # Comparing predicted tags with actual tags
    correctly_predicted_tags = []
    for i in range(len(predicted_classes)):
        predicted_tags = [list(tag2idx.keys())[pred] for pred in predicted_classes[i]]
        actual_tags = [tag for _, tag in test_data[i]]
        correctly_predicted_tags.append([(word, predicted_tag, actual_tag) for word, predicted_tag, actual_tag in zip(test_data[i], predicted_tags, actual_tags)])

    # Calculate precision, recall, and F1-score---
    # Flatten the lists of actual and predicted tags
    actual_tags_flat = [tag for sentence in y_test_padded for tag in sentence]
    predicted_tags_flat = [tag for sentence in predicted_classes for tag in sentence]

    # Convert tag indices to tag names
    actual_tags_flat = [list(tag2idx.keys())[tag] for tag in actual_tags_flat]
    predicted_tags_flat = [list(tag2idx.keys())[tag] for tag in predicted_tags_flat]

    # Get unique tag names from the actual tags
    unique_actual_tags = set(actual_tags_flat)

    # get the scores
    target_names = list(unique_actual_tags)  # unique tag names as target names
    print(classification_report(actual_tags_flat, predicted_tags_flat, target_names=target_names))


    # # Calculate accuracy on predicted test set-----------------
    # def calculate_accuracy_on_predicted(predicted_tags, actual_tags):
    #     total_tags = len(actual_tags)
    #     correct_predictions = sum(1 for pred, actual in zip(predicted_tags, actual_tags) if pred == actual)
    #     accuracy = correct_predictions / total_tags if total_tags > 0 else 0
    #     # Print predicted tags next to actual tags for comparison
    #     print("Predicted Tags\tActual Tags")
    #     for i in range(min(150, len(predicted_tags))):
    #         print(f"{predicted_tags[i]}\t\t{actual_tags[i]}")
    
    #     return accuracy

    # # Calculate accuracy on predicted test set
    # accuracy_on_predicted = calculate_accuracy_on_predicted(predicted_tags_flat, actual_tags_flat)
    # print("Accuracy on predicted test set:", accuracy_on_predicted)



    # Printing a sample of sentences w/ predicted tags-------------------
    sample_size = min(10, len(correctly_predicted_tags))  
    for i in range(sample_size):
        sample_sentence = correctly_predicted_tags[i]
        if sample_sentence and all(isinstance(item, tuple) and len(item) == 3 and isinstance(item[0], tuple) and isinstance(item[0][0], str) for item in sample_sentence):  # Check if the sample sentence is not empty and contains valid tuples
            print("Sentence:", ' '.join([word[0] for word, _, _ in sample_sentence]))
            print("Predicted Tags:", ' '.join([predicted_tag for _, predicted_tag, _ in sample_sentence]))
            print("Actual Tags:", ' '.join([actual_tag for _, _, actual_tag in sample_sentence]))
            print()
        else:
            print("Invalid sample sentence:", sample_sentence)
else:
    print("No sentences found in training or testing data.")
