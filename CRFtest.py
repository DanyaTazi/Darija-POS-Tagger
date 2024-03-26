#CRF with two separate testing and training sets dyal conllu


from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics
from preconllu import parse_corpus 

# Load preprocessed data
corpus_file = 'TrainArabizi.conllu'
test_file = 'TestArabizi.conllu'
Xtrain, Ytrain = parse_corpus(corpus_file)
Xtest, Ytest = parse_corpus(test_file)


# Initialize and train CRF with contextual features
model = CRF()
model.fit([Xtrain], [Ytrain]) 

# Predict POS tags for the test data
predicted_tags = model.predict([Xtest])  

# Evaluate the performance of the model
print(metrics.flat_accuracy_score([Ytest], predicted_tags))  
