def parse_corpus_with_context(text_file):
    X = []  # Initialize list for feature vectors
    y = []  # Initialize list for POS tags

    with open(text_file, 'r', encoding='utf-8') as file:
        prev_word = None
        next_word = None
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split('\t')
                # Skip compound tokens
                if '-' in parts[0]:
                    continue
                word = parts[1]
                # Additional features
                features = {
                    'word': word,
                    'prev_word': prev_word,
                    'next_word': next_word,
                }
                X.append(features)
                y.append(parts[3])  
                prev_word = word
                # Update next word for the next iteration
                next_word = None if parts[0][-1] == '.' else parts[1]  # Skip compound tokens          

    return X, y

corpus_file = 'TrainArabizi.conllu'
X, y = parse_corpus_with_context(corpus_file)
print("Successfully Preprocessed")

