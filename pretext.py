def parse_corpus(text_file):
    X = []  # Initialize list for words
    y = []  # Initialize list for POS tags

    with open(text_file, 'r', encoding='utf-8') as file:
        first_line_skipped = False  # Flag if first line has been skipped
        for line in file:
            if not first_line_skipped:  # Skip first line
                first_line_skipped = True
                continue
            
            line = line.strip()
            if line:  # Skip empty lines
                parts = line.split('\t')
                X.append(parts[4])  # Append word
                y.append(parts[6])  # Append POS tag

    return X, y

# Example usage:
corpus_file = 'twitt.txt'
X, y = parse_corpus(corpus_file)
print("Successfully Preprocessed")
