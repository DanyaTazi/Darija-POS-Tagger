# def parse_corpus(text_file):
#     X = []  # Initialize list for words
#     y = []  # Initialize list for POS tags

#     with open(text_file, 'r', encoding='utf-8') as file:
#         for line in file:
#             line = line.strip()
#             if line and not line.startswith("#"):  # Skip empty lines and comments
#                 parts = line.split('\t')
#                 if '-' not in parts[0]:  # Exclude compound tokens
#                     X.append(parts[1])  # Append word
#                     y.append(parts[3])  # Append POS tag

#     return X, y

# # Example usage:
# corpus_file = 'TrainArabizi.conllu'
# X, y = parse_corpus(corpus_file)
# print("Successfully Preprocessed")
# # for i in range(3):
# #     print("Word:", X[i])
# #     print("POS Tag:", y[i])
# #     print()


#to wrok with crf test since theres a formatting error with the testfile conllu
def parse_corpus(test_file):
    X = []  # Initialize list for tokens
    Y = []  # Initialize list for POS tags

    with open(test_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line and not line.startswith("#"):  # Skip metadata lines
                parts = line.split('\t')
                # Ensure that the line has enough elements
                if len(parts) >= 2:
                    word = parts[1]  # Extract word
                    pos_tag = parts[3]  # Extract POS tag (assuming it's in column 4)
                    X.append(word)  # Append word to X
                    Y.append(pos_tag)  # Append POS tag to Y
                else:
                    print(f"Issue with line: {line}")  # Debugging message
                    # Add error handling as needed
    return X, Y

