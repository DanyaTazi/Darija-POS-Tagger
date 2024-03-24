import xml.etree.ElementTree as ET

def parse_corpus(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    X = []  # Initialize list for words
    y = []  # Initialize list for POS tags
    
    for word in root.findall('LE'):
        word_text = word.get('word')
        pos_tag = word.get('pos')
        
        # Ignore words with missing POS tags
        if word_text and pos_tag:
            X.append(word_text)
            y.append(pos_tag)
    
    return X, y

# Example usage:
corpus_file = 'morv.xml'
X_train, y_train = parse_corpus(corpus_file)
print("Successfully Preprocessed")
