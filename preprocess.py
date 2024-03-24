import xml.etree.ElementTree as ET

def parse_corpus(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    corpus_data = []

    
    for word in root.findall('LE'):
        word_data = {
            'word': word.get('word'),
            'msa': word.get('msa'),
            'ma': word.get('ma'),
            'pos': word.get('pos'),
            'transitivity': word.get('transitivity'),
            'root': word.get('root'),
            'origin': word.get('origin'),
            'suffix': word.get('suffix'),
            'prefix': word.get('prefix'),
            'negation': word.get('negation'),
            'tense': word.get('tense'),
            'number': word.get('number'),
            'gender': word.get('gender'),
            'pers': word.get('pers')
        }
        corpus_data.append(word_data)
    
    return corpus_data

# Example usage:
corpus_file = 'test.xml'
parsed_corpus = parse_corpus(corpus_file)
print("Succesfully Pre Processed")
