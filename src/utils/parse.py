import json
import re
import pandas as pd
MAX_LENGTH = 64
OVERLAP = 20
train_files_path = '../input/coleridgeinitiative-show-us-the-data/train'
to_return = "heading"
def count_in_json(json_id,label):
    path_to_json = os.path.join(train_files_path,(json_id+'.json'))
    count_dict = {}
    with open(path_to_json,'r') as f:
        json_decode = json.load(f)
        for data in json_decode:
            heading = data.get('section_title')
            content = data.get('text')
            count_dict[heading] = content.count(heading)
    return count_dict

def shorten_sentences(sentences):
    short_sentences = []
    for sentence in sentences:
        words = sentence.split()
        if len(words) > MAX_LENGTH:
            for p in range(0, len(words), MAX_LENGTH - OVERLAP):
                short_sentences.append(' '.join(words[p:p+MAX_LENGTH]))
        else:
            short_sentences.append(sentence)
    return short_sentences


def clean_training_text(txt):
    """
    similar to the default clean_text function but without lowercasing.
    """
    txt = re.sub('[^A-Za-z0-9]+', ' ', str(txt)).strip()


    return txt


def find_sublist(big_list, small_list):
    all_positions = []
    for i in range(len(big_list) - len(small_list) + 1):
        if small_list == big_list[i:i + len(small_list)]:
            all_positions.append(i)

    return all_positions


def tag_sentence(sentence, labels):  # requirement: both sentence and labels are already cleaned
    sentence_words = sentence.split()

    if labels is not None and any(re.findall(f'\\b{label}\\b', sentence)
                                  for label in labels):  # positive sample
        nes = ['O'] * len(sentence_words)
        for label in labels:
            label_words = label.split()

            all_pos = find_sublist(sentence_words, label_words)
            for pos in all_pos:
                nes[pos] = 'B'
                for i in range(pos + 1, pos + len(label_words)):
                    nes[i] = 'I'

        return True, list(zip(sentence_words, nes))

    else:  # negative sample
        nes = ['O'] * len(sentence_words)
        return False, list(zip(sentence_words, nes))


def parse_for_bert(dataset_label,content):
    labels = dataset_label.split(('|'))
    labels = [clean_training_text(label) for label in labels]
    sentences = [clean_training_text(sent) for sent in content.split('.')]
    sentences = shorten_sentences(sentences)
    sentences = [sentence for sentence in sentences if len(sentence) > 10]
    all_sent = []
    for sentence in sentences:
        is_positive, tags = tag_sentence(sentence, labels)
        if is_positive:
            all_sent.append(tags)
        elif any(word in sentence.lower() for word in ['data', 'study']):
            all_sent.append(tags)
    return all_sent

def parse_json(json_id):
    path_to_json = os.path.join(train_files_path,(json_id+'.json'))
    heading = []
    content = []
    with open(path_to_json,'r') as f:
        json_decode = json.load(f)
        for data in json_decode:
            heading.append(data.get('section_title'))
            content.append(data.get('text'))
    if to_return == "heading":
        all_heading = ",".join(heading)
        return all_heading
    if to_return == "content":
        all_content = ".".join(content)
        return all_content
