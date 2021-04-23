import json
import re
import os
import pandas as pd


class ParseUtils:
    # Needs explanation
    MAX_LENGTH = 64

    # Needs explanation
    OVERLAP = 20

    DATA_PATH = os.path.dirname(__file__) + '/data/coleridgeinitiative-show-us-the-data/'
    TRAIN_DATA_PATH = DATA_PATH + 'train'
    TRAIN_DF_PATH = DATA_PATH + 'train.csv'
    TEST_DATA_PATH = DATA_PATH + 'test'

    @staticmethod
    def count_in_json(json_id, label):
        path_to_json = os.path.join(ParseUtils.TRAIN_DATA_PATH, (json_id+'.json'))
        count_dict = {}
        with open(path_to_json, 'r') as f:
            json_decode = json.load(f)
            for data in json_decode:
                heading = data.get('section_title')
                content = data.get('text')
                count_dict[heading] = content.count(heading)
        return count_dict

    @staticmethod
    def shorten_sentences(sentences):
        short_sentences = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) > ParseUtils.MAX_LENGTH:
                for p in range(0, len(words), ParseUtils.MAX_LENGTH - ParseUtils.OVERLAP):
                    short_sentences.append(' '.join(words[p:p + ParseUtils.MAX_LENGTH]))
            else:
                short_sentences.append(sentence)
        return short_sentences

    @staticmethod
    def clean_training_text(txt):
        """
        similar to the default clean_text function but without lowercasing.
        """
        txt = re.sub('[^A-Za-z0-9]+', ' ', str(txt)).strip()


        return txt

    @staticmethod
    def find_sublist(big_list, small_list):
        all_positions = []
        for i in range(len(big_list) - len(small_list) + 1):
            if small_list == big_list[i:i + len(small_list)]:
                all_positions.append(i)

        return all_positions

    @staticmethod
    def tag_sentence(sentence, labels):  # requirement: both sentence and labels are already cleaned
        sentence_words = sentence.split()

        if labels is not None and any(re.findall(f'\\b{label}\\b', sentence)
                                      for label in labels):  # positive sample
            nes = ['O'] * len(sentence_words)
            for label in labels:
                label_words = label.split()

                all_pos = ParseUtils.find_sublist(sentence_words, label_words)
                for pos in all_pos:
                    nes[pos] = 'B'
                    for i in range(pos + 1, pos + len(label_words)):
                        nes[i] = 'I'

            return True, list(zip(sentence_words, nes))

        else:  # negative sample
            nes = ['O'] * len(sentence_words)
            return False, list(zip(sentence_words, nes))

    @staticmethod
    def parse_for_bert(dataset_label,content):
        labels = dataset_label.split(('|'))
        labels = [ParseUtils.clean_training_text(label) for label in labels]
        sentences = [ParseUtils.clean_training_text(sent) for sent in content.split('.')]
        sentences = ParseUtils.shorten_sentences(sentences)
        sentences = [sentence for sentence in sentences if len(sentence) > 10]
        all_sent = []
        for sentence in sentences:
            is_positive, tags = ParseUtils.tag_sentence(sentence, labels)
            if is_positive:
                all_sent.append(tags)
            elif any(word in sentence.lower() for word in ['data', 'study']):
                all_sent.append(tags)
        return all_sent

    @staticmethod
    def parse_json(json_id, to_return):
        path_to_json = os.path.join(ParseUtils.TRAIN_DATA_PATH, (json_id+'.json'))
        heading = []
        content = []
        with open(path_to_json, 'r') as f:
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

    @staticmethod
    def get_train_df():
        return pd.read_csv(ParseUtils.TRAIN_DF_PATH)

    @staticmethod
    def read_append_return(filename, train_files_path=TRAIN_DATA_PATH, output='text'):
        """
        Function to read json file and then return the text data from them and append to the dataframe

        Basicall parse json but then from https://www.kaggle.com/prashansdixit/coleridge-initiative-eda-baseline-model
        """
        json_path = os.path.join(train_files_path, (filename + '.json'))
        headings = []
        contents = []
        combined = []
        with open(json_path, 'r') as f:
            json_decode = json.load(f)
            for data in json_decode:
                headings.append(data.get('section_title'))
                contents.append(data.get('text'))
                combined.append(data.get('section_title'))
                combined.append(data.get('text'))

        all_headings = ' '.join(headings)
        all_contents = ' '.join(contents)
        all_data = '. '.join(combined)

        if output == 'text':
            return all_contents
        elif output == 'head':
            return all_headings
        else:
            return all_data

