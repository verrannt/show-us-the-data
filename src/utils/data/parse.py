import json
import re
import os
import pandas as pd
import pickle
from tqdm import tqdm


class ParseUtils:

    @staticmethod
    def count_in_json(json_id, label, train_data_path):
        path_to_json = os.path.join(train_data_path, (json_id + '.json'))
        count_dict = {}
        with open(path_to_json, 'r') as f:
            json_decode = json.load(f)
            for data in json_decode:
                heading = data.get('section_title')
                content = data.get('text')
                count_dict[heading] = content.count(heading)
        return count_dict

    @staticmethod
    def shorten_sentences(sentences, max_len, overlap):
        short_sentences = []
        for sentence in sentences:
            words = sentence.split()
            if len(words) > max_len:
                for p in range(0, len(words), max_len - overlap):
                    short_sentences.append(' '.join(words[p:p + max_len]))
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
    def tag_sentence(sentence, labels, ignore_case):  # requirement: both sentence and
        re_flags = re.IGNORECASE if ignore_case else None

        # labels are already cleaned
        sentence_words = sentence.split()

        if labels is not None and any(re.findall(f'\\b{label}\\b', sentence,
                                                 flags=re_flags) for label in labels):  # positive sample
            nes = ['O'] * len(sentence_words)
            for label in labels:
                label_words = label.split()

                if ignore_case:
                    nocase_label_words = list(map(lambda word: word.lower(), label_words))
                    nocase_sentence_words = list(map(lambda word: word.lower(), sentence_words))
                    all_pos = ParseUtils.find_sublist(nocase_sentence_words, nocase_label_words)
                else:
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
    def read_append_return(filename, train_data_path, output='text'):
        """
        Function to read json file and then return the text data from them and append to the dataframe

        Basically parse json but then from https://www.kaggle.com/prashansdixit/coleridge-initiative-eda-baseline-model
        """
        json_path = os.path.join(train_data_path, (filename + '.json'))
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

    @staticmethod
    def save_extracted(ner_data, data_path, file_name):
        with open(os.path.join(data_path, file_name), 'w') as f:
            for row in ner_data:
                words, nes = list(zip(*row))
                row_json = {'tokens': words, 'tags': nes}
                json.dump(row_json, f)
                f.write('\n')

    @staticmethod
    def load_extracted(data_path, file_name):

        ner_data = []
        f = open(os.path.join(data_path, file_name), 'r')

        for line in f.readlines():
            # Each line is formatted in JSON format, e.g.
            # { "tokens" : ["A", "short", "sentence"],
            #   "tags"   : ["0", "0", "0"] }
            sentence = json.loads(line)

            # From the tokens and tags, we create a list of 
            # tuples of the form
            # [ ("A", "0"), ("short", "0"), ("sentence", "0")]
            sentence_tuple_list = [
                (token, tag) for token, tag
                in zip(sentence["tokens"], sentence["tags"])
            ]

            # Each of these parsed sentences becomes an entry
            # in our overall data list
            ner_data.append(sentence_tuple_list)

        f.close()
        return ner_data

    @staticmethod
    def load_auxiliary_datasets(data_path, file_name):
        with open(os.path.join(data_path, file_name), 'r') as f:
            return f.read().split('\n')

    @staticmethod
    def load_tokenized_auxiliary_datasets(data_path, file_name):
        with open(os.path.join(data_path, file_name), 'r') as f:
            datasets = f.read().split('\n')
            return [[int(item) for item in row.split(',') if item != ''] for row in datasets if len(row) > 0]

    @staticmethod
    def save_file(output, data_path, file_name):
        with open(os.path.join(data_path, file_name), 'wb') as f:
            pickle.dump(output, f)

    @staticmethod
    def load_file(data_path, file_name):
        with open(os.path.join(data_path, file_name), 'rb') as f:
            output = pickle.load(f)
        return output

    @staticmethod
    def all_labels_mentioned(data):
        """
        Method that can be applied to a dataframe and check, for all dataset labels, if they occur in the text at least
        once. Case insensitive
        """
        labels = data['dataset_label'].split("|")
        return all(list(map(lambda label: data['text'].lower().count(label.lower()) > 0, labels)))

    @staticmethod
    def extract(
            max_len,
            overlap,
            max_sample,
            max_text_tokens,
            train_df_path,
            train_data_path,
            ignore_label_case,
            exclude_non_exact_label_match

    ):
        """
        Reads the training data from storage using the train.csv file as well
        as all json files inside the train folder, and computes a list,
        where each element is a sentence. Each sentence is itself a list,
        consisting of tuples, where the first element is the word (token) and
        the second is the label (tag).

        This is an example of the data list returned:

            ner_data = [
                ...
                [
                    ("This", "0"),
                    ("is", "0"),
                    ("New", "LOC"),
                    ("York", "LOC"),
                ],
                ...
            ]

        If `save` is True, the data will be stored on disk in the DATA_PATH
        directory in a single text file, where each line is in JSON format, e.g.

            { "tokens" : ["A", "short", "sentence"], "tags" : ["0", "0", "0"] }
        """

        # Read data in CSV file
        train = pd.read_csv(train_df_path)
        train = train[:max_sample]
        print(f'Found {len(train)} raw training rows')

        # Group rows by publication ID
        train = train.groupby('Id').agg({
            'pub_title': 'first',
            'dataset_title': '|'.join,
            'dataset_label': '|'.join,
            'cleaned_label': '|'.join
        }).reset_index()
        print(f'Found {len(train)} unique training rows')

        print('Loading texts, this might take a while...')
        # Read texts for text length analysis
        train['text'] = train['Id'].apply(lambda ID: ParseUtils.read_append_return(ID, train_data_path))
        train['text_token_length'] = train['text'].apply(lambda text: len(text))

        # Remove texts that have more tokens than max_text_tokens
        train = train[train['text_token_length'] <= max_text_tokens]
        print(f'Removed texts exceeding max length, {len(train)} training rows left')

        if exclude_non_exact_label_match:
            # Count label mentions in text
            train["all_labels_mentioned"] = train.apply(ParseUtils.all_labels_mentioned, axis=1)

            # Remove texts that have 0 label count for at least 1 label
            train = train[train['all_labels_mentioned']]
            print(f'Removed texts that had at least one label with 0 exact (case insensitive) matches in the text, '
                  f'{len(train)} training rows left')

        # Read individual papers by ID from storage
        papers = {}
        for paper_id in train['Id'].unique():
            with open(f'{train_data_path}/{paper_id}.json', 'r') as f:
                paper = json.load(f)
                papers[paper_id] = paper

        cnt_pos, cnt_neg = 0, 0  # number of sentences that contain/not contain labels
        ner_data = []

        pbar = tqdm(total=len(train))
        for i, id, dataset_label in train[['Id', 'dataset_label']].itertuples():
            # paper
            paper = papers[id]

            # labels
            labels = dataset_label.split('|')
            labels = [ParseUtils.clean_training_text(label) for label in labels]

            # sentences
            sentences = set([
                ParseUtils.clean_training_text(sentence)
                for section in paper
                for sentence in section['text'].split('.')
            ])
            sentences = ParseUtils.shorten_sentences(
                sentences, max_len, overlap)

            # only accept sentences with length > 10 chars
            sentences = [sentence for sentence in sentences if len(sentence) > 10]

            # positive sample
            for sentence in sentences:
                is_positive, tags = ParseUtils.tag_sentence(sentence, labels, ignore_label_case)
                if is_positive:
                    cnt_pos += 1
                else:
                    cnt_neg += 1
                ner_data.append(tags)

            # process bar
            pbar.update(1)
            pbar.set_description(f"Training data size: {cnt_pos} positives + {cnt_neg} negatives")

        # shuffling
        # random.shuffle(ner_data)

        return ner_data
