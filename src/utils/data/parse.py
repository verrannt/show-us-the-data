import json
import re
import os
import pandas as pd
from tqdm import tqdm

class ParseUtils:
    # Needs explanation
    MAX_LENGTH = 64

    # Needs explanation
    OVERLAP = 20

    DATA_PATH = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                             'data'), 'coleridgeinitiative-show-us-the-data')
    TRAIN_DATA_PATH = os.path.join(DATA_PATH, 'train')
    TRAIN_DF_PATH = os.path.join(DATA_PATH, 'train.csv')
    TEST_DATA_PATH = os.path.join(DATA_PATH, 'test')

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

    @staticmethod
    def compute_ner_data(
        max_sample:int, # Maximum number of samples taken from dataframe for
                        # development purposes
        save:str,       # Save parsed data to storage under the name provided
                        # here as string ('.json' will be appended). 
                        # If not provided, will not save.
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
        directory in a single text file, where is line is in JSON format, e.g.
        { "tokens" : ["A", "short", "sentence"], "tags"   : ["0", "0", "0"] }
        """

        # Read data in CSV file
        train = pd.read_csv(ParseUtils.TRAIN_DF_PATH)
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

        # Read individual papers by ID from storage
        papers = {}
        for paper_id in train['Id'].unique():
            with open(f'{ParseUtils.TRAIN_DATA_PATH}/{paper_id}.json', 'r') as f:
                paper = json.load(f)
                papers[paper_id] = paper

        cnt_pos, cnt_neg = 0, 0 # number of sentences that contain/not contain labels
        ner_data = []

        pbar = tqdm(total=len(train))
        for i, id, dataset_label in train[['Id', 'dataset_label']].itertuples():
            # paper
            paper = papers[id]
            
            # labels
            labels = dataset_label.split('|')
            labels = [ParseUtils.clean_training_text(label) for label in labels]
            
            # sentences
            sentences = set([ParseUtils.clean_training_text(sentence) for section in paper 
                        for sentence in section['text'].split('.') 
                        ])
            sentences = ParseUtils.shorten_sentences(sentences) # make sentences short
            # only accept sentences with length > 10 chars
            sentences = [sentence for sentence in sentences if len(sentence) > 10] 
            
            # positive sample
            for sentence in sentences:
                is_positive, tags = ParseUtils.tag_sentence(sentence, labels)
                if is_positive:
                    cnt_pos += 1
                    ner_data.append(tags)
                elif any(word in sentence.lower() for word in ['data', 'study']): 
                    ner_data.append(tags)
                    cnt_neg += 1
            
            # process bar
            pbar.update(1)
            pbar.set_description(f"Training data size: {cnt_pos} positives + {cnt_neg} negatives")
            
        # shuffling
        #random.shuffle(ner_data)

        # Write data to file
        if save:
            with open(os.path.join(ParseUtils.DATA_PATH,f'{save}.json'), 'w') as f:
                for row in ner_data:
                    words, nes = list(zip(*row))
                    row_json = {'tokens' : words, 'tags' : nes}
                    json.dump(row_json, f)
                    f.write('\n')

        return ner_data

    @staticmethod
    def load_ner_data_from_json(filename):
        
        ner_data = []
        f = open(os.path.join(ParseUtils.DATA_PATH, filename), 'r')
        
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
                in zip(sentence["tokens"],sentence["tags"])
            ]
            
            # Each of these parsed sentences becomes an entry
            # in our overall data list
            ner_data.append(sentence_tuple_list)
            
        f.close()
        return ner_data

if __name__=='__main__':

    print('Testing parse functionality.')

    # The following are some testing capabilities for the funcitionality
    # of the components provided in this script
    ner_data = ParseUtils.compute_ner_data(
        max_sample=1000, save='testing_functionality')

    assert 'testing_functionality.json' in os.listdir(ParseUtils.DATA_PATH), \
        "Something went wrong storing the data. Maybe the wrong path?"

    del ner_data

    ner_data = ParseUtils.load_ner_data_from_json('testing_functionality.json')

    os.remove(os.path.join(
        ParseUtils.DATA_PATH,
        'testing_functionality.json'))

    print('Done.')