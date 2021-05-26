import os

from utils.data.parse import ParseUtils
from utils.generic import timer

import json
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
from rich.console import Console
from rich.progress import track
from tqdm import tqdm
from transformers import BertTokenizerFast


class Pipeline:

    def __init__(self, configs):
        self.configs = configs
        self.tokenizer = None

    def set_tokenizer(self, tokenizer):
        """
        Set a custom tokenizer to be used when running the pipeline.
        If not, this will default to BERTTokenizerFast in `run()`
        """
        self.tokenizer = tokenizer

    def tokenize_and_preserve_labels(self, tupled_sentence):
        tokenized_sentence = []
        labels = []

        for (word, label) in tupled_sentence:

            # Tokenize the word
            tokenized_word = self.tokenizer.tokenize(word)
            tokenized_sentence.extend(tokenized_word)
            
            # Repeat the label for words that are broken up into several tokens
            labels.extend([label]*len(tokenized_word))
            
        # Add the tokenized word and its label to the final tokenized word list
        return list(zip(tokenized_sentence, labels))

    def add_start_end_tokens(self, tupled_sentence):
        tupled_sentence.insert(0, ('[CLS]', 'O'))
        tupled_sentence.append(('[SEP]', 'O'))
        return tupled_sentence

    def add_padding(self, tokenized_sentences, labels):
        # Note that this implicitly converts to an array of objects (strings)
        
        padded_sentences = pad_sequences(
            tokenized_sentences, 
            value='[PAD]', 
            dtype=object, 
            maxlen=self.configs.MAX_LENGTH, 
            truncating='post', 
            padding='post')

        padded_labels = pad_sequences(
            labels, 
            value='O', 
            dtype=object, 
            maxlen=self.configs.MAX_LENGTH, 
            truncating='post', 
            padding='post')
        
        return padded_sentences, padded_labels

    def get_attention_mask(self, input_ids, ignore_tokens=[0,101,102]):
        """
        Compute the attention marks for the tokens in `input_ids`, which is
        assumed to be a list (batch) of lists (sentences) of integer tokens.
        Tokens that should be masked out can be specified using the 
        `ignore_tokens` parameter. By default, these are supposed to be 0, 101,
        and 102, representing [PAD], [CLS], and [SEP] tokens, respectively.
        """

        return [
            [ float(token not in ignore_tokens) for token in sent ] 
                for sent in input_ids
        ]

    def run(self, ner_data):
        """
        Run extracted sentence data through the pipeline.
        """

        console = Console()

        # Initialize tokenizer
        if not self.tokenizer:
            self.tokenizer = BertTokenizerFast.from_pretrained(
                'bert-base-cased', do_lower_case=False)
            console.log('Initialized default BERT tokenizer')
        else:
            console.log('Using custom tokenizer')

        # Tokenize into known tokens
        ner_data = [
            self.tokenize_and_preserve_labels(sentence) for sentence in 
                track(ner_data, description='Tokenizing words...')
        ]
        console.log('Tokenized words')

        with console.status("[bold green]Running pipeline...") as status:

            # Add [CLS] and [SEP] tokens to beginning and end
            ner_data = [
                self.add_start_end_tokens(sentence)
                    for sentence in ner_data
            ]
            console.log('Added [CLS] and [SEP] tokens')

            # Get only sentences, not labels
            tokenized_sentences = [
                [token_label_tuple[0] for token_label_tuple in sent]
                    for sent in ner_data
            ]

            # Get only labels, not sentences
            labels = [
                [token_label_tuple[1] for token_label_tuple in sent] 
                    for sent in ner_data 
            ]

            # Pad sentences and labels 
            padded_sentences, padded_labels = self.add_padding(
                tokenized_sentences, labels)
            console.log('Padded sentences and labels')

            # Convert to integer ids
            input_ids = [
                self.tokenizer.convert_tokens_to_ids(text) 
                    for text in padded_sentences
            ]
            tags = [
                self.tokenizer.convert_tokens_to_ids(text) 
                    for text in padded_labels
            ]
            console.log('Converted to integer ids')

            # Compute attention mask from input tokens
            attention_mask = self.get_attention_mask(
                input_ids,
                # Only ignore [PAD] tokens (integer 0)
                ignore_tokens=[0]
            )
            
            console.log('Computed attention mask')

        if self.configs.SAVE:
            ParseUtils.save_file(
                {
                    'input_ids': input_ids, 
                    'tags': tags,
                    'attention_mask': attention_mask
                },
                self.configs.DATA_PATH,
                self.configs.TOKENIZED_FILENAME
            )

        return input_ids, tags, attention_mask

    def load_outputs(self):
        """
        Recover the outputs of a previously completed run from storage.
        """
        output_dict = ParseUtils.load_file(
            self.configs.DATA_PATH,
            self.configs.TOKENIZED_FILENAME,
        )

        return output_dict['input_ids'], \
               output_dict['tags'], \
               output_dict['attention_mask']

    def extract(self):
        train_ner_data, val_ner_data = ParseUtils.extract(
            max_len = self.configs.MAX_LENGTH,
            overlap = self.configs.OVERLAP,
            max_sample = self.configs.MAX_SAMPLE,
            max_text_tokens = self.configs.MAX_TEXT_TOKENS,
            train_df_path = self.configs.TRAIN_DF_PATH,
            train_data_path = self.configs.TRAIN_DATA_PATH,
            ignore_label_case = self.configs.IGNORE_LABEL_CASE,
            exclude_non_exact_label_match = self.configs.EXCLUDE_NON_EXACT_LABEL_MATCH,
            val_split = self.configs.VAL_SPLIT
        )

        # Write data to file
        if self.configs.SAVE:
            ParseUtils.save_extracted(
                train_ner_data,
                val_ner_data,
                self.configs.DATA_PATH, 
                self.configs.EXTRACTED_FILENAME
            )

        return train_ner_data, val_ner_data

    def load_extracted(self):
        return ParseUtils.load_extracted(
            self.configs.DATA_PATH, 
            self.configs.EXTRACTED_FILENAME
        )


class PipelineConfigs:

    def __init__(
        self,
        DATA_PATH,
        SAVE,
        EXTRACTED_FILENAME,
        TOKENIZED_FILENAME,
        MAX_SAMPLE,
        MAX_LENGTH = 64,
        OVERLAP = 20,
        MAX_TEXT_TOKENS=200000,
        IGNORE_LABEL_CASE=True,
        EXCLUDE_NON_EXACT_LABEL_MATCH=True,
        VAL_SPLIT = 0.10
    ):

        # Maximum number of words for each sentence
        self.MAX_LENGTH = MAX_LENGTH

        # If a sentence exceeds MAX_LENGTH, we split it to multiple sentences 
        # with overlapping
        self.OVERLAP = OVERLAP

        # During development, you may want to only load part of the data. Leave
        # uninitialized during production
        self.MAX_SAMPLE = MAX_SAMPLE

        self.DATA_PATH = DATA_PATH
        #self.DATA_PATH = \
        #    os.path.join(
        #        os.path.join(
        #            os.path.dirname(
        #                os.path.dirname(
        #                    os.path.dirname(
        #                        os.path.dirname(__file__)
        #                    )
        #                )
        #            ),
        #            'data'
        #        ), 
        #        'coleridgeinitiative-show-us-the-data'
        #    )
        self.TRAIN_DATA_PATH = os.path.join(self.DATA_PATH, 'train')
        self.TRAIN_DF_PATH = os.path.join(self.DATA_PATH, 'train.csv')
        self.TEST_DATA_PATH = os.path.join(self.DATA_PATH, 'test')

        # If SAVE is true, will save the extracted and/or the tokenized data
        # under the provided filename(s)
        self.SAVE = SAVE
        self.EXTRACTED_FILENAME = EXTRACTED_FILENAME
        self.TOKENIZED_FILENAME = TOKENIZED_FILENAME
        # Maximum amount of tokens in training texts. Longer texts will be discarded
        self.MAX_TEXT_TOKENS = MAX_TEXT_TOKENS
        # Whether the tagger should ignore the case of the label when matching labels to the text
        self.IGNORE_LABEL_CASE = IGNORE_LABEL_CASE
        # Whether to exclude texts that do not have a single one-on-one (case insensitve) label match
        self.EXCLUDE_NON_EXACT_LABEL_MATCH = EXCLUDE_NON_EXACT_LABEL_MATCH

        self.VAL_SPLIT = VAL_SPLIT