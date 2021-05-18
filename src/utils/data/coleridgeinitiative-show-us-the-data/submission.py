from transformers import BertForTokenClassification, AdamW, BertTokenizerFast
import re
import pandas as pd
import numpy as np
import torch
import json
from keras.preprocessing.sequence import pad_sequences

class SubmitPred:

    def __init__(self, test_csv_path, test_path, model_path, tokenizer_path, batch_size=64):
        self.test_csv_path = test_csv_path
        self.test_path = test_path
        self.model = BertForTokenClassification.from_pretrained(model_path, num_labels=3, output_attentions=False,
                                                                output_hidden_states=False)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, do_lower_case=False)
        self.MAX_LENGTH = 64  # max no. words for each sentence.
        self.OVERLAP = 20
        self.tag2str = {2: 'O', 1: 'I', 0: 'B'}
        self.batch_size = batch_size

    def load_submission(self):
        self.sample_submission = pd.read_csv(self.test_csv_path)

    def tokenize_sent(self, sentence):
        tokenized_sentence = []
        sentence = sentence.split()
        for word in sentence:
            tokenized_word = self.tokenizer.tokenize(word)
            tokenized_sentence.extend(tokenized_word)
        return tokenized_sentence

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
        return re.sub('[^A-Za-z0-9]+', ' ', str(txt)).strip()

    def add_padding(self, tokenized_sentences):
        padded_sentences = pad_sequences(
            tokenized_sentences,
            value='[PAD]',
            dtype=object,
            maxlen=self.MAX_LENGTH,
            truncating='post',
            padding='post')
        return padded_sentences

    @staticmethod
    def get_attention_mask(input_ids, ignore_tokens=[0, 101, 102]):
        return [[float(token not in ignore_tokens) for token in sent] for sent in input_ids]

    @staticmethod
    def jaccard_similarity(s1, s2):
        l1 = s1.split(" ")
        l2 = s2.split(" ")
        intersection = len(list(set(l1).intersection(l2)))
        union = (len(l1) + len(l2)) - intersection
        return float(intersection) / union

    @staticmethod
    def add_start_end_tokens(tupled_sentence):
        # print(tupled_sentence)
        # tupled_sentence= tupled_sentence.split()
        tupled_sentence.insert(0, ('[CLS]'))
        tupled_sentence.append(('[SEP]'))
        return tupled_sentence

    def run(self):
        self.load_submission()
        self.model.cuda()
        paper_length = []
        sentences_e = []
        papers = {}
        for paper_id in self.sample_submission['Id']:
            with open(f'{self.test_path}/{paper_id}.json', 'r') as f:
                paper = json.load(f)
                papers[paper_id] = paper
        for id in self.sample_submission['Id']:
            paper = papers[id]
            sentences = set(
                [self.clean_training_text(sentence) for section in paper for sentence in section['text'].split('.')])
            sentences = self.shorten_sentences(sentences, self.MAX_LENGTH, self.OVERLAP)
            sentences = [sentence for sentence in sentences if len(sentence) > 10]
            ner_data = [sentence for sentence in sentences if
                        any(word in sentence.lower() for word in ['data', 'study'])]
            sentences_e.extend(ner_data)
            paper_length.append(len(ner_data))
            print(len(ner_data))
        tokenized_words = [self.tokenize_sent(sentence) for sentence in sentences_e]
        start_end = [self.add_start_end_tokens(sentence) for sentence in tokenized_words]
        padding_sentences = self.add_padding(start_end)
        input_ids = [self.tokenizer.convert_tokens_to_ids(text) for text in padding_sentences]
        attention_mask = self.get_attention_mask(input_ids, ignore_tokens=[0, 101, 102])
        predicts = torch.tensor(input_ids)
        masks = torch.tensor(attention_mask)
        predict_data = TensorDataset(predicts, masks)
        predict_dataloader = DataLoader(predict_data, batch_size=self.batch_size)
        all_predictions = []
        for step, batch in enumerate(predict_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch
            with torch.no_grad():
                output = self.model(b_input_ids, attention_mask=b_input_mask)
            label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
            all_predictions.extend(label_indices)

        all_preds_str = [[self.tag2str[token] for token in pred] for pred in all_predictions]
        all_sent_str = [self.tokenizer.convert_ids_to_tokens(sent) for sent in input_ids]
        all_sent_int = [ids for ids in input_ids]
        final_predics = []
        all_sent_str_1 = all_sent_int
        all_preds_str_1 = all_preds_str
        for pap_len in paper_length:
            print(pap_len)
            labels = []
            for sentence, pred in zip(all_sent_str_1[:pap_len], all_preds_str_1[:pap_len]):
                phrase = []
                for word, tag in zip(sentence, pred):
                    if tag == "I" or tag == "B":
                        phrase.append(word)
                    else:
                        if len(phrase) != 0:
                            labels.append(self.tokenizer.decode(phrase))
                            phrase = []

            final_predics.append(labels)
            # print(lol)
            del all_sent_str_1[:pap_len], all_preds_str_1[pap_len]
        self.sample_submission['PredictionString'] = final_predics
        self.sample_submission['PredictionString'] = self.sample_submission.apply(
            lambda x: "|".join(x.PredictionString), axis=1)

    def save_csv(self):
        self.sample_submission.to_csv(f'submission.csv', index=False)