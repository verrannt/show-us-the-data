from transformers import BertForTokenClassification, AdamW, BertTokenizerFast
import re
import pandas as pd
import numpy as np
import torch
import json
from keras.preprocessing.sequence import pad_sequences


class SubmitPred:

    def __init__(self, test_csv_path, test_path, model_path, tokenizer_path, batch_size=128):
        self.test_csv_path = test_csv_path
        self.test_path = test_path
        self.model = BertForTokenClassification.from_pretrained(model_path, num_labels=3, output_attentions=False,
                                                                output_hidden_states=False).to(device)
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

    def read_and_create_csv(self):
        all_test_papers = os.listdir('../input/coleridgeinitiative-show-us-the-data/test')
        self.submission = pd.DataFrame({'Id': all_test_papers})

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
        return list(map(lambda sent: list(map(lambda token: float(token not in ignore_tokens), sent)), input_ids))

    @staticmethod
    def jaccard_similarity(list1, list2):
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return float(intersection) / union

    @staticmethod
    def add_start_end_tokens(tupled_sentence):
        tupled_sentence.insert(0, '[CLS]')
        tupled_sentence.append('[SEP]')
        return tupled_sentence

    def read_papers(self):
        paper_length = []
        sentences_e = []
        papers = {}
        for paper_id in self.submission['Id']:
            with open(f'{self.test_path}/{paper_id}', 'r') as f:
                paper = json.load(f)
                papers[paper_id] = paper
            sentences = set(
                [self.clean_training_text(sentence) for section in paper for sentence in section['text'].split('.')])
            sentences = self.shorten_sentences(sentences, self.MAX_LENGTH, self.OVERLAP)
            sentences = [sentence for sentence in sentences if len(sentence) > 10]
            sentences_e.extend(sentences)
            paper_length.append(len(sentences))
        
        return papers, paper_length, sentences_e

    def run(self):
        #self.load_submission()
        self.read_and_create_csv()
        
        papers, paper_length, sentences_e = self.read_papers()
        
        tokenized_words = list(map(lambda sentence: self.tokenize_sent(sentence), sentences_e))
        start_end = list(map(lambda sentence: self.add_start_end_tokens(sentence), tokenized_words))
        padding_sentences = self.add_padding(start_end)
        input_ids = list(map(lambda text: self.tokenizer.convert_tokens_to_ids(text), padding_sentences))
        attention_mask = self.get_attention_mask(input_ids, ignore_tokens=[0])
        
        predicts = torch.tensor(input_ids, requires_grad=False).to(device)
        masks = torch.tensor(attention_mask, requires_grad=False).to(device)

        predict_data = TensorDataset(predicts, masks)
        predict_dataloader = DataLoader(predict_data, batch_size=self.batch_size)

        all_predictions = torch.empty((0, self.MAX_LENGTH, 3), device=device, requires_grad=False)

        for step, batch in enumerate(predict_dataloader):
            b_input_ids, b_input_mask = batch
            with torch.no_grad():
                output = self.model(b_input_ids, attention_mask=b_input_mask)
            all_predictions = torch.vstack((all_predictions, output[0]))
        
        all_predictions = np.argmax(all_predictions.to('cpu').numpy(), axis=2)
        
        
        all_preds_str = list(map(lambda pred: map(lambda token: self.tag2str[token], pred), all_predictions))
        #all_preds_str = [[self.tag2str[token] for token in pred] for pred in all_predictions]
        all_sent_int = [ids for ids in input_ids]
        final_predics = []
        
        for pap_len in paper_length:
            labels = []
            for sentence, pred in zip(all_sent_int[:pap_len], all_preds_str[:pap_len]):
                phrase = []
                phrase_test = []
                for word, tag in zip(sentence, pred):

                    if tag == "I" or tag == "B":
                        phrase_test.append(word)
                        if word != 0 and word != 101 and word != 102:
                            phrase.append(word)
                    else:
                        if len(phrase) != 0:
                            labels.append(self.tokenizer.decode(phrase))
                            phrase_test = []
                            phrase = []

            final_predics.append(labels)
            del all_sent_int[:pap_len], all_preds_str[:pap_len]
        final_predics = [[pred for pred in preds if not pred.startswith("#")] for preds in final_predics]

        filtered = []
        for final_predic in final_predics:
            filt = []
            for pred in final_predic:
                if len(filt) == 0:
                    filt.append(pred)
                else:
                    flag = 0
                    for filtered_pred in filt:
                        if self.jaccard_similarity(filtered_pred.split(), pred.split()) > 0.70:
                            flag = 1
                        if flag == 0:
                            filt.append(pred)

            filtered.append(filt)

        self.final_predics = final_predics
        filtered = ["|".join(filt) if len(filt) != 0 else filt for filt in filtered]
        self.filtered = filtered
        self.submission['PredictionString'] = filtered
        # self.submission['PredictionString'] = self.submission.apply(lambda x:"|".join(x.PredictionString),axis=1)

    def save_csv(self):
        self.submission.to_csv(f'submission.csv', index=False)