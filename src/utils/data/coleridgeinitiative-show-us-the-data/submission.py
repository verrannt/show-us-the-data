from transformers import BertForTokenClassification, AdamW, BertTokenizerFast
import re
import pandas as pd
import numpy as np
import torch
import json
class SubmitPred:
    def __init__(self, test_csv_path, test_path, model_path, tokenizer_path):
        self.test_csv_path = test_csv_path
        self.test_path = test_path
        self.model = BertForTokenClassification.from_pretrained(model_path, num_labels=3, output_attentions=False,
                                                                output_hidden_states=False)
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, do_lower_case=False)

    def retrieve_label(self, sentence, model):
        tokenized_sentence = self.tokenizer.encode(sentence)
        input_ids = torch.tensor([tokenized_sentence]).cuda()
        with torch.no_grad():
            output = model(input_ids)
        label_indices = np.argmax(output[0].to('cpu').numpy(), axis=2)
        predicted_string = np.array(tokenized_sentence)[np.where(label_indices != 2)[1]]
        predicted_string = self.tokenizer.decode(predicted_string)
        return predicted_string

    def load_submission(self, test_path):
        self.sample_submission = pd.read_csv(test_path)

    def predict_from_text(self, sentence_list, model):
        all_preds = []
        for sentence in sentence_list:
            preds = self.retrieve_label(sentence, model)
            if preds != '':
                if not preds.startswith('#'):
                    all_preds.append(preds)
        print(all_preds)
        return "|".join(all_preds)

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

    def extract_and_clean(self, list_sentences):
        list_sentences = [self.clean_training_text(sentence) for sentence in list_sentences]
        list_sentences = self.shorten_sentences(list_sentences, max_len=64, overlap=20)
        sentences = [sentence for sentence in list_sentences if len(sentence) > 10]
        return sentences

    def parse_json(self, json_id, to_return):
        path_to_json = os.path.join(self.test_path, (json_id + '.json'))
        heading = []
        content = []
        print(path_to_json)
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

    def generate_submission(self):
        self.load_submission(self.test_csv_path)
        self.model.cuda()

        print(self.sample_submission)
        self.sample_submission['content_list'] = self.sample_submission.apply(
            lambda x: self.parse_json(x.Id, "content").split("."), axis=1)
        self.sample_submission['content_list'] = self.sample_submission.apply(
            lambda x: self.extract_and_clean(x.content_list), axis=1)
        self.sample_submission['PredictionString'] = self.sample_submission.apply(
            lambda x: self.predict_from_text(x.content_list, self.model), axis=1)
        self.sample_submission.drop('content_list', axis=1)
        self.sample_submission.to_csv(f'submission.csv', index=False)

    def print_submission(self):
        print(self.sample_submission.head())