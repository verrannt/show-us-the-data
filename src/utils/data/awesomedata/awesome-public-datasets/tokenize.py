from src.utils import ParseUtils
from transformers import *


data_path = os.path.join(
        os.getcwd(),
        'data/coleridgeinitiative-show-us-the-data/'
        )
datasets = ParseUtils.load_auxiliary_datasets(data_path, 'auxiliary_datasets.txt')

tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_cased', do_lower_case=False) #BertTokenizerFast.from_pretrained('bert-base-cased', do_lower_case=False)

datasets = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(dataset)) for dataset in datasets]

with open(f'{data_path}/auxiliary_datasets_scibert_tokenized.txt', 'w') as output:
    for dataset in datasets:
        output.write(",".join(list(map(lambda it: str(it), dataset))) + "\n")

# test = ParseUtils.load_tokenized_auxiliary_datasets(data_path, 'auxiliary_datasets_bert_tokenized.txt')