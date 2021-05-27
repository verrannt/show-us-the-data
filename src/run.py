# append to path to allow relative imports
import os, sys
sys.path.append("..")
from rich.console import Console
from utils.data.preproc import Pipeline, PipelineConfigs
from utils.data.parse import ParseUtils
from utils.generic import timer


@timer
def main():

    console = Console()

    # Only works when run from root of project (i.e. outside of 'src/')
    data_path = 'D:\\Kynntec\\Git_repositories\\show-us-the-data\data\\coleridgeinitiative-show-us-the-data'
    # data_path = os.path.join(
    #     os.getcwd(),
    #     'data/coleridgeinitiative-show-us-the-data/'
    # )

    USE_PREVIOUSLY_EXTRACTED = True
    USE_PREVIOUSLY_TOKENIZED = True
    USE_PREVIOUS_MENTIONPOS = False

    # Configure the variables using PipelineConfigs
    # If SAVE == True, the EXTRACTED_FILENAME and TOKENIZED_FILENAME
    # will be used to load/save the processed data, depending on what 
    # function will be called. If SAVE == False, these will be ignored
    configs = PipelineConfigs(
        DATA_PATH = data_path,
        MAX_LENGTH = 64,
        OVERLAP = 20,
        MAX_SAMPLE = None,
        SAVE = False,
        EXTRACTED_FILENAME = 'train_ner.data',
        TOKENIZED_FILENAME = 'train_ner.data.scibert-tokenized',
        MENTION_POS_FILENAME = 'train_ner.data.scibert-mentpos',
        MAX_TEXT_TOKENS=200000
    )

    pipeline = Pipeline(configs)

    # If we have already tokenized and preprocessed data in storage
    # we can load it
    if USE_PREVIOUSLY_TOKENIZED:
        input_ids, tags, attention_mask = pipeline.load_outputs()
    # Otherwise need to get raw sentence data extracted from text
    else:
        # If we have extracted data in storage, use it
        if USE_PREVIOUSLY_EXTRACTED:
            ner_data = pipeline.load_extracted()
            console.log('Loaded extracted data')
        # Else extract it
        else:
            ner_data = pipeline.extract()
            console.log('Extracted data')

        # If you want to set a custom tokenizer, call this function before
        # calling `run()`
        #pipeline.set_tokenizer(BertTokenizerFast.from_pretrained(
        #    'bert-base-cased', do_lower_case=False))

        # Using the extracted data, we can compute the input data for the 
        # model by running it through the pipeline
        input_ids, tags, attention_mask = pipeline.run(ner_data)

    if USE_PREVIOUS_MENTIONPOS:
        mention_positions = pipeline.load_mention_pos()
    else:
        ParseUtils.save_mention_positions(tags, data_path, "train_ner.data.scibert-mentpos", o_id=303)  # Normal BERT: 152. SciBERT: 303
        mention_positions = pipeline.load_mention_pos()

    # Train BERT model here


if __name__ =='__main__':
    main()
