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
    data_path = os.path.join(
        os.getcwd(),
        'data/coleridgeinitiative-show-us-the-data/'
    )

    USE_PREVIOUSLY_EXTRACTED = False
    USE_PREVIOUSLY_TOKENIZED = False

    # Configure the variables using PipelineConfigs
    # If SAVE == True, the EXTRACTED_FILENAME and TOKENIZED_FILENAME
    # will be used to load/save the processed data, depending on what 
    # function will be called. If SAVE == False, these will be ignored
    configs = PipelineConfigs(
        DATA_PATH = data_path,
        MAX_LENGTH = 64,
        OVERLAP = 20,
        MAX_SAMPLE = None,
        SAVE = True,
        EXTRACTED_FILENAME = 'train_ner.data',
        TOKENIZED_FILENAME = 'train_ner.data.tokenized',
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

    # Train BERT model here
    

if __name__=='__main__':
    main()