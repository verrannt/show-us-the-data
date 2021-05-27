# append to path to allow relative imports
import os, sys
sys.path.append("..")
import re


def main():

    # Only works when run from root of project (i.e. outside of 'src/')
    data_path = os.path.join(
        os.getcwd(),
        'data/awesomedata/awesome-public-datasets'
    )

    with open(f'{data_path}/README.rst', 'r') as input:
        with open(f'{data_path}/processed.txt', 'w') as output:
            for line in input.readlines():
                if '* |OK_ICON|' in line:
                    dataset_match = re.search(r'(`)[\s\S]+(`)', line)

                    if dataset_match:
                        dataset_string = dataset_match.group()

                        # Remove tildes
                        dataset_string = re.sub(r'(`)', '', dataset_string).strip()

                        # Only keep first half of string
                        dataset_string = dataset_string.split("-")[0].strip()

                        # Remove hyperlinks
                        dataset_string = re.sub(r'(<)[\s\S]+', '', dataset_string).strip()

                        # Only keep dataset names longer than 1 word
                        if len(dataset_string.split()) > 1:
                            output.write(dataset_string+"\n")


if __name__ == "__main__":
    main()
    # Then, manually clean mistakes and odd explanations
