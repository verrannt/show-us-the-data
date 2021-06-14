# Show US The Data

Code for the Kaggle challenge by the Coleridge Initiative: [Show US The Data](https://www.kaggle.com/c/coleridgeinitiative-show-us-the-data).

## Code

While we provide Python files for our code implementations, the important parts that are required for training and submitting models for the competition can be found as Jupyter notebooks under [`src/notebooks/modeling`](src/notebooks/modeling):

- We provide a [`Training.ipynb`](src/notebooks/modeling/Training.ipynb) notebook that trains a BERT-based model of choice on the competition data, using the input processing pipeline defined in [`src/utils/data/preproc.py`](src/utils/data/preproc.py). 
- Two further notebooks are provided for the data augmentation ([`Training (data augmentation).ipynb`](src/notebooks/modeling/Training (data augmentation).ipynb)) and oversampling ([`Training (oversampling).ipynb`](src/notebooks/modeling/Training (oversampling).ipynb)) parts of our experiments.
- The in-depth analysis of our best model can be found in [`SciBERT Result Analysis.ipynb`](src/notebooks/modeling/SciBERT%20Result%20Analysis.ipynb).
- We have a separate notebook used for submission of all our models in [`Submission.ipynb`](src/notebooks/modeling/Submission.ipynb).

Additional notebooks we used to conduct our initial exploratory data analysis are to be found in [`src/notebooks/EDA`](src/notebooks/EDA).

## Team

- [Aashutosh](https://www.kaggle.com/aerigon) ([aerigon](https://github.com/aerigon/))
- [Thomas](https://www.kaggle.com/thomasrood) ([thomasroodnl](https://github.com/thomasroodnl))
- [Pascal](https://www.kaggle.com/pascalschroder) ([verrannt](https://github.com/verrannt))

## Acknowledgements & License

Parts of our code, mainly relating to the input processing pipeline and the training loop, have been taken from or inspired by [this blog post](https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/) by [Tobias Sterbak](https://www.depends-on-the-definition.com/about/). Like the rest of our code, it is licensed under the MIT license. A copy of the license can be found in the `LICENSE` document. 
