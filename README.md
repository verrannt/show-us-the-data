# Show US The Data

Code for the Kaggle challenge by the Coleridge Initiative: Show US The Data.

## Directory structure

* `vis/`: all types of visualisations, like plots and images of model architectures
* `model/`: model related outputs, like training and testing logs, model weights and results
* `src/`: contains all source code in different subdirectories:
  * `models/`: code that implements model architectures
  * `notebooks/`: Jupyter Notebooks for quick testing and visualization that are not part of the main workflow
  * `utils/`: utility modules:
    - `model/`: modules for loading, training and testing your models
    - `data/`: modules for loading and preprocessing/augmenting your data
    - `generic.py`: module for generic helper functions, e.g. status printing for scripts
  * `run.py`: main file in which to set up the logical flow of the project and run different interactions, i.e. call the functions defined in the different modules above
  
**Note:** there is no central `data/` directory as data might be placed outside of this repository. If not, I recommend to place it at the root. 

## Acknowledgements & License

Parts of our code, mainly relating to the input processing pipeline and the training loop, have been taken from or inspired by [this blog post](https://www.depends-on-the-definition.com/named-entity-recognition-with-bert/) by [Tobias](https://www.depends-on-the-definition.com/about/). Like the rest of our code, it is licensed under the MIT license. A copy of the license can be found in the `LICENSE` document. 