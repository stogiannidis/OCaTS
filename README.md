# Cache me if you can: an Online Cost-aware Teacher-Student framework to Reduce the Calls to Large Language Models
This repository is the official implementation of the [OCaTS Framework](https://arxiv.org/abs/2310.13395) that was introduced in the paper: Cache me if you can: an Online Cost-aware Teacher-Student framework to Reduce the Calls to Large Language Models.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cache-me-if-you-can-an-online-cost-aware/intent-detection-on-banking77)](https://paperswithcode.com/sota/intent-detection-on-banking77?p=cache-me-if-you-can-an-online-cost-aware)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cache-me-if-you-can-an-online-cost-aware/sentiment-analysis-on-imdb)](https://paperswithcode.com/sota/sentiment-analysis-on-imdb?p=cache-me-if-you-can-an-online-cost-aware)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Installation

In order to use the framework, you need to clone the repository and
install the packages in the `requirements.txt` file. We recommend using a virtual environment to install the packages. You can use the following commands to create a virtual environment and install the packages:

* Conda:
```setup
conda create -n ocats python=3.11
conda activate ocats
pip install -r requirements.txt
```

* Virtualenv:
```setup
virtualenv ocats
source ocats/bin/activate
pip install -r requirements.txt
```

* Pyenv:
```setup
pyenv virtualenv 3.11 ocats
pyenv activate ocats
pip install -r requirements.txt
```

We recommend using conda to install the packages as it is the easiest way to install the packages. If you don't have conda installed, you can install it from [here](https://docs.conda.io/en/latest/miniconda.html).

## Usage
In order to use the framework, you need to first train the student model, if you are using a $k$-NN, you can skip this step. Then you need to tune the decision thesholds of the framework and then use the framework. The framework is implemented in the `main.py` file.

**Important notes:**
* 🚩 This repository is specific to experiments employed for the paper and we only provide the code for the MLP and $k$-NN students. However, the framework can be used with any teacher and student model. Please feel free to create a fork of this repository and add your own teacher and student models. We would be happy to merge your pull request.
* 🚩 We plan to make the code more user-friendly in the future. If you have any suggestions, please feel free to open an issue or create a pull request.
* 🚩 Finally, we plan to implement the full abstraction of the framework in the future. Currently, the framework is implemented in the `main.py` file and it is not abstracted. We also plan to open-source the code for the abstracted framework in the future for you to use.

### Training
To train the MLP on top of MPNet embeddings, you can use the following command:
```train
python train.py --config <config_file> --train_path <train_path> --dev_path <dev_path> --model_dir <model_dir> 
```

### Tuning
To tune the decision thresholds with the $k$-NN student, you can use the following command:
```tune
python tune_knn.py --config <config_file> --lambdas <lambdas> --train_path <train_path> --dev_path <dev_path> --study_name <study_name> 
```
To tune the decision thresholds with the MLP student, you can use the following command:
```tune
python tune.py --config <config_file> --lambdas <lambdas> --train_path <train_path> --dev_path <dev_path> --study_name <study_name> 
```

### Evaluation
To evaluate the framework, you can use the following command:
```eval
python main.py --config <config_file> --lambdas <lambdas> --train_path <train_path> --test_path <test_path> --model <model> 
```

You can also use the scripts without any arguments. In this case, the scripts will use the default values which are for the main experiment in the paper. 

### Prompting LLMs
To prompt GPT-3.5-turbo and GPT-4 with this framework, you can use the `prompt.py` script in the `misc` folder. You can use the following commandz inside the misc folder:
```prompt
python prompt.py --model <model> --prompt <prompt> --train_data <train_data> --test_data <test_data> 
```

## Contributing

As mentioned above, we plan to make the code more general user-friendly in the future. If you have any suggestions, please feel free to open an issue or create a pull request with your changes.

## License

This repository is licensed under the MIT license. See [LICENSE](LICENSE) for details.

## Citation

```
@article{stogiannidis_etal2023,
  title={Cache me if you can: an Online Cost-aware Teacher-Student framework to Reduce the Calls to Large Language Models},
  author={Ilias Stogiannidis and Stavros Vassos and Prodromos Malakasiotis and Ion Androutsopoulos},
  journal={Findings of 2023 Conference on Empirical Methods in Natural Language Processing},
  year={2023},
  publisher={Association for Computational Linguistics}
}
```
