# Cache me if you can: an Online Cost-aware Teacher-Student framework to Reduce the Calls to Large Language Models
This repository is the official implementation of the [OCaTS Framework](https://arxiv.org/abs/2310.13395) that was introduced in the paper: Cache me if you can: an Online Cost-aware Teacher-Student framework to Reduce the Calls to Large Language Models.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cache-me-if-you-can-an-online-cost-aware/intent-detection-on-banking77)](https://paperswithcode.com/sota/intent-detection-on-banking77?p=cache-me-if-you-can-an-online-cost-aware)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cache-me-if-you-can-an-online-cost-aware/sentiment-analysis-on-imdb)](https://paperswithcode.com/sota/sentiment-analysis-on-imdb?p=cache-me-if-you-can-an-online-cost-aware)

## Acknowledgements 
This work is supported from Google's [TPU Research Cloud (TRC) program](https://sites.research.google/trc/about/)

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)

## Installation

In order to use the framework, you need to clone the repository and
install the packages in the `pyproject.toml` file. We recommend using a virtual environment to install the packages. You can use the following commands to create a virtual environment and install the packages with [uv](https://docs.astral.sh/uv/):

```setup
uv sync
```

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
@inproceedings{stogiannidis-etal-2023-cache,
    title = "Cache me if you Can: an Online Cost-aware Teacher-Student framework to Reduce the Calls to Large Language Models",
    author = "Stogiannidis, Ilias  and
      Vassos, Stavros  and
      Malakasiotis, Prodromos  and
      Androutsopoulos, Ion",
    editor = "Bouamor, Houda  and
      Pino, Juan  and
      Bali, Kalika",
    booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2023",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-emnlp.1000",
    doi = "10.18653/v1/2023.findings-emnlp.1000",
    pages = "14999--15008",
    abstract = "Prompting Large Language Models (LLMs) performs impressively in zero- and few-shot settings. Hence, small and medium-sized enterprises (SMEs) that cannot afford the cost of creating large task-specific training datasets, but also the cost of pretraining their own LLMs, are increasingly turning to third-party services that allow them to prompt LLMs. However, such services currently require a payment per call, which becomes a significant operating expense (OpEx). Furthermore, customer inputs are often very similar over time, hence SMEs end-up prompting LLMs with very similar instances. We propose a framework that allows reducing the calls to LLMs by caching previous LLM responses and using them to train a local inexpensive model on the SME side. The framework includes criteria for deciding when to trust the local model or call the LLM, and a methodology to tune the criteria and measure the tradeoff between performance and cost. For experimental purposes, we instantiate our framework with two LLMs, GPT-3.5 or GPT-4, and two inexpensive students, a $k$-NN classifier or a Multi-Layer Perceptron, using two common business tasks, intent recognition and sentiment analysis. Experimental results indicate that significant OpEx savings can be obtained with only slightly lower performance.",
}
```
