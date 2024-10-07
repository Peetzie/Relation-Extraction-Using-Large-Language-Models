# Relation Extraction Using Large Language Models

This repository is part of my Master's Thesis for Human-Centered Artificial Intelligence at the Technical University of Denmark. It compares state-of-the-art classifiers with a novel prompt engineering approach for relation extraction.

## Project Overview

-**Duration:** Approximately 5 months

-**Computational Resources:** Tested with 2x NVIDIA A100 (40GB) and 2x TESLA V100 (32GB). Note that the full memory capacity is not required, but at least 40GB is necessary for model training.

## Repository Structure

The repository is organized into the following main directories:

1.**`databuilding/`**

   Contains scripts and instructions for downloading and parsing datasets.

2.**`prompt/`**

   Contains scripts and relevant information to create a relation extraction pipeline using [DSPY](https://github.com/stanfordnlp/dspy).

3.**`SOTA/`**

   Includes two subfolders for state-of-the-art models:

   -[**DREEAM**](https://github.com/YoumiMa/dreeam)

   -[**REBEL**](https://github.com/Babelscape/rebel)

   **Note:** Alterations have been made to the source code to fit the dataset format from `databuilding/`. For reproducibility, use the source files provided here instead of pulling them directly from the original repositories.

## Setting Up Virtual Environments

Each solution (`databuilding`, `prompt`, `SOTA/DREEAM`, and `SOTA/REBEL`) requires a separate virtual environment. Detailed instructions for setting up these environments, including hotfixes for dependencies, are provided in the respective sections below.

**NOTE:** If you wanna skip setting up the enviroments, I have created a zip folder with the enviroments where all fixes are incorporated.
The folder can be found here from my [Personal Proton Drive with Environments](Viewerhttps://drive.proton.me/urls/A5RZV32TVW#Kwomimt2z5xm)

## Computational Resources

-**Hardware Requirements:**

  -*DREEAM*: Requires approximately **270GB** memory for DocRED data and **800GB** for the combined dataset.

-**Training Note:** A minimum of 40GB memory is required for model training.

## Databuilding

All scripts related to data building are located in the `databuilding/` directory.

### Environment Setup

-**Python Version:**`3.10.13`

-**Dependencies:** Install via `requirements.txt`

```sh

  pip install -r requirements.txt

```

### Running Scripts

- Scripts are located in `components/scripts/`.
- You can run these scripts using the command line, or utilize the provided terminal user interface (`runner.py`) for ease of use.

  **Using the Automated script provides a terminal user interface that handles all input/output arguments required for the functions.** 

  **Default values** are incorporated in the development process, but these can be modified within the relevant `.py` files.

### REBEL Dataset

- Parsing the REBEL dataset is optional.
- Predefined mappings of entity types and relations are found in `components/types/`.
- If parsed, default mappings are provided as a `pkl` file, although you may calculate mappings yourself using the [SPARQL API](https://query.wikidata.org/) (approximately 20 hours).

## REBEL

### References

-**Paper:**[REBEL: Relation Extraction By End-to-End Language Generation](https://aclanthology.org/2021.findings-emnlp.204)

-**Original Repository:**[GitHub - Babelscape/rebel](https://github.com/Babelscape/rebel)

### Environment Setup

-**Python Version:**`3.7.17`

-**Dependencies:** Install via `requirements.txt`

```sh

  pip install -r requirements.txt

```

### Important Hotfixes

-**PyTorch Lightning Fix:**

- Locate `saving.py` in your virtual environment (`.venv/lib/python3.7/site-packages/pytorch_lightning/core/saving.py`).
- Uncomment line 165:

  ```python

  # checkpoint[cls.CHECKPOINT_HYPER_PARAMS_KEY].update(kwargs)

  ```

### Training and Testing

- Training and testing scripts are located in `src/`.
- Example command to train the model:

  ```sh

  python src/train.py model=rebel_model data=custom_ReDocRED train=custom_ReDocRED

  ```
- To evaluate a trained checkpoint:

  ```sh

  python src/test.py model=rebel_model data=custom_ReDocRED train=custom_ReDocRED do_predict=True checkpoint_path="path_to_checkpoint"

  ```
- Hyperparameter tuning can be performed using `optimizer.py` and `optimizer_REBEL.py`.

## DREEAM

### References

-**Paper:**[DREEAM: Guiding Attention with Evidence for Improving Document-Level Relation Extraction](https://arxiv.org/abs/2301.00001)

-**Original Repository:**[GitHub - YoumiMa/dreeam](https://github.com/YoumiMa/dreeam)

### Environment Setup

-**Python Version:**`3.8.17`

-**Dependencies:** Install via `requirements.txt`

```sh

  pip install -r requirements.txt

```

### Training

- Training scripts are located in `training/`.
- The baseline can be run with:

  ```sh

  python training/baseline/baseline.py

  ```
  Make sure to update paths in line 175 of the script to match your config file in `training/config/`.
- For training on the combined dataset, use the segmented training scripts: `part_1.py`, `part_2.py`, etc.

  Example command:

  ```sh

  python training/baseline_roberta.py

  ```
  Running training using the fine-tuned model can be performed automatically using
  ```sh
  python training/finalized/finalized_RoBERTa.py -p <no 1-4>
  ```
  Where the input is an integer 1-4 depending on the training part. 

  ### Testing
  - Testing is performed by running the test script, including a path to the student checkpoint. First it must be run with argument "dev" and later "test" to apply the tresholds. 
  ```
  python training/test.py "checkpoint dir" "dev"
  ```
  ```
  python training/test.py "checkpoint dir" "test"
  ```
  Where checkpoint refers to the path of the student folder containing the last.cpt file. 