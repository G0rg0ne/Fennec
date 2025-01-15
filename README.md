
# fennec

This repository provides a robust training pipeline for solving classification tasks on the FSD50K dataset, a comprehensive collection of labeled sound events. The pipeline leverages advanced audio preprocessing, feature extraction, and deep learning techniques to achieve high classification performance. It supports configurable hyperparameters, model architectures, and training strategies, making it adaptable to various audio classification scenarios. Explore the code and documentation to train, evaluate, and customize the pipeline for your specific needs.

## Dataset 
Freesound Dataset 50k (or [FSD50K](https://paperswithcode.com/dataset/fsd50k) for short) is an open dataset of human-labeled sound events containing 51,197 Freesound clips unequally distributed in 200 classes drawn from the AudioSet Ontology.

## Docker

First launch the Mlflow server : 
```bash
cd docker/
docker build -t mlflow_server .
docker-compose up -d
```
Build the main project conttainer:
```bash
docker build -t fennec .
```

Run the container and install dependencies
```bash
docker run -it --rm --gpus all -v $(pwd):/app fennec /bin/bash
```
Training pipeline
```bash
kedro run -p audio_classification
```
You can run a specific node of the pipeline by adding `-t <tag_name>` 

```bash
kedro run -p audio_classification -t feature_extraction
```


[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is your new Kedro project, which was generated using `kedro 0.19.10`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.


## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)


Usefull links: 
https://github.com/jaquielajoie/MelSpectrograms/tree/main/src/notebooks

https://github.com/FilipTirnanic96/mfcc_extraction

https://paperswithcode.com/task/audio-classification

https://zenodo.org/records/4060432
