# LesionDetector
## Project Description
LesionDetector is a program written for my master's thesis, that incorporates deep reinforcement learning along with convolutional neural networks to detect lesions in CT scans.

## Virtual Environment Activation
```bash
# Initialize the environment
conda init

# Create a new conda environment
conda env create -f environment.yml

# Activate the environment
conda activate lesion_detector

# Reload the environment
conda env update --name lesion_detector_env --file environment.yml --prune
```

## Virtual Environment Removal
```bash
# Deactivate the environment
conda deactivate

# Remove the existing environment
conda remove --name lesion_detector_env --all
```

## Pre-commit Hooks Configuration
### Install Pre-commit Hooks
```bash
pre-commit install
```
### Pre-commit Hooks manual run
```bash
pre-commit run --all-files
```