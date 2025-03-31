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
conda activate lesion_detector_env

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

## Run training
```bash
python main.py --task train_localizer
```

## Plot results
To plot the obtained results from a training or evaluation run use the plot_results.py script located in the scripts/ folder in the root directory.
```bash
python plot_results.py --csv_file <csv_file_path>
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

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.