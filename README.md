The following was derived from a YouTube tutorial published by Mosh Hamedani.

# Anaconda setup
## Assumptions
This README file was written based on an environment with the following versions loaded:
* Windows 11 Home 23H2 OS Build 22631.4890
* Anaconda Navigator 2.6.4 to 2.6.5
* Visual Studio Code 1.97.2 to 1.98.2
## Setup
* Start Anaconda Navigator.
* From the Home panel, launch Visual Studio Code (VSC).
* Open the `audio-deepfake-detection` project in VSC.
* Start a terminal in VSC.
* Run:
  ```
  conda env create -f .\environment.yml
  ```
* After the environment has finished loading, run:
  ```
  conda activate audio-deepfake-detection
  ```
* Run unit tests from `conda` command-line
  ```
  python -m unittest
  ```
  * NOTE: If the unit tests do not run from the `conda` command-line in Visual Studio Code, try exiting and relaunching from Anaconda Navigator.
## Configure
### Options
* `job-defaults`:
  * `input-file-batch-size`:
  * `output-folder`:
  * `data-path-root`:
  * `data-extension`: ie: .flac
  # Assign 'None' to training-split-random-state to get random split.
  # Pass same int for reproducible split.
  * `training-split-random-state`:
  * `labels-execute-to-categorical`:
  * `num-classes`:        # Number of classes (bonafide and spoof)
  * `sample-rate`:
  * `duration`:
  * `num-mels`:
  * `max-time-steps`:
  * `kernel-size`: Tuple ie: (3, 3)
  * `pool-size`: Tuple ie: (2, 2)
  * `batch-size`:
  * `num-epochs`:
  * `optimizer`:
  * `loss`:
  * `metrics:
    - one metric per line - ie: `accuracy`
  * `preprocessor`:
* `jobs`: A list of jobs each with their own parameters.
  * `{{name}}`: name of job.
    * Any values entered at the job level overrides what is provided in defaults above.



## Update `environment.yml` and `requirements.txt` for commit to source control.
The following commands can be used to save changes to dependencies in the environment:
* Using Anaconda:
  ```
  conda env export -n audio-deepfake-detection > environment.yml
  ```
* Using PIP:
  ```
  pip freeze > requirements.txt
  ```


## References

* Anagha, R., Arya, A., Narayan, V. H., Abhishek, S., & Anjali, T. (2023). Audio Deepfake Detection Using Deep Learning. *2023 12th International Conference on System Modeling & Advancement in Research Trends (SMART), System Modeling & Advancement in Research Trends (SMART), 2023 12th International Conference On*, 176–181. https://doi.org/10.1109/SMART59791.2023.10428163

* Chan, J. (2021). *Machine learning with Python for beginners: A step-by-step guide to hands-on projects* [Kindle edition].

* Hamedani, M. (2020, September 17). *Python machine learning tutorial (Data science)*. YouTube. https://www.youtube.com/watch?v=7eh4d6sabA0

* Koul, N. (2024). *Ultimate Deepfake detection using Python* [Kindle edition]. Orange Education Pvt Ltd.
