The following was derived from a YouTube tutorial published by Mosh Hamedani.

# Anaconda setup
## Assumptions
This README file was written based on an environment with the following versions loaded:
* Windows 11 Home 23H2 OS Build 22631.4890
* Anaconda Navigator 2.6.4
* Visual Studio Code 1.97.2
## Setup
* Start Anaconda Navigator.
* From the Home panel, launch Visual Studio Code (VSC).
* Open the `main-audio-deepfake-detection` project in VSC.
* Start a terminal in VSC.
* Run:
  ```
  conda env create -p ./.venv -f .\environment.yml
  ```
* After the environment has finished loading, run:
  ```
  conda activate main-audio-deepfake-detection
  ```
## Configure
### Options
* `jobs`: A list of jobs each with their own parameters.
  * `{{name}}`: name of job.
    * `training-data-filename`: name of source data file to use for training.
    * `persisted-model`: model name to save or load.
    * `visualize-feature-names`: feature names to include on model visualization.


## Update `environment.yml` and `requirements.txt` for commit to source control.
The following commands can be used to save changes to dependencies in the environment:
* Using Anaconda:
  ```
  conda env export -n main-audio-deepfake-detection > environment.yml
  ```
* Using PIP:
  ```
  pip freeze > requirements.txt
  ```


## References

Hamedani, M. (2020, September 17). *Python machine learning tutorial (Data science)*. YouTube. https://www.youtube.com/watch?v=7eh4d6sabA0
