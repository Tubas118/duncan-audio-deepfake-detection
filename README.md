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
## Configure
### Options
* `jobs`: A list of jobs each with their own parameters.
  * `{{name}}`: name of job.
    * `data-path`:
    * `training-data-path`: 
    * `training-data-extension`:
    * `training-label-filename`: 
    * `num-classes`:
    * `sample-rate`:
    * `duration`:
    * `num-mels`:
    * `max-time-steps`:
    * `optimizer`:
    * `loss`:
    * `metrics`:
      - <<one metric per line>>
    * `batch-size`:
    * `num-epochs`:



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

Hamedani, M. (2020, September 17). *Python machine learning tutorial (Data science)*. YouTube. https://www.youtube.com/watch?v=7eh4d6sabA0
