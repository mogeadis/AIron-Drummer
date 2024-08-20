# AIron Drummer 🥁


## Table of Contents

- [Description](#description)
    - [Project](#project)
    - [Repository](#repository)
    - [Technical Details](#technical-details)
- [Installation](#installation)
    - [Step 1: Download Repository](#step-1-download-repository)
    - [Step 2: Navigate to Project Directory](#step-2-navigate-to-project-directory)
    - [Step 3: Create Virtual Environment](#step-3-create-virtual-environment)
    - [Step 4: Install Requirements](#step-4-install-requirements)
- [Usage](#usage)
    - [Prerequisites](#prerequisites)
    - [Train the Neural Network](#train-the-neural-network)
    - [Generate a Drum Track](#generate-a-drum-track)
- [License](#license)


## Description

### Project
**AIron Drummer** generates a MIDI drum track to accompany a musical excerpt contained in an audio (`.wav`) file by utilizing a neural network model. To do this, a MIDI (`.mid`) file containing information about the tempo and time signature changes of the track is also required. Such a file can be created using either a MIDI editor or a Digital Audio Workstation (DAW), while ensuring that the audio and MIDI events are aligned in time. It is crucial that no tempo automations are present within any musical measure; in other words, all measures *must* maintain a constant tempo value. The neural network was trained using a total of 120 MIDI tracks from [Iron Maiden](https://www.ironmaiden.com)'s discography. You can listen to some of the generated samples from tracks not included in the training set at the following [link](https://youtu.be/JYg5VLjR_FE).

### Repository
The code in this repository is organized into two directories, namely [*Scripts*](Scripts) and [*Modules*](Modules). The former contains three *Python* scripts and one *Jupyter Notebook*, which facilitates the integration of its associated code parts with a computing environment such as *Google Colaboratory*. The latter consists of several modules, one of which serves as a configuration file for all but the neural network parameters, which can be tuned in the aforementioned notebook. A third directory contains a handful of [*Files*](Files) for demonstration purposes and, last but not least, the full text of my [*thesis*](thesis.pdf) and an abridged published [*paper*](paper.pdf) are included in the repository for reference. Note that both of these texts are written in Greek, with abstracts provided in English.

### Technical Details
This project was developed using *Visual Studio Code* on `Windows 10`, with `Python 3.9.7` from the standard *CPython* distribution. The rest of this guide assumes that you are working in a similar environment, although it's not by any means necessary. You can use any available tools to run the code on any operating system, but you should at least have a compatible version of *Python* installed on your machine. 


## Installation

### Step 1: Download Repository
Download the repository as a ZIP file and extract its contents to a directory of your choice.

### Step 2: Navigate to Project Directory
Open a command-line interface and navigate to your chosen project directory, using the `cd` command, like this:
```
cd C:\Users\mogeadis\Projects\GitHub\AIron-Drummer
```

### Step 3: Create Virtual Environment
Create and activate a virtual environment, e.g. `.venv`:
```
python -m venv .venv
```
```
.venv\Scripts\activate
```

### Step 4: Install Requirements
Install the package requirements using `pip`:
```
python -m pip install -U pip setuptools wheel
```
```
python -m pip install -r requirements.txt
```


## Usage

### Prerequisites

Before proceeding, please make sure that you have carefully read the project description and correctly followed the installation instructions above. To avoid warnings in *Visual Studio Code*, you will first need to open the *Command Palette* (Ctrl+Shift+P) and select *Preferences: Open Workspace Settings (JSON)*. A `.vscode` folder containing a `settings.json` file will be automatically created, where you will have to add the following configuration:
```json
{
    "python.analysis.extraPaths": ["./Modules"]
}
```

### Train the Neural Network

There is no need to build a neural network from scratch, as a pretrained model ([`test_model.h5`](Files/test_model.h5)) is already included in the [*Files*](Files) directory. However, if you wish to train a new model, you will first need to provide a dataset of your own with the following directory structure:

> Dataset
>> - Audio
>>> - Track 1.wav
>>> - Track 2.wav
>>> - ...
>> - MIDI
>>> - Track 1.mid
>>> - Track 2.mid
>>> - ...

Then, you will have to assign the path of your dataset directory to the `dataset_path` variable inside the [`config.py`](Modules/config.py) module, where you can change any other configuration you wish. Finally, you will have to run [`create_dataset.py`](Scripts/create_dataset.py), [`preprocess_dataset.py`](Scripts/preprocess_dataset.py) and [`train_model.ipynb`](Scripts/train_model.ipynb) in succession. Please make sure beforehand that your `.wav` and `.mid` files satisfy the conditions specified in the project description. 

### Generate a Drum Track

To generate a drum track you will simply need to run [`demo.py`](Scripts/demo.py) after setting the paths to your files accordingly. Of course, you can use the example files ([`test_model.h5`](Files/test_model.h5), [`test_midi.mid`](Files/test_midi.mid) & [`test_audio.wav`](Files/test_audio.wav)) which are included in the [*Files*](Files) directory. The generated drum track MIDI file ([`test_drums.mid`](Files/test_drums.mid)) is included as well, along with the associated synthesized audio sample ([`test_sample.wav`](Files/test_sample.wav)).


## License

*AIron Drummer* © *2022* by *Alexandros Iliadis* is licensed under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/).

This license requires that reusers give credit to the creator. It allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, for noncommercial purposes only. If others modify or adapt the material, they must license the modified material under identical terms.

See the [LICENSE.md](LICENSE.md) file for more details.