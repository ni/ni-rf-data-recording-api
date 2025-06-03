# AI-based Spectrum Sensing Example

## Introduction

This example implements a spetrum sensing application using USRP RF hardware, the NI RF Data Recording API and a YOLO-based image detection model

# Getting Started
*Note: It is recommended to use a Python virtual environment for package installation and execution*

1. Install Python dependencies from the NI RF Data Recording API if not already done
    > `pip install -r ../../requirements.txt`

2. Install Python dependencies for this example
    > `pip install -r ./requirements.txt`

3. Clone the `Yolov5` repository, e.g. outside of this repository
    > `git clone https://github.com/ultralytics/yolov5 ../../yolov5`

4. Change the settings of the example using the files found in the `config` folder
    - Modify `config_spectrum_sensing_1Rx.json` for RX only setups
    - Modify `config_spectrum_sensing_1Tx1_1Rx.json` for setups with TX1 and RX
    - Modify `config_spectrum_sensing_1Tx1_1Rx.json` for setups with TX2 and RX
    - Modify `config_spectrum_sensing_2Tx_1Rx.json` for setups with TX1, TX2 and RX

# Execution

1. Open a terminal, activate your Python virtual environment if needed and run the inference script
    > `python ./inference.py`
2. Open another terminal, activate your Python virtual environment if needed and run the UI application
    > `python ./spectrum_sensing.py`
3. Open your browser and go to http://127.0.0.1:8050 (or use the provided link shown in the output of step #2)
4. Configure the settings for TX1, TX2 and RX according to your needs
5. Click on **Start** to start the Spectrum Sensing application example, click **Stop** before applying changes to the current settings


# Notes

* If a GPU with CUDA support is available, it will be used automatically instead of the CPU. Refer to https://pytorch.org/get-started/locally/ for how to enable PyTorch with CUDA support.
* Additional settings can be changed directly in the Python scripts, e.g. to configure different locations for data files, yolov5, etc.
* For debugging, it is recommended to enable the developer UI in `spectrum_sensing.py`