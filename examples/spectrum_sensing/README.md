# AI-based Spectrum Sensing Example

## Introduction

This example implements a spetrum sensing application using USRP RF hardware, the NI RF Data Recording API and a YOLO-based image detection model.

# Getting Started

💡 Note: Install UHD and the USRP drivers corresponding to your used OS. See [USRP Hardware Driver and USRP Manual: Binary Installation](https://files.ettus.com/manual/page_install.html) for more details.

💡 Note: It is recommended to use a Python virtual environment for package installation and execution

1. Clone the `NI RF Data Recording API` repository
    > `git clone https://github.com/ni/ni-rf-data-recording-api`
2. Clone the `Yolov5` repository
    > `git clone https://github.com/ultralytics/yolov5`
3. Change to the `NI RF Data Recording API` folder
    > `cd ni-rf-data-recording-api`
4. Install Python dependencies for the `NI RF Data Recording API`
    > `pip install -r requirements.txt`
5. Change to the `Spectrum Sensing` example folder
    > `cd examples/spectrum_sensing`
6. Install Python dependencies for the `Spectrum Sensing`example
    > `pip install -r requirements.txt`
7. Change the settings of the example using the files found in the `config` folder
    - Modify `config_spectrum_sensing_1Rx.json` for RX only setups
    - Modify `config_spectrum_sensing_1Tx1_1Rx.json` for setups with TX1 and RX
    - Modify `config_spectrum_sensing_1Tx1_1Rx.json` for setups with TX2 and RX
    - Modify `config_spectrum_sensing_2Tx_1Rx.json` for setups with TX1, TX2 and RX

# Execution

1. Open a terminal, activate your Python virtual environment if needed and run the inference script
    > `python inference.py`
2. Open another terminal, activate your Python virtual environment if needed and run the UI application
    > `python spectrum_sensing.py`
3. Open your browser and go to http://127.0.0.1:8050 (or use the provided link shown in the output of step #2)
4. Configure the settings for TX1, TX2 and RX according to your needs
5. Click on **Start** to start the Spectrum Sensing application example, click **Stop** before applying changes to the current settings

# Notes

* If a GPU with CUDA support is available, it will be used automatically instead of the CPU. Refer to https://pytorch.org/get-started/locally/ for how to enable PyTorch with CUDA support.
* Additional settings can be changed directly in the Python scripts, e.g. to configure different locations for data files, yolov5, etc.