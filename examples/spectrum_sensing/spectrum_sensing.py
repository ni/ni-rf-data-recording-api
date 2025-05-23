#
# Copyright 2025 Northeastern University and National Instruments Corporation
#
# SPDX-License-Identifier: MIT
#
import glob
import os
import sys

import dash
import dash_bootstrap_components as dbc
import dash_daq as daq
from dash import Input, Output, dcc, html
from PIL import Image

# We need the path to this file to base of other paths
curr_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent 'src' folder to the system path
sys.path.append(os.path.join(curr_dir, '../../src'))
import data_recording  # noqa: E402
from lib import sync_settings  # noqa: E402

# *** Titles ***
short_title = "Spectrum Sensing with AI"
long_title = "AI-based Spectrum Sensing with NI USRP SDR Devices"

# *** Update Rates ***
gui_status_update_rate_ms = 100
gui_fig_update_rate = 800

# *** Paths ***
datasets_path = os.path.join(curr_dir, 'datasets/')
rx_recorded_data_path = os.path.join(datasets_path, 'records/')
inference_results_folder_source = os.path.join(datasets_path, 'results/')
ni_rf_data_recording_api_path = os.path.join(curr_dir, '../../')

# *** Images ***
block_diagram_img = 'GUI_API_USRP_block_diagram.png'
system_diagram_img = 'GUI_API_System_block_diagram.png'
default_inference_img = os.path.join(curr_dir, 'assets/default.jpg')

# *** Enabled Waveforms ***
waveform_list = [
    "5GNR_FR1_DL_FDD_SISO_BW-10MHz_CC-1_SCS-30kHz_TM2",
    # "5GNR_FR1_DL_TDD_SISO_BW-20MHz_CC-1_SCS-30kHz_Mod-64QAM_OFDM_TM3.1",
    "LTE_FDD_DL_10MHz_CC-1_E-UTRA_E-TM2",
    "LTE_TDD_DL_20MHz_CC-1_E-UTRA_E-TM3.1",
    # "Radar_Waveform_BW_2M",
    "IEEE_tx11ac_legacy_20MHz_80MSps_MCS7_27bytes_1frame",
    "OFF"
]

# *** DASH App Initialization ***
app = dash.Dash(
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title=short_title,
    update_title=None
)

# *** Global Variable Initialization ***
global general_config_gui
global txs_config_gui
global rxs_config_gui
general_config_gui = {}
txs_config_gui = []
rxs_config_gui = []

# *** Sync Settings (used by UI indicators) ***
sync_settings.init()

# *** UI Components ***

# Title
page_title = html.H1(long_title, className="text-primary text-center fw-bold")

# Tx 1 Config
tx01_Cfg = html.Div(children=[
    html.H2('TX1 Config'),
    html.Label('Frequency [GHz]'),
    dbc.Input(id="tx1_freq_GHz", type="number", placeholder="", value=3.590, className="mb-2"),
    html.Label('Gain [dB]'),
    dbc.Input(id="tx1_gain_dB", type="number", placeholder="", value=40.0, className="mb-2"),
    html.Label('Waveform'),
    dcc.Dropdown(waveform_list, waveform_list[0], id='tx1_waveform')
    ],
    className="p-1 border border-2 border-dark m-1 rounded"
)

# Tx 2 Config
tx02_Cfg = html.Div(children=[
    html.H2('TX2 Config'),
    html.Label('Frequency [GHz]'),
    dbc.Input(id="tx2_freq_GHz", type="number", placeholder="", value=3.610, className="mb-2"),
    html.Label('Gain [dB]'),
    dbc.Input(id="tx2_gain_dB", type="number", placeholder="", value=40.0, className="mb-2"),
    html.Label('Waveform'),
    dcc.Dropdown(waveform_list, waveform_list[-1], id='tx2_waveform')
    ],
    className="p-1 border border-2 border-dark m-1 rounded"
)

# RX Config
rx_cfg = html.Div(children=[
    html.H2('Rx Config'),
    html.Label('Frequency [GHz]'),
    dbc.Input(id="rx_freq_GHz", type="number", placeholder="", value=3.60, className="mb-2"),
    html.Label('Sample Rate [MHz]'),
    dbc.Input(id="rx_sample_rate_MHz", type="number", placeholder="", value=50.0, className="mb-2"),
    html.Label('Gain [dB]'),
    dbc.Input(id="rx_gain_dB", type="number", placeholder="", value=40.0, className="mb-2"),
    html.Label('Duration [ms]'),
    dbc.Input(id="rx_duration_ms", type="number", placeholder="", value=20.0, className="mb-2"),
    html.Label('Number of records'),
    dbc.Input(id="rx_nrecords", type="number", placeholder="", value=100000.0, className="mb-2")
    ],
    className="p-1 border border-2 border-dark m-1 rounded"
)

# control Config
start_button = html.Div(children=[
    dbc.Button('Start', id="startBtn", n_clicks=0, color="primary", className="w-100"),
    ],
)

stop_button = html.Div(children=[
    dbc.Button('Stop', id="stopBtn", n_clicks=0, color="danger", className="w-100"),
    ],
)

# Indications
usrp_init_status = html.Div(children=[
    daq.Indicator(id='usrp_init_status', label="SDR Init", labelPosition="right",
                  width=30, height=30, color="#FF9616", className="ms-1 w-100"),
    dcc.Interval(id='interval_component_usrp_init',
                 interval=1*gui_status_update_rate_ms,
                 n_intervals=0),
    ],
)

rf_data_recording_status = html.Div(children=[
    daq.Indicator(id='rf_data_recording_status', label="Recording Status", labelPosition="right",
                  width=30, height=30, color="#FF9616", className="ms-1 w-100"),
    dcc.Interval(id='interval_component_rf_data_recording_status',
                 interval=1*gui_status_update_rate_ms,
                 n_intervals=0),
    ],
)

# Using Pillow to read the image
pil_img = Image.open(default_inference_img)
inference_result_figure = html.Div(children=[
    html.H2('Spectrum sensing AI - Inference results'),
    html.Img(id='live-update-graph', src=pil_img,  # src = assets_path,
             style={'height': '100%', 'width': '100%'}),
    dcc.Interval(id='interval_component_inference',
                 interval=1*gui_fig_update_rate,
                 n_intervals=0),

    ],
    className="p-1 border border-2 border-dark rounded"
)

system_block_diagram = html.Div(
    children=[
        html.Img(id='system_block_diagram',
                 src=app.get_asset_url(system_diagram_img),
                 style={'height': '100%', 'width': '100%', 'object-fit': 'contain'}),
    ],
    style={'height': '100%'}
)

usrp_block_diagram = html.Div(
    children=[
        html.Img(id='usrp_block_diagram',
                 src=app.get_asset_url(block_diagram_img),
                 style={'height': '100%', 'width': '100%', 'object-fit': 'contain'}),
    ],
    style={'height': '100%'}
)

# *** UI Layout ***
app.layout = dbc.Container([
    dbc.Row(
        [
            dbc.Col(page_title, width=True)
        ],
        className="mb-4",
    ),
    dbc.Row(
        [
            dbc.Col(start_button, width=1),
            dbc.Col(stop_button, width=1),
            dbc.Col(usrp_init_status, width="auto", align="center"),
            dbc.Col(rf_data_recording_status, width="auto", align="center"),
        ],
        className="mb-4",
    ),
    dbc.Row(
        [
            dbc.Col([
                dbc.Row(dbc.Col(tx01_Cfg, width=12)),
                dbc.Row(dbc.Col(tx02_Cfg, width=12)),
            ]),
            dbc.Col(usrp_block_diagram, width=3, align="center"),
            dbc.Col(rx_cfg, width=2),  # "auto"),
            dbc.Col(inference_result_figure, width=4),  # "auto"),
        ],
        justify="center"
    ),
    dbc.Row(
        [
            dbc.Col(system_block_diagram, width=8),
        ],
        justify="center",
        style={"flex-grow": "1"}
    ),
    html.Div(id='dummy-output_get_gui_config'),
    html.Div(id='dummy-output_start'),
    html.Div(id='dummy-output_stop')
    ],
    fluid=True,
    className="p-5",
    style={"display": "flex", "flex-direction": "column", "height": "100vh"}
)


# *** Callbacks ***
@app.callback(Output('dummy-output_get_gui_config', 'children'),
              [Input("tx1_freq_GHz", "value"),
               Input("tx1_gain_dB", "value"),
               Input('tx1_waveform', "value"),
               Input("tx2_freq_GHz", "value"),
               Input("tx2_gain_dB", "value"),
               Input('tx2_waveform', "value"),
               Input("rx_freq_GHz", "value"),
               Input("rx_sample_rate_MHz", "value"),
               Input("rx_gain_dB", "value"),
               Input("rx_duration_ms", "value"),
               Input("rx_nrecords", "value")
               ])
def get_gui_config(tx1_freq_GHz, tx1_gain_dB, tx1_waveform,
                   tx2_freq_GHz, tx2_gain_dB, tx2_waveform,
                   rx_freq_GHz, rx_sample_rate_MHz, rx_gain_dB, rx_duration_ms, rx_nrecords):
    global general_config_gui
    global txs_config_gui
    global rxs_config_gui
    general_config_gui = {"nrecords": rx_nrecords,
                          "rx_recorded_data_path": rx_recorded_data_path}
    # Unit conversion
    # TX 1
    tx1_config = {}
    tx1_config = {"freq": tx1_freq_GHz * 1e9,
                  "gain": tx1_gain_dB,
                  "waveform_file_name": tx1_waveform
                  }
    # TX 2
    tx2_config = {}
    tx2_config = {"freq": tx2_freq_GHz * 1e9,
                  "gain": tx2_gain_dB,
                  "waveform_file_name": tx2_waveform
                  }
    txs_config_gui = [tx1_config, tx2_config]
    # Rx
    rx_config = {}
    rx_config = {"freq": rx_freq_GHz * 1e9,
                 "gain": rx_gain_dB,
                 "rate": rx_sample_rate_MHz * 1e6,
                 "duration": float(rx_duration_ms * 1e-3),
                 }
    rxs_config_gui = [rx_config]
    return html.Div([''])


# Execute RF Data Recording API
@app.callback(Output('dummy-output_start', 'children'),
              [Input('startBtn', 'n_clicks')])
def start_selected(n_clicks):
    runFlag = data_recording.get_recording_status()
    if n_clicks > 0 and runFlag is False:
        print('start_selected n_clicks ', n_clicks)
        print('Start NI RF Data Recording API:')
        data_recording.run_ni_rf_data_recording_api(
            general_config_gui, txs_config_gui, rxs_config_gui, ni_rf_data_recording_api_path)
    return html.Div([''])


# Stop RF Data Recoording API
@app.callback(Output(component_id='dummy-output_stop', component_property='children'),
              Input(component_id='stopBtn', component_property='n_clicks'))
def stop_selected(n_clicks):
    if n_clicks > 0:
        print('stop_selected n_clicks:', n_clicks)
        data_recording.stop_rf_data_recording_api()
        data_recording.delete_previous_records(
            rx_recorded_data_path, inference_results_folder_source)
    return html.Div([''])


# Update recording Status
@app.callback(Output(component_id='rf_data_recording_status', component_property='color'),
              Input(component_id='interval_component_rf_data_recording_status',
                    component_property='n_intervals'))
def update_recording_ind(n):
    retVal = data_recording.get_recording_status()
    if retVal:
        return "#16FF32"
    else:
        return "#FF9616"


# Update SDR INIT
@app.callback(Output(component_id='usrp_init_status', component_property='color'),
              Input(component_id='interval_component_usrp_init', component_property='n_intervals'))
def update_usrp_init_status(n):
    retVal = data_recording.get_usrp_init_status()
    if retVal:
        return "#16FF32"
    else:
        return "#FF9616"


# Multiple components can update everytime interval gets fired.
@app.callback(Output('live-update-graph', component_property='src'),
              Input(component_id='interval_component_inference', component_property='n_intervals'))
def update_graph_live(n):
    # get list of all inference results
    img_result_list = glob.glob(inference_results_folder_source + '*' + '.jpg')
    # run only if the list is not empty
    if not img_result_list or len(img_result_list) < 2:
        fig = Image.open(default_inference_img)
    else:
        # sort the files by date/time in ascending order (oldest first)
        img_result_list = sorted(img_result_list, key=os.path.getmtime)
        if len(img_result_list) > 2:
            # remove the most recent file from the list
            img_result_list.pop()
        # get path of most recent file remaining
        img_result_file_path = img_result_list.pop()
        fig = Image.open(img_result_file_path)
        # delete older files
        [os.remove(file) for file in img_result_list]
        print("Show new result:", img_result_file_path)

    return fig


# *** Main Function ***
if __name__ == "__main__":
    app.run(debug=True, dev_tools_ui=False)
