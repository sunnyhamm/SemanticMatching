import dash
from dash import dcc, html, dash_table
import dash.dependencies as dd
import pandas as pd
import os
import flask

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the directory path for CSV and Excel files
local_directory = './data'  # Adjust this path as necessary

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
