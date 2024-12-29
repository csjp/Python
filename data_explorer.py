import pandas as pd
import numpy as np
import ipywidgets as widgets
from ipywidgets import FileUpload
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
import seaborn as sns
import io

# Output area
output = widgets.Output()

# Create a FileUpload widget
upload_widget = FileUpload(
    accept='.csv',  # Accept CSV files
    multiple=False,  # Do not allow multiple uploads
    description='Upload CSV File'
)

# Initialize an empty DataFrame
df = pd.DataFrame()
column_options = []

def on_file_upload(change):
    global df, column_options
    with output:
        clear_output()
        uploaded_file = change['new'][0]  # Access the first item in the tuple
        content = uploaded_file['content']
        # Read the CSV file
        df = pd.read_csv(io.BytesIO(content))
        print("File uploaded successfully!")
        print(f"DataFrame shape: {df.shape}")
        # Update column options for dropdowns
        column_options = df.columns.tolist()
        update_dropdown_options()

def update_dropdown_options():
    # Update options for column selection widgets
    value_counts_column.options = column_options
    unique_values_column.options = column_options
    histogram_column.options = column_options
    boxplot_column.options = column_options

# Data exploration functions with checks
def show_head():
    with output:
        clear_output()
        if df.empty:
            print("Please upload a CSV file.")
        else:
            display(df.head())

def show_tail():
    with output:
        clear_output()
        if df.empty:
            print("Please upload a CSV file.")
        else:
            display(df.tail())

def show_dtypes():
    with output:
        clear_output()
        if df.empty:
            print("Please upload a CSV file.")
        else:
            display(df.dtypes)

def show_describe():
    with output:
        clear_output()
        if df.empty:
            print("Please upload a CSV file.")
        else:
            display(df.describe())

def show_missing_values():
    with output:
        clear_output()
        if df.empty:
            print("Please upload a CSV file.")
        else:
            display(df.isnull().sum())

def show_corr():
    with output:
        clear_output()
        if df.empty:
            print("Please upload a CSV file.")
        else:
            # Select only numeric columns for correlation
            numeric_df = df.select_dtypes(include=[np.number])
            if numeric_df.empty:
                print("No numeric columns available for correlation.")
            else:
                corr = numeric_df.corr()
                display(corr)
                plt.figure(figsize=(10, 8))
                sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
                plt.show()


def show_value_counts(column):
    with output:
        clear_output()
        if df.empty:
            print("Please upload a CSV file.")
        else:
            counts = df[column].value_counts()
            display(counts)

def show_unique_values(column):
    with output:
        clear_output()
        if df.empty:
            print("Please upload a CSV file.")
        else:
            uniques = df[column].unique()
            print(f"Unique values in '{column}':")
            display(uniques)

def show_histogram(column):
    with output:
        clear_output()
        if df.empty:
            print("Please upload a CSV file.")
        else:
            plt.figure(figsize=(8,6))
            sns.histplot(df[column].dropna(), kde=True)
            plt.title(f"Histogram of {column}")
            plt.show()

def show_boxplot(column):
    with output:
        clear_output()
        if df.empty:
            print("Please upload a CSV file.")
        elif df[column].dtype not in [np.float64, np.int64]:
            print(f"The selected column '{column}' is not numeric. Please select a numeric column.")
        else:
            plt.figure(figsize=(8, 6))
            sns.boxplot(y=df[column].dropna())
            plt.title(f"Boxplot of {column}")
            plt.show()


# Buttons for data exploration options
button_head = widgets.Button(description="First Rows")
button_tail = widgets.Button(description="Last Rows")
button_dtypes = widgets.Button(description="Data Types")
button_describe = widgets.Button(description="Statistical Summary")
button_missing = widgets.Button(description="Missing Values")
button_corr = widgets.Button(description="Correlation Matrix")

# Initialize dropdowns with empty options
value_counts_column = widgets.Dropdown(
    options=column_options,
    description='Select Column:'
)

unique_values_column = widgets.Dropdown(
    options=column_options,
    description='Select Column:'
)

histogram_column = widgets.Dropdown(
    options=column_options,
    description='Select Column:'
)

boxplot_column = widgets.Dropdown(
    options=column_options,
    description='Select Column:'
)

# Buttons for functions requiring column selection
button_value_counts = widgets.Button(description="Show Value Counts")
button_unique_values = widgets.Button(description="Show Unique Values")
button_histogram = widgets.Button(description="Show Histogram")
button_boxplot = widgets.Button(description="Show Boxplot")

# Button click handlers
def on_button_head_clicked(b):
    show_head()

def on_button_tail_clicked(b):
    show_tail()

def on_button_dtypes_clicked(b):
    show_dtypes()

def on_button_describe_clicked(b):
    show_describe()

def on_button_missing_clicked(b):
    show_missing_values()

def on_button_corr_clicked(b):
    show_corr()

def on_button_value_counts_clicked(b):
    show_value_counts(value_counts_column.value)

def on_button_unique_values_clicked(b):
    show_unique_values(unique_values_column.value)

def on_button_histogram_clicked(b):
    show_histogram(histogram_column.value)

def on_button_boxplot_clicked(b):
    show_boxplot(boxplot_column.value)

# Connect buttons to handlers
button_head.on_click(on_button_head_clicked)
button_tail.on_click(on_button_tail_clicked)
button_dtypes.on_click(on_button_dtypes_clicked)
button_describe.on_click(on_button_describe_clicked)
button_missing.on_click(on_button_missing_clicked)
button_corr.on_click(on_button_corr_clicked)

button_value_counts.on_click(on_button_value_counts_clicked)
button_unique_values.on_click(on_button_unique_values_clicked)
button_histogram.on_click(on_button_histogram_clicked)
button_boxplot.on_click(on_button_boxplot_clicked)

# Observe the upload widget
upload_widget.observe(on_file_upload, names='value')

# Group buttons without column selection
button_row1 = widgets.HBox([button_head, button_tail, button_dtypes])
button_row2 = widgets.HBox([button_describe, button_missing, button_corr])

# Group widgets for value counts
value_counts_widget = widgets.VBox([value_counts_column, button_value_counts])

# Group widgets for unique values
unique_values_widget = widgets.VBox([unique_values_column, button_unique_values])

# Group widgets for histogram
histogram_widget = widgets.VBox([histogram_column, button_histogram])

# Group widgets for boxplot
boxplot_widget = widgets.VBox([boxplot_column, button_boxplot])

# Arrange all widgets
ui = widgets.VBox([
    upload_widget,
    button_row1,
    button_row2,
    value_counts_widget,
    unique_values_widget,
    histogram_widget,
    boxplot_widget
])

# Display the UI and output
display(ui, output)
__name__ = "__main__" 