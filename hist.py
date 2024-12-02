import numpy as np 
import pandas as pd 
import warnings
from plotnine import ggplot, aes, geom_histogram, theme_minimal, labs
import pandas as pd
import matplotlib.pyplot as plt
import io

# Load the datasets
train_df = pd.read_csv(r'C:\shree project\p\PRODIGY_DS_1\test.csv')
test_df = pd.read_csv(r'C:\shree project\p\PRODIGY_DS_1\test.csv')

# First five rows of the dataframe
train_df.head()

# Basic info of train dataframe
train_df.info()

# Number of missing values in columns
train_df.isna().sum()

# Number of duplicates values in dataframe
train_df.duplicated().sum()

# First five rows of test data
test_df.head()

# Basic info of test dataframe
test_df.info()

# Missing values
test_df.isna().sum()

train_df.duplicated().sum()

col = train_df[['Age']]

def plot_numerical(df, column_name):
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame")
    
    # Extract the series from the DataFrame
    series = df[column_name]

    # Create histogram plot
    histogram = (ggplot(df, aes(x=series.name)) +
                 geom_histogram(binwidth=1, fill='skyblue', color='black') +
                 theme_minimal() +
                 labs(title=f'Histogram of {series.name}', x=series.name, y='Frequency'))

    # Save plot to in-memory buffer
    histogram_buffer = io.BytesIO()
    
    # Suppress plotnine warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        
        # Save plotnine plot to buffer
        histogram.save(histogram_buffer, format='png')
    
    # Load image from buffer
    histogram_buffer.seek(0)
    histogram_img = plt.imread(histogram_buffer)
    
    # Create a subplot
    fig, ax = plt.subplots(figsize=(7, 5))
    
    # Display histogram
    ax.imshow(histogram_img)
    ax.axis('off')  # Hide axes
    ax.set_title('Histogram')
    
    plt.tight_layout()
    
    return fig

# For the 'Age' column
fig = plot_numerical(train_df, 'Age')
plt.show()

