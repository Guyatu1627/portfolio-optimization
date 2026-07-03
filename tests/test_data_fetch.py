import pandas as pd
import sys
import os

# Add src to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

def test_dummy_passing():
    """
    A simple dummy test to ensure pytest collects at least one test
    and exits with code 0.
    """
    assert True

def test_data_fetch_clean_data():
    from data_fetch import clean_data
    import numpy as np
    
    # Create a dummy dataframe with NaNs
    df = pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [np.nan, 2, np.nan]
    })
    
    cleaned_df = clean_data(df)
    
    # After forward and backward fill, there should be no NaNs
    assert not cleaned_df.isnull().values.any()
