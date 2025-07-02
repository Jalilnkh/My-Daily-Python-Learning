import pandas as pd


def convert_tsv_to_csv(tsv_filename="train.tsv", columns_name=["sentence"]):
    """
    Convert a TSV file to CSV format.
    Assumes the TSV file is named 'train.tsv' and contains a column named 'sentence'.
    The output will be saved as 'sentences.csv'.
    """
    # Read the TSV file
    df = pd.read_csv(tsv_filename, sep="\t")
    #breakpoint()

    # Select the column you want, e.g., "sentence"
    selected = df[columns_name]

    return selected
