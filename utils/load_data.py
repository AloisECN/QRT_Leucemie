import pandas as pd


def load_target_dataset(path):

    """
    INPUT : 
        - path (string) -> path of the target dataset
    OUTPUT :
        - target dataset
    
    """

    target_df = pd.read_csv(path)

    # Drop rows where 'OS_YEARS' is NaN if conversion caused any issues
    target_df.dropna(subset=['OS_YEARS', 'OS_STATUS'], inplace=True)

    # Convert target_df 'OS_YEARS' to numeric if it isn’t already
    target_df['OS_YEARS'] = pd.to_numeric(target_df['OS_YEARS'], errors='coerce')

    # Ensure 'OS_STATUS' is boolean
    #target_df['OS_STATUS'] = target_df['OS_STATUS'].astype(bool)

    #we swap os_status and os_years
    #target_df['OS_STATUS'], target_df['OS_YEARS'] = target_df['OS_YEARS'], target_df['OS_STATUS']

    return target_df