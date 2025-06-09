import pandas as pd
import re

def extract_cytogenetic_features(df):
    """Extract features from cytogenetics column"""
    
    # -- Initialize new columns --

    # check if no abnormalities in chromosomes
    df['is_normal'] = df['CYTOGENETICS'].fillna('').astype(str).str.startswith('46').astype(int)
    # check if deletion in chromosomes
    df['has_deletion'] = df['CYTOGENETICS'].fillna('').astype(str).str.contains('del').astype(int)
    # check if translocation in chromosomes
    df['has_translocation'] = df['CYTOGENETICS'].fillna('').astype(str).str.contains('t\(').astype(int)
    # check if inversion in chromosomes
    df['has_inversion'] = df['CYTOGENETICS'].fillna('').astype(str).str.contains('inv').astype(int)
    # check if affition in chromosomes
    df['has_addition'] = df['CYTOGENETICS'].fillna('').astype(str).str.contains('add').astype(int)
    
    # -- Check for specific chromosome abnormalities --
    df['has_chr7_abnormal'] = df['CYTOGENETICS'].fillna('').astype(str).str.contains('-7|del\(7\)').astype(int)
    df['has_chr5_abnormal'] = df['CYTOGENETICS'].fillna('').astype(str).str.contains('-5|del\(5\)').astype(int)
    df['has_trisomy8'] = df['CYTOGENETICS'].fillna('').astype(str).str.contains('\+8').astype(int)
    
    # Count total abnormalities
    df['total_abnormalities'] = (
        df['has_deletion'] + 
        df['has_translocation'] + 
        df['has_inversion'] + 
        df['has_addition'] +
        df['has_chr7_abnormal'] +
        df['has_chr5_abnormal'] +
        df['has_trisomy8']
    )
    
    return df



def extract_cell_proportions(cytogenetics_str):
    """Extract the proportion of abnormal cells from cytogenetics string"""
    if pd.isna(cytogenetics_str):
        return 0, 0
    
    # Split different cell populations
    populations = str(cytogenetics_str).split('/')
    total_cells = 0
    abnormal_cells = 0
    
    for pop in populations:
        # Extract number of cells in square brackets
        cells = re.findall(r'\[(\d+)\]', pop)
        if cells:
            count = int(cells[0])
            total_cells += count
            # If not starting with 46, it's abnormal
            if not pop.strip().startswith('46'):
                abnormal_cells += count
    
    if total_cells == 0:
        return 0, 0
        
    return abnormal_cells, total_cells



def add_severity_features(df):
    """Add severity features based on cell proportions"""
    
    # Extract cell counts
    cell_counts = df['CYTOGENETICS'].apply(extract_cell_proportions)
    
    # Calculate proportion of abnormal cells
    df['abnormal_cell_count'] = cell_counts.apply(lambda x: x[0])
    df['total_cell_count'] = cell_counts.apply(lambda x: x[1])
    df['abnormal_cell_proportion'] = df.apply(
        lambda row: row['abnormal_cell_count'] / row['total_cell_count'] 
        if row['total_cell_count'] > 0 else 0, 
        axis=1
    )
    
    return df