import pandas as pd

high_impact = [
    "stop_gained", "frameshift_variant", "initiator_codon_change", 
    "stop_lost", "splice_site_variant", "ITD", "PTD"
]
moderate_impact = [
    "non_synonymous_codon", "inframe_codon_loss", "inframe_codon_gain", 
    "inframe_variant", "complex_change_in_transcript"
]
low_impact = [
    "synonymous_codon", "stop_retained_variant", 
    "3_prime_UTR_variant", "2KB_upstream_variant"
]


def process_molecular_data_effect(df):
    """
    Process molecular data to create features based on gene mutations.

    Args:
        df_mol (pd.DataFrame): Molecular data DataFrame.

    Returns:
        pd.DataFrame: DataFrame with aggregated gene mutation features.
    """

    # Group by 'ID' and 'GENE', then count the occurrences of each gene for each ID
    gene_counts = df.groupby(['ID', 'GENE']).size().unstack(fill_value=0)

    # Add a total mutation count for each ID
    gene_counts['total_mutations'] = gene_counts.sum(axis=1)
    effect_scores = df.groupby("ID")["EFFECT_LEVEL"].sum().reset_index()

    gene_counts = gene_counts.reset_index().merge(effect_scores, on="ID").set_index("ID")

    return gene_counts


def process_molecular_data(df):
    """
    Process molecular data to create features based on gene mutations.

    Args:
        df_mol (pd.DataFrame): Molecular data DataFrame.

    Returns:
        pd.DataFrame: DataFrame with aggregated gene mutation features.
    """

    # Group by 'ID' and 'GENE', then count the occurrences of each gene for each ID
    gene_counts = df.groupby(['ID', 'GENE']).size().unstack(fill_value=0)

    # Add a total mutation count for each ID
    gene_counts['total_mutations'] = gene_counts.sum(axis=1)

    return gene_counts


def classify_impact(effect):
    impact = 0
    if effect in high_impact:
        impact = 3
    if effect in moderate_impact:
        impact = 1
    return impact