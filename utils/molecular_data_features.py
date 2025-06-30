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

    # Create an impact score based on the effect of the mutation
    effect_scores = df.groupby("ID")["EFFECT_LEVEL"].sum().reset_index()
    gene_counts = gene_counts.reset_index().merge(effect_scores, on="ID").set_index("ID")

    # Compact numerical values
    max_VAF = df.groupby("ID")["VAF"].max().reset_index()
    gene_counts = gene_counts.reset_index().merge(max_VAF, on="ID").set_index("ID")

    mean_VAF = df.groupby("ID")["VAF"].mean().reset_index()
    gene_counts = gene_counts.reset_index().merge(mean_VAF, on="ID").set_index("ID")

    max_DEPTH = df.groupby("ID")["DEPTH"].max().reset_index()
    gene_counts = gene_counts.reset_index().merge(max_DEPTH, on="ID").set_index("ID")

    mean_DEPTH = df.groupby("ID")["DEPTH"].mean().reset_index()
    gene_counts = gene_counts.reset_index().merge(mean_DEPTH, on="ID").set_index("ID")

    # is high risk
    df["HIGH_RISK"] = df.apply(lambda x: is_high_risk(x["REF"], x["ALT"], x["PROTEIN_CHANGE"]), axis=1)
    high_risk_counts = df.groupby("ID")["HIGH_RISK"].sum().reset_index()
    gene_counts = gene_counts.reset_index().merge(high_risk_counts, on="ID", how="left").set_index("ID")

    start = df.groupby("ID")["START"].sum().reset_index()
    end = df.groupby("ID")["END"].sum().reset_index()
    diff = {"ID" : end["ID"], "DIFF" : end["END"] - start["START"]}
    diff = pd.DataFrame(diff)
    gene_counts = gene_counts.reset_index().merge(diff, on="ID").set_index("ID")

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

def is_high_risk(ref, alt, protein_change):
    # A simple heuristic to determine if a mutation is high risk based on reference and alternate alleles
    risky_pairs = {("C", "T"), ("G", "A"), ("A", "T"), ("T", "A")}
    risky_protein_signatures = ["*", "fs", "R132", "R882", "D835"]
    

    ref_alt_risky = (ref, alt) in risky_pairs
    if protein_change is None or pd.isna(protein_change):
        prot_risky = False
    prot_risky =  any(risk in str(protein_change) for risk in risky_protein_signatures) #For NaN

    # We will use PolyPhen-2 in the future
    return int(ref_alt_risky and prot_risky)


def classify_impact(effect):
    impact = 1
    if effect in high_impact:
        impact = 5
    if effect in moderate_impact:
        impact = 2
    return impact