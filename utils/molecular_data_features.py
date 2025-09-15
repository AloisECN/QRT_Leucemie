import pandas as pd
"""
Hyperparameters
"""

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


def process_molecular_data_effect_bis(df):
    """
    Process molecular data to create features based on gene mutations.

    Args:
        df_mol (pd.DataFrame): Molecular data DataFrame.

    Returns:
        pd.DataFrame: DataFrame with aggregated gene mutation features.
    """

    # Group by 'ID' and 'GENE', then count the occurrences of each gene for each ID
    #gene_counts = df.groupby(['ID', 'GENE']).size().unstack(fill_value=0)
    # Add a total mutation count for each ID
    #gene_counts['total_mutations'] = gene_counts.sum(axis=1)
    gene_counts = df.groupby("ID")["GENE"].nunique().reset_index(name="total_mutations")

    # Create an impact score based on the effect of the mutation
    effect_scores = df.groupby("ID")["EFFECT_LEVEL"].sum().reset_index()
    gene_counts = gene_counts.reset_index().merge(effect_scores, on="ID").set_index("ID")

    # Compact numerical values
    max_VAF = df.groupby("ID")["VAF"].max().reset_index()
    max_VAF.rename(columns = {"VAF": "max_VAF"})
    gene_counts = gene_counts.reset_index().merge(max_VAF, on="ID").set_index("ID")

    mean_VAF = df.groupby("ID")["VAF"].mean().reset_index()
    mean_VAF.rename(columns = {"VAF": "mean_VAF"})
    gene_counts = gene_counts.reset_index().merge(mean_VAF, on="ID").set_index("ID")

    max_DEPTH = df.groupby("ID")["DEPTH"].max().reset_index()
    max_DEPTH .rename(columns = {"DEPTH": "max_DEPTH"})
    gene_counts = gene_counts.reset_index().merge(max_DEPTH, on="ID").set_index("ID")

    mean_DEPTH = df.groupby("ID")["DEPTH"].mean().reset_index()
    mean_DEPTH .rename(columns = {"DEPTH": "mean_DEPTH"})
    gene_counts = gene_counts.reset_index().merge(mean_DEPTH, on="ID").set_index("ID")

    # is high risk
    df["HIGH_RISK"] = df.apply(lambda x: is_high_risk(x["REF"], x["ALT"], x["PROTEIN_CHANGE"]), axis=1)
    high_risk_counts = df.groupby("ID")["HIGH_RISK"].sum().reset_index()
    gene_counts = gene_counts.reset_index().merge(high_risk_counts, on="ID", how="left").set_index("ID")

    #Has hotspots 
    df["HOTSPOTS"] = df.apply(lambda x: has_hotspot_mutation(x["PROTEIN_CHANGE"]), axis = 1)
    hostspots_counts = df.groupby("ID")["HOTSPOTS"].sum().reset_index()
    gene_counts = gene_counts.reset_index().merge(hostspots_counts, on="ID", how="left").set_index("ID")


    start = df.groupby("ID")["START"].sum().reset_index()
    end = df.groupby("ID")["END"].sum().reset_index()
    diff = {"ID" : end["ID"], "DIFF" : end["END"] - start["START"]}
    diff = pd.DataFrame(diff)
    gene_counts = gene_counts.reset_index().merge(diff, on="ID").set_index("ID")

    return gene_counts

def process_molecular_data_effect(df):
    """
    Process molecular data to create features based on gene mutations,
    including HIGH_RISK and HOTSPOTS using user-defined functions.
    """

    genes_of_interest = ["TP53", "NPM1", "CEBPA", "FLT3", "ASXL1", "DNMT3A", "TET2", "IDH1", "IDH2", "RUNX1"]

    # Appliquer les fonctions personnalisées
    df["HIGH_RISK"] = df.apply(lambda x: is_high_risk(x["REF"], x["ALT"], x["PROTEIN_CHANGE"]), axis=1)
    df["HOTSPOTS"] = df.apply(lambda x: has_hotspot_mutation(x["PROTEIN_CHANGE"]), axis=1)

    # Agrégation par patient (ID)
    gene_counts = df.groupby("ID").agg(
        total_mutations=("GENE", "nunique"),        # nombre de gènes altérés distincts
        effect_score=("EFFECT_LEVEL", "sum"),
        max_VAF=("VAF", "max"),
        mean_VAF=("VAF", "mean"),
        max_DEPTH=("DEPTH", "max"),
        mean_DEPTH=("DEPTH", "mean"),
        HIGH_RISK=("HIGH_RISK", "sum"),             # somme des mutations high-risk
        HOTSPOTS=("HOTSPOTS", "sum"),               # somme des hotspots
        start_sum=("START", "sum"),
        end_sum=("END", "sum")
    )

    # Calcul de la différence entre start et end
    gene_counts["DIFF"] = gene_counts["end_sum"] - gene_counts["start_sum"]

    # Supprimer les colonnes intermédiaires si pas nécessaires
    gene_counts = gene_counts.drop(columns=["start_sum", "end_sum"])

    gene_indicators = pd.crosstab(df["ID"], df["GENE"])
    # Garder seulement ceux qui nous intéressent
    gene_indicators = gene_indicators.reindex(columns=genes_of_interest, fill_value=0)
    gene_counts = gene_counts.merge(gene_indicators, on="ID", how="left")

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

hotspot_keywords = ["R882", "R132", "R140", "R172", "D835", "K700", "P95", "S34", "Q157", "W288", "R175", "R248"]

"""
Helpers for main function below
"""

def has_hotspot_mutation(protein_change):
    return int(any(hot in str(protein_change) for hot in hotspot_keywords))


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
        impact = 10
    if effect in moderate_impact:
        impact = 2
    return impact


def process_molecular_data_effect(df):
    """
    Process molecular data to create features based on gene mutations.

    Args:
        df (pd.DataFrame): Molecular data DataFrame.

    Returns:
        pd.DataFrame: DataFrame with aggregated gene mutation features.
    """
    
    # Group by 'ID' and count the total number of mutations for each ID
    gene_counts = df.groupby('ID').size().to_frame('total_mutations').reset_index()

    # Create an impact score based on the effect of the mutation
    df["IMPACT"] = df.apply(lambda x: classify_impact(x["EFFECT"]), axis=1)
    impact_sum = df.groupby("ID")["IMPACT"].sum().reset_index()
    gene_counts = gene_counts.merge(impact_sum, on="ID")

    # Compact numerical values
    max_VAF = df.groupby("ID")["VAF"].max().reset_index().rename(columns={"VAF": "max_VAF"})
    gene_counts = gene_counts.merge(max_VAF, on="ID")

    mean_VAF = df.groupby("ID")["VAF"].mean().reset_index().rename(columns={"VAF": "mean_VAF"})
    gene_counts = gene_counts.merge(mean_VAF, on="ID")

    max_DEPTH = df.groupby("ID")["DEPTH"].max().reset_index().rename(columns={"DEPTH": "max_DEPTH"})
    gene_counts = gene_counts.merge(max_DEPTH, on="ID")

    mean_DEPTH = df.groupby("ID")["DEPTH"].mean().reset_index().rename(columns={"DEPTH": "mean_DEPTH"})
    gene_counts = gene_counts.merge(mean_DEPTH, on="ID")

    # is high risk
    df["HIGH_RISK"] = df.apply(lambda x: is_high_risk(x["REF"], x["ALT"], x["PROTEIN_CHANGE"]), axis=1)
    high_risk_counts = df.groupby("ID")["HIGH_RISK"].sum().reset_index()
    gene_counts = gene_counts.merge(high_risk_counts, on="ID", how="left")

    # Has hotspots 
    df["HOTSPOTS"] = df.apply(lambda x: has_hotspot_mutation(x["PROTEIN_CHANGE"]), axis=1)
    hotspots_counts = df.groupby("ID")["HOTSPOTS"].sum().reset_index()
    gene_counts = gene_counts.merge(hotspots_counts, on="ID", how="left")

    # Difference between END and START positions
    start = df.groupby("ID")["START"].sum().reset_index()
    end = df.groupby("ID")["END"].sum().reset_index()
    diff = pd.DataFrame({"ID": end["ID"], "DIFF": end["END"] - start["START"]})
    gene_counts = gene_counts.merge(diff, on="ID")

    gene_counts = gene_counts.set_index("ID")

    return gene_counts