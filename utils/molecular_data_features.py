import pandas as pd
"""
Function
"""

def classify_impact(effect):
    impact = 1
    if effect in high_impact:
        impact = 10
    if effect in moderate_impact:
        impact = 2
    return impact


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



hotspot_keywords = ["R882", "R132", "R140", "R172", "D835", "K700", "P95", "S34", "Q157", "W288", "R175", "R248"]


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
        #mean_VAF=("VAF", "mean"),
        max_DEPTH=("DEPTH", "max"),
        #mean_DEPTH=("DEPTH", "mean"),
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

