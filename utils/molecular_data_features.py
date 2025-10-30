import pandas as pd

class MolecularDataFeature:
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
    genes_of_interest = ["TP53", "NPM1", "CEBPA", "FLT3", "ASXL1", "DNMT3A", "TET2", "IDH1", "IDH2", "RUNX1"]

    def classify_impact(self, effect):
        impact = 1
        if effect in self.high_impact:
            impact = 10
        if effect in self.moderate_impact:
            impact = 2
        return impact

    def has_hotspot_mutation(self, protein_change):
        return int(any(hot in str(protein_change) for hot in self.hotspot_keywords))

    def is_high_risk(self, ref, alt, protein_change):
        risky_pairs = {("C", "T"), ("G", "A"), ("A", "T"), ("T", "A")}
        risky_protein_signatures = ["*", "fs", "R132", "R882", "D835"]
        ref_alt_risky = (ref, alt) in risky_pairs
        if protein_change is None or pd.isna(protein_change):
            prot_risky = False
        else:
            prot_risky = any(risk in str(protein_change) for risk in risky_protein_signatures)
        return int(ref_alt_risky and prot_risky)

    def process(self, df):
        df = df.copy()
        df["HIGH_RISK"] = df.apply(lambda x: self.is_high_risk(x["REF"], x["ALT"], x["PROTEIN_CHANGE"]), axis=1)
        df["HOTSPOTS"] = df.apply(lambda x: self.has_hotspot_mutation(x["PROTEIN_CHANGE"]), axis=1)
        df["EFFECT_LEVEL"] = df.apply(lambda x: self.classify_impact(x["EFFECT"]), axis=1)

        gene_counts = df.groupby("ID").agg(
            total_mutations=("GENE", "nunique"),
            effect_score=("EFFECT_LEVEL", "sum"),
            max_VAF=("VAF", "max"),
            max_DEPTH=("DEPTH", "max"),
            HIGH_RISK=("HIGH_RISK", "sum"),
            HOTSPOTS=("HOTSPOTS", "sum"),
            start_sum=("START", "sum"),
            end_sum=("END", "sum")
        )

        gene_counts["DIFF"] = gene_counts["end_sum"] - gene_counts["start_sum"]
        gene_counts = gene_counts.drop(columns=["start_sum", "end_sum"])

        gene_indicators = pd.crosstab(df["ID"], df["GENE"])
        gene_indicators = gene_indicators.reindex(columns=self.genes_of_interest, fill_value=0)
        gene_counts = gene_counts.merge(gene_indicators, on="ID", how="left")

        return gene_counts
