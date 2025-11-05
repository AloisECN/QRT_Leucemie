import pandas as pd
import re
from typing import Tuple


class ClinicalDataFeatures:
    """
    Refactored clinical features extractor as a class.
    Usage:
        cdf = ClinicalDataFeatures()
        df_out = cdf.process(df_in)
    """

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full feature extraction pipeline and return modified DataFrame."""
        if df is not None:
            self.df = df.copy()
        if self.df is None:
            raise ValueError("No DataFrame supplied to process.")
        self._extract_info()
        self._add_severity_features()
        return self.df

    def _extract_info(self) -> None:
        """Extract features from CYTOGENETICS column into new columns."""
        df = self.df
        cyto = df["CYTOGENETICS"].fillna("").astype(str)

        df["46_chromo"] = cyto.str.startswith("46").astype(int)

        df["has_deletion"] = cyto.str.contains("del").astype(int)
        df["has_translocation"] = cyto.str.contains(r"t\(").astype(int)
        df["has_inversion"] = cyto.str.contains("inv").astype(int)
        df["has_addition"] = cyto.str.contains("add").astype(int)

        df["has_chr7_abnormal"] = cyto.str.contains(r"-7|del\(7\)").astype(int)
        df["has_chr5_abnormal"] = cyto.str.contains(r"-5|del\(5\)").astype(int)
        df["has_trisomy8"] = cyto.str.contains(r"\+8").astype(int)
        df["has_monosomy7"] = cyto.str.contains(r"-7(?![0-9])").astype(int)
        df["has_del7q"] = cyto.str.contains(r"del\(7.*?q.*?\)").astype(int)

        df["total_abnormalities"] = (
            df["has_deletion"]
            + df["has_translocation"]
            + df["has_inversion"]
            + df["has_addition"]
            + df["has_chr7_abnormal"]
            + df["has_chr5_abnormal"]
            + df["has_trisomy8"]
            + df["has_monosomy7"]
            + df["has_del7q"]
        )

        df["has_high_risk_marker"] = (
            df["has_chr7_abnormal"] | df["has_chr5_abnormal"] | df["has_trisomy8"]
        ).astype(int)

        df["is_missing_cytogenetics"] = df["CYTOGENETICS"].isna().astype(int)

        self.df = df

    @staticmethod
    def _extract_cell_proportions(cytogenetics_str: str) -> Tuple[int, int]:
        """Return (abnormal_cell_count, total_cell_count) for a cytogenetics string."""
        if pd.isna(cytogenetics_str):
            return 0, 0

        populations = str(cytogenetics_str).split("/")
        total_cells = 0
        abnormal_cells = 0

        for pop in populations:
            cells = re.findall(r"\[(\d+)\]", pop)
            if cells:
                count = int(cells[0])
                total_cells += count
                if not pop.strip().startswith("46"):
                    abnormal_cells += count

        return (abnormal_cells, total_cells)

    def _add_severity_features(self) -> None:
        """Add severity features based on cytogenetics cell proportions."""
        df = self.df

        cell_counts = df["CYTOGENETICS"].apply(self._extract_cell_proportions)
        df["abnormal_cell_count"] = cell_counts.apply(lambda x: x[0])
        df["total_cell_count"] = cell_counts.apply(lambda x: x[1])

        df["abnormal_cell_proportion"] = df.apply(
            lambda row: (
                row["abnormal_cell_count"] / row["total_cell_count"]
                if row["total_cell_count"] > 0
                else 0
            ),
            axis=1,
        )

        df["abnormal_cell_fraction_bin"] = pd.cut(
            df["abnormal_cell_proportion"],
            bins=[-0.01, 0.01, 0.25, 0.75, 1.0],
            labels=[0, 1, 5, 10],
        )

        # clean up intermediate columns and remove CYTOGENETICS if desired
        df.drop(
            columns=["abnormal_cell_proportion", "abnormal_cell_count"], inplace=True
        )
        df.drop(columns=["CYTOGENETICS"], inplace=True)

        self.df = df
