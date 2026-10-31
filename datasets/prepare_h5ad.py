from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.io import mmread


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    accession: str
    cells: int
    genes: int
    proteins: int


SPECS = {
    "cbmc": DatasetSpec("CBMC", "GSE100866", 8617, 36280, 13),
    "eccite": DatasetSpec("ECCITE-5P", "GSE164378", 49147, 33538, 54),
    "gse3p": DatasetSpec("GSE3P-3P", "GSE164378", 161764, 33538, 228),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert the GEO RNA and ADT count matrices to scRAGFlow H5AD input."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(SPECS))
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def require_files(raw_dir: Path, names: list[str]) -> list[Path]:
    paths = [raw_dir / name for name in names]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        joined = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing required files:\n{joined}")
    return paths


def make_unique(values: list[str]) -> list[str]:
    counts: dict[str, int] = {}
    result: list[str] = []
    for value in values:
        count = counts.get(value, 0)
        result.append(value if count == 0 else f"{value}-{count}")
        counts[value] = count + 1
    return result


def read_feature_names(path: Path) -> list[str]:
    frame = pd.read_csv(path, sep="\t", header=None, compression="infer")
    column = 1 if frame.shape[1] > 1 else 0
    return make_unique(frame.iloc[:, column].astype(str).tolist())


def read_barcodes(path: Path) -> list[str]:
    frame = pd.read_csv(path, sep="\t", header=None, compression="infer")
    return frame.iloc[:, 0].astype(str).tolist()


def read_10x_triplet(
    matrix_path: Path,
    features_path: Path,
    barcodes_path: Path,
) -> tuple[sp.csr_matrix, list[str], list[str]]:
    matrix = mmread(matrix_path).tocsr().astype(np.float32).transpose().tocsr()
    features = read_feature_names(features_path)
    barcodes = read_barcodes(barcodes_path)
    if matrix.shape != (len(barcodes), len(features)):
        raise ValueError(
            f"Matrix shape {matrix.shape} does not match "
            f"{len(barcodes)} barcodes and {len(features)} features."
        )
    if matrix.data.size and float(matrix.data.min()) < 0:
        raise ValueError("Count matrices must be nonnegative.")
    return matrix, features, barcodes


def align_modalities(
    rna: sp.csr_matrix,
    rna_barcodes: list[str],
    protein: sp.csr_matrix,
    protein_barcodes: list[str],
) -> tuple[sp.csr_matrix, sp.csr_matrix, list[str]]:
    protein_lookup = {barcode: index for index, barcode in enumerate(protein_barcodes)}
    rna_indices: list[int] = []
    protein_indices: list[int] = []
    common: list[str] = []
    for rna_index, barcode in enumerate(rna_barcodes):
        protein_index = protein_lookup.get(barcode)
        if protein_index is not None:
            rna_indices.append(rna_index)
            protein_indices.append(protein_index)
            common.append(barcode)
    if not common:
        raise ValueError("RNA and protein matrices have no shared cell barcodes.")
    return rna[rna_indices], protein[protein_indices], common


def load_cbmc(raw_dir: Path) -> tuple[sp.csr_matrix, list[str], np.ndarray, list[str], list[str]]:
    rna_path, protein_path = require_files(
        raw_dir,
        [
            "GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz",
            "GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv.gz",
        ],
    )
    rna_frame = pd.read_csv(rna_path, index_col=0, compression="infer")
    protein_frame = pd.read_csv(protein_path, index_col=0, compression="infer")
    common = [barcode for barcode in rna_frame.columns if barcode in protein_frame.columns]
    if not common:
        raise ValueError("RNA and protein CSV files have no shared cell barcodes.")
    rna = sp.csr_matrix(rna_frame.loc[:, common].to_numpy(dtype=np.float32).T)
    protein = protein_frame.loc[:, common].to_numpy(dtype=np.float32).T
    if rna.data.size and float(rna.data.min()) < 0:
        raise ValueError("RNA count matrix must be nonnegative.")
    if protein.size and float(protein.min()) < 0:
        raise ValueError("Protein count matrix must be nonnegative.")
    return (
        rna,
        make_unique(rna_frame.index.astype(str).tolist()),
        protein,
        make_unique(protein_frame.index.astype(str).tolist()),
        common,
    )


def load_gse164378(
    raw_dir: Path,
    rna_prefix: str,
    protein_prefix: str,
) -> tuple[sp.csr_matrix, list[str], np.ndarray, list[str], list[str]]:
    names = [
        f"{rna_prefix}-matrix.mtx.gz",
        f"{rna_prefix}-features.tsv.gz",
        f"{rna_prefix}-barcodes.tsv.gz",
        f"{protein_prefix}-matrix.mtx.gz",
        f"{protein_prefix}-features.tsv.gz",
        f"{protein_prefix}-barcodes.tsv.gz",
    ]
    (
        rna_matrix,
        rna_features,
        rna_barcodes,
        protein_matrix,
        protein_features,
        protein_barcodes,
    ) = require_files(raw_dir, names)
    rna, genes, rna_cells = read_10x_triplet(
        rna_matrix, rna_features, rna_barcodes
    )
    protein, proteins, protein_cells = read_10x_triplet(
        protein_matrix, protein_features, protein_barcodes
    )
    rna, protein, cells = align_modalities(rna, rna_cells, protein, protein_cells)
    return rna, genes, protein.toarray().astype(np.float32), proteins, cells


def validate_dimensions(
    spec: DatasetSpec,
    rna: sp.csr_matrix,
    protein: np.ndarray,
) -> None:
    observed = (rna.shape[0], rna.shape[1], protein.shape[1])
    expected = (spec.cells, spec.genes, spec.proteins)
    if observed != expected:
        raise ValueError(
            f"{spec.label} dimensions are {observed}; the paper configuration expects {expected}."
        )


def write_h5ad(
    output: Path,
    spec: DatasetSpec,
    rna: sp.csr_matrix,
    genes: list[str],
    protein: np.ndarray,
    proteins: list[str],
    cells: list[str],
) -> None:
    import anndata as ad

    validate_dimensions(spec, rna, protein)
    output.parent.mkdir(parents=True, exist_ok=True)
    obs = pd.DataFrame(index=pd.Index(cells, name="cell_barcode"))
    var = pd.DataFrame(index=pd.Index(genes, name="gene"))
    adata = ad.AnnData(X=rna, obs=obs, var=var)
    adata.obsm["protein_expression"] = pd.DataFrame(
        protein,
        index=adata.obs_names.copy(),
        columns=proteins,
    )
    adata.uns["dataset_name"] = spec.label
    adata.uns["geo_accession"] = spec.accession
    adata.uns["matrix_type"] = "raw_umi_counts"
    adata.write_h5ad(output, compression="gzip")
    print(
        f"wrote {output} with {adata.n_obs} cells, "
        f"{adata.n_vars} genes, and {protein.shape[1]} proteins"
    )


def main() -> None:
    args = parse_args()
    spec = SPECS[args.dataset]
    if args.dataset == "cbmc":
        data = load_cbmc(args.raw_dir)
    elif args.dataset == "eccite":
        data = load_gse164378(
            args.raw_dir,
            "GSM5008740_RNA_5P",
            "GSM5008741_ADT_5P",
        )
    else:
        data = load_gse164378(
            args.raw_dir,
            "GSM5008737_RNA_3P",
            "GSM5008738_ADT_3P",
        )
    write_h5ad(args.output, spec, *data)


if __name__ == "__main__":
    main()
