# Dataset Download and Preparation

The benchmark uses three paired RNA–protein datasets:

| Benchmark name | GEO accession | Assay | Expected cells | Expected genes | Expected proteins |
|---|---|---:|---:|---:|---:|
| CBMC | GSE100866 | CITE-seq | 8,617 | 36,280 | 13 |
| ECCITE | GSE164378 | ECCITE-seq 5′ | 49,147 | 33,538 | 54 |
| GSE3P | GSE164378 | CITE-seq 3′ | 161,764 | 33,538 | 228 |

The model reads raw UMI counts from `.h5ad` files. RNA counts are stored in
`adata.X`, and protein counts are stored in
`adata.obsm["protein_expression"]`. Normalization and gene screening are
performed by the experiment code within each fold.

## Directory layout

```text
datasets/
├── README.md
├── download_geo.sh
├── prepare_h5ad.py
├── raw/
│   ├── cbmc/
│   ├── eccite_5p/
│   └── gse3p_3p/
└── processed/
    ├── cbmc.h5ad
    ├── eccite_5p.h5ad
    └── gse3p_3p.h5ad
```

Downloaded matrices and generated `.h5ad` files are excluded from the release
archive by `.gitignore`.

## Automated download

The download script retrieves the processed count matrices from the official
NCBI GEO supplementary-file service.

Download every dataset:

```bash
bash datasets/download_geo.sh all
```

Download one dataset:

```bash
bash datasets/download_geo.sh cbmc
bash datasets/download_geo.sh eccite
bash datasets/download_geo.sh gse3p
```

The downloads require approximately 1.5 GB of compressed storage. The script
keeps existing files and resumes interrupted transfers when the local `curl`
version supports continuation.

## Manual download

### CBMC — GSE100866

GEO record:

```text
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE100866
```

Required supplementary files:

```text
GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz
GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv.gz
```

Place both files in `datasets/raw/cbmc/`.

### ECCITE-5′ — GSE164378

GEO series record:

```text
https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE164378
```

RNA sample `GSM5008740`:

```text
GSM5008740_RNA_5P-barcodes.tsv.gz
GSM5008740_RNA_5P-features.tsv.gz
GSM5008740_RNA_5P-matrix.mtx.gz
```

Protein sample `GSM5008741`:

```text
GSM5008741_ADT_5P-barcodes.tsv.gz
GSM5008741_ADT_5P-features.tsv.gz
GSM5008741_ADT_5P-matrix.mtx.gz
```

Place the six files in `datasets/raw/eccite_5p/`.

### GSE3P-3′ — GSE164378

RNA sample `GSM5008737`:

```text
GSM5008737_RNA_3P-barcodes.tsv.gz
GSM5008737_RNA_3P-features.tsv.gz
GSM5008737_RNA_3P-matrix.mtx.gz
```

Protein sample `GSM5008738`:

```text
GSM5008738_ADT_3P-barcodes.tsv.gz
GSM5008738_ADT_3P-features.tsv.gz
GSM5008738_ADT_3P-matrix.mtx.gz
```

Place the six files in `datasets/raw/gse3p_3p/`.

## Build the `.h5ad` inputs

After downloading the matrices, run:

```bash
python datasets/prepare_h5ad.py \
  --dataset cbmc \
  --raw-dir datasets/raw/cbmc \
  --output datasets/processed/cbmc.h5ad

python datasets/prepare_h5ad.py \
  --dataset eccite \
  --raw-dir datasets/raw/eccite_5p \
  --output datasets/processed/eccite_5p.h5ad

python datasets/prepare_h5ad.py \
  --dataset gse3p \
  --raw-dir datasets/raw/gse3p_3p \
  --output datasets/processed/gse3p_3p.h5ad
```

The converter aligns RNA and protein matrices by cell barcode, retains raw UMI
counts, records protein names, and checks the dataset dimensions reported in the
paper.

## Run the benchmark

```bash
python experiments/run_5fold.py \
  --data datasets/processed/cbmc.h5ad \
  --dataset-name CBMC \
  --output outputs/cbmc \
  --device cuda
```

Use `ECCITE` with `datasets/processed/eccite_5p.h5ad` and `GSE3P` with
`datasets/processed/gse3p_3p.h5ad` for the other benchmarks.
