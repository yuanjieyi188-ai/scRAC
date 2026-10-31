#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="${SCRIPT_DIR}/raw"
TARGET="${1:-all}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required." >&2
  exit 1
fi

download() {
  local url="$1"
  local destination="$2"
  mkdir -p "$(dirname "$destination")"
  if [[ -s "$destination" ]]; then
    printf 'exists  %s\n' "$destination"
    return
  fi
  printf 'fetch   %s\n' "$destination"
  curl --fail --location --retry 3 --retry-delay 2 --continue-at - \
    --output "$destination" "$url"
}

download_cbmc() {
  local base="https://ftp.ncbi.nlm.nih.gov/geo/series/GSE100nnn/GSE100866/suppl"
  local out="${RAW_DIR}/cbmc"
  download "${base}/GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz" \
    "${out}/GSE100866_CBMC_8K_13AB_10X-RNA_umi.csv.gz"
  download "${base}/GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv.gz" \
    "${out}/GSE100866_CBMC_8K_13AB_10X-ADT_umi.csv.gz"
}

download_eccite() {
  local rna_base="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5008nnn/GSM5008740/suppl"
  local adt_base="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5008nnn/GSM5008741/suppl"
  local out="${RAW_DIR}/eccite_5p"
  for suffix in barcodes.tsv.gz features.tsv.gz matrix.mtx.gz; do
    download "${rna_base}/GSM5008740_RNA_5P-${suffix}" \
      "${out}/GSM5008740_RNA_5P-${suffix}"
    download "${adt_base}/GSM5008741_ADT_5P-${suffix}" \
      "${out}/GSM5008741_ADT_5P-${suffix}"
  done
}

download_gse3p() {
  local rna_base="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5008nnn/GSM5008737/suppl"
  local adt_base="https://ftp.ncbi.nlm.nih.gov/geo/samples/GSM5008nnn/GSM5008738/suppl"
  local out="${RAW_DIR}/gse3p_3p"
  for suffix in barcodes.tsv.gz features.tsv.gz matrix.mtx.gz; do
    download "${rna_base}/GSM5008737_RNA_3P-${suffix}" \
      "${out}/GSM5008737_RNA_3P-${suffix}"
    download "${adt_base}/GSM5008738_ADT_3P-${suffix}" \
      "${out}/GSM5008738_ADT_3P-${suffix}"
  done
}

case "$TARGET" in
  all)
    download_cbmc
    download_eccite
    download_gse3p
    ;;
  cbmc)
    download_cbmc
    ;;
  eccite)
    download_eccite
    ;;
  gse3p)
    download_gse3p
    ;;
  *)
    echo "usage: bash datasets/download_geo.sh [all|cbmc|eccite|gse3p]" >&2
    exit 2
    ;;
esac
