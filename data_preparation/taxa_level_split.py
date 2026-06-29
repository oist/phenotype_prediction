import os
import sys
import random
import logging
import argparse
import numpy as np
import polars as pl

import itertools
from collections import defaultdict

"""
This script splits the input data into train and test with respect to the desired taxonomy group (e.g. phylum, class, family).

Inputs:
    - gtdb metadata files (stored in ~/gene-context/phenotype_prediction/data_preparation/gtdb_files/),
    - input_annotation_csv: input csv with genome names and annotations,
    - input_data_csv: input csv with COG counts,
    - tax_level: desired taxonomy level for splitting OR "random" for random train/test split that doesn't take into account any taxonomy;
    - output_dir: desired output directory

Outputs:
    - txt file with taxonomy groups for training,
    - txt file with taxonomy groups for testing;
    - test/train csv's with annotations,
    - test/train csv's with COG counts,
    - test/train csv's with the corresponding taxa group names.

How to run this script?
    cd ~/gene-context/phenotype_prediction
    python3 -m data_preparation.taxa_level_split  --tax_level [tax_level] --input_annotation_csv [input_annotation_csv] --input_data_csv [input_data_csv] --output_dir [output_dir] --train_val_test_split_flag 

E.g.
    python3 -m data_preparation.taxa_level_split --tax_level phylum --input_annotation_csv data_diderm/gold_standard1.tsv --input_data_csv data_diderm/all_gene_annotations.tsv --output_dir data_diderm/input_data 
    python3 -m data_preparation.taxa_level_split --tax_level phylum --input_annotation_csv data_host/all_annotations.csv --input_data_csv data_host/all_genes.csv --output_dir data_host/input_data 
    python3 -m data_preparation.taxa_level_split --tax_level phylum --input_annotation_csv data_ogt/ogt_annot.csv --input_data_csv data_ogt/kegg.csv --output_dir data_ogt/input_data 
    python3 -m data_preparation.taxa_level_split  --tax_level phylum --input_annotation_csv data_aerob/bacdive_scrape_20230315.json.parsed.anaerobe_vs_aerobe.with_cyanos.csv --input_data_csv data_aerob/all_gene_annotations.tsv --output_dir data_aerob/input_data 

"""

BAC_TSV = 'data_preparation/gtdb_files/bac120_metadata_r202.tsv'
ARC_TSV = 'data_preparation/gtdb_files/ar122_metadata_r202.tsv'
TEST_DATA_SIZE = 0.2  # for 80/20 or 60/20/20
VAL_DATA_SIZE = 0.2   # only used if --train_val_test_split_flag is True
RANDOM_SEED = 42
NUM_SPLITS = 30

logging.basicConfig(level=logging.ERROR, format='%(levelname)s: %(message)s')


def process_args():
    parser = argparse.ArgumentParser(description="Split data into train/test or train/val/test with respect to taxonomy.")
    parser.add_argument("--tax_level", type=str, required=True, help="Taxonomic level for splitting (or 'random').")
    parser.add_argument('-annot', '--input_annotation_csv', required=True, type=str, help="Input CSV with genome annotations.")
    parser.add_argument('-data', '--input_data_csv', required=True, type=str, help="Input CSV with COG counts.")
    parser.add_argument('--output_dir', required=True, type=str, help="Output directory for split files.")
    parser.add_argument('--train_val_test_split_flag', action='store_true',
                        help="If set, perform 60/20/20 train/val/test split instead of 80/20 train/test.")  ### NEW
    return parser.parse_args()


def save_selected_data_and_annot(df, groups, tax_level, filename_data, filename_annot, filename_taxa):
    df_filter = df.filter(pl.col(tax_level).is_in(groups))
    df_filter = df_filter.unique(subset=["accession"], keep="first")
    print(f"The set has {len(df_filter)} rows")

    df_filter.select(["accession", "annotation"]).write_csv(filename_annot, separator="\t")
    df_filter.select(["accession", tax_level]).write_csv(filename_taxa, separator="\t")

    numeric_cols = [c for c in df_filter.columns if c not in ["annotation", tax_level, "accession"]]
    df_filter.select(["accession"] + numeric_cols).write_csv(filename_data, separator="\t")



if __name__ == '__main__':
    args = process_args()
    tax_level = args.tax_level
    random.seed(RANDOM_SEED)

    tax_levels = {"domain": 0, "phylum": 1, "class": 2, "order": 3, "family": 4, "genus": 5, "species": 6}

    out_dir = f"{args.output_dir}/{tax_level}"
    os.makedirs(out_dir, exist_ok=True)

    if tax_level not in tax_levels.keys() and tax_level != "random":
        logging.error(f"Invalid tax_level '{tax_level}'. Choose one of {list(tax_levels.keys())} or 'random'.")
        sys.exit(1)

    if tax_level in tax_levels.keys():
        gtdb_df = pl.concat([
            pl.read_csv(BAC_TSV, separator="\t"),
            pl.read_csv(ARC_TSV, separator="\t")
        ])
        gtdb_df = gtdb_df.with_columns(
            pl.col("gtdb_taxonomy").str.split(';').list.get(tax_levels[tax_level]).alias(tax_level)
        )
        gtdb_df = gtdb_df[['accession', tax_level]]

        input_df_annot = pl.read_csv(args.input_annotation_csv, separator="\t") #<------here
        old_name = input_df_annot.columns[-1]
        input_df_annot = input_df_annot.rename({old_name: "annotation"})

        input_df_counts = pl.read_csv(args.input_data_csv, separator="\t")  #<------here
        print(f"Reading input count table with {len(input_df_counts)} rows...")

        joined_df = input_df_counts.join(gtdb_df.join(input_df_annot, on="accession", how="left"), on="accession", how="left")
        joined_df = joined_df.filter(pl.col("annotation").is_not_null() & pl.col(tax_level).is_not_null())

        all_groups = list(set(joined_df[tax_level].to_list()))
        group_to_size = {g: len(joined_df.filter(pl.col(tax_level) == g)) for g in all_groups}
        total_samples = len(joined_df)

        for split_idx in range(NUM_SPLITS):
            shuffled = all_groups[:]
            random.shuffle(shuffled)

            if args.train_val_test_split_flag:  ### NEW
                # compute thresholds
                max_test = total_samples * TEST_DATA_SIZE
                max_val = total_samples * VAL_DATA_SIZE
                test_groups, val_groups, train_groups = set(), set(), set()

                cum_size = 0
                for g in shuffled:
                    if cum_size < max_test:
                        test_groups.add(g)
                    elif cum_size < max_test + max_val:
                        val_groups.add(g)
                    else:
                        train_groups.add(g)
                    cum_size += group_to_size[g]

                print(f"[Split {split_idx}] Train={len(train_groups)}, Val={len(val_groups)}, Test={len(test_groups)}")

                # Save all 3 splits
                save_selected_data_and_annot(joined_df, train_groups, tax_level,
                                             f"{out_dir}/train_data_{tax_level}_split_{split_idx}",
                                             f"{out_dir}/train_annot_{tax_level}_split_{split_idx}",
                                             f"{out_dir}/train_taxa_{tax_level}_split_{split_idx}")
                save_selected_data_and_annot(joined_df, val_groups, tax_level,
                                             f"{out_dir}/val_data_{tax_level}_split_{split_idx}",
                                             f"{out_dir}/val_annot_{tax_level}_split_{split_idx}",
                                             f"{out_dir}/val_taxa_{tax_level}_split_{split_idx}")
                save_selected_data_and_annot(joined_df, test_groups, tax_level,
                                             f"{out_dir}/test_data_{tax_level}_split_{split_idx}",
                                             f"{out_dir}/test_annot_{tax_level}_split_{split_idx}",
                                             f"{out_dir}/test_taxa_{tax_level}_split_{split_idx}")

            else:
                # Default 80/20 split
                max_test = total_samples * TEST_DATA_SIZE
                test_groups, train_groups = set(), set()
                cum_size = 0
                for g in shuffled:
                    if cum_size < max_test:
                        test_groups.add(g)
                    else:
                        train_groups.add(g)
                    cum_size += group_to_size[g]

                print(f"[Split {split_idx}] Train={len(train_groups)}, Test={len(test_groups)}")

                save_selected_data_and_annot(joined_df, train_groups, tax_level,
                                             f"{out_dir}/train_data_{tax_level}_split_{split_idx}",
                                             f"{out_dir}/train_annot_{tax_level}_split_{split_idx}",
                                             f"{out_dir}/train_taxa_{tax_level}_split_{split_idx}")
                save_selected_data_and_annot(joined_df, test_groups, tax_level,
                                             f"{out_dir}/test_data_{tax_level}_split_{split_idx}",
                                             f"{out_dir}/test_annot_{tax_level}_split_{split_idx}",
                                             f"{out_dir}/test_taxa_{tax_level}_split_{split_idx}")

        print("Finished!")

    else:
        # Random split (no taxonomy)
        for split_idx in range(NUM_SPLITS):
            input_df_annot = pl.read_csv(args.input_annotation_csv, separator=",")
            old_name = input_df_annot.columns[-1]
            input_df_annot = input_df_annot.rename({old_name: "annotation"})
            input_df_counts = pl.read_csv(args.input_data_csv, separator=",")

            joined_df = input_df_counts.join(input_df_annot, on="accession", how="left")
            joined_df = joined_df.filter(pl.col("annotation").is_not_null())
            shuffled_idx = np.random.permutation(joined_df.height)
            joined_df = joined_df[shuffled_idx.tolist()]

            if args.train_val_test_split_flag:  ### NEW
                test_size = int(len(joined_df) * TEST_DATA_SIZE)
                val_size = int(len(joined_df) * VAL_DATA_SIZE)
                table_test = joined_df[:test_size]
                table_val = joined_df[test_size:test_size + val_size]
                table_train = joined_df[test_size + val_size:]

                print(f"[Split {split_idx}] Train={len(table_train)}, Val={len(table_val)}, Test={len(table_test)}")

                # Save each set
                for subset, table in zip(["train", "val", "test"], [table_train, table_val, table_test]):
                    table.select(["accession", "annotation"]).write_csv(
                        f"{out_dir}/{subset}_annot_{tax_level}_split_{split_idx}", separator="\t")
                    table.drop("annotation").write_csv(
                        f"{out_dir}/{subset}_data_{tax_level}_split_{split_idx}", separator="\t")
            else:
                # Default 80/20 split
                test_size = int(len(joined_df) * TEST_DATA_SIZE)
                table_test = joined_df[:test_size]
                table_train = joined_df[test_size:]
                print(f"[Split {split_idx}] Train={len(table_train)}, Test={len(table_test)}")

                for subset, table in zip(["train", "test"], [table_train, table_test]):
                    table.select(["accession", "annotation"]).write_csv(
                        f"{out_dir}/{subset}_annot_{tax_level}_split_{split_idx}", separator="\t")
                    table.drop("annotation").write_csv(
                        f"{out_dir}/{subset}_data_{tax_level}_split_{split_idx}", separator="\t")

        print("Finished!")

