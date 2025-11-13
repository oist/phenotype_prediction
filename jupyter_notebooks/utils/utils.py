import os
import random 
import torch
import logging 
import pandas as pd
import numpy as np
import polars as pl
import argparse
from matplotlib import cm
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.metrics import accuracy_score

from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MaxAbsScaler

from sklearn.model_selection import cross_val_predict, KFold, GroupKFold
from xgboost import XGBRegressor, XGBClassifier

from sklearn.metrics import mean_squared_error,r2_score

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def read_ogt_data(X_filename, y_filename, taxa_filename, device):
    # --- Read X ---
    df_x_data = pd.read_csv(X_filename, sep="\t")
    if df_x_data.empty:
        print(f"[WARNING] X file '{X_filename}' is empty. Skipping.")
        return None, None, None, None

    X_train_column_names = df_x_data.columns
    if "accession" not in df_x_data.columns:
        print(f"[WARNING] 'accession' column missing in X file '{X_filename}'. Skipping.")
        return None, None, None, None

    X_val = df_x_data.drop(columns=['accession']).apply(pd.to_numeric, errors='coerce').fillna(0).values
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)

    # --- Read Y ---
    df_y_labels = pd.read_csv(y_filename, sep="\t")
    if df_y_labels.empty:
        print(f"[WARNING] Y file '{y_filename}' is empty. Skipping.")
        return None, None, None, None
    if "accession" not in df_y_labels.columns:
        print(f"[WARNING] 'accession' column missing in Y file '{y_filename}'. Skipping.")
        return None, None, None, None

    y_label = df_y_labels.drop(columns=['accession'])
    if y_label.empty:
        print(f"[WARNING] Y file '{y_filename}' has no columns after dropping 'accession'. Skipping.")
        return None, None, None, None

    y_label = torch.tensor(y_label.values, dtype=torch.float32).to(device)

    # --- Read taxa if present ---
    if taxa_filename is not None:
        try:
            taxa_label = pd.read_csv(taxa_filename, sep="\t")
            if taxa_label.empty:
                print(f"[WARNING] Taxa file '{taxa_filename}' is empty. Setting taxa_label = None.")
                taxa_label = None
            else:
                taxa_label = taxa_label.iloc[:, -1].tolist()
        except Exception as e:
            print(f"[WARNING] Could not read taxa file '{taxa_filename}': {e}")
            taxa_label = None
    else:
        taxa_label = None

    return X_val, y_label, X_train_column_names[1:], taxa_label


def read_diderm_data(X_filename, y_filename, taxa_filename, device):
    # --- Read X ---
    df_x_data = pd.read_csv(X_filename, sep="\t")
    if df_x_data.empty:
        print(f"[WARNING] X file '{X_filename}' is empty. Skipping.")
        return None, None, None, None
    df_x_data = df_x_data.drop_duplicates(subset='accession', keep='first')
    if "accession" not in df_x_data.columns:
        print(f"[WARNING] 'accession' column missing in X file '{X_filename}'. Skipping.")
        return None, None, None, None
    X_train_column_names = df_x_data.columns

    # --- Read Y ---
    df_y_labels = pd.read_csv(y_filename, sep="\t")
    if df_y_labels.empty:
        print(f"[WARNING] Y file '{y_filename}' is empty. Skipping.")
        return None, None, None, None
    df_y_labels = df_y_labels.drop_duplicates(subset='accession', keep='first')
    if "accession" not in df_y_labels.columns:
        print(f"[WARNING] 'accession' column missing in Y file '{y_filename}'. Skipping.")
        return None, None, None, None

    df_merged = pd.merge(df_x_data, df_y_labels, on='accession', how='inner')
    if df_merged.empty:
        print(f"[WARNING] Merge between X and Y produced no rows. Skipping.")
        return None, None, None, None

    # --- Read taxa if provided ---
    if taxa_filename is not None:
        taxa_label_df = pd.read_csv(taxa_filename, sep="\t")
        if taxa_label_df.empty:
            print(f"[WARNING] Taxa file '{taxa_filename}' is empty. Setting taxa_label = None.")
            taxa_label = None
        else:
            taxa_label_df = taxa_label_df.drop_duplicates(subset='accession', keep='first')
            df_merged = pd.merge(df_merged, taxa_label_df, on='accession', how='inner')
            if df_merged.empty:
                print(f"[WARNING] Merge with taxa produced no rows. Skipping.")
                return None, None, None, None
            taxa_label = df_merged.iloc[:, -1].tolist()
            df_merged = df_merged.drop(columns=df_merged.columns[-1])
    else:
        taxa_label = None

    X_val = df_merged.drop(columns=['annotation', 'accession']).apply(pd.to_numeric, errors='coerce').fillna(0).values
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)

    y_label = df_merged["annotation"].map({'Diderm': 0, 'Monoderm': 1})
    y_label = torch.tensor(y_label.values, dtype=torch.float32).to(device)

    return X_val, y_label, X_train_column_names[1:], taxa_label








def read_aerob_data(X_filename, y_filename, taxa_filename, device):
    df_x_data = pd.read_csv(X_filename, sep="\t")
    df_y_labels = pd.read_csv(y_filename, sep="\t")

    # --- Sanity checks ---
    if df_x_data.empty:
        print(f"[WARNING] X file '{X_filename}' is empty or could not be read. Skipping.")
        return None, None, None, None

    if df_y_labels.empty:
        print(f"[WARNING] Y file '{y_filename}' is empty or could not be read. Skipping.")
        return None, None, None, None

    if "accession" not in df_x_data.columns or "accession" not in df_y_labels.columns:
        print(f"[WARNING] Missing 'accession' column in one of the files. Skipping.")
        return None, None, None, None

    # --- Merge and check result ---
    df_merged = pd.merge(df_x_data, df_y_labels, on="accession", how="inner")
    if df_merged.empty:
        print(f"[WARNING] Merge between {X_filename} and {y_filename} produced no overlapping accessions.")
        return None, None, None, None

    # --- Extract numeric features ---
    try:
        X_val = df_merged.drop(columns=["annotation", "accession"]).apply(pd.to_numeric, errors="coerce").fillna(0).values
    except Exception as e:
        print(f"[ERROR] Failed to extract numeric features from {X_filename}: {e}")
        return None, None, None, None

    X_train_column_names = df_x_data.columns

    # --- Convert to tensor ---
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)

    # --- Map annotation labels ---
    if "annotation" not in df_merged.columns:
        print(f"[WARNING] No 'annotation' column found after merge in {y_filename}.")
        return None, None, None, None

    y_label = df_merged["annotation"].map({"anaerobe": 0, "aerobe": 1})
    if y_label.isnull().any():
        print(f"[WARNING] Some labels could not be mapped to 0/1 in {y_filename}.")
        y_label = y_label.fillna(-1)  # optional fallback
    y_label = torch.tensor(y_label.values, dtype=torch.float32).to(device)

    # --- Read taxa file if available ---
    if taxa_filename is not None:
        try:
            taxa_label = pd.read_csv(taxa_filename, sep="\t")
            taxa_label = taxa_label.iloc[:, -1].tolist()
        except Exception as e:
            print(f"[WARNING] Could not read taxa file {taxa_filename}: {e}")
            taxa_label = None
    else:
        taxa_label = None

    return X_val, y_label, X_train_column_names[1:], taxa_label


def read_sporulat_data(X_filename, y_filename, taxa_filename, device):
    # --- Read input files ---
    df_x_data = pd.read_csv(X_filename, sep="\t")
    if df_x_data.empty:
        print(f"[WARNING] X file '{X_filename}' is empty. Skipping.")
        return None, None, None, None

    df_y_labels = pd.read_csv(y_filename, sep="\t")
    if df_y_labels.empty:
        print(f"[WARNING] Y file '{y_filename}' is empty. Skipping.")
        return None, None, None, None

    # --- Merge on accession and check result ---
    if "accession" not in df_x_data.columns or "accession" not in df_y_labels.columns:
        print(f"[WARNING] Missing 'accession' column in one of the files. Skipping.")
        return None, None, None, None

    df_merged = pd.merge(df_x_data, df_y_labels, on='accession', how='inner')
    if df_merged.empty:
        print(f"[WARNING] Merge between {X_filename} and {y_filename} produced no overlapping rows. Skipping.")
        return None, None, None, None

    # --- Extract features ---
    if "annotation" not in df_merged.columns:
        print(f"[WARNING] 'annotation' column missing in merged data. Skipping.")
        return None, None, None, None

    X_val = df_merged.drop(columns=['annotation', 'accession']).apply(pd.to_numeric, errors='coerce').fillna(0).values
    X_train_column_names = df_x_data.columns

    # --- Convert to torch tensor ---
    X_val = torch.tensor(X_val, dtype=torch.float32).to(device)

    # --- Map labels ---
    y_label = df_merged["annotation"].map({'no': 0, 'yes': 1})
    if y_label.isnull().any():
        print(f"[WARNING] Some labels in '{y_filename}' could not be mapped to 0/1.")
        y_label = y_label.fillna(-1)
    y_label = torch.tensor(y_label.values, dtype=torch.float32).to(device)

    # --- Read taxa if provided ---
    if taxa_filename is not None:
        try:
            taxa_label = pd.read_csv(taxa_filename, sep="\t")
            if taxa_label.empty:
                print(f"[WARNING] Taxa file '{taxa_filename}' is empty. Setting taxa_label = None.")
                taxa_label = None
            else:
                taxa_label = taxa_label.iloc[:, -1].tolist()
        except Exception as e:
            print(f"[WARNING] Could not read taxa file '{taxa_filename}': {e}")
            taxa_label = None
    else:
        taxa_label = None

    return X_val, y_label, X_train_column_names[1:], taxa_label


def read_aerob_data11(
    X_data_path='../data_aerob/all_gene_annotations.tsv', 
    y_data_path = '../data_aerob/bacdive_scrape_20230315.json.parsed.anaerobe_vs_aerobe.with_cyanos.csv',
    bac_phylogeny_data_path='../data_preparation/gtdb_files/bac120_metadata_r202.tsv', 
    arch_phylogeny_data_path='../data_preparation/gtdb_files/ar122_metadata_r202.tsv',
    target_column = 'oxytolerance'):
    """
    Read aerobicity data:
    :param str X_data_path: Path to the feature table
    :param str y: Path to the labels table
    :param str bac_phylogeny_data_path: Path to the phylogenetic annotation table for bacteria
    :param arch_phylogeny_data_path: Path to the phylogenetic annotation table for archaeae
    :param target_column: Column name of the target
    :return: pandas.DataFrame
    """
    
    # Read GTDB phylogenetic annotation table
    gtdb = pl.concat([
        pl.read_csv(bac_phylogeny_data_path, separator='\t'),
        pl.read_csv(arch_phylogeny_data_path, separator='\t')
    ])
    gtdb = gtdb.filter(pl.col("gtdb_representative") == "t")
    print("Read in {} GTDB representatives".format(len(gtdb)))
    
    gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(0).alias("domain"))
    gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(1).alias("phylum"))
    gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(2).alias("class"))
    gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(3).alias("order"))
    gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(4).alias("family"))
    gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(5).alias("genus"))
    
    # Read feature table
    X_data = pl.read_csv(X_data_path, separator='\t')
    
    # Read y
    y_data = pl.read_csv(y_data_path, separator='\t')
    
    # Add phylogenetic annotation (join based on accession)
    full_data = X_data.join(gtdb.select(['accession','domain', 'phylum','class','order','family','genus']), on="accession", how="left")
    full_data = full_data.join(y_data, on="accession", how="inner") # Inner join because test accessions are in y1 but not in full_data
    print(f'Data without noise: {len(full_data)}')

    # Map y labels
    classes_map = {
        'anaerobe': 0,
        'aerobe': 1,
    }      
    
    y = full_data.with_columns(
        pl.col(target_column)
        .replace_strict(classes_map, default='unknown')
        .alias(target_column)
    )
    y = y.with_columns(
        pl.col(target_column).cast(pl.Int32)
    )    
    print("\nCounts of y:", y.group_by(target_column).agg(pl.len()))
    y = y.to_pandas().iloc[:, -1]

    # Make X dataframe
    X = full_data.select(pl.exclude(['accession',target_column,'domain','phylum','class','order','family','genus','false_negative_rate','false_positive_rate'])).to_pandas()

    return X, y, full_data

def process_aerob_dataset(X_filename, y_filename, device, remove_noise):
    d3_train, X_train, y_train = read_xy_data(X_filename, y_filename, remove_noise)
    d_gtdb_train = d3_train.to_pandas()

   # X_train = X_train.drop(columns=["family_right", "phylum_right", "class_right", "order_right", "genus_right"])
    X_train_column_names = X_train.columns

    matrix = X_train.values
    X_data = torch.tensor(matrix)
    X_train = X_data.float().to(device)
    X_train_numpy = X_train.cpu().numpy()
    scaler = MaxAbsScaler()
   # X_train_scaled = scaler.fit_transform(X_train_numpy)
    X_train = torch.tensor(X_train_numpy, dtype=torch.float32).to(device).float()

    y_train = torch.tensor(y_train.values).to(device)
    y_train = y_train.squeeze(1)
    y_train = y_train.float()

    return X_train, X_train_column_names, y_train, d_gtdb_train


def read_xy_data(data_filename, y_filename, remove_noise = True):

    try:
        gtdb = pl.concat([
            pl.read_csv('data_aerob/bac120_metadata_r202.tsv', separator="\t"),
            pl.read_csv('data_aerob/ar122_metadata_r202.tsv', separator="\t")
        ])
    except FileNotFoundError as e:  
        gtdb = pl.concat([
            pl.read_csv('../data_aerob/bac120_metadata_r202.tsv', separator="\t"),
            pl.read_csv('../data_aerob/ar122_metadata_r202.tsv', separator="\t")
        ])
    gtdb = gtdb.filter(pl.col("gtdb_representative") == "t")
    logging.info("Read in {} GTDB reps".format(len(gtdb)))
    gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(1).alias("phylum"))
    gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(2).alias("class"))
    gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(3).alias("order"))
    gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(4).alias("family"))
    gtdb = gtdb.with_columns(pl.col("gtdb_taxonomy").str.split(';').list.get(5).alias("genus"))

    
    target_column = "oxytolerance"
    # Read y
    y0 = pl.read_csv(y_filename, separator="\t")
    y1 = y0.unique() # There are some duplicates in the cyanos, so dedup
    logging.info("Read y: %s", y1.shape)
    # Log counts of each class
    logging.info("Counts of each class amongst unique accessions: %s", y1.group_by(target_column).agg(pl.len()))
    
    # Read the data
    d = pl.read_csv(data_filename,separator="\t")
    d2 = d.join(gtdb.select(['accession','phylum','class','order','family','genus']), on="accession", how="left")

    d3 = d2.join(y1, on="accession", how="inner") # Inner join because test accessions are in y1 but not in d2

    if remove_noise == True:
        d3 = d3.filter(pl.col("false_negative_rate") == 0)
        d3 = d3.filter(pl.col("false_positive_rate") == 0)

    print(f"Counts of each class in training/test data: {d3.group_by(target_column).agg(pl.len())}")
    
    d_gtdb = d3.to_pandas()
    X = d3.select(pl.exclude(['accession',target_column,'phylum','class','order','family','genus','false_negative_rate','false_positive_rate'])).to_pandas()

    # Blacklist these as they aren't in the current ancestral file, not sure why
    X = X.drop(['COG0411', 'COG0459', 'COG0564', 'COG1344', 'COG4177'], axis=1)


    # Generate y vector with 0s, and 1s
    y = d3.select(
        pl.when(pl.col(target_column) == 'anaerobe').then(0)
        .when(pl.col(target_column) == 'aerobe').then(1)
        .when(pl.col(target_column) == 'anaerobic_with_respiration_genes').then(2)
        .otherwise(None)  # Handle cases not in the map
        .alias(target_column)
    )
    y = y.to_pandas()
    return d3, X, y

def table_row_subsampling(d3):

   target_column = "oxytolerance"

   X = d3.select(pl.exclude([target_column])).to_pandas() #'phylum','class','order','family','genus'
   
   # Generate y vector with 0s, and 1s
   y = d3.select(
       pl.when(pl.col(target_column) == 'anaerobe').then(0)
       .when(pl.col(target_column) == 'aerobe').then(1)
       .when(pl.col(target_column) == 'anaerobic_with_respiration_genes').then(2)
       .otherwise(None)  # Handle cases not in the map
       .alias(target_column)
   )
   y = y.to_pandas()

   num_aerobs = y['oxytolerance'].sum()
   num_anaerobs = len(y['oxytolerance']) - num_aerobs
   
   # Sub-sampling training data
   indices_aerobs = y[y['oxytolerance'] == 1].index.tolist()
   indices_anaerobs = y[y['oxytolerance'] == 0].index.tolist()

   if num_aerobs > num_anaerobs:
       print(f"Sub-sampling {num_aerobs} aerobs to {num_anaerobs} anaerobs")
       subsampled_aerobs = random.sample(indices_aerobs, num_anaerobs)
       final_row_indices = subsampled_aerobs + indices_anaerobs
   else:
       print(f"Sub-sampling {indices_anaerobs} aerobs to {num_aerobs} anaerobs")
       subsampled_anaerobs = random.sample(indices_anaerobs, num_aerobs)
       final_row_indices = subsampled_anaerobs + indices_aerobs

   X_subsampled = X.iloc[final_row_indices].reset_index(drop=True)
   y_subsampled = y.iloc[final_row_indices].reset_index(drop=True)

   print(f"Sub-sampled table length = {len(y_subsampled)} with { y_subsampled['oxytolerance'].sum()} aerobs and  {len(y_subsampled['oxytolerance']) - y_subsampled['oxytolerance'].sum()} anaerobs")
   
   return X_subsampled, y_subsampled


def generate_colors_from_colormap(colormap_name, N):
   # Get the colormap
   colormap = plt.cm.get_cmap(colormap_name, N)
   
   # Generate the N colors from the colormap
   colors = [colormap(i) for i in range(N)]
   
   # Create a ListedColormap from the N colors
   listed_cmap = ListedColormap(colors)
   
   return listed_cmap, colors

def pca_run_and_plot(X_train_val, n_compon, y_train_val = None, category_names = None,  colors = None, legend = False, colorbar = False, alpha=0.6, s = 10):
   scaler = MaxAbsScaler()

   # Fit and transform the data
   X_train_val = scaler.fit_transform(X_train_val)

   # Run PCA on the X-data
   pca = PCA(n_components=n_compon)
   X_train_pca = pca.fit_transform(X_train_val)

   # Find the explained variance
   explained_variance_ratio = pca.explained_variance_ratio_

   listed_cmap = None

  # plt.grid(True, zorder=1)

  # plt.figure(figsize=(4, 4))
   if y_train_val is not None:
       #Ensure 'y_train_val' has unique, valid labels
       unique_ids = np.unique(y_train_val)

       # Check if 'colors' is already a ListedColormap or needs to be generated
       if isinstance(colors, ListedColormap):
           listed_cmap = colors  # Use the passed ListedColormap directly
       else:
           # Generate colors based on the unique labels in y_train_val
           
           listed_cmap = ListedColormap(cm.nipy_spectral(np.linspace(0, 1, len(unique_ids))))
       scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=y_train_val, alpha=alpha, s = s, label = category_names, cmap=listed_cmap, zorder=2)

       if category_names is not None:
           categ_name_dict = defaultdict(int)
           for i in range(len(y_train_val)):
               categ_id = y_train_val[i]
               #if categ_id not in categ_name_dict.keys():
               categ_id = int(categ_id)
               categ_name_dict[categ_id] = category_names[i]
           labels = [categ_name_dict[unique_id] for unique_id in unique_ids]       

           # Create legend handles and labels based on unique labels
           handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=listed_cmap(i / len(unique_ids)), markersize=10) for i in range(len(unique_ids))]
       
           if legend:
               plt.legend(handles=handles, labels=labels ,loc='upper center', title="Categories", ncol=5, bbox_to_anchor=(0.5, -0.25)) #, bbox_to_anchor=(1.05, 1)
    #    else:
    #        plt.colorbar()    
   else:
       scatter = plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], alpha=alpha, s = s, zorder=2)
   if colorbar is True:
       plt.colorbar()            
   plt.xlabel(f"PC1")
   plt.ylabel(f"PC2")
   plt.title("PCA")
   
   #plt.show()

   return listed_cmap    


false_posit_uniq = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
false_negat_uniq = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def tsne_plot(X_train, perplexity, learning_rate, random_seed, y_train = None, colors = None, colorbar = False, alpha=0.6, s = 10):
    scaler = MaxAbsScaler()

    # Fit and transform the data
    X_train_scal = scaler.fit_transform(X_train)


    # Initialize and apply t-SNE
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=learning_rate, max_iter=3000, init='pca', random_state=random_seed) 

    if colors is None:
        listed_cmap = ListedColormap(cm.nipy_spectral(np.linspace(0, 1, len(np.unique(y_train)))))
        #colors = ListedColormap(["tab:green", "tab:purple"])
    else:
        listed_cmap = colors    


    X_tsne = tsne.fit_transform(X_train_scal) 

    print(f"Shape of the projected data = {X_tsne.shape}")

    # Visualize the t-SNE output
    if y_train is not None:
        sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, alpha=alpha, s = s, cmap=listed_cmap, zorder=2)
    else:
        sc = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=alpha, s = s, zorder=2)
    if colorbar is True and y_train is not None:
        plt.colorbar(sc)        
    plt.xlabel("tSNE1")
    plt.ylabel("tSNE2")
    plt.title("tSNE")    
   # plt.grid(True, zorder=1)      


def generate_tables(grouped):
    tables = []
    
    false_positive_unique = grouped['false_positive_rate'].unique()
    
    # Loop over each unique value in false_positive_rate
    for false_positive_value in false_positive_unique:
        # Filter the rows with this false_positive_rate value
        filtered_df = grouped[grouped['false_positive_rate'] == false_positive_value]
        
        # Optionally, you can add the false_positive_rate as a column in the filtered DataFrame
        filtered_df = filtered_df[['false_negative_rate', 'matching_probability', 'mean_fp_prediction', 'mean_fn_prediction']]
    
        # Append to the tables list
        tables.append(filtered_df)
    return tables    

def find_aver_accuracy(table_dict):
    average_arr = []
    for key in table_dict.keys():
        average_arr.append(table_dict[key]['matching_probability'].mean())
    return np.mean(average_arr)   

def find_average_table(csv_files_cross_valid, result_directory):
    tables_all_folds = defaultdict(list)
    tables_average_folds = defaultdict(pd.DataFrame)
    
    for csv_file in csv_files_cross_valid:
        file_path = result_directory + csv_file
        grouped = group_matching_probab(file_path)
        tables = generate_tables(grouped)
        for i in range(len(tables)):
            tables_all_folds[false_posit_uniq[i]].append(tables[i])
    
    for key in tables_all_folds.keys():
        tables = tables_all_folds[key]
        
        # Concatenate all tables into one DataFrame
        combined = pd.concat(tables)
        
        # Group by 'false_negative_rate' and calculate the mean of 'matching_probability'
        average_table = (
            combined.groupby('false_negative_rate', as_index=False)
            .agg({
                'matching_probability': 'mean',
                'mean_fp_prediction': 'mean',
                'mean_fn_prediction': 'mean'
            })
        )
       
        tables_average_folds[key] = average_table
    return tables_average_folds   

def group_matching_probab(file_path):
    df = pd.read_csv(file_path, delimiter='\t')
    # Assuming df is your DataFrame
    df['prediction_correct'] = df['prediction'] == df['y_actual']

    df['fp_prediction'] = (df['prediction'] == 1) & (df['y_actual'] == 0)
    df['fn_prediction'] = (df['prediction'] == 0) & (df['y_actual'] == 1)

  #  print(df)
    
    grouped = (
        df.groupby(['false_negative_rate', 'false_positive_rate'])
        .agg({
            'prediction_correct': 'mean',
            'fp_prediction': 'mean',
            'fn_prediction': 'mean'
        })
        .reset_index()
    )
    
    # Rename columns for clarity if needed
    grouped.rename(
        columns={
            'prediction_correct': 'matching_probability',
            'fp_prediction': 'mean_fp_prediction',
            'fn_prediction': 'mean_fn_prediction'
        },
        inplace=True
    )
    
    return grouped

def find_accuracies(num_ind_points, result_directory):
    # List of all csv files with cross_validation results
    csv_files_cross_valid = [f for f in os.listdir(result_directory) if f.endswith('.csv') and "cross_valid" in f and f"indPoints_{num_ind_points}" in f]
    
    csv_files_holdout_test = [f for f in os.listdir(result_directory) if f.endswith('.csv') and "holdout_test" in f and f"indPoints_{num_ind_points}" in f]
    

    grouped = group_matching_probab(result_directory+csv_files_holdout_test[0])
    holdout_test_accur_aver = grouped['matching_probability'].mean()

    print(f"\nHold-out (test) dataset results for {num_ind_points} inducing points:")
    print(f"Average accuracy: {round(grouped['matching_probability'].mean(),3)};")
    print(f"Average false_positive predictions: {round(grouped['mean_fp_prediction'].mean(),3)};")
    print(f"Average false_negative predictions: {round(grouped['mean_fn_prediction'].mean(),3)}")
    
    
    tables_average_folds = find_average_table(csv_files_cross_valid, result_directory)
    cross_valid_aver = find_aver_accuracy(tables_average_folds)
    
    print(f"\nCross-validation accuracy = {round(cross_valid_aver,3)} for {num_ind_points} inducing points")
    return tables_average_folds

def plot_results(column_name, num_ind_points, fp_to_plot, tables_average_folds):
    plt.figure(figsize=(6, 4))
    idx = 0
    for false_posit in tables_average_folds.keys():
        false_positive_value = false_posit_uniq[idx]
        if false_positive_value in fp_to_plot:
            table = tables_average_folds[false_posit]
            plt.scatter(table['false_negative_rate'], table[column_name])
            plt.plot(table['false_negative_rate'], table[column_name], label=f'extra genes rate = {false_positive_value}')
        idx += 1    
    
  #  plt.ylim([0.83, 1])
    plt.xlabel('gene removal rate')
    plt.ylim([0.85,1])
    plt.ylabel('accuracy')
    plt.legend()

    plt.title(f"{column_name} for SetTransformer with {num_ind_points} inducing points")
    
    plt.grid(True, zorder=1)
    plt.show()  

import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import KFold, StratifiedKFold


def train_xgboost_classification(X_train, y_train, X_test, y_test, num_classes=50):
    
    model = XGBClassifier(
        n_jobs=-1,
        tree_method="hist",
        objective="multi:softmax",   # Multi-class classification
        num_class=num_classes,       # Number of target classes
        eval_metric="mlogloss",      # Suitable for multi-class
    )

    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_true_list = []
    y_pred_list = []

    for train_idx, test_idx in kf.split(X_train, y_train):
        X_fold_train, X_fold_test = X_train[train_idx], X_train[test_idx]
        y_fold_train, y_fold_test = y_train[train_idx], y_train[test_idx]

        model.fit(X_fold_train, y_fold_train)
        y_pred_fold = model.predict(X_fold_test)

        y_true_list.append(y_fold_test)
        y_pred_list.append(y_pred_fold)

    y_true_cv = np.concatenate(y_true_list)
    y_pred_cv = np.concatenate(y_pred_list)

    # Re-initializa the model
    model = XGBClassifier(
        n_jobs=-1,
        tree_method="hist",
        objective="multi:softmax",   # Multi-class classification
        num_class=num_classes,       # Number of target classes
        eval_metric="mlogloss",      # Suitable for multi-class
    )
    model.fit(X_train.cpu(), y_train.cpu())

    y_pred_test = model.predict(X_test.cpu())

    return y_true_cv, y_pred_cv, y_pred_test



def train_xgboost(X_train, y_train, X_test, y_test, weights = None, model = None, taxonomy_labels=None):
    # model = XGBRegressor(
    # n_jobs=-1,                # Use all CPU cores
    # tree_method="hist",   # Use "hist" for CPU, "gpu_hist" for GPU
    # objective="reg:squarederror",  # Default loss function for regression
    # n_estimators=500, 
    # )

    if model is None:
        model = XGBRegressor(reg_alpha=1.0,reg_lambda=1.0, max_depth=3,subsample=0.8, colsample_bytree=0.8, n_estimators=300,learning_rate=0.05) 

    

    # Define cross-validation (e.g., 5-fold)
   # kf = KFold(n_splits=5, shuffle=True, random_state=42)

    n_splits=5
    # Choose fold strategy
    if taxonomy_labels is not None:
        fold_generator = GroupKFold(n_splits=n_splits).split(X_train, y_train, groups=taxonomy_labels)
        print("Using taxonomy aware CV folds")
    else:
        fold_generator = KFold(n_splits=n_splits, shuffle=True, random_state=42).split(X_train, y_train)



    y_true_list = []
    y_pred_list = []

    if weights is not None:
        weights_tensor_all = torch.tensor(weights, dtype=torch.float32)

    #for train_idx, test_idx in kf.split(X_train):
    for train_idx, test_idx in fold_generator:    
        X_fold_train, X_fold_test = X_train[train_idx], X_train[test_idx]
        y_fold_train, y_fold_test = y_train[train_idx], y_train[test_idx]

        if weights is not None:
            weights_fold_train = weights_tensor_all[train_idx]
            model.fit(X_fold_train, y_fold_train, sample_weight=weights_fold_train)
        else:
            model.fit(X_fold_train, y_fold_train)    

        y_pred_fold = model.predict(X_fold_test)

        y_true_list.append(y_fold_test)
        y_pred_list.append(y_pred_fold)

    # Convert lists to arrays
    y_true_cv = np.concatenate(y_true_list)
    y_pred_cv = np.concatenate(y_pred_list)   

    if weights is not None:
        model.fit(X_train.cpu(), y_train.cpu().numpy(), sample_weight=weights_tensor_all)
    else:
        model.fit(X_train.cpu(), y_train.cpu().numpy())    

    # Make predictions
    y_pred_test = model.predict(X_test.cpu())

    return  y_true_cv, y_pred_cv, y_pred_test, model


from sklearn.preprocessing import LabelEncoder
def xgboost_misture_of_experts(X_train, range_ids, sample_weights, X_test, num_classes = 2, temp_bound = 45):
    
    gating_model = XGBClassifier(
        n_jobs=-1,
        tree_method="hist",
        objective="binary:logistic",   # Binary classification objective
        eval_metric="logloss",         # Suitable for binary classification
    )

    gating_model.fit(X_train.cpu(), range_ids,sample_weight=sample_weights )  # Predicts soft assignments to experts

    # --- 4. Get gating probabilities (soft weights for each expert)
    gate_probs = gating_model.predict_proba(X_test.cpu())  # Shape: (n_samples, 3)
    gate_preds = gating_model.predict(X_test.cpu()) 

    y_train = y_train.squeeze()

    # Define masks (all 1D)
    low_mask  = y_train < temp_bound
    high_mask = y_train >= temp_bound

    # Apply masks correctly
    X_low, y_low   = X_train[low_mask].cpu(), y_train[low_mask].cpu()
    X_high, y_high = X_train[high_mask].cpu(), y_train[high_mask].cpu()

    model_low  = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05).fit(X_low, y_low)
    model_high =  XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.05).fit(X_high, y_high)

    pred_low  = model_low.predict(X_test)
    pred_high = model_high.predict(X_test)

    # --- 5. Weighted combination of expert outputs

    # Ensure correct mapping
    id_low = 0#= np.where(le.classes_ == 'low')[0][0]
    id_high =1#= np.where(le.classes_ == 'high')[0][0]

    final_pred = (
        gate_probs[:, id_low]  * pred_low +
        gate_probs[:, id_high] * pred_high
    )

    return gate_probs, final_pred



def xgboost_accuracy_contin(X_train, X_test, y_train, y_test, sorted_cog_idx, feat_step, feat_removal = False):
    rmse_test_arr = []
    r2_test_arr = []
    rmse_cv_arr = []
    r2_cv_arr = []
    
    num_feat = range(1,len(sorted_cog_idx),feat_step)
    num_feat_plot = []
    for N in num_feat:
        if feat_removal == False:
            select_feat = list(sorted_cog_idx[:N])
        else:
            select_feat = list(sorted_cog_idx[N:])
        num_feat_plot.append(N)#len(select_feat))    
        X_train_select_feat = X_train[:, select_feat]
        X_test_select_feat = X_test[:, select_feat]
        y_true_cv, y_pred_cv, y_pred_test  = train_xgboost(X_train_select_feat, y_train, X_test_select_feat, y_test)
        
        rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
        rmse_test_arr.append(rmse_test)
        r2_test = r2_score(y_test, y_pred_test)
        r2_test_arr.append(r2_test)
        rmse_cv = np.sqrt(mean_squared_error(y_true_cv, y_pred_cv))
        rmse_cv_arr.append(rmse_cv)
        r2_cv = r2_score(y_true_cv, y_pred_cv)
        r2_cv_arr.append(r2_cv)

    return rmse_test_arr, r2_test_arr, rmse_cv_arr, r2_cv_arr, num_feat_plot 


def plot_accuracy_metric(metric, test_accuracy_scores, cv_accuracy_scores, test_accur_arr, test_accur_arr_rem, cv_accur_arr, cv_accur_arr_rem, num_feat, tot_num_feat):
    plt.axhline(y=test_accuracy_scores[metric], color='darkred', linestyle='--', linewidth=1.5, label='baseline test')
    plt.axhline(y=cv_accuracy_scores[metric], color='darkblue', linestyle='--', linewidth=1.5, label='baseline CV')

    plt.plot(num_feat, [scores[metric] for scores in test_accur_arr], c = "tab:red", label = "test | add")
    plt.plot(num_feat, [scores[metric] for scores in cv_accur_arr], c = "tab:blue", label = "cv | add")

    plt.plot([tot_num_feat - n for n in num_feat] ,  [scores[metric] for scores in test_accur_arr_rem], c = "tab:red", label = "test | remove", alpha = 0.5)
    plt.plot([tot_num_feat - n for n in num_feat], [scores[metric] for scores in cv_accur_arr_rem], c = "tab:blue", label = "cv | remove", alpha = 0.5)

    plt.xlabel("number of features added/removed")
    plt.ylabel(metric)


def random_feat_removal_curves_ogt(X_train, X_test, y_train, y_test, num_runs, feat_step, feat_removal):
    tot_num_feat = X_train.cpu().shape[1]
    rmse_test_arr_mi_tot = []
    r2_test_arr_mi_tot = []
    rmse_cv_arr_mi_tot = []
    r2_cv_arr_mi_tot = []
    
    for _ in range(num_runs):
        shuffled_indices = np.random.permutation(tot_num_feat)
        rmse_test_arr_mi, r2_test_arr_mi, rmse_cv_arr_mi, r2_cv_arr_mi, num_feat_plot = xgboost_accuracy_contin(X_train.cpu(), X_test.cpu(), y_train, y_test, shuffled_indices, feat_step, feat_removal)
        rmse_test_arr_mi_tot.append(rmse_test_arr_mi)
        r2_test_arr_mi_tot.append(r2_test_arr_mi)
        rmse_cv_arr_mi_tot.append(rmse_cv_arr_mi)
        r2_cv_arr_mi_tot.append(r2_cv_arr_mi)
        
    rmse_test_arr_mi_mean = np.array(rmse_test_arr_mi_tot).mean(axis=0)  
    rmse_test_arr_mi_std = np.array(rmse_test_arr_mi_tot).std(axis=0)  
    
    r2_test_arr_mi_mean = np.array(r2_test_arr_mi_tot).mean(axis=0)  
    r2_test_arr_mi_std = np.array(r2_test_arr_mi_tot).std(axis=0)  
    
    rmse_cv_arr_mi_mean = np.array(rmse_cv_arr_mi_tot).mean(axis=0)  
    rmse_cv_arr_mi_std = np.array(rmse_cv_arr_mi_tot).std(axis=0)  
    
    r2_cv_arr_mi_mean = np.array(r2_cv_arr_mi_tot).mean(axis=0)  
    r2_cv_arr_mi_std = np.array(r2_cv_arr_mi_tot).std(axis=0)  
    return rmse_test_arr_mi_mean, rmse_test_arr_mi_std, r2_test_arr_mi_mean, r2_test_arr_mi_std, rmse_cv_arr_mi_mean, rmse_cv_arr_mi_std, r2_cv_arr_mi_mean, r2_cv_arr_mi_std    

def calculate_aver_std(y_test_np, diff_np, num_bins):
    # Define bins (adjust bin width as needed)
    bins = np.linspace(y_test_np.min(), y_test_np.max(), num=num_bins)
    bin_centers = 0.5 * (bins[1:] + bins[:-1])
    
    # Digitize y_test to find bin indices
    bin_indices = np.digitize(y_test_np, bins) - 1  # shift to 0-based
    
    # Initialize arrays for mean and std
    mean_diff = []
    std_diff = []
    
    # Compute mean and std of diff per bin
    for i in range(len(bin_centers)):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            mean_diff.append(np.mean(diff_np[bin_mask]))
            std_diff.append(np.std(diff_np[bin_mask]))
        else:
            mean_diff.append(np.nan)
            std_diff.append(np.nan)
    
    # Convert to arrays for plotting
    mean_diff = np.array(mean_diff)
    std_diff = np.array(std_diff)
    return bin_centers, mean_diff, std_diff