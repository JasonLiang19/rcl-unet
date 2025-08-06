import pandas as pd

def train_serpin_classifier(serpin_file, control_file, exclude_list, output_dir):

    # load serpins 
    df_serpins = pd.read_csv(serpin_file)
    df_serpins = df_serpins[df_serpins['sequence'].apply(len).between(200, 1024)]
    df_serpins.drop(['protein_name', 'gene_name', 'organism', 'group', 'is_kw_0722', 'is_serpin_family', 'rcl_seq', 'cterm_seq', 'cluster_id_90', 'cluster_id_50', 'rcl_loc'], axis=1, inplace=True)
    # drop serpins in rcl train and test set
    df_serpins = df_serpins[~df_serpins['id'].isin(exclude_list)]

    # load control proteins (non serpins)
    n = len(df_serpins)
    df_control = df_control.sample(n=n, random_state=42)
    
    # combine into single dataframe 
    df_serpins['is_serpin'] = True
    df_control['is_serpin'] = False

    df = pd.concat([df_serpins, df_control], ignore_index=True)

    X = []
    Y = []

    for protein_name in data_dict:

        sequence = data_dict[protein_name]["sequence"]
        one_hot_encoded = [encoding_map.get(residue, encoding_map["X"]) for residue in sequence]
        data_dict[protein_name]["encoding"] = np.array(one_hot_encoded, dtype=np.float32)

        # Assume ProtTrans features already computed and available
        encoded_protein = data_dict[protein_name]["encoding"]
        label_vector = data_dict[protein_name]["label"]

        x, y = prepare_training_pair(encoded_protein, label_vector)
        X.append(x)
        Y.append(y)

    X = np.stack(X)
    Y = np.stack(Y)

    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    if (scaling):
        N_train, T, D, C = X_train.shape
        N_val = len(X_val)
        print("scaling data")

        # Remove last dimension and flatten to (N*T, D)
        X_train_flat = X_train.reshape(-1, D)  # shape: (N*1024, 1024)
        X_val_flat = X_val.reshape(-1, D)  # shape: (N*1024, 1024)

        scaler = StandardScaler()
        X_train_flat_scaled = scaler.fit_transform(X_train_flat)
        X_val_flat_scaled = scaler.transform(X_val_flat)

        X_train = X_train_flat_scaled.reshape(N_train, T, D, C)
        X_val = X_val_flat_scaled.reshape(N_val, T, D, C)

        with open(os.path.join(run_dir, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)

    return model 

def main():
    train_serpin_classifier
    

if __name__ == "__main__":
    main()