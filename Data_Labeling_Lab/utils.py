import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from snorkel.classification.data import DictDataset, DictDataLoader
from collections import OrderedDict # Needed for preview_tfs

# --- Data Loading ---

def load_dataset(csv_path="data/sentiment_analysis.csv", test_frac=0.2, random_state=42):
    """
    Loads Sentiment CSV and splits into train/test DataFrames with text and label columns.
    Converts polarity: 0 (negative), 4 (positive) --> 0/1 binary.
   
    """
    df = pd.read_csv(csv_path, encoding='latin-1', header=None)
    df.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
    df['label'] = df['polarity'].replace(4, 1) #
    df = df[['text', 'label']]
    # Use train_test_split for potentially cleaner stratified splitting if needed later
    # For now, keeping the user's sample/drop logic
    df_train = df.sample(frac=1 - test_frac, random_state=random_state)
    df_test = df.drop(df_train.index)
    return df_train.reset_index(drop=True), df_test.reset_index(drop=True)

# --- Keras Model Helpers ---

def get_keras_logreg(input_dim, output_dim=2):
    """ Creates a Keras Logistic Regression model. """
    model = tf.keras.Sequential()
    loss = "binary_crossentropy" if output_dim == 1 else "categorical_crossentropy" #
    activation = tf.nn.sigmoid if output_dim == 1 else tf.nn.softmax #
    dense = tf.keras.layers.Dense(
        units=output_dim,
        input_dim=input_dim,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(0.001), #
    )
    model.add(dense)
    # Note: Adam optimizer might benefit from explicit learning_rate in newer TF versions
    opt = tf.keras.optimizers.Adam(learning_rate=0.01) #
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"]) #
    return model

def get_keras_lstm(num_buckets, embed_dim=16, rnn_state_size=64):
    """ Creates a Keras LSTM model. """
    lstm_model = tf.keras.Sequential()
    lstm_model.add(tf.keras.layers.Embedding(num_buckets, embed_dim)) #
    lstm_model.add(tf.keras.layers.LSTM(rnn_state_size, activation=tf.nn.relu)) #
    lstm_model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)) #
    # Note: Optimizer name as string uses default parameters
    lstm_model.compile("Adagrad", "binary_crossentropy", metrics=["accuracy"]) #
    return lstm_model

def get_keras_early_stopping(patience=10, monitor="val_accuracy"):
    """ Returns a Keras EarlyStopping callback. """
    return tf.keras.callbacks.EarlyStopping(
        monitor=monitor, patience=patience, verbose=1, restore_best_weights=True #
    )

# --- Keras Featurization Helpers ---

def map_pad_or_truncate(string, max_length=30, num_buckets=30000):
    """ Hashes text tokens, pads/truncates to max_length. """
    # Uses hashing trick for tokenization/vectorization
    ids = tf.keras.preprocessing.text.hashing_trick(
        string, n=num_buckets, hash_function="md5" #
    )
    # Pad with 0s or truncate
    return ids[:max_length] + [0] * (max_length - len(ids)) #

def featurize_df_tokens(df):
    """ Applies map_pad_or_truncate to the 'text' column of a DataFrame. """
    # Ensure text column is string type
    return np.array([map_pad_or_truncate(str(text)) for text in df.text])

# --- Scikit-learn Featurization Helper ---

def df_to_features(vectorizer, df, split):
    """
    Converts DataFrame text column to features using a vectorizer.
    Returns sparse matrix.
    """
    words = df.text.tolist()
    # Fit vectorizer only on training data
    feats = vectorizer.fit_transform(words) if split == "train" else vectorizer.transform(words) #
    X = feats # Return the sparse matrix directly
    Y = df["label"].values
    return X, Y

# --- PyTorch Model Helpers ---

def get_pytorch_mlp_base(input_dim, hidden_dim, num_layers=1):
    """
    Creates the base MLP layers *without* the final classification layer.
    Outputs features of size hidden_dim.
    """
    layers = []
    layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()]) #
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]) #
    return nn.Sequential(*layers)

def get_pytorch_mlp(input_dim, hidden_dim, num_layers=1):
    """
    Creates the full MLP including the final classification layer.
    Assumes binary classification (output dim 2).
    """
    base_model = get_pytorch_mlp_base(input_dim, hidden_dim, num_layers)
    classifier_layer = nn.Linear(hidden_dim, 2) #
    return nn.Sequential(base_model, classifier_layer)

# Note: The following function signature is different and was provided separately.
# Including it as the last definition means this version will be used if called by name.
def get_pytorch_mlp(hidden_dim, num_layers):
    """ Creates a simple MLP with specified hidden_dim and layers. """
    layers = []
    # This assumes input_dim is also hidden_dim, which might not always be true.
    # Consider adjusting if input dim differs.
    for _ in range(num_layers):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]) #
    # This definition *lacks* the final output layer for classification.
    # It might be intended as a base model similar to get_pytorch_mlp_base.
    return nn.Sequential(*layers)


# --- PyTorch DataLoader Helper ---

def create_dict_dataloader(X, Y, split, **kwargs):
    """
    Creates a DictDataLoader for PyTorch.
    Note: Expects dense torch.FloatTensor for X.
    """
    # Convert X to dense FloatTensor if it's not already
    if not isinstance(X, torch.Tensor):
        try:
            # Attempt conversion assuming X is numpy array or compatible
             X_tensor = torch.FloatTensor(X)
        except TypeError:
             # Handle sparse matrix case if necessary, might require .toarray()
             # Be cautious of memory usage with .toarray()
             if hasattr(X, "toarray"):
                 print("Warning: Converting sparse matrix to dense tensor in create_dict_dataloader. This may use significant memory.")
                 X_tensor = torch.FloatTensor(X.toarray())
             else:
                 raise TypeError("Input X to create_dict_dataloader must be convertible to torch.FloatTensor")
    else:
        # Ensure it's FloatTensor if already a Tensor
        X_tensor = X.float()

    # Ensure Y is LongTensor
    Y_tensor = torch.LongTensor(Y) if not isinstance(Y, torch.Tensor) else Y.long()

    ds = DictDataset.from_tensors(X_tensor, Y_tensor, split) #
    return DictDataLoader(ds, **kwargs) #

# --- Augmentation Helper ---

def preview_tfs(df, tfs):
    """ Displays examples of transformations applied by TFs. """
    transformed_examples = []
    for f in tfs:
        # Iterate through shuffled sample to find one successful transformation per TF
        for i, row in df.sample(frac=1, random_state=2).iterrows(): #
            transformed_or_none = f(row) # Apply TF
            if transformed_or_none is not None:
                # Record successful transformation
                transformed_examples.append(
                    OrderedDict({
                        "TF Name": f.name,
                        "Original Text": row.text, # Assumes original text is needed; TF modifies in place
                        "Transformed Text": transformed_or_none.text, # Access text after modification
                    })
                )
                break # Move to the next TF
    return pd.DataFrame(transformed_examples)