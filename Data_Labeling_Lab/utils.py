import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tensorflow as tf
from sklearn.model_selection import train_test_split
from snorkel.classification.data import DictDataset, DictDataLoader
from collections import OrderedDict
import scipy.sparse # Needed for sparse handling

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
        units=output_dim, input_dim=input_dim, activation=activation,
        kernel_regularizer=tf.keras.regularizers.l2(0.001), #
    )
    model.add(dense)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01) #
    model.compile(optimizer=opt, loss=loss, metrics=["accuracy"]) #
    return model

def get_keras_lstm(num_buckets, embed_dim=16, rnn_state_size=64):
    """ Creates a Keras LSTM model. """
    lstm_model = tf.keras.Sequential()
    lstm_model.add(tf.keras.layers.Embedding(num_buckets, embed_dim)) #
    lstm_model.add(tf.keras.layers.LSTM(rnn_state_size, activation=tf.nn.relu)) #
    lstm_model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid)) #
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
    ids = tf.keras.preprocessing.text.hashing_trick(
        string, n=num_buckets, hash_function="md5" #
    )
    return ids[:max_length] + [0] * (max_length - len(ids)) #

def featurize_df_tokens(df):
    """ Applies map_pad_or_truncate to the 'text' column of a DataFrame. """
    return np.array([map_pad_or_truncate(str(text)) for text in df.text])

# --- Scikit-learn Featurization Helper ---

def df_to_features(vectorizer, df, split):
    """
    Converts DataFrame text column to features using a vectorizer.
    Returns sparse matrix.
    """
    words = df.text.tolist()
    # Ensure vectorizer is fitted only on 'train' split, handle other split names
    if split == "train":
        feats = vectorizer.fit_transform(words)
    elif hasattr(vectorizer, 'vocabulary_'): # Check if already fitted
        feats = vectorizer.transform(words)
    else:
        raise ValueError("Vectorizer is not fitted. Call with split='train' first.")
    X = feats # Return the sparse matrix directly
    Y = df["label"].values
    return X, Y

# --- PyTorch Model Helpers ---

# -- VERSION FOR DENSE INPUT (e.g., from .toarray()) --
def get_pytorch_mlp_base(input_dim, hidden_dim, num_layers=1):
    """
    Creates the base MLP layers for DENSE input *without* the final
    classification layer. Outputs features of size hidden_dim.
   
    """
    layers = []
    layers.extend([nn.Linear(input_dim, hidden_dim), nn.ReLU()]) #
    current_dim = hidden_dim
    for _ in range(num_layers - 1):
        layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()]) #
        current_dim = hidden_dim
    # Ensure output matches hidden_dim if num_layers=0 or input_dim=hidden_dim initially
    if current_dim != hidden_dim and num_layers > 0: # Correction: Only add if needed
         layers.append(nn.Linear(current_dim, hidden_dim))

    return nn.Sequential(*layers)

def get_pytorch_mlp(input_dim, hidden_dim, num_layers=1):
    """
    Creates the full MLP including the final classification layer for DENSE input.
    Assumes binary classification (output dim 2).
   
    """
    base_model = get_pytorch_mlp_base(input_dim, hidden_dim, num_layers)
    classifier_layer = nn.Linear(hidden_dim, 2) #
    return nn.Sequential(base_model, classifier_layer)


# -- VERSION FOR SPARSE INPUT (using nn.EmbeddingBag) --
def get_pytorch_sparse_mlp_base(vocab_size, embedding_dim, hidden_dim, num_layers=1):
    """
    Creates the base MLP layers for SPARSE input using EmbeddingBag,
    *without* the final classification layer. Outputs features of size hidden_dim.
   
    """
    layers = []
    layers.append(nn.EmbeddingBag(vocab_size, embedding_dim, mode='sum', sparse=True)) # Use sparse=True
    current_dim = embedding_dim
    # Add subsequent dense hidden layers
    for _ in range(num_layers):
        layers.extend([nn.Linear(current_dim, hidden_dim), nn.ReLU()])
        current_dim = hidden_dim
    # Ensure final output dim matches hidden_dim if needed
    if current_dim != hidden_dim:
        layers.append(nn.Linear(current_dim, hidden_dim))
    return nn.Sequential(*layers)

def get_pytorch_sparse_mlp(vocab_size, embedding_dim, hidden_dim, num_layers=1):
    """
    Creates the full MLP including the final classification layer, for SPARSE input.
    Assumes binary classification (output dim 2).
   
    """
    base_model = get_pytorch_sparse_mlp_base(vocab_size, embedding_dim, hidden_dim, num_layers)
    classifier_layer = nn.Linear(hidden_dim, 2)
    return nn.Sequential(base_model, classifier_layer)

# --- PyTorch Sparse Conversion Helper ---

def scipy_sparse_to_torch_sparse(sparse_mx):
    """Converts a SciPy sparse matrix (COO format) to a PyTorch sparse tensor."""
    if not isinstance(sparse_mx, scipy.sparse.coo_matrix):
        sparse_mx = sparse_mx.tocoo()
    sparse_mx = sparse_mx.astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape).coalesce() #

# --- PyTorch DataLoader Helpers ---

# -- VERSION FOR DENSE DATA (Original create_dict_dataloader) --
def create_dict_dataloader(X, Y, split, batch_size=64, shuffle=True, **kwargs):
    """
    Creates a DictDataLoader for PyTorch. Expects DENSE torch.FloatTensor for X.
   
    """
    if not isinstance(X, torch.Tensor):
        try:
             if hasattr(X, "toarray"): # Handle sparse input by converting to dense
                 print(f"Warning: Converting sparse matrix to dense tensor in create_dict_dataloader ({split}). This may use significant memory.")
                 X_tensor = torch.FloatTensor(X.toarray())
             else: # Assume numpy array or compatible
                 X_tensor = torch.FloatTensor(X)
        except Exception as e:
             raise TypeError(f"Input X ({type(X)}) to create_dict_dataloader must be convertible to torch.FloatTensor: {e}")
    else:
        X_tensor = X.float()

    Y_tensor = torch.LongTensor(Y) if not isinstance(Y, torch.Tensor) else Y.long()
    dataset = DictDataset.from_tensors(X_tensor, Y_tensor, split) #
    return DictDataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs) #


# -- VERSION FOR SPARSE DATA (using nn.EmbeddingBag) --
def create_sparse_batch(batch):
    """
    Custom collate_fn for DataLoader to prepare sparse batches for EmbeddingBag.
    Expects batch items to be tuples: (sparse_coo_tensor, label_tensor)
    Outputs a tuple: (indices, offsets, labels)
   
    """
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    indices = []
    offsets = [0]
    for sparse_tensor, _ in batch:
        if not sparse_tensor.is_sparse:
             raise TypeError("Expected sparse COO tensors in batch")
        sparse_tensor = sparse_tensor.coalesce() #
        if sparse_tensor._nnz() > 0:
            indices.append(sparse_tensor.indices()[1]) # Feature indices
        offsets.append(offsets[-1] + sparse_tensor._nnz()) #

    if not indices: indices_cat = torch.tensor([], dtype=torch.long)
    else: indices_cat = torch.cat(indices)
    offsets_tensor = torch.tensor(offsets[:-1], dtype=torch.long)
    return indices_cat, offsets_tensor, labels #

class SparseRowDataset(torch.utils.data.Dataset):
    """ Custom PyTorch Dataset for efficiently slicing sparse CSR matrices. """
    def __init__(self, sparse_matrix_csr, labels):
        self.sparse_matrix = sparse_matrix_csr
        self.labels = labels
    def __len__(self):
        return self.sparse_matrix.shape[0]
    def __getitem__(self, idx):
        row_sparse = self.sparse_matrix[idx].tocoo() #
        indices = torch.from_numpy(np.vstack((row_sparse.row, row_sparse.col)).astype(np.int64))
        values = torch.from_numpy(row_sparse.data.astype(np.float32))
        shape = torch.Size(row_sparse.shape)
        sparse_row_tensor = torch.sparse_coo_tensor(indices, values, shape).coalesce() #
        return sparse_row_tensor, self.labels[idx]

def create_dict_dataloader_sparse(X, Y, split, batch_size=64, shuffle=True, **kwargs):
    """
    Creates a standard PyTorch DataLoader for sparse input (SciPy matrix X)
    intended for use with nn.EmbeddingBag. Uses a custom collate_fn.
   
    """
    if not isinstance(X, scipy.sparse.spmatrix):
        raise TypeError("Input X must be a SciPy sparse matrix for create_dict_dataloader_sparse")

    Y_tensor = torch.LongTensor(Y) if not isinstance(Y, torch.Tensor) else Y.long()
    X_csr = X.tocsr() #
    dataset = SparseRowDataset(X_csr, Y_tensor) #

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        collate_fn=create_sparse_batch, #
        **kwargs
    )
    print(f"Created standard PyTorch DataLoader for {split} split with sparse batching.")
    # Returns a standard DataLoader, NOT DictDataLoader.
    return dataloader

# --- Augmentation Helper ---

def preview_tfs(df, tfs):
    """ Displays examples of transformations applied by TFs. """
    transformed_examples = []
    for f in tfs:
        for i, row in df.sample(frac=1, random_state=2).iterrows(): #
            row_copy = row.copy()
            transformed_or_none = f(row_copy) #
            if transformed_or_none is not None:
                transformed_examples.append(
                    OrderedDict({
                        "TF Name": f.name,
                        "Original Text": row.text,
                        "Transformed Text": transformed_or_none.text, #
                    })
                )
                break
    return pd.DataFrame(transformed_examples)