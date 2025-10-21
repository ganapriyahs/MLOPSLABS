# Sentiment Analysis with Snorkel: Weak Supervision, Data Augmentation, and Data Slicing

This project demonstrates how to use the Snorkel library to build a sentiment analysis classifier for the Sentiment140 dataset using weak supervision techniques. It covers data labeling, data augmentation, and data slicing, adapting the official Snorkel spam classification tutorials.

The goal is to train a model to classify tweets as **Positive (1)** or **Negative (0)** without relying on large amounts of hand-labeled data.

**Dataset**: Sentiment140 dataset containing 1.6 million tweets.

---

## 1. Tutorial 1: Data Labeling (`01_sentiment_labeling.ipynb`)

This tutorial focuses on creating training labels programmatically using **weak supervision**.

### Key Steps:
1.  **Load & Clean Data**: Loaded the Sentiment140 dataset using `utils.load_dataset`, mapped labels (0, 4 -> 0, 1), and cleaned the tweet text (lowercase, remove URLs, mentions, punctuation). The training set labels were discarded (set to -1) to simulate an unlabeled scenario.
2.  **Write Labeling Functions (LFs)**: Defined Python functions (`@labeling_function`) as noisy heuristics to assign `POSITIVE`, `NEGATIVE`, or `ABSTAIN` labels. Examples included keyword, pattern (emoticon), negation, and TextBlob polarity LFs.
3.  **Apply LFs & Analysis**: Used `PandasLFApplier` to generate a label matrix (`L_train`) and `LFAnalysis` to check LF statistics. Debugged zero-coverage emoticon LFs by correcting regex.
4.  **Train Label Model**: Trained Snorkel's `LabelModel` on `L_train` to produce **probabilistic labels** (`probs_train`) and filtered out unlabeled data points.
5.  **Train Final Classifier**:
    * Featurized cleaned text using `TfidfVectorizer` (via `utils.df_to_features`) into **sparse matrices**. Resolved a memory error by removing `.toarray()` in `utils.df_to_features`.
    * Converted probabilistic labels to hard labels using `probs_to_preds`.
    * Trained a `LogisticRegression` model.
6.  **Evaluate**: Achieved an accuracy of ~61-67% on the test set, demonstrating the weak supervision approach.

---

## 2. Tutorial 2: Data Augmentation (`02_sentiment_data_augmentation.ipynb`)

This tutorial increases the size of the labeled training set using **data augmentation**.

### Key Steps:
1.  **Load Labeled Data**: Started with the labeled training data (`df_train_labeled`) generated in Tutorial 1. Created a **small subset** (`df_train_labeled_subset`) for faster execution.
2.  **Write Transformation Functions (TFs)**: Defined Python functions (`@transformation_function`) to modify tweets while preserving sentiment. Examples included replacing words with synonyms (using `nltk.wordnet`), replacing mentions, changing person names, and swapping adjectives. Debugged an `UnboundLocalError` in synonym TFs.
3.  **Define Policy**: Used `MeanFieldPolicy` to apply TFs with specified probabilities, prioritizing safer transformations.
4.  **Apply TFs**: Used `PandasTFApplier` to generate an augmented DataFrame (`df_train_augmented_subset`) from the subset.
5.  **Train & Compare Models**:
    * Featurized the original and augmented subsets using `utils.featurize_df_tokens` for an LSTM model.
    * Trained the LSTM model (`utils.get_keras_lstm`) separately on both subsets. Addressed a TensorFlow/Keras `AttributeError` related to `set_session` by removing deprecated TF1.x code.
    * Evaluated both models on the **full test set** to measure the impact of augmentation.

---

## 3. Tutorial 3: Data Slicing (`03_sentiment_data_slicing.ipynb`)

This tutorial focuses on **monitoring and improving** model performance on specific data subsets (slices).

### Key Steps:
1.  **Load Labeled Data (Subset)**: Loaded the Sentiment140 dataset *with* original labels, cleaned the text, and created **small training and testing subsets** (`df_train`, `df_test`) for efficiency.
2.  **Write Slicing Functions (SFs)**: Defined Python functions (`@slicing_function`) returning booleans to identify slices. Examples: `short_tweet`, `has_negation`, `high_positive_polarity`.
3.  **Monitor Baseline Performance**:
    * Trained a baseline `LogisticRegression` on the TF-IDF features of the training subset.
    * Applied SFs to the test subset (`S_test`) using `PandasSFApplier`.
    * Used `Scorer.score_slices` to evaluate the baseline model on each slice of the test subset. Handled `ValueError` caused by empty slices by adding checks.
4.  **Improve Slice Performance (`SliceAwareClassifier`)**:
    * **Attempted Sparse Approach**:
        * Defined a sparse-input PyTorch MLP base (`utils.get_pytorch_sparse_mlp_base`) using `nn.EmbeddingBag`.
        * Initialized `SliceAwareClassifier` with the sparse base.
        * Created sparse PyTorch tensors and used a custom Dataset/collate_fn (`utils.create_dict_dataloader_sparse`).
        * Training failed with a `NotImplementedError` for `aten::view` on sparse tensors, indicating incompatibility within Snorkel's `Trainer`.
    * **Reverted to Dense Approach (on Subset)**:
        * Ensured `utils.py` had a dense base MLP (`utils.get_pytorch_mlp_base`).
        * Initialized `SliceAwareClassifier` with the **dense** base MLP.
        * Converted the **subset's** sparse TF-IDF features to **dense** PyTorch tensors (`.toarray()`).
        * Created slice-aware dataloaders (`train_dl_slice`, `test_dl_slice`) using `make_slice_dataloader` with dense tensors and subset slice matrices.
        * Trained the `SliceAwareClassifier` using `Trainer` on the subset.
    * **Evaluate**: Evaluated the trained `SliceAwareClassifier` using `score_slices` on the test subset dataloader and compared results to the baseline.

---

## Utility Functions (`utils.py`)

This file contains helper functions adapted from the original Snorkel tutorials:
* Loading/preparing the Sentiment140 dataset (`load_dataset`).
* Creating Keras models (`get_keras_logreg`, `get_keras_lstm`) and callbacks.
* Featurizing text for Keras LSTM (`map_pad_or_truncate`, `featurize_df_tokens`).
* Featurizing text into sparse matrices (`df_to_features`).
* Creating PyTorch MLP models for both dense (`get_pytorch_mlp_base`, `get_pytorch_mlp`) and sparse input (`get_pytorch_sparse_mlp_base`, `get_pytorch_sparse_mlp`).
* Converting SciPy sparse matrices to PyTorch sparse tensors (`scipy_sparse_to_torch_sparse`).
* Creating PyTorch DataLoaders for dense (`create_dict_dataloader`) and sparse input (`create_dict_dataloader_sparse`, `create_sparse_batch`, `SparseRowDataset`).
* Previewing transformation functions (`preview_tfs`).

---
