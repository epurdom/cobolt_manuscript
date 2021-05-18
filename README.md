

## code/

This directory contains scripts for the `Cobolt` model inference.

-   `preprocessing_fragment_files.py` preproces the ATAC-seq `.snap` files and write the data to `.mtx` files.
-   `preprocessing_create_example_data.py` generates a random toy dataset for the `Cobolt`  package tutorials.
-   `load_data.py` contains functions for reading datasets. 
-   `model.py` contains a pytorch module class for the variational autoencoder.
-   `multiomicDataset.py` contains a pytorch dataset class for multiomic datasets.
-   `run_snare.py`  runs the analysis for SNARE-seq data.
-   `run_three_datasets.py` runs `Cobolt` for integrating three datasets (SNARE-seq, MOp scRNA-seq, and MOp scATAC-seq)
-   `run_train_test_split.py` perform the train-test split analysis for method comparison.

