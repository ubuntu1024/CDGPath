CDGPath
===========

<details>
<summary>
  <b>CDGPath: Dual-Relation Graph Learning with Confidence-Guided Alignment Fusion for Pathology–Genomics Survival Prediction</b>
</summary>
</details>

**Summary:** We propose CDGPath, a pathology–genomics survival prediction framework that combines dual-relation graph learning with confidence-guided alignment fusion to incorporate tissue relationships and complementary molecular information into patient representation learning.
<img src="support/CDGPath.png" width="1500px" align="center" />

### Downloading TCGA Data
To download diagnostic WSIs (formatted as .svs files), molecular feature data and other clinical metadata, please refer to the [NIH Genomic Data Commons Data Portal](https://portal.gdc.cancer.gov/) and the [cBioPortal](https://www.cbioportal.org/). WSIs for each cancer type can be downloaded using the [GDC Data Transfer Tool](https://docs.gdc.cancer.gov/Data_Transfer_Tool/Users_Guide/Data_Download_and_Upload/).

## Installation Guide for Linux (using anaconda)

### Pre-requisites:
* Linux (Tested on Ubuntu 22.04) 
* NVIDIA GPU (Traning and tested on NVIDIA Tesla A100 GPU) with CUDA 12.1
* Python (3.10.18)


## Processing Whole Slide Images
To process WSIs, first, the tissue regions in each biopsy slide are segmented using Otsu's Segmentation on a downsampled WSI using OpenSlide. The 256 x 256 patches without spatial overlapping are extracted from the segmented tissue regions at the desired magnification. Consequently, a pretrained truncated UNI is used to encode raw image patches into 1024-dim feature vectors, which we then save as .pt files for each WSI. The extracted features then serve as input (in a .pt file) to the network. The following folder structure is assumed for the extracted features vectors:    
```bash
DATA_ROOT_DIR/
    └──TCGA_BLCA/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
    └──TCGA_HNSC/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
    └──TCGA_COREAD/
        ├── slide_1.pt
        ├── slide_2.pt
        └── ...
    └──TCGA_STAD/
        ├── slide_1.ptd
        ├── slide_2.pt
        └── ...
    ...
```
DATA_ROOT_DIR is the base directory of all datasets / cancer type(e.g. the directory to your SSD). Within DATA_ROOT_DIR, each folder contains a list of .pt files for that dataset / cancer type.


## Molecular Features and Genomic Signatures
Processed molecular profile features containing mutation status, copy number variation, and RNA-Seq abundance can be downloaded from the [cBioPortal](https://www.cbioportal.org/), which we include as CSV files in the [following directory](https://github.com/iccv2021anon/10972/tree/master/datasets_csv_sig). For ordering gene features into gene embeddings, we used the following categorization of gene families (categorized via common features such as homology or biochemical activity) from [MSigDB](https://www.gsea-msigdb.org/gsea/msigdb/gene_families.jsp?ex=1). Gene sets for homeodomain proteins and translocated cancer genes were not used due to overlap with transcription factors and oncogenes respectively. The curation of "genomic signatures" can be modified to curate genomic embedding that reflect unique biological functions.

## Training-Validation Splits 
For evaluating the algorithm's performance, we  partitioned each dataset using 5-fold cross-validation (stratified by the site of histology slide collection). Splits for each cancer type are found in the [`splits`]([https://github.com/ubuntu1024/CDGPath/splits/5folds](https://github.com/ubuntu1024/CDGPath/tree/main/splits/5folds)) folder, which each contain **splits_{k}.csv** for k = 1 to 5. In each **splits_{k}.csv**, the first column corresponds to the TCGA Case IDs used for training, and the second column corresponds to the TCGA Case IDs used for validation. Slides from one case are not distributed across training and validation sets. Alternatively, one could define their own splits, however, the files would need to be defined in this format. The dataset loader for using these train-val splits are defined in the `return_splits` function in the `SurvivalDatasetFactory`.

## Running Experiments 
Refer to [`scripts`]([https://github.com/ubuntu1024/CDGPath/scripts](https://github.com/ubuntu1024/CDGPath/scripts)) folder for source files to train CDGPath and the baselines presented in the paper. Refer to the paper to find the hyperparameters required for training.

### Step 1: Generate WSI Graph Structures 
To initiate the graph-building process for WSI patches, you can use the following Python command. This approach involves first storing binary edges, which allows for random sampling and subsequently generating pairwise dual-relation graph based on these stored binary edges: 

``` shell
python extract_graph.py --h5_path H5_PATH --graph_save_path GRAPH_SAVE_PATH
```

`--h5_path H5_PATH`: This parameter specifies the path to your HDF5 file containing WSI patch data. Replace `H5_PATH` with the actual path to your data file.

`--graph_save_path GRAPH_SAVE_PATH`: This parameter defines where the generated graph structures will be saved. Replace `GRAPH_SAVE_PATH` with your desired output directory or file path. 

For a quick start, you can also run the `graph.sh` script to generate WSI graph structures: 

``` shell
bash scripts/graph.sh
``` 

This script will automate the processing of your data and create graph structures suitable for training. Ensure that you check and adjust the parameter settings within the script as needed to ensure compatibility with your specific data format.

### Step 2: Start Model Training 
Next, use the `main.py` script to start training the CDGPath model and other baseline models. 

``` shell 
python main.py
``` 

Before running this script, refer to the hyperparameter settings recommended in the paper, and adjust relevant configurations in the script according to your experimental needs. 
