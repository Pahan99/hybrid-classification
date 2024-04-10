# Hybrid Classification of Sequences 🧬

This is a novel approach which uses the strength of reference databases with machine learning to do a hybrid classification of gene sequences upto a species level.

**Kraken2** is used as the reference database approach.
* Code: https://github.com/DerrickWood/kraken2
* Paper: https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1891-0



## Instructions 📝

### 1️⃣ Dataset Selection
* Datasets must be in the file format .fastq or .fasta.
* For the experiments we are using sequences of length at least 5000bp.

### 2️⃣ Kraken2 Result
Since we are utilizing Kraken2 output for the approach, first we are obtaining the classification results from Kraken2 tool. For this, you can use the notebook at [`notebooks/kraken.ipynb`](https://github.com/Pahan99/hybrid-classification/blob/main/notebooks/kraken.ipynb).

For running the tool, we have to provide a Kraken database to the tool. You can find the databases at https://benlangmead.github.io/aws-indexes/k2. Select the proper database and provide the relevant link in the above-mentioned notebook. For more help you can refer [`Kraken2 manual`](https://github.com/DerrickWood/kraken2/blob/master/docs/MANUAL.markdown) as well.




