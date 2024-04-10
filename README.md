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

Now, the Kraken2 result has to be modified for further experiments. First, the output files of Kraken2 will be merged into a single .csv file. For that use the notebook [`notebooks/kraken_results.ipynb`](https://github.com/Pahan99/hybrid-classification/blob/main/notebooks/kraken_results.ipynb). 

Then, we filter out the sequences where Kraken2 has classified upto species level for training the models. For this use the notebook [`notebooks/kraken_taxonomic_levels.ipynb`](https://github.com/Pahan99/hybrid-classification/blob/main/notebooks/kraken_taxonomic_levels.ipynb). 

### 3️⃣ Obtaining Vector Embeddings for sequences
Sequences have to be converted into vectors before feeding them into ML models. So we are obtaining a vector embedding for sequences based on their k-mer frequencies. We use [`seq2vec`](https://github.com/anuradhawick/seq2vec) for generating the vector embeddings. We are using k=4 in our experiments. You can use [`notebooks/seq2vec.ipynb`](https://github.com/Pahan99/hybrid-classification/blob/main/notebooks/seq2vec.ipynb) notebook as well.







