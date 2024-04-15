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

### 4️⃣ Building the Graph
Graph is constructed using sequences as nodes and edges are made between them considering the overlaps between each sequence pair. Here we use [`seqtk`](https://github.com/lh3/seqtk) and [`Wtdbg2`](https://github.com/ruanjue/wtdbg2) for sequence manipulation and graph construction respectively. This process is implemented in the notebook [`notebooks/graph.ipynb`](https://github.com/Pahan99/hybrid-classification/blob/app/notebooks/graph.ipynb) notebook.

### 5️⃣ Graph Convolution Network (GCN)
Graph generated using in above step is fed into a GCN. Here we employ a node classification task where nodes of are sequences. `data.npz` file generated in above step is used in this step.

#### 5️⃣.1️⃣ Generating the weight vector for modified loss function
We are modifying the categorical cross entropy function using a weight for each sample which is derived using the Kraken2 output. To generate the weight vector, use the notebook [`notebooks/weights.ipynb`](https://github.com/Pahan99/hybrid-classification/blob/app/notebooks/weights.ipynb).

#### 5️⃣.2️⃣ Providing the labels for training & testing
For training we are providing the Kraken2 output itself. Training set will be filtered from the dataset using a mask and for the test set we have to provide the ground truth. 

Ground truth for simulated data can be found in the .fastq file itself which is generated from SimLoRD tool and ground truth for existing datasets can be found using minimap2 tool. Provide the ground truth in a .txt file which contains the true labels of the dataset with same order as in the sequence file.

    Eg: Limosilactobacillus fermentum
        Staphylococcus aureus
        ...
        Limosilactobacillus fermentum
        Salmonella enterica

* SimLoRD : Refer notebook [`notebooks/simlord.ipynb`](https://github.com/Pahan99/hybrid-classification/blob/app/notebooks/simlord.ipynb)
    * Code: https://bitbucket.org/genomeinformatics/simlord/src/master/
    * Paper : https://academic.oup.com/bioinformatics/article/32/17/2704/2450740

* Minimap2 : Refer notebook [`notebooks/minimap.ipynb`](https://github.com/Pahan99/hybrid-classification/blob/app/notebooks/minimap.ipynb)
    * Code: https://github.com/lh3/minimap2
    * Paper: https://academic.oup.com/bioinformatics/article/34/18/3094/4994778


