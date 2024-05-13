**KrakenBlend** is the extension of Kraken2 reference database approach for gene sequence classification. Hybrid classification is performed on gene sequences combining Kraken2 output with Graph Convolution Networks (GCNs). 

## ✨ Directions to use 

### 🚀 Prediction
- Input can be provided in 2 ways.\
 🔴 Sequence file must be in the extension `.fastq` or `.fasta`.\
 🔴 Provide reference data in `.tar.gz` format.

**Uploading files**: 
Upload sequence file along with the reference database to be used in the Kraken2.\
**Providing URL**: 
Provide URL for sequence file and reference database.

### 🔎 Evaluation
- You can evaluate the prediction by giving the ground truth.\
 🔴 Provide true labels in the correct order of sequences in a .txt file.\
 🔴 Label names must match with Kraken2 species naming.