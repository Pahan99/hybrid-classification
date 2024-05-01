OUTPUT_DIR="output"

seq2vec/build/seq2vec -f reads.fasta -k 4 -o $OUTPUT_DIR/4-mers.tsv -t 8 -x csv

echo "Finished seq2vec..."