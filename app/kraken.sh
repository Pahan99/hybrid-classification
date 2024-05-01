echo "Starting Kraken2"

OUTPUT_DIR="output"

mkdir -p $OUTPUT_DIR

mkdir -p db

tar -xvf kraken_db.tar.gz -C db

kraken/kraken2 --db db/ reads.fasta --threads 4 --output $OUTPUT_DIR/output.txt --report $OUTPUT_DIR/report.txt

echo "Finished Kraken2"