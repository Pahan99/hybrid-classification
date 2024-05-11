echo "Starting kbm2"

pip install numpy

OUTPUT_DIR="output"

mkdir -p $OUTPUT_DIR
grep ">" reads.fasta > $OUTPUT_DIR/read_ids

kbm2/wtdbg2/kbm2 -i reads.fasta -d reads.fasta -n 2000 -l 2560 -t 16 > $OUTPUT_DIR/kbm2.txt

echo "Finished kbm2"