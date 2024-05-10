echo "Starting kbm2"
pip install numpy

OUTPUT_DIR="output"
exp="."
mkdir -p $OUTPUT_DIR
grep ">" $exp/reads.fasta > $exp/read_ids
kmb2/wtdbg2/kbm2 -i $exp/reads.fasta -d $OUTPUT_DIR/reads.fasta -n 2000 -l 2560 -t 16 | python3 filter_alignments.py

echo "Finished kbm2"