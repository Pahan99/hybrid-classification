import subprocess

def run_kraken():
    subprocess.run(["bash", "kraken.sh"])

def run_seq2vec():
    subprocess.run(["bash", "seq2vec.sh"])

def run_kbm2():
    subprocess.run(["bash", "kbm2.sh"])
    subprocess.run(["python", "filter_alignments.py"])
    subprocess.run(["python", "graph.py"])