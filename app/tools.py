import subprocess

def run_kraken():
    subprocess.run(["bash", "kraken.sh"])

def run_seq2vec():
    subprocess.run(["bash", "seq2vec.sh"])