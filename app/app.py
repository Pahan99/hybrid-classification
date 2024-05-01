import streamlit as st
import time

from tools import run_kraken,run_seq2vec


def kraken_prediction():
    status_placeholder = st.empty()
    status_placeholder.write("1️⃣ Running Kraken2 prediction...")
    # run_kraken()
    time.sleep(5)
    status_placeholder.write("<span style='color:#00d26a'>✅ Kraken2 prediction completed...</span>", unsafe_allow_html=True)

def vectorize():
    status_placeholder = st.empty()
    status_placeholder.write("2️⃣ Vectorizing sequences...")
    # run_seq2vec()
    time.sleep(5)
    status_placeholder.write("<span style='color:#00d26a'>✅ Sequences vectorized...</span>", unsafe_allow_html=True)

def build_graph():
    status_placeholder = st.empty()
    status_placeholder.write("3️⃣ Building graph...")
    time.sleep(5)
    status_placeholder.write("<span style='color:#00d26a'>✅ Graph built...</span>", unsafe_allow_html=True)

def train_model():
    status_placeholder = st.empty()
    status_placeholder.write("4️⃣ Training model...")
    time.sleep(5)
    status_placeholder.write("<span style='color:#00d26a'>✅ Model trained...</span>", unsafe_allow_html=True)

# Function to perform prediction using KrakenBlend
def perform_prediction(sequence_input, database_input):
    kraken_prediction()
    vectorize()
    build_graph()
    train_model()

def main():
    st.title("KrakenBlend: Hybrid Sequence Classification")

    # Prediction page
    st.header("Get Prediction")

    # Sequence data upload
    sequence_file = st.file_uploader("Upload Sequence File (.fasta, .fastq)", type=["fasta", "fastq"])

    if sequence_file is not None:
        sequence_data = sequence_file.read()
    else:
        sequence_data = None

    # Kraken2 database upload
    database_file = st.file_uploader("Upload Kraken2 Database (.k2db)", type=["k2db"])

    if database_file is not None:
        database_data = database_file.read()
    else:
        database_data = None

    
    if st.button("Predict", type="primary"):
        prediction_result = perform_prediction(sequence_data, database_data)
        

if __name__ == "__main__":
    main()
