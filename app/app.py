import streamlit as st
from streamlit_option_menu import option_menu

from kraken import get_kraken_results, get_kraken_taxonomic, get_weights
import time
import os
import gc

from tools import run_kraken,run_seq2vec,run_kbm2
from util import run_model, get_predictions, evaluate_model, save_uploaded

st.set_page_config(layout="wide")

with open( "styles.css" ) as css:
    st.markdown( f'<style>{css.read()}</style>' , unsafe_allow_html= True)

def download_file():
    file_path = 'output/prediction.csv'
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            file_content = file.read()
        st.download_button(label="Download File", data=file_content, file_name=file_path, mime="text/plain")
    else:
        st.error("File not found.")
        
def kraken_prediction():
    # run_kraken()
    time.sleep(5)


def vectorize():
    # run_seq2vec()
    time.sleep(5)

def build_graph():
    run_kbm2()
    time.sleep(5)

def get_vector():
    # get_kraken_results()
    # get_kraken_taxonomic()
    # get_weights()
    time.sleep(5)

def train_model():
    # run_model()
    time.sleep(5)

# Function to perform prediction using KrakenBlend
def perform_prediction(sequence_input, database_input, pl):
    
    placeholders = [
    "⚪ Running Kraken prediction...",
    "⚪ Vectorizing data...",
    "⚪ Getting vector...",
    "⚪ Building graph...",
    "⚪ Training model..."
]


    for step, placeholder in enumerate(placeholders, start=1):
        # with st.spinner('Running...'):
            # Call your functions for each step here
        if step == 1:
            pl.write(placeholder)
            kraken_prediction()
        elif step == 2:
            pl.write(placeholder)
            vectorize()
        elif step == 3:
            pl.write(placeholder)
        elif step == 4:
            pl.write(placeholder)
            get_vector()
            build_graph()
        elif step == 5:
            pl.write(placeholder)
            train_model()
        pl.write(f"🟢 Completed...")
    # Display a button to download results after completing all steps
    return True
        

def evaluate(df): 
    accuracy, precision, recall, f1 = evaluate_model(df)
    st.write(f"Accuracy: {accuracy}")
    st.write(f"Precision: {precision}")
    st.write(f"Recall: {recall}")
    st.write(f"F1 Score: {f1}")

def main():
    st.title("🧬 KrakenBlend")
    with open("content/home.md", "r", encoding="utf-8") as file:
        markdown_content = file.read()
        st.write(markdown_content)

    # Prediction page
    # st.header("Get Prediction")
    # st.divider()

    with st.sidebar:
        selected = option_menu(
            menu_title="",
            menu_icon = 'app-indicator',
            options=["Predict","Analyse", "Help"],
            icons=["arrow-right-circle","search", "info-circle"],
            default_index=0
        )

    if selected == "Predict":
        # Split the content into two columns
        col1, col2 = st.columns(2)

        # Content for the first column
        with col1:
            sequence_file = st.file_uploader("Upload Sequence File (.fasta, .fastq)", type=["fasta", "fastq"])
            if sequence_file is not None:
                save_uploaded(sequence_file, "reads.fasta")
                print("Sequence file uploaded")
                del sequence_file
                gc.collect()
                # sequence_data = sequence_file.read()
            else:
                sequence_data = None

        # Content for the second column
        with col2:
            database_file = st.file_uploader("Upload Kraken2 Database (.tar.gz)", type=["tar.gz"])
            if database_file is not None:
                save_uploaded(database_file, "kraken_db.tar.gz")
                print("Database file uploaded")
                del database_file
                gc.collect()
                # database_data = database_file.read()
            else:
                database_data = None
        
        pl = st.empty()
        pl.write("Perform Hybrid Classification")
        if st.button("Predict",type="primary"):
            result = perform_prediction(sequence_data, database_data, pl)
        st.divider()
        download_file()

    if selected == "Analyse":
        col1, col2 = st.columns([2,1])
        df = get_predictions()
        with open("content/analyse.md", "r", encoding="utf-8") as file:
            markdown_content = file.read()
            st.write(markdown_content)
        with col1:
            st.subheader("Predictions")
            if os.path.exists("output/prediction.csv"):
                st.dataframe(df, height=300, width=700)
            else:
                st.error("No predictions found. Please run the prediction first.")
        with col2:
            st.subheader("Evaluate")
            st.file_uploader("Upload Ground Truth (.txt)", type=["txt"])
            st.button("Evaluate", type="primary",on_click=evaluate(df))
            

    if selected == "Help":
        with open("content/help.md", "r", encoding="utf-8") as file:
            markdown_content = file.read()
            st.write(markdown_content)
    
        

if __name__ == "__main__":
    main()
