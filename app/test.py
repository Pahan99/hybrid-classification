import streamlit as st

def main():
    st.title("🧬 Hybrid Sequence Classification")

    # Upload files
    st.header("Upload Files")
    uploaded_file_1 = st.file_uploader("Upload Kraken2 Classification Result", type=["txt"])
    uploaded_file_2 = st.file_uploader("Upload Kraken2 Report", type=["txt"])

    # Model selection
    st.header("Select Model")
    model = st.selectbox("Select Classification Model", ["Model 1", "Model 2", "Model 3"])

    if uploaded_file_1 and uploaded_file_2:
        # Perform classification when both files are uploaded
        st.header("Perform Classification")

        # Add code here to perform classification using ML model
        # You can use the uploaded files and selected model for classification

        # Display results
        st.header("Results")
        # Display classification results here

if __name__ == "__main__":
    main()
