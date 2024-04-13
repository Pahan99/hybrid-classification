import streamlit as st

def main():
    st.set_page_config(layout="wide")
    st.markdown('<link href="styles.css" rel="stylesheet">', unsafe_allow_html=True)
    st.title("🧬 Hybrid Sequence Classification")

    # Upload files
    st.header("Upload Kraken2 Output Files")

    # Divide the screen into two columns
    col1, col2 = st.columns(2)

    # File uploader for the first file in the first column
    with col1:
        uploaded_file_1 = st.file_uploader("Upload Kraken2 Classification Result", type=["txt"])

    # File uploader for the second file in the second column
    with col2:
        uploaded_file_2 = st.file_uploader("Upload Kraken2 Report", type=["txt"])

    # if uploaded_file_1 and uploaded_file_2:
        # Button for hybrid classification
    # st.markdown('<button class="button">Perform Hybrid Classification</button>', unsafe_allow_html=True)
    st.button("Perform Hybrid Classification")
    #     # Add code here to perform hybrid classification
    #     st.header("Performing Hybrid Classification...")
    #     # You can use the uploaded files for hybrid classification

    #     # Display results
    #     st.header("Results")


if __name__ == "__main__":
    main()
