import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("d4data/biomedical-ner-all")
model = AutoModelForTokenClassification.from_pretrained("d4data/biomedical-ner-all")

# Create NER pipeline
pipe = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")

# Title and description
st.title("Named Entity Recognition (NER) with Streamlit")
st.write("Enter text below for entity recognition:")

# Text input area
input_text = st.text_area("Input Text")

# Process input text if provided
if input_text:
    # Process input text
    output = pipe(input_text)

    # Display the output using Streamlit
    st.markdown("### Named Entity Recognition Results")
    st.write("Below are the identified entities:")
    st.write("")

    # Define complex style sheet for the table
    table_style = """
    <style>
    table {
        font-family: Arial, sans-serif;
        border-collapse: collapse;
        width: 100%;
        margin-top: 20px;
        margin-bottom: 20px;
    }

    th, td {
        border: 1px solid #dddddd;
        text-align: left;
        padding: 8px;
    }

    th {
        background-color: #f2f2f2;
    }

    tr:nth-child(even) {
        background-color: #f2f2f2;
    }
    </style>
    """

    # Render the table
    st.write(table_style, unsafe_allow_html=True)
    st.write("<table>", unsafe_allow_html=True)
    st.write("<tr><th>Entity Group</th><th>Word</th></tr>", unsafe_allow_html=True)
    for entity in output:
        st.write(f"<tr><td>{entity['entity_group']}</td><td>{entity['word']}</td></tr>", unsafe_allow_html=True)
    st.write("</table>", unsafe_allow_html=True)
    





