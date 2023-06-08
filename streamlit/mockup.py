import streamlit as st
import base64
import pandas as pd
import numpy as np

def process_input(user_input):
    # Simulate server response
    options = [
        '1: Processed concept 1 | Similarity: 0.85',
        '2: Processed concept 2 | Similarity: 0.84',
        '3: Processed concept 3 | Similarity: 0.83',
        '4: Processed concept 4 | Similarity: 0.82',
        '5: Processed concept 5 | Similarity: 0.81',
    ]
    return options

def select_option(options):
    selected_option = st.selectbox("Select a similar concept:", options)
    if selected_option:
        st.write("You selected:", selected_option)
    return selected_option

def get_free_var_search(extracted_string, threshold):
    # Simulate server response
    data = {
        'Equation': ['A + B - C', 'D + E - F', 'G + H - I'],
        'Concept': ['Concept 1', 'Concept 2', 'Concept 3'],
        'Similarity': [0.95, 0.94, 0.93],
        'Equation_mapped': ['A mapped + B mapped - C mapped', 'D mapped + E mapped - F mapped', 'G mapped + H mapped - I mapped'],
        'Concept Description': ['Description 1', 'Description 2', 'Description 3'],
        'Rationale': ['Rationale 1', 'Rationale 2', 'Rationale 3']
    }
    df = pd.DataFrame(data)
    return df

# Set up the Streamlit page
st.title("BioConceptVec Exploration App")

# Get the user's input
user_input = st.text_input("Enter a concept:")

if user_input:
    options = process_input(user_input)
    if options:
        option = select_option(options)
        if option:
            start_index = option.find(":") + 1
            end_index = option.find("|")
            extracted_string = option[start_index:end_index].strip()
            st.write(extracted_string)
            # Make an input box from 0.0 to 1.0 by increments of 0.1 multiselect
            threshold = st.multiselect("Select a threshold:", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])
            if threshold:
                threshold = threshold[0]
                df = get_free_var_search(extracted_string, threshold)

                # Display a download button
                csv = df.to_csv(index=False).encode()
                b64 = base64.b64encode(csv).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="res.csv">Download CSV File</a>'
                st.markdown(href, unsafe_allow_html=True)

                # Show the dataframe
                st.dataframe(df)
