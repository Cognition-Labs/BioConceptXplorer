import streamlit as st
import requests
import pandas as pd
from typing import List, Tuple

def process_input(user_input: str) -> List[str]:
    url = f'https://degtrdg--bioconceptvecxplorer-bert-query.modal.run/?query={user_input}&top_k=5'
    r = requests.get(url)
    options: List[Tuple[str, float]] = r.json()
    options_str: List[str] = [f"{concept}| Similarity: {score}" for concept, score in options]
    if not options_str:
        options_str = ["No similar concepts found. Please try again."]
    return options_str

def get_free_var_search(query, sim_threshold, use_gpt=False):
    base_url = 'https://degtrdg--bioconceptvecxplorer-free-var-search.modal.run'
    n_samples = 100
    if use_gpt:
        gpt = 'gpt-4'
    else:
        gpt = 'none'
    params = {
        'query': query,
        'n': n_samples,
        'sim_threshold': sim_threshold,
        'use_gpt':gpt 
    }

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()

        json_response = response.json()

        if json_response and isinstance(json_response, list):
            df = pd.DataFrame(json_response)
            return df
        else:
            print("Error: Unexpected response format.")
            return None
    except requests.exceptions.RequestException as e:
        print("Error: Request failed:", e)
        return None

# Set up the Streamlit page
st.title("BioConceptVec Exploration App")

if 'user_input' not in st.session_state:
    st.session_state['user_input'] = ''

user_input = st.text_input("Enter a concept:", value=st.session_state['user_input'])

if st.button('Process Concept') or st.session_state['user_input'] != '':
    st.session_state['user_input'] = user_input
    st.session_state['options'] = process_input(user_input)
    st.session_state['gpt'] = False

    if 'options' in st.session_state:
        selected_option = st.selectbox("Select a similar concept:", st.session_state['options'])
        st.write("The option you select will be used to search for concepts in the following form:")
        st.write('[selected concept] + X - Y = Z')
        st.write('where X, Y, and Z are free variables in the latent space.')
        if st.button('Confirm Concept'):
            st.session_state['selected_option'] = selected_option

        if 'selected_option' in st.session_state:
            end_index = st.session_state['selected_option'].find("|") - 1
            extracted_string = st.session_state['selected_option'][:end_index].strip()

            st.write(f"You selected: {extracted_string}")

            threshold = st.selectbox("Select a threshold:", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            gpt = st.checkbox("Use GPT-4")
            if st.button('Submit Threshold'):
                st.session_state['threshold'] = threshold
                st.session_state['gpt'] = gpt
                st.write("Please wait while we process your request...")
                df = get_free_var_search(extracted_string, st.session_state['threshold'], st.session_state['gpt'])
                if df is not None:
                    st.write("Here are the results:")
                    st.download_button(
                        label="Download CSV",
                        data=df.to_csv(index=False),
                        file_name="res.csv",
                        mime="text/csv",
                    )

                    st.dataframe(df)
