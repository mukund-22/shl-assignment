import streamlit as st
import pandas as pd
import requests


# Basic config
st.set_page_config(page_title="SHL Assessment Recommender", layout="centered", initial_sidebar_state="collapsed")

# Styling
st.markdown("""
    <style>
        body { background-color: #fafafa; }
        .main { padding-top: 2rem; }
        .title { text-align: center; color: #222; margin-bottom: 1rem; }
        .subtitle { text-align: center; color: #555; font-size: 1rem; margin-top: -10px; }
        .stButton button {
            background-color: #4B8BBE;
            color: white;
            border-radius: 6px;
            padding: 0.5rem 1rem;
            margin-top: 1rem;
        }
        table.minimal-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.95rem;
        }
        table.minimal-table th, table.minimal-table td {
            border: 1px solid #e0e0e0;
            padding: 8px 10px;
            text-align: left;
        }
        table.minimal-table thead {
            background-color: #4B8BBE;
            color: white;
            font-weight: bold;
        }
        table.minimal-table tr:nth-child(even) {
            background-color: #102c2c;
        }
        a {
            color: #1a73e8;
            text-decoration: none;
        }
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h2 class='title'>🔎 SHL Assessment Recommender</h2>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Get smart test recommendations using your role or JD</p>", unsafe_allow_html=True)

# Input
query = st.text_input("What kind of assessment are you looking for?", placeholder="e.g. Python test under 30 mins, Sales JD")

if st.button("Find Recommendations"):
    if not query.strip():
        st.warning("Please enter a valid query.")
    else:
        with st.spinner("Finding best-matching SHL assessments..."):
            try:
                response = requests.post(
                    "http://127.0.0.1:8000/recommend",
                    json={"query": query, "top_k": 5}
                )

                if response.status_code == 200:
                    result_json = response.json()
                    results = result_json.get("recommended_assessments", [])

                    if results:
                        df = pd.DataFrame(results)

                        if 'duration' in df.columns:
                            df = df.rename(columns={"duration": "Duration (mins)"})
                        if 'description' in df.columns:
                            df = df.rename(columns={"description": "Description"})
                        if 'test_type' in df.columns:
                            df = df.rename(columns={"test_type": "Test Type"})
                        if 'remote_support' in df.columns:
                            df = df.rename(columns={"remote_support": "Remote Testing Support"})
                        if 'adaptive_support' in df.columns:
                            df = df.rename(columns={"adaptive_support": "Adaptive/IRT"})
                        if 'url' in df.columns:
                            df = df.rename(columns={"url": "URL"})

                        df['URL'] = df['URL'].apply(
                            lambda x: f"<a href='{x}' target='_blank'>View</a>" if pd.notna(x) else "")

                        display_cols = ["Description", "Test Type", "Remote Testing Support",
                                        "Adaptive/IRT", "Duration (mins)", "URL"]
                        df = df[[col for col in display_cols if col in df.columns]]

                        # Build and display HTML table
                        html = "<table class='minimal-table'><thead><tr>"
                        html += "".join([f"<th>{col}</th>" for col in df.columns]) + "</tr></thead><tbody>"

                        for _, row in df.iterrows():
                            html += "<tr>" + "".join([f"<td>{cell}</td>" for cell in row]) + "</tr>"

                        html += "</tbody></table>"

                        st.success("Top SHL assessments recommended:")
                        st.markdown(html, unsafe_allow_html=True)

                    else:
                        st.warning("No matching assessments found. Try modifying your query.")
                else:
                    st.error(f"API Error: {response.status_code} - {response.text}")

            except Exception as e:
                st.error(f"Something went wrong: {e}")
