import pandas as pd
import os
import streamlit as st
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
from pandasai.responses.response_parser import ResponseParser
import io
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()

# Initialize the language model
llm = ChatGroq(model_name="llama3-70b-8192", api_key=os.environ["GROQ_API_KEY"])

def load_data(file_path) -> pd.DataFrame:
    """Load data from the local CSV file."""
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        return data
    else:
        st.error(f"File not found: {file_path}")
        return None

def save_plot_to_buffer(fig) -> io.BytesIO:
    """Save the Matplotlib figure to an in-memory buffer."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf

class StreamlitResponse(ResponseParser):
    """Custom Response Parser to handle Streamlit visualization and output formatting."""

    def format_dataframe(self, result):
        """Format and display a dataframe result."""
        st.dataframe(result["value"])

    def format_plot(self, result):
        """Format and display a plot."""
        if isinstance(result["value"], plt.Figure):
            # Save Matplotlib figure to buffer and display
            buf = save_plot_to_buffer(result["value"])
            st.image(buf)
        else:
            st.write("No plot returned.")

    def format_other(self, result):
        """Format and display other types of results."""
        st.write(result["value"])

# Streamlit app title
st.write("# Chat with CSV Data 🦙")

# Set the local file path of the CSV file
csv_file_path = "ai4i2020.csv"  # Replace with the path to your local CSV file

# Load the dataframe from the local file
df = load_data(csv_file_path)

if df is not None:
    # st.write("Data Columns:", df.columns)  # Display the columns for debugging
    # st.write(df)

    # Form to capture user input and submit
    with st.form(key="query_form"):
        query = st.text_area("🗣️ Chat with Dataframe", key="query_input")
        submit_button = st.form_submit_button(label="Generate")

    # Process the query when the form is submitted
    if submit_button and query:
        with st.spinner("Generating response..."):
            # Create a SmartDataframe object
            query_engine = SmartDataframe(df, config={"llm": llm, "response_parser": StreamlitResponse})

            # Get the answer from the query engine
            answer = query_engine.chat(query)

            # Debugging: Display the answer content
            # st.write("Answer:", answer)

            # Display the answer (whether it's a dataframe, plot, or text)
            if answer:
                st.write("Answer:", answer)
            else:
                st.write("No result returned. Please check your query.")
