import streamlit as st
import pandas as pd
import groq
from io import StringIO
import os

# Set page config
st.set_page_config(
    page_title="File-Based Q&A Assistant",
    page_icon="📄",
    layout="wide"
)

# Initialize Groq client with better error handling
@st.cache_resource
def init_groq_client(api_key):
    if not api_key:
        return None
    try:
        return groq.Groq(api_key=api_key)
    except Exception as e:
        st.error(f"Error initializing Groq client: {str(e)}")
        return None

def get_groq_api_key():
    """Get API key from various sources"""
    # Try to get from environment variable first
    api_key = os.getenv("GROQ_API_KEY")
    
    # If not found in env, try streamlit secrets (with error handling)
    if not api_key:
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except:
            api_key = None
    
    return api_key

def read_file_content(uploaded_file):
    """Read and return file content based on file type"""
    try:
        if uploaded_file.type == "text/plain":
            # Read TXT file
            stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
            content = stringio.read()
            return content, "txt"
        
        elif uploaded_file.type == "text/csv" or uploaded_file.name.endswith('.csv'):
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            content = df.to_string(index=False)
            return content, "csv"
        
        else:
            return None, None
    except Exception as e:
        st.error(f"Error reading file: {str(e)}")
        return None, None

def get_answer_from_groq(question, file_content, model_name, client):
    """Get answer from Groq API based on file content"""
    if not client:
        return "Error: Groq client not initialized. Please check your API key."
    
    prompt = f"""
    You are an AI assistant that answers questions based ONLY on the provided file content. 
    
    IMPORTANT RULES:
    1. Only answer questions using information that exists in the provided file content
    2. If the answer is not in the file content, clearly state "I cannot find this information in the uploaded file"
    3. Do not provide information from your general knowledge
    4. Be specific and reference the relevant parts of the file when answering
    
    FILE CONTENT:
    {file_content}
    
    QUESTION: {question}
    
    Please provide an answer based only on the file content above.
    """
    
    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based only on provided file content."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1000,
            temperature=0.1
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error getting response: {str(e)}"

def main():
    st.title("📄 File-Based Q&A Assistant")
    st.markdown("Upload a TXT or CSV file and ask questions about its content!")
    
    # API Key Input Section
    st.sidebar.title("🔑 API Configuration")
    
    # Get API key
    api_key = get_groq_api_key()
    
    if not api_key:
        st.sidebar.warning("⚠️ No API key found!")
        api_key = st.sidebar.text_input(
            "Enter your Groq API Key:",
            type="password",
            help="Get your API key from https://console.groq.com/"
        )
    else:
        st.sidebar.success("✅ API key loaded!")
        # Option to override the API key
        if st.sidebar.checkbox("Use different API key"):
            api_key = st.sidebar.text_input(
                "Enter your Groq API Key:",
                type="password",
                help="Get your API key from https://console.groq.com/"
            )
    
    if not api_key:
        st.error("🚫 Please provide your Groq API key to continue.")
        st.info("💡 You can either:")
        st.info("1. Set GROQ_API_KEY environment variable")
        st.info("2. Create .streamlit/secrets.toml with GROQ_API_KEY")
        st.info("3. Enter it in the sidebar")
        st.stop()
    
    # Initialize Groq client
    client = init_groq_client(api_key)
    if not client:
        st.error("❌ Failed to initialize Groq client. Please check your API key.")
        st.stop()
    
    # Model selection
    st.sidebar.title("⚙️ Settings")
    
    # Available Groq models
    groq_models = [
        "llama3-8b-8192",
        "llama3-70b-8192", 
        "mixtral-8x7b-32768",
        "gemma-7b-it",
        "gemma2-9b-it"
    ]
    
    selected_model = st.sidebar.selectbox(
        "Select Groq Model:",
        groq_models,
        index=0
    )
    
    # File upload section
    st.header("📁 Upload Your File")
    uploaded_file = st.file_uploader(
        "Choose a TXT or CSV file",
        type=['txt', 'csv'],
        help="Upload a text file (.txt) or CSV file (.csv) to ask questions about its content"
    )
    
    if uploaded_file is not None:
        # Display file info
        st.success(f"✅ File uploaded: {uploaded_file.name}")
        st.info(f"📊 File size: {uploaded_file.size} bytes")
        
        # Read file content
        with st.spinner("Reading file content..."):
            file_content, file_type = read_file_content(uploaded_file)
        
        if file_content is not None:
            # Display file preview
            with st.expander("📖 Preview File Content", expanded=False):
                if file_type == "csv":
                    # Re-read for display
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file)
                    st.dataframe(df.head(10))
                    st.info(f"Showing first 10 rows. Total rows: {len(df)}")
                else:
                    st.text_area("File Content", file_content[:1000] + "..." if len(file_content) > 1000 else file_content, height=200)
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Chat interface
            st.header("💬 Ask Questions About Your File")
            
            # Display chat history
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Chat input
            if prompt := st.chat_input("Ask a question about your uploaded file..."):
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Get AI response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = get_answer_from_groq(prompt, file_content, selected_model, client)
                    st.markdown(response)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Alternative input method with button
            st.markdown("---")
            st.subheader("💬 Alternative Input Method")
            col1, col2 = st.columns([4, 1])
            
            with col1:
                question = st.text_input("Type your question here:", key="question_input")
            
            with col2:
                ask_button = st.button("Ask", type="primary")
            
            if ask_button and question:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": question})
                
                # Get and display AI response
                with st.spinner("Getting answer from your file..."):
                    response = get_answer_from_groq(question, file_content, selected_model, client)
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Clear the input and rerun
                st.rerun()
            
            # Clear chat button
            if st.sidebar.button("🗑️ Clear Chat History"):
                st.session_state.messages = []
                st.rerun()
        
        else:
            st.error("❌ Could not read the file. Please make sure it's a valid TXT or CSV file.")
    
    else:
        st.info("👆 Please upload a TXT or CSV file to get started!")
        
        # Show example questions
        with st.expander("💡 Example Questions You Can Ask"):
            st.markdown("""
            Once you upload a file, you can ask questions like:
            - "How many records are there?"
            - "What information is available?"
            - "List all unique values in column X"
            - "What is the average/total of column Y?"
            - "Show me records where condition Z is met"
            """)

if __name__ == "__main__":
    main()
