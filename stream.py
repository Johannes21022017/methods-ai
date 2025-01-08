import os
import streamlit as st
import PyPDF2
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import openai
import glob
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from typing import Dict
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Fetch the OpenAI API key from the environment or Streamlit secrets
if 'OPENAI_API_KEY' in st.secrets:
    openai_api_key = st.secrets['OPENAI_API_KEY']
else:
    openai_api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is available
if not openai_api_key:
    st.error("OpenAI API key not found. Please set the OPENAI_API_KEY in your environment or Streamlit secrets.")
    st.stop()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = []
if "document_stores" not in st.session_state:
    st.session_state.document_stores = {}
if "current_doc_id" not in st.session_state:
    st.session_state.current_doc_id = None
if "doc_chat_history" not in st.session_state:
    st.session_state.doc_chat_history = []
if "current_document" not in st.session_state:
    st.session_state.current_document = None

# Initialize OpenAI embeddings and LLM
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7)

# Initialize textsplitter for documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=25,
    length_function=len,
)

# Function to extract text from PDF files
def extract_text_from_pdf(pdf_path):
    try:
        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error processing PDF: {str(e)}")
        return None

# Function to load initial PDFs from the directory
def load_initial_documents(directory_path):
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    documents = []
    for pdf_file in pdf_files:
        text = extract_text_from_pdf(pdf_file)
        if text:
            chunks = text_splitter.split_text(text)
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={"source": pdf_file, "chunk": i}
                )
                documents.append(doc)
    return documents

# Initialize FAISS VectorStore for document indexing
index_path = "faiss_index"
if os.path.exists(index_path):
    vector_store = FAISS.load_local(
        index_path, 
        embeddings,
        allow_dangerous_deserialization=True
    )
else:
    initial_directory = "/Users/ferdinandschweigert/Coding/research_rag/documents"  # Replace with the path to your documents
    documents = load_initial_documents(initial_directory)
    if documents:
        vector_store = FAISS.from_documents(documents, embeddings)
        vector_store.save_local(index_path)
    else:
        sample_doc = Document(page_content="This is a sample document to initialize the vector store.")
        vector_store = FAISS.from_documents([sample_doc], embeddings)

# Function to load and embed documents from a directory
def initialize_document_store(directory_path: str) -> Dict[str, FAISS]:
    document_stores = {}
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))
    
    for pdf_file in pdf_files:
        file_name = Path(pdf_file).name
        index_path = f"indexes/{file_name}_index"
        
        # Check if index already exists
        if os.path.exists(index_path):
            try:
                document_stores[file_name] = FAISS.load_local(
                    index_path,
                    embeddings,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded existing index for {file_name}")
            except Exception as e:
                print(f"Error loading index for {file_name}: {e}")
        else:
            try:
                # Process and embed the document
                text = extract_text_from_pdf(pdf_file)
                if text:
                    chunks = text_splitter.split_text(text)
                    documents = []
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={"source": file_name, "chunk": i}
                        )
                        documents.append(doc)
                    
                    # Create vector store
                    vector_store = FAISS.from_documents(documents, embeddings)
                    
                    # Save the index
                    os.makedirs("indexes", exist_ok=True)
                    vector_store.save_local(index_path)
                    
                    document_stores[file_name] = vector_store
                    print(f"Created new index for {file_name}")
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    
    return document_stores

# Modify the upload_document function
def upload_document():
    uploaded_file = st.file_uploader("Upload a PDF Document", type=["pdf"])
    if uploaded_file is not None:
        try:
            with st.spinner('Processing document...'):
                # Save and process the document
                with open("temp_uploaded_file.pdf", "wb") as f:
                    f.write(uploaded_file.getbuffer())
                document_text = extract_text_from_pdf("temp_uploaded_file.pdf")
                
                if document_text:
                    # Embed the document
                    chunks = text_splitter.split_text(document_text)
                    documents = []
                    for i, chunk in enumerate(chunks):
                        doc = Document(
                            page_content=chunk,
                            metadata={"source": uploaded_file.name, "chunk": i}
                        )
                        documents.append(doc)
                    
                    # Create vector store and save index
                    document_vector_store = FAISS.from_documents(documents, embeddings)
                    index_path = f"indexes/{uploaded_file.name}_index"
                    os.makedirs("indexes", exist_ok=True)
                    document_vector_store.save_local(index_path)
                    
                    # Add to document stores
                    st.session_state.document_stores[uploaded_file.name] = document_vector_store
                    
                    st.success(f"Document processed successfully! Created {len(chunks)} chunks.")
                
                os.remove("temp_uploaded_file.pdf")
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

# Modify the get_document_answer function to handle multiple documents
def get_document_answer(query, vector_store_or_stores):
    try:
        # Handle single or multiple document stores
        if isinstance(vector_store_or_stores, dict):
            # Combine retrievers from all stores
            retrievers = []
            for doc_id, store in vector_store_or_stores.items():
                retriever = store.as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": 3,  # Reduced per-document to avoid overwhelming
                        "fetch_k": 5,
                        "lambda_mult": 0.7
                    }
                )
                retrievers.append(retriever)
            
            # Merge results from all retrievers
            all_docs = []
            for retriever in retrievers:
                docs = retriever.get_relevant_documents(query)
                all_docs.extend(docs)
            
            # Create a new FAISS store from combined results
            combined_store = FAISS.from_documents(all_docs, embeddings)
            vector_store = combined_store
        else:
            vector_store = vector_store_or_stores

        # Use the selected prompt template
        current_prompt = PROMPT_TEMPLATES[selected_prompt]
        
        # Rest of the function remains similar, but use the selected prompt
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=vector_store.as_retriever()
        )

        # Create QA chain with selected prompt
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=compression_retriever,
            return_source_documents=True,
            combine_docs_chain_kwargs={
                "prompt": PromptTemplate(
                    template=PROMPT_TEMPLATES[selected_prompt],
                    input_variables=["context", "question", "chat_history", "current_document"]
                ),
                "document_prompt": PromptTemplate(
                    template="{page_content}",
                    input_variables=["page_content"]
                ),
                "document_variable_name": "context"
            }
        )
        
        # Get chat history text
        chat_history_text = "\n".join([
            f"User: {msg['user']}\nAssistant: {msg['assistant']}"
            for msg in st.session_state.doc_chat_history
        ])
        
        # Get response
        result = qa_chain({
            "question": query,
            "chat_history": [(msg["user"], msg["assistant"]) for msg in st.session_state.doc_chat_history],
            "current_document": "Multiple Documents" if isinstance(vector_store_or_stores, dict) else st.session_state.current_doc_id
        })
        
        return result["answer"], result.get("source_documents", [])
    except Exception as e:
        return f"Error generating response: {str(e)}", []

# First, let's define different prompt templates
GENERAL_PROMPT = """You are an expert research assistant with a PhD in multiple disciplines.

{context}

Current Document: {current_document}
Chat History: {chat_history}
Question: {question}

When responding:
1. Start with the document title and a brief description
2. Provide a clear summary of the main findings
3. Then elaborate with detailed analysis and evidence

Structure your response as follows:
- Title & Description: [Document title and brief context]
- Main Findings: [Key points and primary conclusions]
- Detailed Analysis: [In-depth discussion with evidence]
- Supporting Evidence: [Specific citations from the documents]

Focus on:
1. Accurate representation of document content
2. Clear organization of information
3. Explicit connections between documents when relevant
4. Academic rigor with accessible language
"""

ACADEMIC_PROMPT = """You are a scholarly research analyst.

{context}

Current Document: {current_document}
Chat History: {chat_history}
Question: {question}

Structure your response as follows:
1. Title and Abstract: Brief overview of relevant documents
2. Key Findings: Primary research outcomes
3. Methodology: How conclusions were reached
4. Discussion: Detailed analysis with citations
5. Implications: Significance of findings

Maintain strict academic rigor and cite all sources explicitly.
"""

SUMMARY_PROMPT = """You are a research summarizer.

{context}

Current Document: {current_document}
Chat History: {chat_history}
Question: {question}

Focus on:
1. Executive Summary (2-3 sentences)
2. Key Points (bullet points)
3. Main Conclusions
4. Evidence Summary

Keep responses concise and highlight the most important information.
"""

SYNTHESIS_PROMPT = """You are a research synthesizer.

{context}

Current Document: {current_document}
Chat History: {chat_history}
Question: {question}

Your role is to:
1. Identify common themes across documents
2. Compare and contrast findings
3. Highlight complementary insights
4. Note any contradictions
5. Suggest integrated conclusions

Focus on connecting information across multiple documents.
"""

FOOD_FORTIFICATION_PROMPT = """You are a food fortification and nutrition expert.

{context}

Current Document: {current_document}
Chat History: {chat_history}
Question: {question}

Analyze the documents to create a comprehensive food fortification strategy. Structure your response as follows:

Short Summary:
- Overview of the current food fortification landscape in the region
- Key challenges and opportunities
- Target populations and their needs

1. Current Situation Analysis:
   - Overview of existing fortification programs
   - Key challenges identified
   - Target populations and their needs

2. Strategic Recommendations:
   - Priority interventions
   - Implementation approach
   - Key stakeholders and their roles
   - Technical specifications for fortification

3. Implementation Framework:
   - Short-term actions (0-6 months)
   - Medium-term actions (6-18 months)
   - Long-term sustainability measures

4. Monitoring & Evaluation:
   - Key performance indicators
   - Quality control measures
   - Impact assessment methods

5. Risk Mitigation:
   - Potential challenges
   - Contingency measures
   - Success factors

When responding:
- Cite specific evidence from the documents
- Provide practical, actionable recommendations
- Consider local context and feasibility
- Include technical and operational details
- Address sustainability and scale-up potential
"""

# Create a dictionary of available prompts
PROMPT_TEMPLATES = {
    "General": GENERAL_PROMPT,
    "Academic": ACADEMIC_PROMPT,
    "Summary": SUMMARY_PROMPT,
    "Synthesis": SYNTHESIS_PROMPT,
    "Food Fortification Strategy": FOOD_FORTIFICATION_PROMPT
}

# Add a new function for document summarization
def summarize_document(document_text):
    try:
        # Create a prompt specifically for summarization
        summary_prompt = PromptTemplate(
            template=GENERAL_PROMPT + "\n\nDocument Content: {text}\n\nTask: Please provide a comprehensive summary of this document.\n\nSummary:",
            input_variables=["text"]
        )
        
        # Split the document into chunks if it's too large
        chunks = text_splitter.split_text(document_text)
        
        # Summarize each chunk and combine
        summaries = []
        for chunk in chunks:
            response = llm.predict(summary_prompt.format(text=chunk))
            summaries.append(response)
        
        # Combine the summaries
        final_summary = "\n\n".join(summaries)
        return final_summary
    except Exception as e:
        return f"Error summarizing document: {str(e)}"

# Update the Streamlit Interface
st.title("Document Q&A Assistant")

# Sidebar for document selection and upload
with st.sidebar:
    st.header("Document Management")
    
    # Initialize document stores if empty
    if not st.session_state.document_stores:
        docs_directory = "/Users/ferdinandschweigert/Coding/research_rag/documents"
        if os.path.exists(docs_directory):
            with st.spinner("Loading documents..."):
                st.session_state.document_stores = initialize_document_store(docs_directory)
                st.success(f"Loaded {len(st.session_state.document_stores)} documents")
    
    # Prompt selector
    selected_prompt = st.selectbox(
        "Select Analysis Style",
        options=list(PROMPT_TEMPLATES.keys()),
        index=0,
        help="Choose how you want to analyze the documents. Select 'Food Fortification Strategy' for specific food fortification recommendations."
    )
    
    # Document selector with "All Documents" option
    if st.session_state.document_stores:
        document_options = ["All Documents"] + list(st.session_state.document_stores.keys())
        selected_doc = st.selectbox(
            "Select document(s) to analyze",
            options=document_options,
            index=0,
            help="Select a specific document or 'All Documents' to analyze everything together"
        )
        
        if selected_doc != st.session_state.current_doc_id:
            st.session_state.current_doc_id = selected_doc
            st.session_state.doc_chat_history = []
            st.rerun()
        
        # Display currently loaded documents
        with st.expander("Currently Loaded Documents"):
            for doc in st.session_state.document_stores.keys():
                st.write(f"📄 {doc}")
    
    st.divider()
    st.header("Upload New Document")
    upload_document()

# Main chat interface
if st.session_state.current_doc_id:
    if st.session_state.current_doc_id == "All Documents":
        st.header("Analyzing All Documents")
        current_store = st.session_state.document_stores
    else:
        st.header(f"Analyzing: {st.session_state.current_doc_id}")
        current_store = st.session_state.document_stores[st.session_state.current_doc_id]
    
    # Display chat history
    for message in st.session_state.doc_chat_history:
        with st.chat_message("user"):
            st.markdown(message["user"])
        with st.chat_message("assistant"):
            st.markdown(message["assistant"])
            if message.get("sources"):
                with st.expander("View sources"):
                    for source in message["sources"]:
                        st.markdown(f"- From chunk {source.metadata.get('chunk', 'unknown')} of {source.metadata.get('source', 'unknown')}")
    
    # Question input
    if doc_question := st.chat_input("Ask a question about the selected document:"):
        with st.chat_message("user"):
            st.markdown(doc_question)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer, sources = get_document_answer(doc_question, current_store)
                st.markdown(answer)
                
                if sources:
                    with st.expander("View sources"):
                        for source in sources:
                            st.markdown(f"- From chunk {source.metadata.get('chunk', 'unknown')} of {source.metadata.get('source', 'unknown')}")
        
        # Add to chat history
        st.session_state.doc_chat_history.append({
            "user": doc_question,
            "assistant": answer,
            "sources": sources
        })
else:
    st.info("Please select a document from the sidebar to start chatting.")

# Clear chat history button
if st.session_state.doc_chat_history:
    if st.button("Clear Chat History"):
        st.session_state.doc_chat_history = []
        st.rerun()

def create_requirements():
    with st.spinner("Creating requirements.txt..."):
        command = "pip freeze > requirements.txt"
        os.system(command)

def create_gitignore():
    gitignore_content = """
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual Environment
.env
.venv
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Streamlit
.streamlit/
"""
    with open(".gitignore", "w") as f:
        f.write(gitignore_content)
