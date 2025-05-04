import os
import streamlit as st
import PyPDF2
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Neo4jVector
from langchain.graphs import Neo4jGraph
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
import tempfile
import traceback
import json
import pandas as pd
from datetime import datetime
from typing import List, Dict, Optional
import google.generativeai as genai

# Configuration
st.set_page_config(
    layout="wide", 
    page_title="Breast Cancer Knowledge Explorer",
    page_icon="logo.png",
    initial_sidebar_state="expanded"
)

load_dotenv()
NEO4J_URI = os.environ.get("NEO4J_URI", "neo4j+s://acbd0f4b.databases.neo4j.io")
NEO4J_USERNAME = os.environ.get("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "V21EnDQ01-wzu3ppYGhSvOElQ7ys6yf85qz3yeD88Ec")

ALLOWED_NODES = [
    "Patient", "Doctor", "Symptom", "Diagnosis", "Treatment", 
    "TumorType", "Stage", "Medication", "SideEffect", "TestResult",
    "Biomarker", "RiskFactor", "Procedure", "Hospital", "Document"
]
ALLOWED_RELATIONSHIPS = [
    "HAS_SYMPTOM", "DIAGNOSED_WITH", "TREATED_WITH", "PRESCRIBED", 
    "EXPERIENCED", "TREATED_BY", "PERFORMED_AT", "HAS_MARKER", 
    "HAS_RISK_FACTOR", "UNDERWENT", "SHOWS", "CONDUCTED_BY",
    "LEADS_TO", "FOLLOWED_BY", "REFERRED_TO", "MENTIONS"
]

def apply_custom_styles():
    """Apply modern UI styles and layout improvements"""
    st.markdown("""
    <style>
    :root {
        --primary: #e91e63;
        --secondary: #6c757d;
        --light: #f8f9fa;
        --dark: #343a40;
        --success: #28a745;
    }
    
    /* Main layout */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Typography */
    h1, h2, h3, h4 {
        color: var(--primary);
        margin-bottom: 0.75rem;
        font-weight: 600;
    }
    
    h1 {
        font-size: 2.2rem;
        border-bottom: 2px solid rgba(233, 30, 99, 0.2);
        padding-bottom: 0.5rem;
    }
    
    h2 {
        font-size: 1.8rem;
    }
    
    h3 {
        font-size: 1.5rem;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: var(--primary);
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1.2rem;
        font-weight: 500;
        transition: all 0.3s;
        border: none;
        box-shadow: 0 2px 5px rgba(233, 30, 99, 0.3);
    }
    
    .stButton>button:hover {
        background-color: #d81b60;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(233, 30, 99, 0.4);
    }
    
    /* Input fields */
    .stTextArea textarea, 
    .stTextInput>div>div>input,
    .stFileUploader label span,
    .stFileUploader span {
        border-radius: 10px;
        border: 1px solid #e0e0e0;
        padding: 0.8rem;
        transition: all 0.3s;
    }
    
    .stTextArea textarea:focus, 
    .stTextInput>div>div>input:focus {
        border-color: var(--primary);
        box-shadow: 0 0 0 2px rgba(233, 30, 99, 0.2);
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        font-weight: 500;
        color: #424242;
        background-color: #fafafa;
        border-radius: 10px;
        padding: 0.7rem 1rem;
        border: 1px solid #e0e0e0;
    }
    
    .streamlit-expanderContent {
        border: 1px solid #e0e0e0;
        border-top: none;
        border-radius: 0 0 10px 10px;
        padding: 1rem;
    }
    
    /* Cards */
    .card {
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
        padding: 1.8rem;
        margin-bottom: 1.5rem;
        background: white;
        border: none;
        transition: all 0.3s;
    }
    
    .card:hover {
        box-shadow: 0 6px 18px rgba(0,0,0,0.12);
        transform: translateY(-2px);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fcf3f6 0%, #fde8f0 100%);
        padding: 1.5rem;
        border-right: 1px solid rgba(233, 30, 99, 0.1);
    }
    
    [data-testid="stSidebar"] h3 {
        margin-top: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid rgba(233, 30, 99, 0.2);
    }
    
    /* Status indicators */
    .metric-container {
        background-color: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    
    /* Footer styling */
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        text-align: center;
        font-size: 0.9rem;
        color: #666;
        border-top: 1px solid #eee;
    }
    
    .footer a {
        color: var(--primary);
        text-decoration: none;
    }
    
    .footer a:hover {
        text-decoration: underline;
    }
    
    /* Pink breast cancer ribbon */
    .pink-ribbon {
        display: inline-block;
        margin-right: 10px;
        opacity: 0.9;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px 10px 0 0;
        padding: 10px 20px;
        background-color: #f8f8f8;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: white !important;
        border-top: 2px solid var(--primary) !important;
        border-left: 1px solid #e0e0e0 !important;
        border-right: 1px solid #e0e0e0 !important;
    }
    
    /* Code blocks */
    code {
        color: var(--primary);
        background-color: #f8f8f8;
        padding: 0.2em 0.4em;
        border-radius: 4px;
        border: 1px solid #eaeaea;
    }
    
    .stMarkdown pre {
        border-radius: 10px;
    }
    
    </style>
    """, unsafe_allow_html=True)


def initialize_session_state():
    """Initialize all required session state variables"""
    session_vars = {
        'GOOGLE_API_KEY': None,
        'embeddings': None,
        'llm': None,
        'graph': None,
        'neo4j_connected': False,
        'qa': None,
        'data_initialized': False,
        'parsed_documents': [],
        'last_cypher_query': None,
        'citations': [],
        'pdf_content': None,
        'pdf_uploaded': False
    }
    
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default

def setup_google_ai(api_key):
    """Initialize Google AI components with validation"""
    try:
        os.environ['GOOGLE_API_KEY'] = api_key
        
        # Initialize embeddings
        st.session_state['embeddings'] = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # Initialize LLM
        st.session_state['llm'] = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-04-17",
            temperature=0,
            google_api_key=api_key,
            max_retries=3
        )
        
        # Configure the direct Google GenAI library as well
        genai.configure(api_key=api_key)
        
        st.session_state['GOOGLE_API_KEY'] = api_key
        return True
    except Exception as e:
        display_error(e, "Google AI initialization failed")
        return False

def connect_neo4j():
    """Establish connection to Neo4j with error handling"""
    try:
        graph = Neo4jGraph(
            url=NEO4J_URI,
            username=NEO4J_USERNAME,
            password=NEO4J_PASSWORD,
            refresh_schema=True
        )
        st.session_state['graph'] = graph
        st.session_state['neo4j_connected'] = True
        
        # Verify connection by running a simple query
        result = graph.query("RETURN 1 AS test")
        if result and len(result) > 0:
            st.success("Neo4j connection verified!")
        return True
    except Exception as e:
        display_error(e, "Neo4j connection failed")
        return False

def setup_qa_chain():
    """Initialize the QA chain with schema validation"""
    try:
        if not st.session_state.get('graph'):
            raise ValueError("No graph connection established")
        
        # First, ensure we can get the schema
        schema = st.session_state['graph'].get_schema
        
        examples = [
            {
                "question": "How many different procedures can lead to operable breast cancer diagnosis?",
                "query": "MATCH (p:Procedure)-[:LEADS_TO]->(d:Diagnosis) WHERE d.id = 'Operable Breast Cancer' RETURN count(DISTINCT p) AS number_of_proc"
            },
            {
                "question": "What are different treatments?",
                "query": "MATCH (n:Treatment) RETURN n.id, n.name LIMIT 25"
            },
            {
                "question": "What procedures might lead to a mastectomy?",
                "query": "MATCH (t:TestResult)-[:LEADS_TO]->(p:Procedure) WHERE p.id = 'Mastectomy' RETURN t.id AS test_result, p.id AS procedure"
            },
            {
                "question": "What are the treatments used?",
                "query": "MATCH p=()-[:TREATED_WITH]->(t) RETURN t.id AS treatment, t.name AS name LIMIT 25"
            },
            {
                "question": "What symptoms are associated with breast cancer?",
                "query": "MATCH (d:Diagnosis)-[:HAS_SYMPTOM]->(s:Symptom) WHERE d.id = 'Breast Cancer' RETURN d.id AS diagnosis, s.id AS symptom"
            },
            {
                "question": "What treatment leads to interstitial brachytherapy procedure?",
                "query": "MATCH (t:Treatment)-[:LEADS_TO]->(p:Procedure) WHERE p.id = 'Interstitial Brachytherapy' RETURN t.id AS treatment, p.id AS procedure"
            }
        ]
        
        example_prompt = PromptTemplate.from_template(
            "User input: {question}\nCypher query: {query}"
        )
        
        # Create a more robust prompt with better error handling guidance
        prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="""
You are a Neo4j expert. Given an input question, create a syntactically correct Cypher query to run against a medical knowledge graph.

Here is the schema information:
{schema}

IMPORTANT: 
1. ALWAYS use property access with caution - use "n.id" only if you're sure the property exists or use "exists(n.id)" in WHERE clauses
2. ALWAYS specify properties in the RETURN clause (NEVER return just node objects)
3. ALWAYS include LIMIT for open-ended queries
4. ALWAYS use explicit aliasing in RETURN statements

Follow these examples carefully for correct syntax:
""",
            suffix="User input: {question}\nCypher query:",
            input_variables=["question", "schema"],
        )
        
        # Create a custom class to capture the last query
        class CaptureQueryGraphCypherQAChain(GraphCypherQAChain):
            def _call(self, inputs):
                result = super()._call(inputs)
                if hasattr(self, 'last_generated_cypher'):
                    st.session_state['last_cypher_query'] = self.last_generated_cypher
                return result
        
        # Create the QA chain with the new class
        qa_chain = CaptureQueryGraphCypherQAChain.from_llm(
            llm=st.session_state['llm'],
            graph=st.session_state['graph'],
            cypher_prompt=prompt,
            verbose=True,
            validate_cypher=True,
            top_k=20,
            max_retries=3,
            return_direct=False,
            allow_dangerous_requests=True
        )
        
        st.session_state['qa'] = qa_chain
        st.session_state['data_initialized'] = True
        return True
    except Exception as e:
        display_error(e, "QA chain setup failed")
        return False

def extract_text_from_pdf(file):
    """Extract text from the uploaded PDF file."""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text_data = []
        for page in pdf_reader.pages:
            text_data.append(page.extract_text())
        full_text = "\n".join(text_data)
        return full_text
    except Exception as e:
        display_error(e, "PDF extraction failed")
        return None

def get_pdf_based_answer(pdf_content, question):
    """Generate an answer to the user's question based on PDF content."""
    input_prompt = f"""
    You are a breast cancer medical expert. The following is the extracted text from a medical document about breast cancer:
    {pdf_content}
    
    Based on this document, answer the question: {question}
    
    If the document doesn't contain relevant information to answer the question, say "The document doesn't provide specific information about this query."
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([input_prompt])
        return response.text
    except Exception as e:
        return f"Error in generating response: {str(e)}"

def is_answer_empty_or_unknowing(answer):
    """Check if the knowledge graph answer is empty or expresses lack of knowledge."""
    empty_indicators = [
        "i don't know", 
        "i do not know",
        "no information", 
        "no data", 
        "cannot find",
        "could not find",
        "no results",
        "doesn't contain",
        "does not contain",
        "no relevant",
        "nothing found",
        "unable to find",
        "no answer"
    ]
    
    if not answer or answer.strip() == "":
        return True
    
    lower_answer = answer.lower()
    for indicator in empty_indicators:
        if indicator in lower_answer:
            return True
            
    return False

def display_graph_stats():
    """Display comprehensive graph statistics"""
    try:
        st.subheader("Current Knowledge Graph Status")
        
        # Node statistics
        node_query = """
        MATCH (n)
        WITH labels(n) AS labels
        UNWIND labels AS label
        RETURN label, count(*) AS count
        ORDER BY count DESC
        """
        node_counts = st.session_state['graph'].query(node_query)
        
        # Relationship statistics
        rel_query = """
        MATCH ()-[r]->()
        RETURN type(r) AS type, count(*) AS count
        ORDER BY count DESC
        """
        rel_counts = st.session_state['graph'].query(rel_query)
        
        # Display in tabs
        tab1, tab2 = st.tabs(["Nodes", "Relationships"])
        
        with tab1:
            if node_counts:
                st.dataframe(node_counts, hide_index=True)
            else:
                st.info("No nodes found in database")
        
        with tab2:
            if rel_counts:
                st.dataframe(rel_counts, hide_index=True)
            else:
                st.info("No relationships found in database")
                
    except Exception as e:
        st.warning(f"Couldn't retrieve graph statistics: {str(e)}")

def process_uploaded_file(uploaded_file):
    """Process and add data from uploaded file"""
    with st.spinner(f"Processing {uploaded_file.name}..."):
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            # Load document
            loader = TextLoader(tmp_path)
            documents = loader.load()
            
            # Split into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = text_splitter.split_documents(documents)
            
            # Clean documents
            cleaned_docs = [
                Document(
                    page_content=doc.page_content.replace("\n", " ").strip(), 
                    metadata={'source': uploaded_file.name}
                ) for doc in docs if doc.page_content.strip()
            ]
            
            # Track parsed documents
            doc_info = {
                'name': uploaded_file.name,
                'chunks': len(cleaned_docs),
                'timestamp': str(datetime.now())
            }
            st.session_state['parsed_documents'].append(doc_info)
            
            # Transform to graph
            transformer = LLMGraphTransformer(
                llm=st.session_state['llm'],
                allowed_nodes=ALLOWED_NODES,
                allowed_relationships=ALLOWED_RELATIONSHIPS,
                node_properties=True,
                relationship_properties=False
            )
            
            # Custom prompt that encourages consistent property usage
            transformer.llm_kwargs = {
                "prompt": """
Extract detailed breast cancer information from this text:
{text}

Focus on:
- Patients: id, name, age, gender, medical history
- Symptoms: type, severity, duration
- Diagnoses: id, type, stage, grade, biomarkers
- Treatments: id, name, type, duration, response
- Medications: id, name, dosage, frequency, side effects
- Procedures: id, type, date, outcomes
- Medical professionals: id, name, specialty
- Facilities: id, name, location

IMPORTANT: Always include an 'id' property for every node. This is required for proper queries.
Create nodes with all available properties.
Establish relationships with context from the text.
"""
            }
            
            graph_docs = transformer.convert_to_graph_documents(cleaned_docs)
            st.session_state['graph'].add_graph_documents(graph_docs)
            
            # Refresh QA chain to include new data
            setup_qa_chain()
            
            os.unlink(tmp_path)
            return True
        except Exception as e:
            display_error(e, "File processing failed")
            return False

def display_answer_with_sources(answer, sources=None, source_type="Knowledge Graph"):
    """Display answer with expandable details and sources"""
    with st.container():
        st.markdown(f"""
        <div class="card">
            <h4>📌 Answer ({source_type})</h4>
            <p>{answer}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if sources:
            with st.expander("🔍 View Sources & References", expanded=False):
                for i, source in enumerate(sources, 1):
                    st.markdown(f"""
                    **Source {i}**
                    - **Document:** {source.get('document', 'N/A')}
                    - **Page:** {source.get('page', 'N/A')}
                    - **Confidence:** {source.get('confidence', 'N/A')}
                    """)
                    if 'excerpt' in source:
                        st.markdown(f"*Excerpt:* {source['excerpt']}")
                    st.divider()

def track_citations(question, answer, cypher_query=None, documents_used=None, source_type="Knowledge Graph"):
    """Track and store citations for queries"""
    citation = {
        "timestamp": datetime.now().isoformat(),
        "question": question,
        "answer": answer,
        "source_type": source_type
    }
    
    if source_type == "Knowledge Graph" and cypher_query:
        citation["cypher_query"] = cypher_query
        
    if documents_used:
        citation["sources"] = [{"document": doc} for doc in documents_used]
    else:
        citation["sources"] = []
    
    if 'citations' not in st.session_state:
        st.session_state.citations = []
    
    st.session_state.citations.append(citation)
    return citation

def display_error(error, context=""):
    """Display user-friendly error messages"""
    error_msg = f"""
    <div style="
        border-radius: 10px;
        padding: 1rem;
        background-color: #fff5f5;
        border-left: 4px solid #ff4b4b;
        margin-bottom: 1rem;
        box-shadow: 0 2px 8px rgba(255, 75, 75, 0.2);
    ">
        <h4 style="color: #ff4b4b; margin-top: 0;">⚠️ Error: {context}</h4>
        <p>{str(error)}</p>
    </div>
    """
    st.markdown(error_msg, unsafe_allow_html=True)
    
    if st.checkbox("Show technical details (for support)"):
        with st.expander("Technical Details"):
            st.code(traceback.format_exc())

def main():
    # Apply modern styles
    apply_custom_styles()
    
    # Initialize session state
    initialize_session_state()
    
    # Modern sidebar layout
    with st.sidebar:
        st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <h2 style="margin: 0; padding: 0;">Breast Cancer Knowledge Explorer</h2>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("ℹ️ About", expanded=False):
            st.markdown("""
            **Clinical Knowledge Graph Explorer**
            - Query breast cancer data using natural language
            - Add clinical notes to expand the knowledge base
            - Upload PDFs for more comprehensive answers
            - Discover patterns and insights in breast cancer data
            - Powered by Neo4j & Google AI technology
            
            For comprehensive information about breast cancer:
            [Indian Cancer Society](https://www.indiancancersociety.org/breast-cancer/index.html)
            """)
        
        # Status indicators
        st.markdown("### System Status")
        
        status_col1, status_col2 = st.columns(2)
        with status_col1:
            conn_status = "Connected" if st.session_state.neo4j_connected else "Disconnected"
            conn_icon = "✅" if st.session_state.neo4j_connected else "❌"
            st.markdown(f"**Neo4j:** {conn_icon} {conn_status}")
        with status_col2:
            ai_status = "Ready" if st.session_state.llm else "Offline"
            ai_icon = "✅" if st.session_state.llm else "❌"
            st.markdown(f"**Google AI:** {ai_icon} {ai_status}")
        
        # Quick actions
        st.markdown("### Quick Actions")
        if st.button("🔄 Refresh Connection"):
            connect_neo4j()
        
        # Database Management
        with st.expander("🛠️ Database Tools", expanded=False):
            if st.button("Initialize QA System"):
                with st.spinner("Setting up query system..."):
                    if setup_qa_chain():
                        st.success("✅ Ready to query!")
                    else:
                        st.error("❌ Initialization failed")
    
    # Main content area with modern layout
    st.image("logo.png", width=100)
    st.title("Breast Cancer Clinical Knowledge Explorer")
    
    st.markdown("""
    <p style="font-size: 1.1rem; color: #555; margin-bottom: 2rem;">
        Explore breast cancer insights through natural language queries to our clinical knowledge graph.
    </p>
    """, unsafe_allow_html=True)
    
    
    # API Key Section with improved UX
    if not st.session_state['GOOGLE_API_KEY']:
        with st.container():
            st.markdown("""
            <div class="card" style="border-left: 4px solid #ff9800;">
                <h3 style="color: #ff9800;">🔑 API Key Required</h3>
                <p>To use this application, you need to provide a Google API key with access to Gemini models.</p>
            </div>
            """, unsafe_allow_html=True)
            
            with st.form("api_key_form"):
                google_api_key = st.text_input("Enter Google API Key:", type="password", 
                                              placeholder="AIza...")
                col1, col2 = st.columns([1, 3])
                with col1:
                    submit_button = st.form_submit_button("Set API Key")
                with col2:
                    if submit_button:
                        if setup_google_ai(google_api_key):
                            st.success("✅ Google AI configured successfully!")
                        else:
                            st.error("❌ Invalid API Key")
            return
    
    # Neo4j Connection with better feedback
    if not st.session_state['neo4j_connected']:
        with st.container():
            st.markdown("""
            <div class="card" style="border-left: 4px solid #ff9800;">
                <h3 style="color: #ff9800;">🔌 Database Connection Required</h3>
                <p>The knowledge graph database needs to be connected to proceed.</p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Connect to Neo4j Database"):
                if connect_neo4j():
                    st.success("✅ Connected to Neo4j knowledge graph database!")
                else:
                    st.error("❌ Connection failed. Please check your credentials and network.")
            return
    
    # Main app functionality with modern cards
    tab1, tab2 = st.tabs(["Query System", "Knowledge Management"])
    
    with tab1:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # PDF Upload Section
            st.subheader("📄 Upload Reference PDF")
            uploaded_pdf = st.file_uploader(
                "Upload medical PDF document", 
                type=["pdf"],
                key="pdf_uploader"
            )
            
            if uploaded_pdf:
                st.info(f"PDF selected: {uploaded_pdf.name}")
                
                if st.button("Process PDF", key="process_pdf"):
                    with st.spinner("Extracting text from PDF..."):
                        pdf_text = extract_text_from_pdf(uploaded_pdf)
                        if pdf_text:
                            st.session_state['pdf_content'] = pdf_text
                            st.session_state['pdf_uploaded'] = True
                            st.success(f"✅ PDF processed successfully: {len(pdf_text.split())} words extracted")
                            with st.expander("View extracted text", expanded=False):
                                st.text_area("PDF Content", pdf_text, height=200)
            
            # Graph Statistics
            display_graph_stats()
            
            # Query section with enhanced UI
            with st.container():
                st.markdown("""
                <h3 style="display: flex; align-items: center; margin-top: 2rem;">
                    <span style="font-size: 1.5rem; margin-right: 10px;">🔍</span>
                    Knowledge Explorer Query
                </h3>
                """, unsafe_allow_html=True)
                
                st.markdown("""
                <div class="card" style="background-color: #fcf3f6; border-left: 4px solid #e91e63; padding: 1.5rem;">
                    <p style="margin-bottom: 1rem;">Ask questions about breast cancer diagnosis, treatments, procedures, symptoms, and more.</p>
                </div>
                """, unsafe_allow_html=True)
                
                question = st.text_area(
                    "Ask your question here:",
                    placeholder="Example: What symptoms are associated with breast cancer?",
                    height=100
                )
                
                search_button = st.button("Search Knowledge Base")
                
                # Process query when button is clicked
                if search_button and question:
                    if not st.session_state['qa']:
                        if setup_qa_chain():
                            st.success("QA system initialized successfully!")
                        else:
                            st.error("Failed to initialize QA system. Please check connection.")
                            return
                    
                    try:
                        with st.spinner("Searching knowledge graph..."):
                            # Query the knowledge graph
                            kg_result = st.session_state['qa'].run(question)
                            kg_query = st.session_state.get('last_cypher_query', "Query not captured")
                            
                            # Check if we should also query the PDF
                            pdf_result = None
                            if (st.session_state.get('pdf_uploaded') and 
                                st.session_state.get('pdf_content') and
                                is_answer_empty_or_unknowing(kg_result)):
                                
                                with st.spinner("Knowledge graph inconclusive. Checking reference PDF..."):
                                    pdf_result = get_pdf_based_answer(
                                        st.session_state['pdf_content'], 
                                        question
                                    )
                            
                            # Display knowledge graph result
                            display_answer_with_sources(kg_result, source_type="Knowledge Graph")
                            
                            # If we have a PDF result, show it
                            if pdf_result:
                                display_answer_with_sources(pdf_result, source_type="PDF Reference")
                            
                            # Store citation
                            track_citations(
                                question=question,
                                answer=kg_result if not pdf_result else f"KG: {kg_result}\nPDF: {pdf_result}",
                                cypher_query=kg_query,
                                source_type="Combined" if pdf_result else "Knowledge Graph"
                            )
                            
                            # Show Cypher query for educational purposes
                            # with st.expander("View Neo4j Cypher Query", expanded=False):
                            #     st.code(kg_query, language="cypher")
                    
                    except Exception as e:
                        display_error(e, "Query execution failed")
        
        with col2:
            # Query history card with a clean look
            st.markdown("""
            <div class="card">
                <h4 style="display: flex; align-items: center; margin-top: 0;">
                    <span style="font-size: 1.2rem; margin-right: 8px;">📜</span>
                    Recent Queries
                </h4>
            """, unsafe_allow_html=True)
            
            if 'citations' in st.session_state and st.session_state.citations:
                for i, citation in enumerate(reversed(st.session_state.citations[-5:])):
                    q_time = datetime.fromisoformat(citation['timestamp']).strftime("%H:%M:%S")
                    st.markdown(f"""
                    <div style="margin-bottom: 1rem; padding-bottom: 0.5rem; border-bottom: 1px solid #eee;">
                        <p style="font-size: 0.9rem; color: #666; margin-bottom: 0.3rem;">{q_time}</p>
                        <p style="font-weight: 500; margin-bottom: 0.3rem;">{citation['question']}</p>
                        <p style="font-size: 0.9rem; color: #888;">Source: {citation['source_type']}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <p style="color: #888; font-style: italic;">No queries yet. Ask your first question!</p>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Quick reference card
            st.markdown("""
            <div class="card" style="margin-top: 1rem;">
                <h4 style="display: flex; align-items: center; margin-top: 0;">
                    <span style="font-size: 1.2rem; margin-right: 8px;">💡</span>
                    Example Questions
                </h4>
                <ul style="padding-left: 1rem; margin-bottom: 0;">
                    <li>How many different procedures can lead to operable breast cancer diagnosis?</li>
                    <li>What treatments are available  breast cancer?</li>
                    <li>Total number of medications</li>
                    <li>List all the risk factors associated </li>
                    <li>What procedures might lead to a mastectomy?</li>
                    <li>How many tumor types are there and list them?</li>
                    <li>what are types of ADJUVANT RADIOTHERAPY ?</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("""
        <h3 style="display: flex; align-items: center; margin-bottom: 1.5rem;">
            <span style="font-size: 1.5rem; margin-right: 10px;">📊</span>
            Knowledge Base Management
        </h3>
        """, unsafe_allow_html=True)
        
        # Add data section with improved styling
        with st.container():
            st.markdown("""
            <div class="card">
                <h4 style="margin-top: 0;">Add Clinical Data</h4>
                <p>Upload clinical notes, research papers, or patient data to expand the knowledge graph.</p>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader(
                "Upload clinical text data (TXT format only)", 
                type=["txt"],
                key="data_uploader"
            )
            
            if uploaded_file:
                if st.button("Process and Add to Knowledge Graph"):
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        if process_uploaded_file(uploaded_file):
                            st.success(f"✅ Successfully added data from {uploaded_file.name} to knowledge graph")
                        else:
                            st.error("❌ Failed to process file")
            
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display existing document list in a more visual format
            if st.session_state.get('parsed_documents'):
                st.markdown("""
                <div class="card" style="margin-top: 1.5rem;">
                    <h4 style="margin-top: 0;">Knowledge Source Documents</h4>
                """, unsafe_allow_html=True)
                
                for i, doc in enumerate(st.session_state['parsed_documents']):
                    st.markdown(f"""
                    <div style="
                        display: flex;
                        justify-content: space-between;
                        padding: 0.8rem;
                        background-color: #f8f9fa;
                        border-radius: 8px;
                        margin-bottom: 0.8rem;
                    ">
                        <div>
                            <p style="font-weight: 500; margin-bottom: 0.2rem;">{doc['name']}</p>
                            <p style="font-size: 0.8rem; color: #666;">Added: {doc['timestamp']}</p>
                        </div>
                        <div style="align-self: center;">
                            <span style="
                                background-color: #e91e63;
                                color: white;
                                padding: 0.2rem 0.5rem;
                                border-radius: 12px;
                                font-size: 0.8rem;
                            ">{doc['chunks']} chunks</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
    
    # Bottom footer
    st.markdown("""
    <div class="footer">
        <p>Breast Cancer Knowledge Explorer | Developed for educational purposes | All data is sourced from medical literature</p>
        <a href="https://www.ncgindia.org/assets/ncg-guidelines-2019/ncg-guidelines-for-breast-cancer-2019.pdf" target="_blank">
            NCG Guidelines for Breast Cancer 2019
        </a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()