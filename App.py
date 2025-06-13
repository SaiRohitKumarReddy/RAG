# CRITICAL: Set environment variables BEFORE any imports
import os
import sys

# Disable Streamlit's file watcher completely
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "false"

# Set event loop policy for Windows/Linux compatibility
import asyncio
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
elif hasattr(asyncio, 'DefaultEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

# Import streamlit FIRST before any heavy libraries
import streamlit as st

# Set page config immediately after streamlit import
st.set_page_config(
    page_title="Advanced PDF Analyzer", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Now import other libraries in order of importance
import tempfile
import io
import traceback
import numpy as np
from PIL import Image

# Lazy imports to avoid conflicts
@st.cache_resource
def get_torch():
    """Lazy load PyTorch"""
    import torch
    return torch

@st.cache_resource  
def get_pdf_reader():
    """Lazy load PyPDF2"""
    from PyPDF2 import PdfReader
    return PdfReader

@st.cache_resource
def get_text_splitter():
    """Lazy load text splitter"""
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    return RecursiveCharacterTextSplitter

@st.cache_resource
def get_faiss():
    """Lazy load FAISS"""
    try:
        from langchain_community.vectorstores import FAISS
        import faiss
        return FAISS, True
    except ImportError:
        return None, False

@st.cache_resource
def get_embeddings():
    """Lazy load embeddings"""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings

@st.cache_resource
def get_groq():
    """Lazy load Groq"""
    from langchain_groq import ChatGroq
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    return ChatGroq, RetrievalQA, PromptTemplate

@st.cache_resource
def get_paddleocr():
    """Lazy load PaddleOCR with proper error handling"""
    try:
        # Set environment variables for PaddleOCR
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        from paddleocr import PaddleOCR
        return PaddleOCR, True
    except Exception as e:
        st.warning(f"PaddleOCR not available: {str(e)}")
        return None, False

@st.cache_resource
def get_fitz():
    """Lazy load PyMuPDF"""
    try:
        import fitz
        return fitz, True
    except ImportError:
        return None, False


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file using PyPDF2"""
    try:
        PdfReader = get_pdf_reader()
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return None


def extract_images_and_ocr(pdf_file):
    """Extract images from PDF and perform OCR"""
    PaddleOCR, ocr_available = get_paddleocr()
    fitz, fitz_available = get_fitz()
    
    if not (ocr_available and fitz_available):
        st.warning("OCR or PDF processing not available")
        return ""

    try:
        # Initialize OCR with minimal settings
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False, use_gpu=False)
        
        pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        ocr_text = ""

        for page_num in range(min(len(pdf_document), 10)):  # Limit to 10 pages
            try:
                page = pdf_document.load_page(page_num)
                mat = fitz.Matrix(1.5, 1.5)  # Reduced resolution
                pix = page.get_pixmap(matrix=mat)
                img_data = pix.tobytes("png")
                image = Image.open(io.BytesIO(img_data))
                img_array = np.array(image)

                ocr_results = ocr.ocr(img_array, cls=True)
                if ocr_results and ocr_results[0]:
                    page_ocr_text = f"\n--- OCR Page {page_num + 1} ---\n"
                    for line in ocr_results[0]:
                        if line[1][0].strip() and line[1][1] > 0.6:  # Higher confidence threshold
                            page_ocr_text += line[1][0] + " "
                    ocr_text += page_ocr_text + "\n"
            except Exception as e:
                st.warning(f"OCR failed for page {page_num + 1}: {str(e)}")
                continue

        pdf_document.close()
        return ocr_text
    except Exception as e:
        st.error(f"Error during OCR: {str(e)}")
        return ""


@st.cache_resource
def create_embeddings():
    """Create embeddings with fallback"""
    try:
        torch = get_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        HuggingFaceEmbeddings = get_embeddings()
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},  # Force CPU to avoid conflicts
            encode_kwargs={'normalize_embeddings': True},
            cache_folder="./cache"
        )
        
        # Test embedding
        test_embedding = embeddings.embed_query("test")
        if test_embedding and len(test_embedding) > 0:
            return embeddings
        else:
            raise ValueError("Empty embedding")
            
    except Exception as e:
        st.error(f"Embedding creation failed: {str(e)}")
        return None


def create_vector_store(text_chunks):
    """Create FAISS vector store"""
    try:
        FAISS, faiss_available = get_faiss()
        if not faiss_available:
            st.error("FAISS not available. Install with: pip install faiss-cpu")
            return None

        # Clean chunks
        valid_chunks = [
            ' '.join(chunk.strip().split()) 
            for chunk in text_chunks 
            if chunk and isinstance(chunk, str) and len(chunk.strip()) > 10
        ]
        
        if not valid_chunks:
            st.error("No valid text chunks")
            return None

        st.info(f"Processing {len(valid_chunks)} chunks...")
        embeddings = create_embeddings()
        if not embeddings:
            return None

        # Create vector store in batches
        batch_size = 20
        vector_store = None
        
        progress_bar = st.progress(0)
        for i in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[i:i + batch_size]
            try:
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch, embedding=embeddings)
                else:
                    batch_vs = FAISS.from_texts(batch, embedding=embeddings)
                    vector_store.merge_from(batch_vs)
                
                progress = min((i + batch_size) / len(valid_chunks), 1.0)
                progress_bar.progress(progress)
            except Exception as e:
                st.warning(f"Batch {i//batch_size + 1} failed: {str(e)}")
                continue

        progress_bar.empty()
        
        if vector_store:
            st.success("✅ Vector store created")
            return vector_store
        else:
            st.error("❌ Vector store creation failed")
            return None
            
    except Exception as e:
        st.error(f"Vector store error: {str(e)}")
        return None


def create_qa_chain(vector_store, api_key):
    """Create QA chain"""
    try:
        ChatGroq, RetrievalQA, PromptTemplate = get_groq()
        
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1,
            max_tokens=1024
        )

        prompt_template = """
You are a document analyst. Answer based on the provided context.

Context: {context}

Question: {question}

Instructions:
- Provide detailed answers when context supports it
- For numerical data, present it clearly
- If information is incomplete, say what you found
- If no relevant info exists, state "Information not found in document"

Answer:
"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        return qa_chain
    except Exception as e:
        st.error(f"QA chain error: {str(e)}")
        return None


def main():
    st.title("📄 Advanced PDF Analyzer")
    st.markdown("Upload PDF and ask questions about content, tables, and data")

    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'processed_file' not in st.session_state:
        st.session_state.processed_file = None

    # API Key
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in secrets!")
        st.stop()

    # System info
    with st.expander("🔧 System Info"):
        torch = get_torch()
        st.info(f"PyTorch: {torch.__version__}")
        st.info(f"CUDA: {torch.cuda.is_available()}")

    # OCR option
    _, ocr_available = get_paddleocr()
    use_ocr = st.checkbox("🔍 Enable OCR", value=False, disabled=not ocr_available)

    # File upload
    uploaded_file = st.file_uploader("Choose PDF", type="pdf")

    if uploaded_file:
        file_changed = (st.session_state.processed_file != uploaded_file.name)

        if file_changed or not st.session_state.vector_store:
            st.session_state.vector_store = None
            st.session_state.qa_chain = None
            st.session_state.processed_file = uploaded_file.name

            with st.spinner("📖 Extracting text..."):
                pdf_text = extract_text_from_pdf(uploaded_file)

            ocr_text = ""
            if use_ocr and ocr_available:
                with st.spinner("🔍 Running OCR..."):
                    ocr_text = extract_images_and_ocr(uploaded_file)

            full_text = ""
            if pdf_text:
                full_text += pdf_text
            if ocr_text:
                full_text += "\n--- OCR CONTENT ---\n" + ocr_text

            if full_text:
                st.success(f"✅ Extracted {len(full_text):,} characters")

                with st.spinner("🔄 Creating chunks..."):
                    RecursiveCharacterTextSplitter = get_text_splitter()
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=100,
                        length_function=len
                    )
                    text_chunks = text_splitter.split_text(full_text)
                    st.info(f"📄 Created {len(text_chunks)} chunks")

                with st.spinner("🧠 Creating vector store..."):
                    st.session_state.vector_store = create_vector_store(text_chunks)

                if st.session_state.vector_store:
                    st.success("✅ Ready for questions!")
                    st.session_state.qa_chain = create_qa_chain(
                        st.session_state.vector_store, api_key
                    )

        # Q&A Interface
        if st.session_state.vector_store and st.session_state.qa_chain:
            st.markdown("---")
            st.subheader("💬 Ask Questions")

            # Example buttons
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("📋 Summary"):
                    question = "Provide a summary of this document"
                    st.session_state.current_question = question
            with col2:
                if st.button("📊 Data"):
                    question = "What numerical data or statistics are mentioned?"
                    st.session_state.current_question = question
            with col3:
                if st.button("🎯 Key Points"):
                    question = "What are the main findings or conclusions?"
                    st.session_state.current_question = question

            # Question input
            question = st.text_input(
                "Your question:",
                value=st.session_state.get('current_question', ''),
                placeholder="Ask about content, tables, or specific information..."
            )

            if question:
                with st.spinner("🤔 Processing..."):
                    try:
                        response = st.session_state.qa_chain.invoke({"query": question})
                        st.markdown("### 💡 Answer:")
                        st.write(response["result"])
                        
                        if response.get("source_documents"):
                            with st.expander("📚 Sources"):
                                for i, doc in enumerate(response["source_documents"][:3]):
                                    st.markdown(f"**Source {i + 1}:**")
                                    content = doc.page_content
                                    st.text(content[:500] + "..." if len(content) > 500 else content)
                                    st.markdown("---")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    else:
        st.info("👆 Upload a PDF to start")
        st.markdown("### Features:")
        st.markdown("""
        - **📄 Text Extraction** - PyPDF2 for regular text
        - **🔍 OCR Support** - PaddleOCR for images and tables
        - **🧠 Smart Q&A** - Groq LLM with document analysis
        - **📊 Data Analysis** - Handles reports, research papers
        - **⚡ Fast Processing** - Optimized for performance
        """)


if __name__ == "__main__":
    main()
