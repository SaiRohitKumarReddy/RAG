import streamlit as st
import os
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import tempfile
import fitz  # PyMuPDF
from PIL import Image
import io
import numpy as np
import torch
import traceback

# Set page config as the first Streamlit command
st.set_page_config(page_title="Advanced PDF Analyzer", layout="wide")

# Defer PaddleOCR import to avoid early st.warning()
def initialize_paddleocr():
    try:
        from paddleocr import PaddleOCR
        return PaddleOCR, True
    except ImportError:
        return None, False

PaddleOCR, PADDLEOCR_AVAILABLE = initialize_paddleocr()
if not PADDLEOCR_AVAILABLE:
    st.warning("⚠️ PaddleOCR not installed. Install with: pip install paddleocr")


def check_faiss_availability():
    """Check if FAISS is available and provide installation guidance"""
    try:
        import faiss
        return True, None
    except ImportError as e:
        error_msg = """
        ❌ FAISS library is not installed. Please install it:

        For CPU-only systems:
        pip install faiss-cpu

        For CUDA GPU systems:
        pip install faiss-gpu

        After installation, restart your Streamlit app.
        """
        return False, error_msg


def extract_text_from_pdf(pdf_file):
    """Extract text from PDF file using PyPDF2"""
    try:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text
        return text
    except Exception as e:
        st.error(f"Error reading PDF with PyPDF2: {str(e)}")
        return None

def extract_images_and_ocr(pdf_file):
    """Extract images from PDF and perform OCR using PaddleOCR"""
    if not PADDLEOCR_AVAILABLE:
        return ""

    try:
        ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        ocr_text = ""

        for page_num in range(len(pdf_document)):
            page = pdf_document.load_page(page_num)
            mat = fitz.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))
            img_array = np.array(image)

            try:
                ocr_results = ocr.ocr(img_array, cls=True)
                if ocr_results and ocr_results[0]:
                    page_ocr_text = f"\n--- OCR from Page {page_num + 1} ---\n"
                    for line in ocr_results[0]:
                        if line[1][0].strip():
                            confidence = line[1][1]
                            if confidence > 0.5:
                                page_ocr_text += line[1][0] + " "
                    ocr_text += page_ocr_text + "\n"
            except Exception as e:
                st.warning(f"OCR failed for page {page_num + 1}: {str(e)}")
                continue

        pdf_document.close()
        return ocr_text
    except Exception as e:
        st.error(f"Error during OCR processing: {str(e)}")
        return ""

def create_embeddings_with_fallback():
    """Create embeddings with fallback to SentenceTransformer"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    embedding_configs = [
        {
            "model_name": "sentence-transformers/all-MiniLM-L6-v2",
            "model_kwargs": {},  # Let sentence-transformers handle device
            "encode_kwargs": {'normalize_embeddings': True},
            "cache_folder": "./huggingface_cache"
        }
    ]

    from langchain_huggingface import HuggingFaceEmbeddings
    for i, config in enumerate(embedding_configs):
        try:
            st.info(f"Attempting to load embedding model {i + 1}/{len(embedding_configs)}: {config['model_name']}")
            print(f"Loading model: {config['model_name']}")  # Debug print
            embeddings = HuggingFaceEmbeddings(
                model_name=config["model_name"],
                model_kwargs=config["model_kwargs"],
                encode_kwargs=config["encode_kwargs"],
                cache_folder=config["cache_folder"]
            )
            print("Model loaded successfully")  # Debug print
            test_text = "This is a test sentence for embedding."
            print("Generating test embedding...")  # Debug print
            test_embedding = embeddings.embed_query(test_text)
            print(f"Test embedding length: {len(test_embedding)}")  # Debug print
            if test_embedding and len(test_embedding) > 0:
                st.success(f"✅ Successfully loaded embedding model: {config['model_name']}")
                return embeddings
            else:
                raise ValueError("Empty embedding generated")
        except Exception as e:
            st.warning(f"❌ Failed to load {config['model_name']}: {str(e)}")
            print(f"Error details: {traceback.format_exc()}")  # Debug print
            continue

    # Fallback to SentenceTransformer directly
    try:
        st.info("Trying direct SentenceTransformer model...")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2", cache_folder="./huggingface_cache")
        def embed_query(text):
            return model.encode(text, normalize_embeddings=True).tolist()
        embeddings = type('obj', (object,), {
            'embed_query': embed_query,
            'embed_documents': lambda texts: [embed_query(text) for text in texts]
        })
        test_embedding = embeddings.embed_query("test")
        print(f"Fallback test embedding length: {len(test_embedding)}")  # Debug print
        if test_embedding and len(test_embedding) > 0:
            st.success("✅ Successfully loaded SentenceTransformer model")
            return embeddings
    except Exception as e:
        st.error(f"❌ SentenceTransformer fallback failed: {str(e)}")
        print(f"Fallback error details: {traceback.format_exc()}")  # Debug print

    st.error("❌ Failed to create any embedding model. Please check your environment.")
    return None


def create_vector_store(text_chunks):
    """Create FAISS vector store from text chunks with improved error handling"""
    try:
        # Check FAISS availability first
        faiss_available, faiss_error = check_faiss_availability()
        if not faiss_available:
            st.error(faiss_error)
            return None

        valid_chunks = []
        for chunk in text_chunks:
            if chunk and isinstance(chunk, str) and chunk.strip():
                cleaned_chunk = ' '.join(chunk.strip().split())
                if len(cleaned_chunk) > 10:
                    valid_chunks.append(cleaned_chunk)

        if not valid_chunks:
            st.error("No valid text chunks found for embedding")
            return None

        st.info(f"Processing {len(valid_chunks)} valid text chunks...")
        embeddings = create_embeddings_with_fallback()
        if embeddings is None:
            st.error("❌ Failed to create embedding model")
            return None

        batch_size = 10  # Reduced batch size
        vector_store = None
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i in range(0, len(valid_chunks), batch_size):
            batch = valid_chunks[i:i + batch_size]
            try:
                status_text.text(
                    f"Processing batch {i // batch_size + 1}/{(len(valid_chunks) + batch_size - 1) // batch_size}")
                if vector_store is None:
                    vector_store = FAISS.from_texts(batch, embedding=embeddings)
                else:
                    batch_vs = FAISS.from_texts(batch, embedding=embeddings)
                    vector_store.merge_from(batch_vs)
                progress = min((i + batch_size) / len(valid_chunks), 1.0)
                progress_bar.progress(progress)
            except Exception as batch_error:
                st.warning(f"⚠️ Error processing batch {i // batch_size + 1}: {str(batch_error)}")
                continue

        progress_bar.empty()
        status_text.empty()

        if vector_store is None:
            st.error("❌ Failed to create any vector store batches")
            return None

        st.success("✅ Successfully created vector store")
        return vector_store
    except Exception as e:
        st.error(f"❌ Error in create_vector_store: {str(e)}")
        with st.expander("🔍 Debug Information"):
            st.error(f"Number of text chunks: {len(text_chunks) if text_chunks else 'None'}")
            if text_chunks:
                st.error(f"First chunk preview: {str(text_chunks[0])[:200] if text_chunks[0] else 'Empty chunk'}")
            st.error(f"PyTorch version: {torch.__version__}")
            st.error(f"CUDA available: {torch.cuda.is_available()}")
            if torch.cuda.is_available():
                st.error(f"CUDA version: {torch.version.cuda}")
        return None

def create_qa_chain(vector_store, api_key):
    """Create QA chain with Groq LLM and robust prompt"""
    try:
        llm = ChatGroq(
            groq_api_key=api_key,
            model_name="llama-3.3-70b-versatile",
            temperature=0.1
        )

        prompt_template = """
You are an expert document analyst. Your task is to answer questions based on the provided context from a PDF document that may contain:
- Regular text content
- Tables with numerical data
- Charts and graphs (described in text)
- Financial reports, annual reports, research papers
- Technical documents with specifications
- OCR-extracted text from images

Instructions:
1. **Analyze the context carefully** - Look for relevant information across all sections
2. **Handle Tables/Data** - If the question involves numerical data, tables, or statistics, extract and present them clearly
3. **Be Comprehensive** - Provide detailed answers when the context supports it
4. **Cite Sections** - When possible, reference which part of the document (e.g., "Page X" or "Table Y") contains the information
5. **Handle Uncertainty** - If information is partially available, explain what you found and what might be missing
6. **Structure Your Response** - Use bullet points, numbers, or clear formatting for complex information

Context from Document:
{context}

Question: {question}

Guidelines for Response:
- If you find complete information: Provide a comprehensive answer with specific details, numbers, and relevant context
- If you find partial information: Share what's available and indicate what additional information might be helpful
- If no relevant information exists: State "I cannot find information about [specific topic] in this document"
- For numerical/tabular data: Present it in a clear, organized format
- For complex topics: Break down your answer into logical sections

Answer:
"""

        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(search_kwargs={"k": 6}),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )
        return qa_chain
    except Exception as e:
        st.error(f"Error creating QA chain: {str(e)}")
        return None

def main():
    st.title("📄 Advanced PDF Document Analyzer")
    st.markdown("Upload any PDF (reports, research papers, technical docs) and ask questions about tables, graphs, and content")

    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'processed_file' not in st.session_state:
        st.session_state.processed_file = None
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    if 'document_stats' not in st.session_state:
        st.session_state.document_stats = {}

    # Get API key
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.error("❌ GROQ_API_KEY not found in secrets!")
        st.stop()

    # System information
    with st.expander("🔧 System Information"):
        st.info(f"PyTorch version: {torch.__version__}")
        st.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            st.info(f"CUDA version: {torch.version.cuda}")
            st.info(f"GPU devices: {torch.cuda.device_count()}")

    # OCR option
    use_ocr = st.checkbox(
        "🔍 Enable OCR for images and tables",
        value=PADDLEOCR_AVAILABLE,
        disabled=not PADDLEOCR_AVAILABLE,
        help="Extract text from images, scanned documents, and complex tables"
    )

    # File upload
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

    if uploaded_file is not None:
        file_changed = (st.session_state.processed_file != uploaded_file.name)

        if file_changed or st.session_state.vector_store is None:
            st.session_state.vector_store = None
            st.session_state.qa_chain = None
            st.session_state.processed_file = uploaded_file.name

            with st.spinner("📖 Extracting text from PDF..."):
                pdf_text = extract_text_from_pdf(uploaded_file)

            ocr_text = ""
            if use_ocr and PADDLEOCR_AVAILABLE:
                with st.spinner("🔍 Performing OCR on images and tables..."):
                    ocr_text = extract_images_and_ocr(uploaded_file)

            full_text = ""
            if pdf_text:
                full_text += pdf_text
            if ocr_text:
                full_text += "\n--- OCR EXTRACTED CONTENT ---\n" + ocr_text

            if full_text:
                st.session_state.document_stats = {
                    'total_chars': len(full_text),
                    'pdf_text_chars': len(pdf_text) if pdf_text else 0,
                    'ocr_text_chars': len(ocr_text) if ocr_text else 0,
                    'has_ocr': len(ocr_text) > 0
                }
                st.success(f"✅ Extracted {len(full_text):,} characters from PDF")

                with st.expander("📊 Extraction Details"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("PDF Text", f"{st.session_state.document_stats['pdf_text_chars']:,} chars")
                    with col2:
                        st.metric("OCR Text", f"{st.session_state.document_stats['ocr_text_chars']:,} chars")
                    with col3:
                        st.metric("Total", f"{st.session_state.document_stats['total_chars']:,} chars")

                with st.spinner("🔄 Creating smart text chunks..."):
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len,
                        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                    )
                    text_chunks = text_splitter.split_text(full_text)
                    st.info(f"📄 Created {len(text_chunks)} text chunks")

                with st.spinner("🧠 Creating embeddings..."):
                    st.session_state.vector_store = create_vector_store(text_chunks)

                if st.session_state.vector_store:
                    st.success("✅ Vector store created successfully!")
                    st.session_state.qa_chain = create_qa_chain(st.session_state.vector_store, api_key)
                else:
                    st.error("❌ Failed to create vector store. Please try again or contact support.")

        if st.session_state.vector_store and st.session_state.qa_chain:
            st.markdown("---")
            st.subheader("💬 Ask Questions About Your Document")

            st.markdown("**💡 Example Questions:**")
            col1, col2 = st.columns(2)

            with col1:
                if st.button("📋 Document Summary"):
                    st.session_state.current_question = (
                        "Provide a comprehensive summary of this document including key findings, main topics, and important data."
                    )
                if st.button("📊 Financial Data"):
                    st.session_state.current_question = (
                        "What financial data, revenue figures, or monetary values are mentioned in this document?"
                    )
                if st.button("📈 Performance Metrics"):
                    st.session_state.current_question = (
                        "What performance metrics, KPIs, or statistical data are presented?"
                    )

            with col2:
                if st.button("🎯 Key Findings"):
                    st.session_state.current_question = (
                        "What are the main conclusions, recommendations, or key findings?"
                    )
                if st.button("📋 Tables & Data"):
                    st.session_state.current_question = (
                        "List all tables, charts, and structured data found in the document with their key information."
                    )
                if st.button("⚠️ Risks & Issues"):
                    st.session_state.current_question = (
                        "What risks, challenges, or issues are identified in this document?"
                    )

            question = st.text_input(
                "Your question:",
                value=st.session_state.current_question,
                placeholder="Ask about specific data, tables, trends, or any content in the document...",
                key="question_input"
            )

            if question != st.session_state.current_question:
                st.session_state.current_question = question

            if question:
                with st.spinner("🤔 Analyzing document and generating answer..."):
                    try:
                        response = st.session_state.qa_chain.invoke({"query": question})
                        st.markdown("### 💡 Answer:")
                        st.write(response["result"])
                        if response.get("source_documents"):
                            with st.expander("📚 Source Context"):
                                for i, doc in enumerate(response["source_documents"]):
                                    st.markdown(f"**Source {i + 1}:**")
                                    content = doc.page_content
                                    if len(content) > 800:
                                        st.text(content[:800] + "...")
                                    else:
                                        st.text(content)
                                    st.markdown("---")
                    except Exception as e:
                        st.error(f"Error processing question: {str(e)}")

            col1, col2 = st.columns([1, 4])
            with col1:
                if st.button("🗑️ Clear"):
                    st.session_state.current_question = ""
                    st.rerun()

    else:
        st.session_state.vector_store = None
        st.session_state.qa_chain = None
        st.session_state.processed_file = None
        st.info("👆 Please upload a PDF file to get started")

        st.markdown("### 🌟 Enhanced Features:")
        st.markdown("""
        - **📄 Comprehensive Text Extraction** - PyPDF2 for regular text
        - **🔍 OCR Support** - PaddleOCR for images, tables, and scanned content 
        - **🧠 Smart Chunking** - Optimized for preserving table and data context
        - **💡 Intelligent Q&A** - Groq LLM with specialized prompts for document analysis
        - **📊 Document Statistics** - See extraction details and content breakdown
        - **🎯 Context-Aware** - Handles annual reports, research papers, technical docs
        - **⚡ Persistent State** - No re-processing on interactions
        - **🔧 Robust Error Handling** - Multiple fallback embedding models
        """)

        if not PADDLEOCR_AVAILABLE:
            st.markdown("### 📦 Installation:")
            st.code("pip install paddleocr", language="bash")

        st.markdown("### 🛠️ Troubleshooting:")
        st.markdown("""
        If you encounter embedding errors, try:
        1. Restarting the Streamlit app
        2. Clearing browser cache
        3. Checking your internet connection
        4. Installing/updating PyTorch: `pip install torch --upgrade`
        5. Clearing Hugging Face cache: `rm -rf ~/.cache/huggingface`
        6. Checking for dependency conflicts: `pip check`
        """)

if __name__ == "__main__":
    main()
