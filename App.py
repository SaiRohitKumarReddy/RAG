import os
import sys

os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"
os.environ["STREAMLIT_SERVER_ENABLE_FILE_WATCHER"] = "false"
os.environ["STREAMLIT_SERVER_ENABLE_STATIC_SERVING"] = "false"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'  # For compatibility

import asyncio

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
elif hasattr(asyncio, 'DefaultEventLoopPolicy'):
    asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

import streamlit as st

st.set_page_config(
    page_title="Advanced Text Extractor(OCR Based)",
    layout="wide",
    initial_sidebar_state="collapsed"
)

import tempfile
import io
import traceback
import numpy as np
from PIL import Image
from docx import Document


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
def get_docx_reader():
    """Lazy load python-docx"""
    from docx import Document
    return Document


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
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings


@st.cache_resource
def get_openai():
    """Lazy load OpenAI"""
    from langchain_openai import ChatOpenAI
    from langchain.chains import RetrievalQA
    from langchain.prompts import PromptTemplate
    return ChatOpenAI, RetrievalQA, PromptTemplate


@st.cache_resource
def get_tesseract():
    """Lazy load Tesseract OCR"""
    try:
        import pytesseract
        # Test Tesseract availability
        pytesseract.get_tesseract_version()
        st.info("Tesseract OCR initialized successfully")
        return pytesseract, True
    except ImportError:
        st.error("Tesseract initialization failed: pytesseract module not found. Install with: pip install pytesseract")
        return None, False
    except Exception as e:
        st.error(f"Tesseract initialization failed: {str(e)}. OCR functionality will be disabled.")
        return None, False


@st.cache_resource
def get_fitz():
    """Lazy load PyMuPDF"""
    try:
        import fitz
        return fitz, True
    except ImportError:
        return None, False


def determine_document_type(pdf_file, use_ocr):
    """Determine the document type based on content analysis"""
    try:
        PdfReader = get_pdf_reader()
        pdf_reader = PdfReader(pdf_file)
        fitz, fitz_available = get_fitz()

        # Check for image content (indicative of scanned or infographic PDF)
        pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        has_images = False
        image_count = 0

        for page_num in range(min(len(pdf_document), 3)):  # Check first 3 pages
            page = pdf_document.load_page(page_num)
            images = page.get_images()
            if images:
                has_images = True
                image_count += len(images)

        pdf_document.close()

        # Get text content
        text = ""
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text += page_text

        text_length = len(text.strip())

        # Logic to determine document type
        if has_images and image_count > 5:
            return "Infographic/Visual PDF"
        elif has_images and text_length < 200:
            return "Scanned PDF (OCR recommended)"
        elif has_images and text_length > 200:
            return "Mixed Content PDF (Text + Images)"
        elif text_length > 100:
            return "Text-based PDF"
        elif text_length > 0:
            return "Sparse Text PDF"
        else:
            return "Image-only PDF (OCR required)"

    except Exception as e:
        st.warning(f"Could not determine document type: {str(e)}")
        return "Unknown"


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


def extract_text_from_docx(docx_file):
    """Extract text from Word document using python-docx"""
    try:
        Document = get_docx_reader()
        docx_file.seek(0)
        doc = Document(docx_file)
        text = ""
        for para_num, paragraph in enumerate(doc.paragraphs):
            if paragraph.text.strip():
                text += f"\n--- Paragraph {para_num + 1} ---\n"
                text += paragraph.text
        return text
    except Exception as e:
        st.error(f"Error reading Word document: {str(e)}")
        return None


def extract_images_and_ocr(pdf_file):
    """Extract images from PDF and perform OCR with Tesseract"""
    pytesseract, tesseract_available = get_tesseract()
    fitz, fitz_available = get_fitz()

    if not fitz_available:
        st.warning("PDF image processing (PyMuPDF) not available. Install with: pip install pymupdf")
        return ""

    if not tesseract_available:
        st.warning("Tesseract OCR not available. Install pytesseract and Tesseract binary.")
        return ""

    try:
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

                page_ocr_text = f"\n--- OCR Page {page_num + 1} ---\n"
                page_ocr_text += pytesseract.image_to_string(image, lang='eng')
                ocr_text += page_ocr_text + "\n"
            except Exception as e:
                st.warning(f"OCR processing failed for page {page_num + 1}: {str(e)}")
                continue

        pdf_document.close()
        if ocr_text:
            st.info("OCR completed successfully")
        return ocr_text
    except Exception as e:
        st.error(f"OCR processing failed: {str(e)}. Continuing without OCR.")
        return ""


@st.cache_resource
def create_embeddings():
    """Create embeddings with OpenAI"""
    try:
        torch = get_torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        OpenAIEmbeddings = get_embeddings()
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=st.secrets.get("OPENAI_API_KEY")
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
                st.warning(f"Batch {i // batch_size + 1} failed: {str(e)}")
                continue

        progress_bar.empty()

        if vector_store:
            st.success("Vector store created")
            return vector_store
        else:
            st.error("Vector store creation failed")
            return None

    except Exception as e:
        st.error(f"Vector store error: {str(e)}")
        return None


def create_qa_chain(vector_store, api_key):
    try:
        ChatOpenAI, RetrievalQA, PromptTemplate = get_openai()

        llm = ChatOpenAI(
            openai_api_key=api_key,
            model_name="gpt-4o-mini",
            temperature=0.3,
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
    st.title("Advanced Text Extractor (OCR Based)")
    st.markdown("Upload PDF or Word document and ask questions about content, tables, and data")

    # Initialize session state
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = None
    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None
    if 'processed_file' not in st.session_state:
        st.session_state.processed_file = None
    if 'document_type' not in st.session_state:
        st.session_state.document_type = None

    api_key = st.secrets.get("OPENAI_API_KEY")
    if not api_key:
        st.error("OPENAI_API_KEY not found in secrets! Add it to .streamlit/secrets.toml")
        st.stop()

    with st.expander("System Info"):
        torch = get_torch()
        st.info(f"PyTorch: {torch.__version__}")
        st.info(f"CUDA: {torch.cuda.is_available()}")

    _, tesseract_available = get_tesseract()
    use_ocr = st.checkbox("Enable OCR (PDF only)", value=tesseract_available,
                          disabled=not tesseract_available)

    uploaded_file = st.file_uploader("Choose PDF or Word document", type=["pdf", "docx"])

    if uploaded_file:
        file_changed = (st.session_state.processed_file != uploaded_file.name)

        if file_changed or not st.session_state.vector_store:
            st.session_state.vector_store = None
            st.session_state.qa_chain = None
            st.session_state.processed_file = uploaded_file.name

            # Determine and store document type
            if uploaded_file.name.lower().endswith('.docx'):
                st.session_state.document_type = "Word Document"
            else:
                st.session_state.document_type = determine_document_type(uploaded_file, use_ocr)

            with st.spinner("Extracting text..."):
                if uploaded_file.name.lower().endswith('.pdf'):
                    text = extract_text_from_pdf(uploaded_file)
                    ocr_text = ""
                    if use_ocr and tesseract_available:
                        with st.spinner("Running OCR..."):
                            ocr_text = extract_images_and_ocr(uploaded_file)
                    elif use_ocr and not tesseract_available:
                        st.warning(
                            "OCR is not available due to missing Tesseract dependencies. Continuing with text extraction only.")
                elif uploaded_file.name.lower().endswith('.docx'):
                    text = extract_text_from_docx(uploaded_file)
                    ocr_text = ""  # OCR not applicable for Word documents
                    if use_ocr:
                        st.warning("OCR is only supported for PDF files. Skipping OCR for Word document.")
                else:
                    st.error("Unsupported file type. Please upload a PDF or Word document.")
                    text = None
                    ocr_text = ""

            full_text = text or ""
            if ocr_text:
                full_text += "\n--- OCR CONTENT ---\n" + ocr_text

            if full_text:
                st.success(f"Extracted {len(full_text):,} characters")

                with st.spinner("Creating chunks..."):
                    RecursiveCharacterTextSplitter = get_text_splitter()
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=800,
                        chunk_overlap=100,
                        length_function=len
                    )
                    text_chunks = text_splitter.split_text(full_text)
                    st.info(f"Created {len(text_chunks)} chunks")

                with st.spinner("Creating vector store..."):
                    st.session_state.vector_store = create_vector_store(text_chunks)

                if st.session_state.vector_store:
                    st.success("Ready for questions!")
                    st.session_state.qa_chain = create_qa_chain(
                        st.session_state.vector_store, api_key
                    )

        # Display document type persistently
        if st.session_state.document_type:
            st.markdown("### Document Type:")
            st.markdown(f"**{st.session_state.document_type}**")

        if st.session_state.vector_store and st.session_state.qa_chain:
            st.markdown("---")
            st.subheader("Ask Questions")

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Summary"):
                    question = "Provide a summary of this document"
                    st.session_state.current_question = question
            with col2:
                if st.button("Data"):
                    question = "What numerical data or statistics are mentioned?"
                    st.session_state.current_question = question
            with col3:
                if st.button("Key Points"):
                    question = "What are the main findings or conclusions?"
                    st.session_state.current_question = question

            question = st.text_input(
                "Your question:",
                value=st.session_state.get('current_question', ''),
                placeholder="Ask about content, tables, or specific information..."
            )

            if question:
                with st.spinner("Processing..."):
                    try:
                        response = st.session_state.qa_chain.invoke({"query": question})
                        st.markdown("Answer:")
                        st.write(response["result"])

                        if response.get("source_documents"):
                            with st.expander("Sources"):
                                for i, doc in enumerate(response["source_documents"][:3]):
                                    st.markdown(f"**Source {i + 1}:**")
                                    content = doc.page_content
                                    st.text(content[:500] + "..." if len(content) > 500 else content)
                                    st.markdown("---")
                    except Exception as e:
                        st.error(f"Error: {str(e)}")

    else:
        st.info("Upload a PDF or Word document to start")
        st.markdown("### Features:")
        st.markdown("""
        - **Text Extraction** - PyPDF2 for PDFs, python-docx for Word documents
        - **OCR Support** - Tesseract for images and tables (PDF only)
        - **Smart Q&A** - OpenAI GPT-4o-mini with document analysis
        - **Data Analysis** - Handles reports, research papers
        - **Fast Processing** - Optimized for performance
        """)


if __name__ == "__main__":
    main()
