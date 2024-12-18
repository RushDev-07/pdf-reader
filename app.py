import os
import re
from langchain.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from pytesseract import image_to_string
from pdf2image import convert_from_path
from dotenv import load_dotenv

load_dotenv()

# Initialize Hugging Face token
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# OCR fallback for scanned PDFs
def extract_text_with_ocr(pdf_path):
    from pdf2image import convert_from_path
    try:
        images = convert_from_path(pdf_path)
        text = " ".join([image_to_string(image) for image in images])
        return text
    except Exception as e:
        print(f"Error extracting text with OCR: {e}")
        return ""

# Extract text from PDFs
def get_pdf_text(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        if not text.strip():  # Fallback to OCR if text is empty
            print(f"Using OCR for {pdf_path} as no text was extracted.")
            text = extract_text_with_ocr(pdf_path)
        return text
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return ""

# Extract text from all PDFs in a folder
def get_pdf_text_from_folder(folder_path):
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing: {pdf_path}")
            text = get_pdf_text(pdf_path)
            all_text += text + "\n"
    return all_text

# Split text into chunks
def get_text_chunks(text):
    try:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_text(text)
        return chunks
    except Exception as e:
        print(f"Error during text chunking: {e}")
        return []

# Create FAISS vector store
def create_vector_store(chunks):
    try:
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

# Set up the conversational retrieval chain
def setup_conversation_chain(vectorstore):
    try:
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-xxl",  # Switch to "flan-t5-large" if memory issues occur
            model_kwargs={"temperature": 0.5, "max_length": 512}
        )
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever()
        )
        return conversation_chain
    except Exception as e:
        print(f"Error setting up conversation chain: {e}")
        return None

# Process PDFs and return conversation chain
def process_pdfs_from_folder(folder_path):
    pdf_text = get_pdf_text_from_folder(folder_path)
    if not pdf_text.strip():
        print("No text extracted from PDFs.")
        return None

    print("Splitting text into chunks...")
    chunks = get_text_chunks(pdf_text)
    if not chunks:
        print("No chunks created from the text.")
        return None

    print("Creating FAISS vector store...")
    vectorstore = create_vector_store(chunks)
    if not vectorstore:
        print("Failed to create vector store.")
        return None

    print("Setting up conversational retrieval chain...")
    conversation_chain = setup_conversation_chain(vectorstore)
    if not conversation_chain:
        print("Failed to set up conversation chain.")
        return None

    return conversation_chain

# Handle user input and interact with the conversation chain
def handle_user_input(conversation_chain, question):
    if not conversation_chain:
        print("Conversation chain is not initialized.")
        return None
    try:
        response = conversation_chain.run(question)
        return response
    except Exception as e:
        print(f"Error handling user input: {e}")
        return None

# Main function
if __name__ == "__main__":
    folder_path = "SOURCE_DOCUMENTS"  # Replace with your folder path
    conversation_chain = process_pdfs_from_folder(folder_path)

    if conversation_chain:
        print("Conversation chain is ready. Ask your questions.")
        while True:
            question = input("\nEnter your question (or type 'exit' to quit): ")
            if question.lower() == "exit":
                break
            response = handle_user_input(conversation_chain, question)
            print("\nResponse:", response)
