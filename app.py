import os
import re
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from transformers import LlamaForCausalLM, LlamaTokenizer
from langchain.llms import HuggingFacePipeline
from dotenv import load_dotenv
import torch

load_dotenv()

# Load environment variables
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Function to initialize LLaMA 70B
def load_llama_model(model_name="meta-llama/Llama-2-70b-chat-hf", device="cuda"):
    try:
        print(f"Loading {model_name} model...")
        tokenizer = LlamaTokenizer.from_pretrained(model_name)
        model = LlamaForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for efficient inference
            device_map="auto",  # Automatically map model across GPUs
        )
        print("Model successfully loaded!")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading LLaMA model: {e}")
        return None, None

# Convert LLaMA into a LangChain-compatible pipeline
def get_llama_pipeline(model, tokenizer):
    try:
        from transformers import pipeline

        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            max_length=512,
            temperature=0.7,
            top_p=0.95,
            repetition_penalty=1.2,
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        return llm
    except Exception as e:
        print(f"Error setting up LLaMA pipeline: {e}")
        return None

# Create FAISS vector store
def create_vector_store(chunks):
    try:
        from langchain.embeddings import HuggingFaceInstructEmbeddings
        embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        return vectorstore
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None

# Set up the conversational retrieval chain
def setup_conversation_chain(vectorstore, llm):
    try:
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever()
        )
        return conversation_chain
    except Exception as e:
        print(f"Error setting up conversation chain: {e}")
        return None

# Main function to process PDFs and run the pipeline
def main():
    folder_path = "SOURCE_DOCUMENTS"  # Replace with your folder path

    # Load and process LLaMA model
    tokenizer, model = load_llama_model()
    if not model or not tokenizer:
        print("Failed to load LLaMA model.")
        return

    llm = get_llama_pipeline(model, tokenizer)
    if not llm:
        print("Failed to set up LLaMA pipeline.")
        return

    # Process PDFs
    pdf_text = get_pdf_text_from_folder(folder_path)
    if not pdf_text.strip():
        print("No text extracted from PDFs.")
        return

    chunks = get_text_chunks(pdf_text)
    if not chunks:
        print("No chunks created from the text.")
        return

    # Create vector store
    vectorstore = create_vector_store(chunks)
    if not vectorstore:
        print("Failed to create vector store.")
        return

    # Set up conversational retrieval chain
    conversation_chain = setup_conversation_chain(vectorstore, llm)
    if not conversation_chain:
        print("Failed to set up conversation chain.")
        return

    # Interact with the user
    print("Conversation chain is ready. Ask your questions.")
    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ")
        if question.lower() == "exit":
            break
        response = conversation_chain.run(question)
        print("\nResponse:", response)

# Helper functions
def get_pdf_text_from_folder(folder_path):
    from PyPDF2 import PdfReader
    all_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(folder_path, filename)
            print(f"Processing: {pdf_path}")
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                all_text += page.extract_text()
    return all_text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

if __name__ == "__main__":
    main()
