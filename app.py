import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login
import torch
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

# Load environment variables
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACEHUB_API_TOKEN")

# Load Falcon model instead of LLaMA
def load_falcon_model(model_name="tiiuae/falcon-7b-instruct", device="cuda"):
    try:
        print(f"Loading {model_name} model...")

        # Ensure HuggingFace authentication for downloading private models
        login(HUGGINGFACE_API_KEY)

        # Load the tokenizer and model using the Auto class
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # Use float16 for efficient inference
            device_map="auto",         # Automatically map model across GPUs
            low_cpu_mem_usage=True     # Optimize memory usage for large models
        )

        # Load model weights separately for security
        model.load_state_dict(torch.load('path_to_your_model/pytorch_model.bin', map_location=torch.device('cpu'), weights_only=True))

        print("Model successfully loaded!")
        return tokenizer, model
    except Exception as e:
        print(f"Error loading Falcon model: {e}")
        return None, None


# Set up Falcon pipeline
def get_falcon_pipeline(model, tokenizer):
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
        print(f"Error setting up Falcon pipeline: {e}")
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


# Process PDF files
def get_pdf_text_from_folder(folder_path):
    try:
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
    except Exception as e:
        print(f"Error reading PDFs: {e}")
        return ""


# Chunk large text into smaller pieces
def get_text_chunks(text):
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        return splitter.split_text(text)
    except Exception as e:
        print(f"Error splitting text into chunks: {e}")
        return []


# Single input mode
def single_input_mode(conversation_chain):
    try:
        # Your query to extract information
        question = (
            'Get the information from both the documents and store it into the following attributes in a JSON format:' + \
            'isCompany       lastModified    lat     lng     rera    builderName     builderPhone    builderUrl      builderBlockNumber      builderBuildingName     builderStreet   builderLocality builderLandMark builderState    builderDivision builderDistrict builderTaluka   builderVillage  builderPincode  projectName     status  proposedClosingDate     revisedClosingDate      litigation      projectType     projectPlotNo   projectState    projectDivision projectDistrict projectTaluka   projectVillage  projectStreet   projectLocality projectPincode  area    totalBuildings  sanctionedBuildings     unsanctionedBuildings   openArea        sanctionedFSI   proposedFSI     permissibleFSI  developmentWork buildingDetailsWithAppartments  brokerReras rera_date' + \
            'This is a sample data in with the above attributes' + \
            'TRUE    Last Modified                   P99000053639    METRO DEVELOPERS        9619304514              OFFICE NO. 1    LAKE VIEW HEIGHTS 1     LAKE VIEW HEIGHTS 1     RAJIWALI VILLAGE        MARUAAI TEMPLE  MAHARASHTRA     Konkan  Palghar Vasai   Vasai-Virar City (M Corp)       401208  HONEST HEIGHTS  New Project     31-12-2027              No      Others  SURVEY NO 232B, PLOT NO 3       MAHARASHTRA     Konkan  Palghar Vasai   Pelhar  PELHAR  PELHAR  401208  162.57  1       1       0       16.26   1198.66 0       1198.66 [{"name":"Internal Roads & Footpaths :","details":"-"},{"name":"Water Conservation, Rain water Harvesting :","details":"-"},{"name":"Energy management :","details":"NA"},{"name":"Fire Protection And Fire Safety \nRequirements :","details":"--"},{"name":"Electrical Meter Room, Sub-Station, Receiving Station :","details":"-"},{"name":"Aggregate area of recreational Open Space  :","details":"-"},{"name":"Open Parking :","details":"-"},{"name":"Water Supply :","details":"-"},{"name":"Sewerage (Chamber, Lines, Septic Tank , STP) :","details":"-"},{"name":"Storm Water Drains :","details":"-"},{"name":"Landscaping & Tree Planting :","details":"-"},{"name":"Street Lighting :","details":"-"},{"name":"Community Buildings :","details":"NA"},{"name":"Treatment And Disposal Of Sewage And Sullage Water :","details":"NA"},{"name":"Solid Waste Management And Disposal :","details":"-"}]      [{"name":"BUILDING NO 7","proposedCompletionDate":"31/12/2027","basements":"0","plinths":"0","podiums":"0","floors":"7","stilts":"0","openParkings":"0","closedParkings":"0","appartments":[{"name":"Shop","carpetArea":"11.15","totalAppartments":"1","bookedAppartments":"0"},{"name":"Shop","carpetArea":"18.59","totalAppartments":"1","bookedAppartments":"0"},{"name":"Shop","carpetArea":"14.87","totalAppartments":"1","bookedAppartments":"0"},{"name":"Shop","carpetArea":"16.26","totalAppartments":"1","bookedAppartments":"0"},{"name":"1RK","carpetArea":"23.23","totalAppartments":"7","bookedAppartments":"0"},{"name":"1BHK","carpetArea":"32.06","totalAppartments":"7","bookedAppartments":"0"},{"name":"1BHK","carpetArea":"32.99","totalAppartments":"7","bookedAppartments":"0"},{"name":"1BHK","carpetArea":"34.39","totalAppartments":"7","bookedAppartments":"0"}],"tasks":"0,0,0,0,0,0,0,0,0,0,0"}]   []'
        )
        
        # Pass chat_history and question as a dictionary
        response = conversation_chain.run({
            "chat_history": [],  # If no previous conversation, start with an empty list
            "question": question
        })
        
        print("\nResponse:", response)
    except Exception as e:
        print(f"Error generating response: {e}")

# Main function
def main():
    folder_path = "SOURCE_DOCUMENTS"  # Replace with your folder path

    # Load and process Falcon model
    tokenizer, model = load_falcon_model()
    if not model or not tokenizer:
        print("Failed to load Falcon model.")
        return

    llm = get_falcon_pipeline(model, tokenizer)
    if not llm:
        print("Failed to set up Falcon pipeline.")
        return

    # Extract and process text from PDFs
    pdf_text = get_pdf_text_from_folder(folder_path)
    if not pdf_text.strip():
        print("No text extracted from PDFs.")
        return

    chunks = get_text_chunks(pdf_text)
    if not chunks:
        print("No chunks created from the text.")
        return

    # Create FAISS vector store
    vectorstore = create_vector_store(chunks)
    if not vectorstore:
        print("Failed to create vector store.")
        return

    # Set up conversational chain
    conversation_chain = setup_conversation_chain(vectorstore, llm)
    if not conversation_chain:
        print("Failed to set up conversation chain.")
        return

    # Enable single input mode for querying
    single_input_mode(conversation_chain)

if __name__ == "__main__":
    main()
