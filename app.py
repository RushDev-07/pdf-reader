import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from huggingface_hub import HfApi
from langchain.llms import HuggingFaceHub

# Set up Hugging Face API token

def get_pdf_files_from_folder(folder_path):
    """
    Get all PDF file paths from the specified folder.
    """
    pdf_files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith('.pdf')
    ]
    return pdf_files


def get_pdf_text_from_folder(folder_path):
    """
    Extract text from all PDFs in the specified folder.
    """
    pdf_files = get_pdf_files_from_folder(folder_path)
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    """
    Split the text into manageable chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    """
    Create a vector store for text chunks.
    """
    # You can choose OpenAI embeddings or HuggingFace embeddings
   # embeddings = OpenAIEmbeddings(openai_api_key="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")
    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


def get_conversation_chain(vectorstore):
    """
    Create a conversational retrieval chain using vectorstore.
    """
    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature": 0.5, "max_length": 512})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def process_pdfs_from_folder(folder_path):
    """
    Process PDFs from a folder and create a conversation chain.
    """
    # Extract text from all PDFs in the folder
    raw_text = get_pdf_text_from_folder(folder_path)

    # Get text chunks
    text_chunks = get_text_chunks(raw_text)

    # Create a vector store
    vectorstore = get_vectorstore(text_chunks)

    # Create a conversation chain
    conversation_chain = get_conversation_chain(vectorstore)
    return conversation_chain


def handle_user_input(conversation, user_question):
    """
    Handle the user's input and get a response from the conversation chain.
    """
    response = conversation({'question': user_question})
    return response


# Main function to run the chatbot
if __name__ == '__main__':
    load_dotenv()
    hf_token = os.getenv("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")  
    folder_path = "SOURCE_DOCUMENTS"  # Replace with the path to your folder containing PDFs
    print("Processing PDFs in the folder...")

    try:
        conversation_chain = process_pdfs_from_folder(folder_path)
        print("PDFs processed successfully. You can now ask questions.")
    except Exception as e:
        print(f"Error processing PDFs: {e}")
        exit()

    # Chat loop
    while True:
        user_question = 'Get the information from both the documents and store it into the following attributes in a JSON format:' + \
        'isCompany       lastModified    lat     lng     rera    builderName     builderPhone    builderUrl      builderBlockNumber      builderBuildingName     builderStreet   builderLocality builderLandMark builderState    builderDivision builderDistrict builderTaluka   builderVillage  builderPincode  projectName     status  proposedClosingDate     revisedClosingDate      litigation      projectType     projectPlotNo   projectState    projectDivision projectDistrict projectTaluka   projectVillage  projectStreet   projectLocality projectPincode  area    totalBuildings  sanctionedBuildings     unsanctionedBuildings   openArea        sanctionedFSI   proposedFSI     permissibleFSI  developmentWork buildingDetailsWithAppartments  brokerReras rera_date' + \
        'This is a sample data in with the above attributes' + \
        'TRUE    Last Modified                   P99000053639    METRO DEVELOPERS        9619304514              OFFICE NO. 1    LAKE VIEW HEIGHTS 1     LAKE VIEW HEIGHTS 1     RAJIWALI VILLAGE        MARUAAI TEMPLE  MAHARASHTRA     Konkan  Palghar Vasai   Vasai-Virar City (M Corp)       401208  HONEST HEIGHTS  New Project     31-12-2027              No      Others  SURVEY NO 232B, PLOT NO 3       MAHARASHTRA     Konkan  Palghar Vasai   Pelhar  PELHAR  PELHAR  401208  162.57  1       1       0       16.26   1198.66 0       1198.66 [{"name":"Internal Roads & Footpaths :","details":"-"},{"name":"Water Conservation, Rain water Harvesting :","details":"-"},{"name":"Energy management :","details":"NA"},{"name":"Fire Protection And Fire Safety \nRequirements :","details":"--"},{"name":"Electrical Meter Room, Sub-Station, Receiving Station :","details":"-"},{"name":"Aggregate area of recreational Open Space  :","details":"-"},{"name":"Open Parking :","details":"-"},{"name":"Water Supply :","details":"-"},{"name":"Sewerage (Chamber, Lines, Septic Tank , STP) :","details":"-"},{"name":"Storm Water Drains :","details":"-"},{"name":"Landscaping & Tree Planting :","details":"-"},{"name":"Street Lighting :","details":"-"},{"name":"Community Buildings :","details":"NA"},{"name":"Treatment And Disposal Of Sewage And Sullage Water :","details":"NA"},{"name":"Solid Waste Management And Disposal :","details":"-"}]      [{"name":"BUILDING NO 7","proposedCompletionDate":"31/12/2027","basements":"0","plinths":"0","podiums":"0","floors":"7","stilts":"0","openParkings":"0","closedParkings":"0","appartments":[{"name":"Shop","carpetArea":"11.15","totalAppartments":"1","bookedAppartments":"0"},{"name":"Shop","carpetArea":"18.59","totalAppartments":"1","bookedAppartments":"0"},{"name":"Shop","carpetArea":"14.87","totalAppartments":"1","bookedAppartments":"0"},{"name":"Shop","carpetArea":"16.26","totalAppartments":"1","bookedAppartments":"0"},{"name":"1RK","carpetArea":"23.23","totalAppartments":"7","bookedAppartments":"0"},{"name":"1BHK","carpetArea":"32.06","totalAppartments":"7","bookedAppartments":"0"},{"name":"1BHK","carpetArea":"32.99","totalAppartments":"7","bookedAppartments":"0"},{"name":"1BHK","carpetArea":"34.39","totalAppartments":"7","bookedAppartments":"0"}],"tasks":"0,0,0,0,0,0,0,0,0,0,0"}]   []'

        if user_question.lower() in ["exit", "quit"]:
            print("Exiting the chatbot. Goodbye!")
            break
        try:
            response = handle_user_input(conversation_chain, user_question)
            print("Bot:", response['answer'])
        except Exception as e:
            print(f"Error during conversation: {e}")
