Here's a **README** file for the provided code along with **usage instructions**.

---

# **GestureBridge PDF Conversational Bot**

This project is a conversational bot that processes PDF files, extracts text, stores it in a vector database, and allows users to query the information in a conversational manner. It leverages **LangChain**, **FAISS**, and pre-trained models from **HuggingFace** to build an intelligent retrieval-based conversational system.

## **Features**
- Extracts and processes text from PDF files in a folder.
- Splits large texts into manageable chunks for efficient processing.
- Embeds text chunks into vector space for similarity-based retrieval using FAISS.
- Supports conversational retrieval using a HuggingFace pre-trained language model.
- Provides real-time responses to user queries, maintaining conversational context.

---

## **Setup and Installation**

### **1. Prerequisites**
Ensure you have the following installed:
- **Python 3.8+**
- Pip for managing Python packages.

### **2. Clone the Repository**
```bash
git clone <repository-url>
cd <repository-folder>
```

### **3. Install Required Python Packages**
Install dependencies from the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### **4. Configure Environment Variables**
Create a `.env` file in the root directory and add your **HuggingFace API token**:
```env
HF_TOKEN=<your_huggingface_api_token>
```

Replace `<your_huggingface_api_token>` with your HuggingFace API key.

---

## **Usage Instructions**

### **1. Organize PDFs**
Place all PDF files to be processed in a folder named `SOURCE_DOCUMENTS` or update the `folder_path` variable in the code to point to your folder.

### **2. Run the Script**
Run the chatbot script using the following command:
```bash
python chatbot.py
```

### **3. Ask Questions**
Once the bot processes the PDFs, you can start asking questions based on the content of the documents.

Example:
```text
User: What is the builder's name for the project in the document?
Bot: The builder's name is METRO DEVELOPERS.
```

### **4. Exit**
Type `exit` or `quit` to terminate the chatbot.

---

## **Code Workflow**

1. **PDF Processing**:
   - The bot reads and extracts text from all PDFs in the specified folder.
   - Text is split into smaller chunks using LangChain's `CharacterTextSplitter`.

2. **Embedding Creation**:
   - Uses HuggingFace's `Instructor-XL` embeddings to convert text chunks into vector representations.
   - The vectors are stored in a FAISS vector database for efficient similarity-based retrieval.

3. **Conversational Retrieval**:
   - Queries are processed by a conversational retrieval chain using `Flan-T5-XXL`, a HuggingFace model.
   - Memory is maintained to provide contextual responses.

4. **Response Handling**:
   - The bot generates responses based on user queries and maintains context throughout the conversation.

---

## **Key Components**

### **1. Libraries Used**
- **LangChain**: For text chunking, embeddings, and conversational chains.
- **FAISS**: For vector database storage and retrieval.
- **PyPDF2**: For extracting text from PDFs.
- **HuggingFace Hub**: For pre-trained embeddings and LLMs.

### **2. Models**
- **HuggingFace Instructor-XL**: For generating text embeddings.
- **Flan-T5-XXL**: For conversational retrieval and response generation.

---

## **Extending the Bot**
- **Add New Models**: Replace `HuggingFaceInstructEmbeddings` or `Flan-T5-XXL` with other embeddings or language models.
- **Custom Retrieval Logic**: Modify `get_conversation_chain` to use different retrievers or chains.

---

## **Troubleshooting**

1. **Missing API Key**:
   - Ensure your `.env` file contains the correct HuggingFace API token.

2. **PDF Parsing Errors**:
   - Some PDFs may not extract text correctly. Ensure your PDFs are not scanned images.

3. **Slow Response**:
   - Use smaller models or reduce the chunk size in `CharacterTextSplitter` for faster processing.

---

## **Contributors**
- **Author**: Your Name
- **Email**: your.email@example.com

---

## **License**
This project is licensed under the [MIT License](LICENSE).

--- 

This README provides detailed instructions to set up, use, and extend the project while addressing potential issues and features.
