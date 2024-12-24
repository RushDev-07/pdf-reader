from transformers import pipeline
import pdfplumber

# Extract text from PDF
def extract_pdf_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Query Hugging Face model for a response based on PDF content
def chat_with_pdf(pdf_text, query):
    # Load a larger pre-trained question-answering model (e.g., BERT or RoBERTa)
    qa_pipeline = pipeline("question-answering", model="bert-large-uncased")
    
    # Perform question-answering on the document content
    result = qa_pipeline(question=query, context=pdf_text)
    return result["answer"]

# Main chatbot loop
def pdf_chatbot(pdf_path):
    pdf_text = extract_pdf_text(pdf_path)
    print("You can now start chatting with the PDF content. Type 'exit' to stop.")
    
    while True:
        user_query = input("Ask a question: ")
        
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break
        
        response = chat_with_pdf(pdf_text, user_query)
        print("Answer:", response)

# Start the chatbot
pdf_chatbot('SOURCE_DOCUMENTS/application_1.pdf')
