import openai
import pdfplumber

# Initialize OpenAI API key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Extract text from PDF
def extract_pdf_text(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Query OpenAI's GPT for a response based on PDF content
def chat_with_pdf(pdf_text, query):
    prompt = f"Here is a PDF document content:\n{pdf_text}\n\nThe user asks: {query}\nAnswer the question based on the above document."

    response = openai.Completion.create(
        model="text-davinci-003",  # Or use the latest available model
        prompt=prompt,
        max_tokens=200
    )

    return response.choices[0].text.strip()

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
