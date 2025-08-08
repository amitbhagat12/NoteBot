import streamlit as st
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit import sidebar
from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

OpenAI_API_KEY="Enter Your API Key"


st.header("NoteBot")

with sidebar:
    st.title("My Notes")
    file = st.file_uploader("Upload Notes PDF & Start Questioning",type="pdf")

# Extracting the text from pdf file
if file is not None:
    my_pdf = PdfReader(file)
    text=""
    for page in my_pdf.pages:
        text += page.extract_text()
        #st.write(text)

    # Break It Into Chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50,length_function=len)
    chunks = splitter.split_text(text)
    #st.write(chunks)

    # creating Object of OpenAIEmbeddings class that let us connect with OpenAI's Embedding Models
    embeddings = OpenAIEmbeddings(api_key=OpenAI_API_KEY)

    #Creating VectorDB & Storing embeddings into it
    vector_store = FAISS.from_texts(chunks,embeddings)

    #get user query
    user_query = st.text_input("Type your query here")

    #semantic search from vector store
    if user_query:
        matching_chunks = vector_store.similarity_search(user_query)

        #define our LLM
        llm = ChatOpenAI(
            api_key=OpenAI_API_KEY,
            max_tokens=300,
            temperature=0,
            model="gpt-3.5-turbo"
        )

        # Approach 1 Generate Response using load_qa_chain() function

        #chain = load_qa_chain(llm,chain_type="stuff")
        #output = chain.run(question=user_query,input_documents=matching_chunks)
        #st.write(output)

        # Approach 2 : Generate Response create_stuff_documents_chain() function
        customized_prompt = ChatPromptTemplate.from_template(
            """ You are my assistant tutor. Answer the question based on the following context and
            if you did not get the context simply say "I don't know Sir/Mam" :
            {context}
            Question: {input}
            """
        )

        chain = create_stuff_documents_chain(llm,customized_prompt)  # It is a function , and not a method
        output = chain.invoke({"input": user_query, "context": matching_chunks})
        st.write(output)





































