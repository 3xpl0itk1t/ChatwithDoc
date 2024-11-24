import streamlit as st
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_together import ChatTogether
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_together import TogetherEmbeddings
import PyPDF2
import io
import os


model_langchain = os.getenv("model_langchain")
model_embedding = os.getenv("model_embedding")
api_key = os.getenv("api_key")


# Initialize Together AI model
chat_model = ChatTogether(
    together_api_key=api_key,
    model=model_langchain
)

# ChromaDB setup
embeddings = TogetherEmbeddings(
    model=model_embedding,
    together_api_key=api_key
)
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

# Prompt engineering
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "You are an expert assistant. Use the provided context to answer questions accurately, focusing on numerical data.\n"
        "Context: {context}\n\n"
        "Question: {question}\n\n"
        "Answer (include specific rates or details if mentioned):"
    )
)
qa_chain = LLMChain(llm=chat_model, prompt=prompt_template)

# Streamlit UI
st.title("📝 File Q&A with AI")
uploaded_files = st.file_uploader("Upload files (txt, pdf)", type=("txt", "pdf"), accept_multiple_files=True)

if "file_contents" not in st.session_state:
    st.session_state["file_contents"] = {}
if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = vectorstore

# Disable chat input until a file is uploaded
chat_disabled = len(st.session_state["file_contents"]) == 0
question = st.chat_input("Ask about the documents")

# Process uploaded files
if uploaded_files:
    for uploaded_file in uploaded_files:
        if uploaded_file.name not in st.session_state["file_contents"]:
            try:
                # Extract content based on file type
                if uploaded_file.type == "text/plain":
                    file_content = uploaded_file.getvalue().decode("utf-8")
                elif uploaded_file.type == "application/pdf":
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.getvalue()))
                    file_content = "".join(page.extract_text() for page in pdf_reader.pages)
                else:
                    st.error(f"Unsupported file type: {uploaded_file.name}")
                    continue

                # Split content into chunks
                document = Document(page_content=file_content, metadata={"source": uploaded_file.name})
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
                chunks = text_splitter.split_documents([document])

                # Store file content and save embeddings in ChromaDB
                st.session_state["file_contents"][uploaded_file.name] = file_content
                st.session_state["vectorstore"].add_documents(chunks)

                st.success(f"Processed and indexed {uploaded_file.name}")
            except Exception as e:
                st.error(f"Error processing {uploaded_file.name}: {str(e)}")

# Display processed files
if st.session_state["file_contents"]:
    st.write("Processed Files:")
    for filename in st.session_state["file_contents"]:
        st.write(f"- {filename}")

# Answer user questions
if question:
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            # Retrieve relevant documents
            retriever = st.session_state["vectorstore"].as_retriever(
                search_type="mmr",  # Use Maximal Marginal Relevance for diversity
                search_kwargs={"k": 5}  # Fetch more relevant chunks
            )
            docs = retriever.get_relevant_documents(question)
            context = "\n\n".join([doc.page_content for doc in docs])

            # Debug retrieved context
            # st.write("Retrieved context:", context)

            # Generate answer
            if context.strip():
                answer = qa_chain.run(context=context, question=question)
                st.write(answer)
            else:
                st.error("No relevant context found. Please check the document indexing.")
        except Exception as e:
            st.error(f"Error querying the AI model: {str(e)}")
