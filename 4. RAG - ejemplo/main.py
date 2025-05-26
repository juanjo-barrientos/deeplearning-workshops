from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
import os

# Configura tu API key de OpenAI
os.environ["OPENAI_API_KEY"] = "tu_api_key_aqui"

# Paso 1: Cargar el texto
loader = PyPDFLoader("la_singularidad_esta_cerca.pdf")
documents = loader.load()

# Paso 2: Dividir el texto en chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
docs = text_splitter.split_documents(documents)

# Paso 3: Crear embeddings con OpenAI
embeddings = OpenAIEmbeddings()

# Paso 4: Crear o cargar vector store con Chroma
db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")
db.persist()

# Paso 5: Crear la cadena RAG con LangChain
retriever = db.as_retriever(search_kwargs={"k": 4})

qa_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(temperature=0),
    chain_type="stuff",  # Usa los chunks como contexto
    retriever=retriever,
    return_source_documents=True
)

# Paso 6: Hacer preguntas
print("Pregúntame sobre el libro (escribe 'salir' para terminar):")
while True:
    pregunta = input("\nTu pregunta: ")
    if pregunta.lower() == "salir":
        break
    resultado = qa_chain(pregunta)
    print("\n💡 Respuesta:\n", resultado["result"])
