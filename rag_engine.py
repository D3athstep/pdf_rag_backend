from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import chromadb
from chromadb.config import Settings 
from dotenv import  load_dotenv
import os 
from openai import OpenAI
load_dotenv ()
client=OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

def processpdf(pdf_path):
    #get the pdf file and extract the data from it 
    reader = PdfReader(pdf_path)
    num_pages = len(reader.pages)
    alltext=""

    for i in range(num_pages):
        page= reader.pages[i]
        text=page.extract_text()
        if text:
            alltext += text

        print(alltext[:1000])
    
    return  alltext
def split_into_chunks(text):
    #data is chunked in this piece of code
    clean_chunk=[]
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size= 1000,
        chunk_overlap =200
    )
    # data is divided into chunks here
    chunks = text_splitter.split_text(text)
    for i,chunk in enumerate(chunks[:5]):
        print("chunks",i+1)
        print (chunk)
        print("-" * 50)
    #white spaces are removed here
    for chunk in chunks:
        chunk = chunk.strip()
    #removed unwanted  text like  \n
        if len(chunk) <30 :
            continue
        #chunk= chunk.replace('\n',' ').replace('\t', ' ')
        chunk= re.sub(r"[^a-zA-Z0-9.,;:!?()\"\'\s]", "", chunk)
        if chunk.lower().startswith("o"):
            continue
        chunk= " ".join(chunk.split())
        clean_chunk.append(chunk)
        print(f"Filtered {len(chunks)} → {len(clean_chunk)} chunks after removing short ones.")

        print("Before:", repr(chunk))
        print("After :", repr(clean_chunk))

def get_chroma_collection(name):
    #connect to chroma client locally
    client = chromadb.Client(Settings())
    collection=  client.get_or_create_collection(name)
    return collection

dummy_document=["This is a sample document.", "ChromaDB is awesome!", "AI models are fun."]
dummy_id=["doc1","doc2","doc3"]

def dummy_chroma_collection(collection,dummy_document):
    # appending data into dummy_embedding 
    dummy_embeddings=[]
    for i,doc in enumerate(dummy_document):
        try:
            response = client.embeddings.create(input=doc,model="text-embedding-ada-002")
            embedding = response.data[0].embedding
            dummy_embeddings.append(embedding)
            print(f"Embedded doc{i+1}")
        except Exception as e:
            print("Error in embedding doc", i+1)
            print(e)
            continue

def query_chroma_collection(collection,query_texts):
    query_embedding=client.embeddings.create(input=query_texts,model="text-embedding-ada-002")
    vector = response.data[0].embedding
    results= collection.query(
        query_embedding=[vector],n_results=3
    )
    print("Query result")
    for result in results['documents']:
        print(result)

def embed_and_store(chunks,collection,documents,ids,embeddings):
    #dummy_document=["This is a sample document.", "ChromaDB is awesome!", "AI models are fun."]
   # dummy_id=["doc1","doc2","doc3"]
    print("adding data into chroma")
    collection.add(
    documents= dummy_document,
    embeddings=dummy_embeddings,
    ids=dummy_id)
    print("added data successfully")

#Add and learn data cleaning
#Embedding it with gemni or chroma

if __name__ == "__main__":
    pdf_file = "E:\\AI learning\\pdf-rag-backend\\pdf\\sample.pdf"
    collection= get_chroma_collection("pdf_chunks")
    dummy_chroma_collection(collection,dummy_document)
    query_chroma_collection(collection,"What is ChromaDB")
   # alltext=processpdf(pdf_file)
  #  chunks= split_into_chunks(alltext)
    