from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import chromadb
from chromadb.config import Settings 
#from dotenv import  load_dotenv
#import os 
#from openai import OpenAI
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-V2")
#load_dotenv ()
#client=OpenAI(api_key= os.getenv("OPENAI_API_KEY"))

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
    return clean_chunks(chunks)

def clean_chunks(chunks):
        cleaned_chunks=[]
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
            cleaned_chunks.append(chunk)
            print(f"Filtered {len(chunks)} → {len(cleaned_chunks)} chunks after removing short ones.")
        return cleaned_chunks
        print("Before:", repr(chunk))
        print("After :", repr(cleaned_chunks))

def get_chroma_collection(name):
    #connect to chroma client locally
    client = chromadb.Client(Settings())
    collection=  client.get_or_create_collection(name)
    return collection


def dummy_chroma_collection(collection,dummy_document):
    # appending data into dummy_embedding 
    dummy_embeddings=[]
    for i,doc in enumerate(dummy_document):
        try:
            embedding = model.encode(doc)
            dummy_embeddings.append(embedding)
            print(f"Embedded doc{i+1}")
        except Exception as e:
            print("Error in embedding doc", i+1)
            print(e)
            continue
    return dummy_embeddings

def query_chroma_collection(collection,query_texts):
    vector = model.encode(query_texts)
    results= collection.query(
        query_embeddings=[vector],n_results=3
    )
    print("Query result")
    for result in results['documents']:
        print(result)

def generate_ids(chunks):
    if chunks is None:
        print("Chunk is none-  Check  split_into_chunks funtions")
        return
    return [f"chunk_{i}" for i in range(len(chunks))]

def embed_chunks(chunks,model):
    embeddings=[]

    for chunk in chunks:
        embedding= model.encode(chunk)
        embeddings.append(embedding)
    return embeddings

def embed_and_store(collection,chunks,ids,embeddings):
    print("adding data into chroma")
    collection.add(
    documents= chunks,
    embeddings=embeddings,
    ids=ids)
    print("added data successfully")
    print(f"Collection now has {collection.count()} documents.")

#Add and learn data cleaning
#Embedding it with gemni or chroma

if __name__ == "__main__":
    pdf_file = "E:\\AI learning\\pdf-rag-backend\\pdf\\sample.pdf"
    collection= get_chroma_collection("pdf_chunks")
    alltext=processpdf(pdf_file)
    chunks= split_into_chunks(alltext)
    ids=generate_ids(chunks)
    embeddings=embed_chunks(chunks,model)
    dummy_chroma_collection(collection,dummy_document)
    embed_and_store(collection=collection,chunks=chunks,ids=ids,embeddings=embeddings)
    query_chroma_collection(collection,"What is the purpose of this document?")
    
    