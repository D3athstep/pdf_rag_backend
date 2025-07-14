from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import chromadb
from chromadb.config import Settings 
from dotenv import  load_dotenv
import os 
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-V2")
load_dotenv ()
genai.configure(api_key= os.getenv("GOOGLE_API_KEY"))
#for m in genai.list_models():
#    print(m.name, "→ supports", m.supported_generation_methods)

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
            for chunk in cleaned_chunks:
                print(repr(chunk))
                if not cleaned_chunks:
                    print("no chunks remained cleaning")
            return chunks
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
    print("Raw chroma result",results)
    documents = results.get('documents')
    if documents and documents[0]:
        context= "".join(documents[0])
        print("📄 Final context from ChromaDB (preview):", context[:300])
    else:
        print("No relevant document found. Cannot build context")
        context=""
    return context

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

def generate_answer_gemni(context,query):
    prompt =  f"""use the context below to answer the question concisely
    print ("full prompt sent to gemni",prompt)
    Context:
    {context}
    Question:
    {query}
    Answer : """

    model =genai.GenerativeModel(model_name='gemini-1.5-flash')
    try:
        response = model.generate_content(prompt)
        print("Raw gemini response object",response)

        try:
            print("Candidate response",response.candidates[0].content.parts[0].text)
            return response.candidates[0].content.parts[0].text
        except Exception as e:
            print("unable to extract partss[0].text",e)
            return "Could not extract a clean answer from the response."
    except Exception as e:
        print("Error Generating response",e)
        return "Sorry, I could not generate an answer."

def test_gemini_basic():
    prompt="""Use the context below to answer the question concisely.
    Context:
    ChromaDB is an open-source vector database used to store and search text embeddings
    Question:
    What is ChromaDB? 
    Answer:"""

    try:
        model= genai.GenerativeModel('gemini-1.5-flash')
        response= model.generate_content(prompt)
        print ("raw response",response)

        try:
            print("Gemini Answer",response.text)
        except AttributeError:
            print("Trying alternate format")
            print("Answer from parts",response.candidates[0].content.parts[0].text)
    except Exception as e:
        print("Gemini api error",e)

def prepare_chroma():
    

#Add and learn data cleaning
#Embedding it with gemni or chroma
collection= get_chroma_collection("pdf_chunks")

if __name__ == "__main__":


    pdf_file = "E:\\AI learning\\pdf-rag-backend\\pdf\\sample.pdf"
    alltext=processpdf(pdf_file)
    chunks= split_into_chunks(alltext)
    cleaned_chunks=  clean_chunks(chunks)
    ids=generate_ids(chunks)
    embeddings=embed_chunks(chunks,model)
    embed_and_store(collection=collection,chunks=cleaned_chunks,ids=ids,embeddings=embeddings)
    query = "What is the purpose of this document?"
    context= query_chroma_collection(collection,query)
    print("Final context from chromadb",context[:500])
    answer= generate_answer_gemni(context,query)
    print("Answer",answer)
   
    #test_gemini_basic()
   