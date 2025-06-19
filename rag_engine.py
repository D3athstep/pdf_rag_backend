from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def processpdf(pdf_path):
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
    text_splitter= RecursiveCharacterTextSplitter(
        chunk_size= 1000,
        chunk_overlap =200
    )
    chunks = text_splitter.split_text(text)
    for i,chunk in enumerate(chunks[:5]):
        print("chunks",i+1)
        print (chunk)
        print("-" * 50)

#Add and learn data cleaning
#Embedding it with gemni or chroma

if __name__ == "__main__":
    pdf_file = "E:\\AI learning\\pdf-rag-backend\\pdf\\sample.pdf"
    alltext=processpdf(pdf_file)
    chunks= split_into_chunks(alltext)
    