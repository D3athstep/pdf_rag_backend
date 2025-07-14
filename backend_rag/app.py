from flask import Flask,request,jsonify
from backend_rag.rag_engine import query_chroma_collection,generate_answer_gemni,collection
app = Flask(__name__)
print("Collection count:", collection.count())

@app.route('/ask',methods=['POST'])
def ask():
    data = request.json
    query=data.get('query')
    context = query_chroma_collection(collection,query)
    print("Query:", query)
    print("Context (first 300 chars):", context[:300] if context else "No context found")
    answer= generate_answer_gemni(context,query)
    return jsonify({"answer":answer})

if __name__ == "__main__":
    app.run()