import os
from flask import Flask, render_template, request
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

os.chdir("C:/Users/Chuyue Shen/OneDrive/Desktop/GCCC_data_science/quiz2/quiz2")

app = Flask(__name__)

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# allow deserialization
embedding = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.load_local(
    "flower_doc_qa_index",
    embedding,
    allow_dangerous_deserialization=True
)

llm = ChatOpenAI(model_name="gpt-4", openai_api_key=api_key)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=False
)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        query = request.form["question"]
        result = qa_chain(query)
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
