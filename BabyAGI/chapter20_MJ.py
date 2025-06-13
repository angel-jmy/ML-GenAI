from dotenv import load_dotenv
load_dotenv()

from langchain_experimental.autonomous_agents.baby_agi import BabyAGI
from langchain_experimental.autonomous_agents.baby_agi.task_creation import TaskCreationChain
from langchain_experimental.autonomous_agents.baby_agi.task_execution import TaskExecutionChain
from langchain_experimental.autonomous_agents.baby_agi.task_prioritization import TaskPrioritizationChain

from langchain.chains.llm import LLMChain
from langchain.chains import SimpleSequentialChain

from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from langchain.schema import SystemMessage
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever

# === Embeddings and Vectorstore Setup ===
embedding = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(
    texts=["Make a todo list"],  # ✅ Add a non-empty doc to avoid KeyError
    embedding=embedding,
    metadatas=[{"task": "Make a todo list"}],  # ✅ This fixes the missing "task" key
    ids=["result_0_0"]
)

retriever = vectorstore.as_retriever()

# === LLM Setup ===
llm = ChatOpenAI(temperature=0, model_name="gpt-4")

# === Task Chains ===
task_creation_chain = TaskCreationChain.from_llm(llm)
task_execution_chain = TaskExecutionChain.from_llm(llm)
task_prioritization_chain = TaskPrioritizationChain.from_llm(llm)

# === BabyAGI Agent ===
baby_agi = BabyAGI(
    llm=llm,
    execution_chain=task_execution_chain,
    task_creation_chain=task_creation_chain,
    task_prioritization_chain=task_prioritization_chain,
    vectorstore=vectorstore,
    verbose=True,
    max_iterations=6,
)

# === Run Agent ===
objective = "Analyze the weather in San Francisco today and write a flower storage strategy."
baby_agi.invoke({"objective": objective})
