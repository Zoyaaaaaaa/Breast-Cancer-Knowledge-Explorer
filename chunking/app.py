import os
from typing import List
from PyPDF2 import PdfReader
from rich import print

from langchain_core.embeddings import Embeddings
from langchain_core.pydantic_v1 import BaseModel
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import create_extraction_chain_pydantic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub

from agentic_chunker import AgenticChunker
from google import genai
from dotenv import load_dotenv
load_dotenv()
# --------------- CONFIG --------------- #
# GEMINI_API_KEY = "AIzaSyAOkCBY-kR4giHrV6l8XrZzVha50I9f6ZM"
GEMINI_API_KEY=os.environ.get("GEMINI_API_KEY")
# genai.configure(api_key=GEMINI_API_KEY)

# --------------- Gemini Embedding Wrapper --------------- #
class GeminiEmbeddingWrapper(Embeddings):
    def __init__(self, model="gemini-embedding-exp-03-07", api_key=None):
        self.client = genai
        self.model = model
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for text in texts:
            res = self.client.embed_content(
                model=self.model,
                content=text
            )
            embeddings.append(res['embedding'])
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        res = self.client.embed_content(
            model=self.model,
            content=text
        )
        return res['embedding']

# --------------- PDF Text Extraction --------------- #
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n\n"
    return text

# --------------- Proposition Extraction --------------- #
class Sentences(BaseModel):
    sentences: List[str]

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.4
)

from langchain_core.prompts import ChatPromptTemplate

# Simple prompt to extract key propositions/sentences
prompt = ChatPromptTemplate.from_template("""
Extract key informative propositions or factual sentences from the following text:

{text}

Return them as a list of short standalone statements.
""")

chain = prompt | llm | StrOutputParser()

def get_propositions(text):
    output = chain.invoke({"text": text})
    return [s.strip() for s in output.split("\n") if s.strip()]


# --------------- RAG Chain --------------- #
def rag(documents, collection_name):
    embeddings = GeminiEmbeddingWrapper(api_key=GEMINI_API_KEY)

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name
    )

    retriever = vectorstore.as_retriever()

    prompt_template = """Answer the question based only on the following context:
    {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(prompt_template)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    result = chain.invoke("What is the use of Text Splitting?")
    print(f"[bold green]RAG Result:[/bold green] {result}")

# --------------- Main Flow --------------- #
def process_pdf(pdf_path, output_txt_path="chunked_output.txt"):
    print(f"[bold]Reading PDF:[/bold] {pdf_path}")
    text = extract_text_from_pdf(pdf_path)
    paragraphs = text.split("\n\n")

    text_propositions = []
    for i, para in enumerate(paragraphs[:10]):
        try:
            propositions = get_propositions(para)
            text_propositions.extend(propositions)
            print(f"✔️ Done with paragraph {i}")
        except Exception as e:
            print(f"[red]❌ Failed on paragraph {i}:[/red] {e}")

    print(f"\n[bold cyan]Total Propositions:[/bold cyan] {len(text_propositions)}")

    print("\n[bold magenta]#### Agentic Chunking ####[/bold magenta]")
    ac = AgenticChunker()
    ac.add_propositions(text_propositions)
    ac.pretty_print_chunks()
    chunks = ac.get_chunks(get_type='list_of_strings')

    # Save to text file
    with open(output_txt_path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk + "\n\n")
    print(f"[bold green]✅ Chunks saved to {output_txt_path}[/bold green]")

    documents = [Document(page_content=chunk, metadata={"source": "local"}) for chunk in chunks]
    rag(documents, "agentic-chunks-gemini")

# --------------- Run It --------------- #
if __name__ == "__main__":
    process_pdf("bc.pdf")
