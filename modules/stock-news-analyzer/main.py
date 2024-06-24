import json
import os
from dotenv import dotenv_values
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama

ollama = Ollama(
    base_url='http://localhost:11434',
    model="llama3"
)
cfg = dotenv_values(".env")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)


def read_json(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        return data


def get_json_files_in_articles_dir(articles_dir):
    json_files = []
    for root, dirs, files in os.walk(articles_dir):
        for file in files:
            if file.endswith(".json"):
                json_files.append(os.path.join(root, file))
    return json_files


def get_ollama_response(system_prompt, query):
    llm_response = ollama.generate(
        model=cfg["OLLAMA_MODEL"],
        prompt=f"{system_prompt}\n{query}",
    )
    return llm_response["response"]


def get_article_urls_from_news(articles_json_files):
    article_urls = []
    for article_file in articles_json_files:
        json_data = read_json(article_file)
        all_articles = json_data["articles"]
        for article in all_articles:
            article_urls.append(article["url"])
    return article_urls


def scrape_articles(article_urls):
    articles_content = None
    for article in article_urls:
        try:
            loader = WebBaseLoader(article)
            data = loader.load()
            if articles_content is None:
                articles_content = data
            articles_content.extend(data)
        except Exception:
            continue
    return articles_content


if __name__ == "__main__":
    articles_json_files = get_json_files_in_articles_dir("./articles")
    all_article_urls = get_article_urls_from_news(articles_json_files)
    articles_docs = scrape_articles(all_article_urls)
    all_splits = text_splitter.split_documents(articles_docs)
    oembed = OllamaEmbeddings(
        base_url="http://localhost:11434", model="nomic-embed-text"
    )
    vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)
    qachain = RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
    result = qachain.invoke(
        {
            "query": "Summarize the NVIDIA articles and provide a brief summary of the articles"
                     "in about 1000 words"
        }
    )
    print(result['result'])
