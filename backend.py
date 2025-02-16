# backend.py
import logging
import functools
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import AIMessage, BaseMessage, SystemMessage, HumanMessage
from langchain_core.messages import SystemMessage, HumanMessage, RemoveMessage
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS, Annoy
from langgraph.graph import StateGraph, START, END
from langchain.retrievers.document_compressors import FlashrankRerank
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.document_loaders import WikipediaLoader
from langchain_community.tools import TavilySearchResults


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SOURCES = {
    "web": [
        "https://lilianweng.github.io/posts/2017-06-21-overview/",
        "https://lilianweng.github.io/posts/2021-05-31-contrastive/",
        "https://lilianweng.github.io/posts/2021-09-25-train-large/",
        "https://lilianweng.github.io/posts/2024-07-07-hallucination/",
        "https://www.azavea.com/blog/2017/05/30/deep-learning-on-aerial-imagery/"
    ],
    "pdf": []
}

# State + Node Helper Classes
class AgentState(MessagesState):
    summary: str
    mode: str

def llm_chain_node(state, llm_chain, name):
    """
    Generic LLM chain node. 
    Adds any existing conversation summary as context, 
    then calls the chain to produce an AIMessage.
    """
    summary = state.get("summary", "")
    if summary:
        system_message = f"Previous conversation summary: {summary}"
        messages = [SystemMessage(content=system_message)] + state["messages"]
    else:
        messages = state["messages"]
        
    result = llm_chain.invoke({"messages": messages})
    return {"messages": [AIMessage(content=result.content, name=name)]}

def web_retriever_node(state, name="WebRetriever"):
    """
    Retrieves relevant information from web sources and Wikipedia.
    Returns formatted context with clean, readable output.
    """
    question = state["messages"][-1].content
    
    # Initialize result sections
    web_results = []
    wiki_results = []
    
    # 1) Search the web using Tavily
    try:
        tavily_search = TavilySearchResults(max_results=2)
        docs = tavily_search.invoke(question)
        
        if isinstance(docs, (list, tuple)):
            for doc in docs:
                if isinstance(doc, dict):
                    url = doc.get("url", "")
                    content = doc.get("content", "").strip()
                    if content:
                        # Clean the content - remove excessive whitespace and normalize spacing
                        content = " ".join(content.split())
                        web_results.append({
                            "url": url,
                            "content": content
                        })
    except Exception as e:
        logger.error(f"Web search error: {str(e)}")
        web_results.append({
            "error": "Web search temporarily unavailable"
        })

    # 2) Search Wikipedia
    try:
        wiki_loader = WikipediaLoader(query=question, load_max_docs=1)
        wiki_docs = wiki_loader.load()
        
        for doc in wiki_docs:
            title = doc.metadata.get("source", "").replace("https://en.wikipedia.org/wiki/", "")
            content = " ".join(doc.page_content.split())  # Clean whitespace
            if content:
                wiki_results.append({
                    "title": title,
                    "content": content
                })
    except Exception as e:
        logger.error(f"Wikipedia search error: {str(e)}")
        wiki_results.append({
            "error": "Wikipedia search temporarily unavailable"
        })

    # Format the combined output
    output_parts = []
    
    # Add web results section
    output_parts.append("WEB SEARCH RESULTS:")
    if web_results:
        for i, result in enumerate(web_results, 1):
            if "error" in result:
                output_parts.append(f"[!] {result['error']}")
            else:
                output_parts.append(f"[{i}] {result['content']}")
                if result['url']:
                    output_parts.append(f"Source: {result['url']}")
            output_parts.append("")  # Add spacing between results
    else:
        output_parts.append("No relevant web results found.")
        output_parts.append("")

    # Add Wikipedia results section
    output_parts.append("WIKIPEDIA RESULTS:")
    if wiki_results:
        for i, result in enumerate(wiki_results, 1):
            if "error" in result:
                output_parts.append(f"[!] {result['error']}")
            else:
                output_parts.append(f"[{i}] {result['title']}")
                output_parts.append(result['content'])
            output_parts.append("")  # Add spacing between results
    else:
        output_parts.append("No relevant Wikipedia results found.")

    # Combine all parts with proper spacing
    combined_output = "\n".join(output_parts).strip()

    return {
        "messages": [
            AIMessage(content=combined_output, name=name)
        ]
    }

def web_generator_node(state, llm_chain, name="WebGenerator"):
    """
    After retrieving context from 'WebRetriever',
    use the provided chain to generate an answer.
    """
    result = llm_chain.invoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result.content, name=name)]}


# RAG Nodes
def retrieval_node(state, retriever, reranker, name="Retriever"):
    question = state["messages"][-1].content
    
    retrieved_docs = retriever.invoke(question)
    reranked_docs = reranker.compress_documents(retrieved_docs, query=question)
    reranked_docs = [doc for doc in reranked_docs if doc.metadata.get('relevance_score', 0) > 0.7]

    if not reranked_docs:
        context = "No relevant context found. Respond with 'I don't know.'"
    else:
        context = "\n\n".join(doc.page_content for doc in reranked_docs)
    
    return {
        "messages": [AIMessage(content=f" - Context:\n{context}", name=name)]
    }

# Summarization
def summarize_node(state, llm, name="Summarizer"):
    """
    Summarizes the conversation so far. 
    Removes older messages from memory except the last two.
    """
    summary = state.get("summary", "")
    if summary:
        summary_prompt = (
            f"Previous conversation summary: {summary}\n\n"
            "Extend or refine the summary by incorporating the new messages above:"
        )
    else:
        summary_prompt = "Create a concise summary of the conversation above:"
    
    # Create summarization chain
    summary_chain = create_llm_chain(llm, summary_prompt)
    messages = state["messages"] + [HumanMessage(content=summary_prompt)]
    
    # Generate new summary
    result = summary_chain.invoke({"messages": messages})
    
    # Keep only the last 2 messages in memory
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    
    return {
        "summary": result.content,
        "messages": delete_messages
    }

def should_continue(state):
    """Determine whether to continue conversation or summarize (after ~ messages)."""
    messages = state["messages"]
    if len(messages) > 10:
        return "Summarizer"
    return END

# General Chat Node
def chat_node(state, llm_chain, name="Chat"):
    """
    A simple chat node that doesn't use retrieval at all.
    It just calls the LLM with the user messages.
    """
    result = llm_chain.invoke({"messages": state["messages"]})
    return {"messages": [AIMessage(content=result.content, name=name)]}

# Create LLM Chains
def create_llm_chain(llm, system_prompt):
    """
    Creates a ChatPromptTemplate -> LLM pipeline
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages")
    ])
    return prompt | llm

# Check mode
def choose_mode_node(state):
    mode = state.get("mode", "CHAT").upper()
    if mode == "CHAT":
        return "Chat"
    elif mode == "WEB":
        return "WebRetriever"
    else:
        return "Retriever"

# Document Helpers
def load_and_split_documents(sources, chunk_size=1000, chunk_overlap=200):
    all_chunks = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    for source_type, loader_class in {
        'web': WebBaseLoader,
        'pdf': PyPDFLoader,
        'text': TextLoader
    }.items():
        for path in sources.get(source_type, []):
            loader = loader_class(path)
            docs = loader.load()
            for doc in docs:
                doc.metadata = {'id': path}  # Add metadata
            all_chunks.extend(text_splitter.split_documents(docs))

    return all_chunks

def create_vector_store(splits, method="faiss"):
    embeddings = HuggingFaceBgeEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L12-v2",
        model_kwargs={'device': 'mps', 'trust_remote_code': True},
        encode_kwargs={'normalize_embeddings': True}
    )
    stores = {
        "chroma": Chroma,
        "faiss": FAISS,
        "annoy": Annoy
    }
    if method not in stores:
        raise ValueError("Invalid method. Use 'chroma', 'faiss', or 'annoy'.")
    return stores[method].from_documents(splits, embedding=embeddings)

# Main Pipeline
def main_pipeline(llm, sources, vector_store_method="faiss", chunk_size=1000, chunk_overlap=200):
    # Create memory
    memory = MemorySaver()
    
    # Prepare RAG resources
    splits = load_and_split_documents(sources, chunk_size, chunk_overlap)
    vector_store = create_vector_store(splits, method=vector_store_method)
    retriever = vector_store.as_retriever(k=10)
    reranker = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=5)
    
    # 1) RAG chain
    rag_system_prompt = (
        "You are a question-answering assistant. "
        "Answer the 'User' query based on the context provided by 'Retriever'. "
        "If context is insufficient or nonexistent, respond 'I don't know'. "
        "Do not make assumptions or add external knowledge beyond the provided context."
    )
    rag_chain = create_llm_chain(llm, rag_system_prompt)
    rag_node = functools.partial(llm_chain_node, llm_chain=rag_chain, name="Generator")
    
    # 2) General Chat chain
    chat_system_prompt = (
        "You are a helpful AI assistant. Have a conversation with the user. "
        "You do not have any external context. Respond to the best of your knowledge."
    )
    chat_chain = create_llm_chain(llm, chat_system_prompt)
    chat_node_partial = functools.partial(chat_node, llm_chain=chat_chain, name="Chat")
    
    # 3) Summarization partial
    summarize = functools.partial(summarize_node, llm=llm)
    
    # ---------------- NEW: WEB chain ---------------- #
    web_system_prompt = (
        "You are a question-answering assistant. "
        "You have web search results and Wikipedia excerpts as context. "
        "Answer the user's question using only that context. "
        "If context is insufficient or nonexistent, respond 'I don't know'. "
        "Do not make things up. Provide helpful, concise answers."
    )
    web_chain = create_llm_chain(llm, web_system_prompt)
    web_node_partial = functools.partial(web_generator_node, llm_chain=web_chain, name="WebGenerator")
    # ------------------------------------------------ #
    
    # Build the workflow
    workflow = StateGraph(AgentState)
    
    # Register existing nodes
    workflow.add_node("Retriever", functools.partial(retrieval_node, retriever=retriever, reranker=reranker))
    workflow.add_node("Generator", rag_node)
    workflow.add_node("Chat", chat_node_partial)
    workflow.add_node("Summarizer", summarize)
    
    # REGISTER NEW WEB NODES
    workflow.add_node("WebRetriever", web_retriever_node)
    workflow.add_node("WebGenerator", web_node_partial)
    
    # Edges
    # START -> CheckMode
    workflow.add_conditional_edges(START, choose_mode_node)
    
    # RAG path: Retriever -> Generator -> Summarizer or END
    workflow.add_edge("Retriever", "Generator")
    workflow.add_conditional_edges("Generator", should_continue)

    # Chat path: Chat -> Summarizer or END
    workflow.add_conditional_edges("Chat", should_continue)
    
    # WEB path: WebRetriever -> WebGenerator -> Summarizer or END
    workflow.add_edge("WebRetriever", "WebGenerator")
    workflow.add_conditional_edges("WebGenerator", should_continue)

    # Summarizer -> END
    workflow.add_edge("Summarizer", END)
    
    # Compile and return
    return workflow.compile(checkpointer=memory)
