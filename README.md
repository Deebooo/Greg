# Agentic-Greg Chatbot

## Overview
Agentic-Greg is an AI-powered chatbot designed to facilitate natural conversations, provide web-based knowledge retrieval, and perform retrieval-augmented generation (RAG) using vector-based embeddings. The chatbot leverages `LangChain`, `Gradio`, and `LlamaCpp` to deliver intelligent responses.

## Features
- **Conversational AI**: Utilizes LLM-based responses for free-flowing conversations.
- **Web Search Integration**: Fetches real-time information using web search APIs.
- **Retrieval-Augmented Generation (RAG)**: Improves answer accuracy using embedded document retrieval.
- **Interactive UI**: Built using `Gradio` for an intuitive user experience.
- **Persistent Conversation Memory**: Summarizes long conversations and maintains context.

## Installation
### Prerequisites
Ensure you have Python 3.8+ installed. Then, install dependencies using:

```sh
pip install -r requirements.txt
```

## Project Structure
```
.
├── app.py                # Main entry point for Gradio-based chatbot UI
├── backend.py            # Core chatbot logic and pipeline definition
├── requirements.txt      # Python dependencies
```

## Usage
### Running the Chatbot
To start the chatbot, execute the following command:

```sh
python app.py
```

This will launch a `Gradio` interface where users can interact with the chatbot.

## Configuration
### API Keys
Ensure you set up the required API keys in the `app.py` file:

```python
os.environ.update({
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_ENDPOINT": "https://api.smith.langchain.com",
    "LANGCHAIN_API_KEY": "your-langchain-api-key",
    "LANGCHAIN_PROJECT": "Agentic-Greg",
    "TAVILY_API_KEY": "your-tavily-api-key"
})
```
Replace `your-langchain-api-key` and `your-tavily-api-key` with actual keys.

### Model Configuration
Modify the Llama model path in `app.py`:

```python
llm = ChatLlamaCpp(
    model_path="path/to/your/llama-model.gguf",
    n_ctx=20000,
    n_gpu_layers=-1,
    verbose=False
)
```

## Available Modes
- **CHAT**: Standard conversation mode with no external retrieval.
- **WEB**: Uses web search to enhance responses with real-time information.
- **RAG**: Uses vector database retrieval to provide more accurate and context-rich answers.

## Technologies Used
- **LangChain** - Manages conversation flow, retrieval, and LLM interactions.
- **Gradio** - Provides a user-friendly web-based interface.
- **FAISS** - Enables efficient vector-based document retrieval.
- **LlamaCpp** - Runs lightweight LLM models efficiently.
- **Tavily API** - Fetches real-time web search results.

## Troubleshooting
If you encounter issues:
- Ensure all dependencies are installed via `pip install -r requirements.txt`.
- Verify API keys are correctly set.
- Check model paths and available system memory.
- Use logging (`app.log`) for debugging errors.

## License
This project is licensed under the MIT License.
