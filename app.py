import os
import logging
import gradio as gr
from backend import main_pipeline, SOURCES
from langchain_community.chat_models import ChatLlamaCpp
from langchain.schema import AIMessage, HumanMessage

# Configure environment variables
os.environ.update({
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_ENDPOINT": "https://example-endpoint.com",
    "LANGCHAIN_API_KEY": "<YOUR_LANGCHAIN_API_KEY>", 
    "LANGCHAIN_PROJECT": "<YOUR_PROJECT_NAME>",
    "TAVILY_API_KEY": "<YOUR_TAVILY_API_KEY>"  
})

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Load LLM model
llm = ChatLlamaCpp(
    model_path=(
        "/Users/<USERNAME>/.cache/huggingface/hub/models--<MODEL_ORG>--<MODEL_NAME>/"
        "snapshots/<SNAPSHOT_ID>/<MODEL_FILE>.gguf"
    ),
    n_ctx=20000,
    n_gpu_layers=-1,
    verbose=False
)

graph = main_pipeline(llm, SOURCES)

def chat(user_message, chat_history, mode_selection):
    """
    Gradio function to handle user input and produce AI responses.
    """
    if not user_message.strip():
        yield gr.update(), chat_history, ""
        return

    # Display user message in the chat
    chat_history.append([user_message, None])
    yield gr.update(value=chat_history), chat_history, ""

    # Show typing indicator
    chat_history[-1][1] = '<div class="typing-indicator"></div>'
    yield gr.update(value=chat_history), chat_history, ""

    # Prepare conversation state with chosen mode
    initial_state = {
        "messages": [HumanMessage(content=user_message, name="User")],
        "mode": mode_selection.upper()  # "CHAT", "WEB", or "RAG"
    }
    
    # Stream the LLM response
    for s in graph.stream(initial_state, config={"configurable": {"thread_id": "1"}}):
        if "__end__" not in s:
            print(s)
            print('\n')
            chat_data = s.get("Generator", {}) or s.get("Chat", {}) or s.get("WebGenerator", {})

            if "messages" in chat_data:
                for m in chat_data["messages"]:
                    if isinstance(m, AIMessage):
                        chat_history[-1][1] = m.content
                        yield gr.update(value=chat_history), chat_history, ""

# Updated CSS to match the desired style
css = '''
#chat-window {
    background-color: #f8f9fa;
    height: 450px !important;
    overflow-y: auto;
    border: 1px solid #dee2e6;
    padding: 15px;
    border-radius: 10px;
    display: flex;
    flex-direction: column;
    box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.1);
    position: relative;
    max-width: 800px !important;    /* Decreased width */
    margin: 0 auto !important;     /* Centers the chat window */
}

.typing-indicator {
    width: 40px;
    height: 30px;
    position: relative;
    display: inline-block;
    text-align: center;
}

.typing-indicator::after {
    content: '...';
    font-size: 24px;
    font-weight: bold;
    animation: blink 1s infinite;
}

@keyframes blink {
    0% { opacity: .2; }
    20% { opacity: 1; }
    100% { opacity: .2; }
}

#user-input-container {
    display: flex;
    align-items: center;
    margin-top: 10px;
    padding: 0 10px;
    box-sizing: border-box;
    max-width: 800px !important;    /* Decreased width */
    margin: 0 auto !important;     /* Centers the chat window */

}

#user-input {
    flex-grow: 1;
    height: 80px !important;
    border: 1px solid #007bff;
    border-top-left-radius: 25px;
    border-bottom-left-radius: 25px;
    padding: 12px 15px;
    font-size: 16px;
    outline: none;
    box-shadow: none;
}

#submit-query {
    height: 70px !important;
    width: 80px;
    border: 1px solid #007bff;
    border-left: none;
    border-top-right-radius: 25px;
    border-bottom-right-radius: 25px;
    background-color: #007bff;
    color: white;
    font-size: 16px;
    font-weight: bold;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: background-color 0.3s ease;
}

#submit-query:hover {
    background-color: #0056b3;
}

/* Container around the radio buttons */
.mode-toggle-container {
    display: flex;
    align-items: center;
    justify-content: center;
    margin: 10px 0;
    padding: 5px;
    background: #f0f0f0;
    border-radius: 25px;
    width: fit-content;
    margin-left: auto;
    margin-right: auto;
}

/* The main wrapper for the radio components */
.mode-toggle {
    display: flex;
    gap: 4px; /* space between the radio options */
    justify-content: center;
    align-items: center;
}

/* Each radio item is typically wrapped like:
   <label class="gradio-radio">
       <input type="radio" ...>
       <span>CHAT</span>
   </label>
*/
.mode-toggle label.gradio-radio {
    position: relative;
    cursor: pointer;
    border-radius: 16px;
    color: #666;
    font-size: 14px;
    font-weight: 500;
    padding: 8px 20px;
    transition: color 0.3s ease, background-color 0.3s ease;
}

/* Hide the native radio input */
.mode-toggle label.gradio-radio input[type="radio"] {
    display: none;
}

/* Highlight the <span> when radio is checked */
.mode-toggle label.gradio-radio input[type="radio"]:checked + span {
    background-color: #007bff !important;
    color: #fff !important;
    border-radius: 16px;
    padding: 8px 20px;
}
'''

# Build Gradio UI
with gr.Blocks(css=css) as demo:
    gr.HTML("""
    <div style="text-align: center; margin-bottom: 20px;">
        <h1 style="margin-bottom: 5px;">Hi, I'm Greg.</h1>
        <p style="font-size: 18px;">How can I help you today?</p>
    </div>
    """)

    chatbot = gr.Chatbot(elem_id="chat-window")

    with gr.Row(elem_id="user-input-container"):
        user_input = gr.Textbox(
            placeholder="Type your question here...",
            elem_id="user-input",
            show_label=False
        )

    # Mode selection: keep "CHAT", replace 'Search' with 'WEB', add 'RAG'
    with gr.Row(elem_id="mode-toggle-container", elem_classes=["mode-toggle-container"]):
        mode_selector = gr.Radio(
            choices=["CHAT", "WEB", "RAG"],  # 'Search' replaced with 'WEB'
            value="CHAT",
            label="",
            elem_classes=["mode-toggle"],
            container=False
        )

    chat_history = gr.State([])

    user_input.submit(
        chat,
        inputs=[user_input, chat_history, mode_selector],
        outputs=[chatbot, chat_history, user_input]
    )

demo.launch()
