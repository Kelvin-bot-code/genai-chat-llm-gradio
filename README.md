## Ex:8 Development and Deployment of a 'Chat with LLM' Application Using the Gradio Blocks Framework

### AIM:
To design and deploy a "Chat with LLM" application by leveraging the Gradio Blocks UI framework to create an interactive interface for seamless user interaction with a large language model.

### PROBLEM STATEMENT:
The objective is to develop an intuitive and interactive interface that enables users to communicate with a large language model (LLM) effectively. The application should handle user inputs, generate responses from the LLM, and display the results in real-time.
### DESIGN STEPS:

#### STEP 1:Set Up the Environment
+ Install the necessary libraries, such as gradio and transformers.
+ Choose an appropriate LLM (e.g., OpenAI's GPT, Hugging Face's GPT-2/3.5/4).
+ Ensure GPU support for faster inference, if required.
#### STEP 2:Create the Gradio Blocks Interface
+ Use gr.Blocks to design a modular and interactive layout.
+ Define the components such as Textbox for user input, Chatbot for displaying messages, and Button for interaction.

#### STEP 3: Integrate the LLM with the Interface
+ Test the application locally to ensure smooth functionality.
+ Deploy the app using platforms like Hugging Face Spaces, Streamlit Cloud, or a custom server.

### PROGRAM:
```py
import gradio as gr
import os
import io
import IPython.display
from PIL import Image
import base64 
import requests 
requests.adapters.DEFAULT_TIMEOUT = 60

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file
hf_api_key = os.environ['HF_API_KEY']

# Helper function
import requests, json
from text_generation import Client

#FalcomLM-instruct endpoint on the text_generation library
client = Client(os.environ['HF_API_FALCOM_BASE'], headers={"Authorization": f"Basic {hf_api_key}"}, timeout=120)

def format_chat_prompt(message, chat_history, instruction):
    prompt = f"System:{instruction}"
    for turn in chat_history:
        user_message, bot_message = turn
        prompt = f"{prompt}\nUser: {user_message}\nAssistant: {bot_message}"
    prompt = f"{prompt}\nUser: {message}\nAssistant:"
    return prompt
def respond(message, chat_history, instruction, temperature=0.7):
    prompt = format_chat_prompt(message, chat_history, instruction)
    chat_history = chat_history + [[message, ""]]
    stream = client.generate_stream(prompt,
                                      max_new_tokens=1024,
                                      stop_sequences=["\nUser:", "<|endoftext|>"],
                                      temperature=temperature)
                                      #stop_sequences to not generate the user answer
    acc_text = ""
    #Streaming the tokens
    for idx, response in enumerate(stream):
            text_token = response.token.text

            if response.details:
                return

            if idx == 0 and text_token.startswith(" "):
                text_token = text_token[1:]

            acc_text += text_token
            last_turn = list(chat_history.pop(-1))
            last_turn[-1] += acc_text
            chat_history = chat_history + [last_turn]
            yield "", chat_history
            acc_text = ""

with gr.Blocks() as demo:
    chatbot = gr.Chatbot(height=240) #just to fit the notebook
    msg = gr.Textbox(label="text")
    with gr.Accordion(label="Advanced options",open=False):
        system = gr.Textbox(label="System message", lines=2, value="A conversation between a user and an LLM-based AI assistant. The assistant gives helpful and honest answers.")
        temperature = gr.Slider(label="temperature", minimum=0.1, maximum=1, value=0.7, step=0.1)
    btn = gr.Button("CHATt")
    clear = gr.ClearButton(components=[msg, chatbot], value="Clear console")

    btn.click(respond, inputs=[msg, chatbot, system], outputs=[msg, chatbot])
    msg.submit(respond, inputs=[msg, chatbot, system], outputs=[msg, chatbot]) #Press enter to submit

gr.close_all()
demo.queue().launch()    
```
### OUTPUT:

![Screenshot 2025-05-21 001338](https://github.com/user-attachments/assets/c4694a88-66e8-4d5b-8a96-b0668f8ad5c8)


### RESULT:
The "Chat with LLM" application was successfully designed and deployed. The Gradio Blocks framework provided a user-friendly interface, ensuring seamless communication with the large language model.
