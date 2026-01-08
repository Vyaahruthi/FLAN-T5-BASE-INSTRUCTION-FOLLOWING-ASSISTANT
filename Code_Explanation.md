
# 📘 `CODE_EXPLANATION.md`

```md
# Code Explanation – FLAN-T5 Instruction-Following Assistant

This document provides a **complete, line-by-line explanation** of the code used in this project.
It explains **what each part does**, **why it exists**, and **how it contributes to the overall system**.


##📁 Project Structure

```

.
├── app.py
├── requirements.txt
├── README.md
├── CODE_EXPLANATION.md
└── LICENSE

````

---

## 📄 app.py

This file contains the **entire application logic**, including:
- Model loading
- Prompt construction
- Inference
- Web interface creation

---

## 1️⃣ Import Statements

```python
import gradio as gr
````

* Imports **Gradio**, a Python library for building web interfaces.
* Used to create the chat UI without writing HTML, CSS, or JavaScript.
* Handles user input, sessions, and output display.

---

```python
import torch
```

* Imports **PyTorch**, the deep learning framework used by Hugging Face models.
* Required for running the model and managing inference.
* Used to disable gradient computation during inference.

---

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
```

* Imports Hugging Face’s **automatic model loaders**.
* `AutoTokenizer` converts text into tokens and back.
* `AutoModelForSeq2SeqLM` loads encoder–decoder language models like FLAN-T5.

---

## 2️⃣ Model Configuration

```python
MODEL_NAME = "google/flan-t5-base"
```

* Specifies which pretrained model to use.
* FLAN-T5 is **instruction-tuned**, meaning it is trained to follow commands.
* The `base` version is chosen because it balances:

  * Performance
  * Accuracy
  * CPU compatibility for free hosting

---

## 3️⃣ InstructionEngine Class

```python
class InstructionEngine:
```

* Defines a class that encapsulates **all AI-related logic**.
* Separates model logic from UI logic.
* Improves code readability, reuse, and maintainability.

---

## 4️⃣ Constructor (`__init__` method)

```python
def __init__(self, model_name: str):
```

* Runs **once** when the application starts.
* Responsible for loading the model and tokenizer.

---

```python
self.device = "cpu"
```

* Forces the model to run on the CPU.
* Hugging Face free Spaces do not provide GPUs.
* Prevents CUDA-related runtime crashes.

---

```python
self.tokenizer = AutoTokenizer.from_pretrained(model_name)
```

* Loads the tokenizer associated with the selected model.
* The tokenizer:

  * Splits text into tokens
  * Maps tokens to numerical IDs
  * Handles decoding model output back to text

---

```python
self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
```

* Loads the pretrained FLAN-T5 model.
* Moves the model to the CPU device.
* The model weights are fixed and not modified (inference only).

---

## 5️⃣ Prompt Construction

```python
def build_prompt(self, user_message: str) -> str:
```

* Creates an **instruction-style prompt** for the model.
* FLAN-T5 expects clear instructions, not chat history.

---

```python
return (
    "Answer the following question clearly and concisely:\n\n"
    f"{user_message}"
)
```

* Prepends an instruction to the user input.
* Example final prompt sent to the model:

```
Answer the following question clearly and concisely:

Tell me a fun fact about space.
```

* This improves response quality and relevance.

---

## 6️⃣ Generating a Response (Inference Logic)

```python
def generate_reply(self, user_message: str) -> str:
```

* Main inference function.
* Converts user input into a model response.

---

```python
prompt = self.build_prompt(user_message)
```

* Converts raw user input into a structured instruction.
* Ensures consistent input format for the model.

---

```python
inputs = self.tokenizer(
    prompt,
    return_tensors="pt",
    truncation=True,
    max_length=256
)
```

* Tokenizes the prompt into numerical tensors.
* `return_tensors="pt"` → PyTorch format.
* `truncation=True` prevents overly long inputs.
* `max_length=256` sets a safe upper bound.

---

```python
with torch.no_grad():
```

* Disables gradient computation.
* Signals that the model is in **inference mode**, not training.
* Reduces memory usage and speeds up execution.

---

```python
outputs = self.model.generate(
    **inputs,
    max_length=120,
    do_sample=False
)
```

* Generates the model’s output tokens.
* `max_length=120` limits response length.
* `do_sample=False` ensures deterministic output.
* Suitable for factual and instructional responses.

---

```python
return self.tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
).strip()
```

* Converts output tokens back into readable text.
* Removes special tokens like `<pad>` or `<eos>`.
* Strips unnecessary whitespace.

---

## 7️⃣ Engine Initialization

```python
engine = InstructionEngine(MODEL_NAME)
```

* Creates a single instance of the AI engine.
* The model is loaded only once and reused.
* Improves efficiency and performance.

---

## 8️⃣ Chat Function (UI–Model Bridge)

```python
def chat_function(message, history):
```

* This function is called by Gradio for each user message.
* `message` → current user input.
* `history` → previous chat history (ignored intentionally).

---

```python
try:
    return engine.generate_reply(message)
```

* Calls the model inference logic.
* Returns the generated response.

---

```python
except Exception as e:
    print("Inference error:", e)
    return "Sorry, something went wrong. Please try again."
```

* Prevents the app from crashing.
* Logs the error for debugging.
* Displays a user-friendly error message.

---

## 9️⃣ Gradio Chat Interface

```python
demo = gr.ChatInterface(
```

* Creates a chat-style web interface.
* Automatically handles:

  * Input box
  * Message history
  * UI rendering

---

```python
fn=chat_function,
```

* Connects the UI to the backend logic.

---

```python
title="FLAN-T5 Instruction-Following Assistant",
```

* Title displayed at the top of the app.

---

```python
description="Instruction-following AI assistant powered by Google's FLAN-T5 model.",
```

* Short description shown under the title.

---

```python
examples=[
    "Which color will look good to wear on Diwali?",
    "Tell me a fun fact about space.",
    "Explain what machine learning is."
]
```

* Example prompts to guide users.
* Clickable and auto-filled in the input box.

---

## 🔟 Application Launch

```python
if __name__ == "__main__":
    demo.launch()
```

* Starts the Gradio server.
* On Hugging Face Spaces, this exposes the app publicly.
* Makes the application accessible via a URL.

---

## 📄 requirements.txt Explanation

```txt
gradio==6.2.0
torch
transformers
sentencepiece
```

* `gradio` → Web UI
* `torch` → Model execution
* `transformers` → Model and tokenizer
* `sentencepiece` → Required for T5 tokenization

---



