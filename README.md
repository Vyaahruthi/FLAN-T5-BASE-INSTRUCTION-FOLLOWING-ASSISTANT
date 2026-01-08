# FLAN-T5 Instruction-Following Assistant

An instruction-following AI assistant built using **Google’s FLAN-T5** model and deployed as a web application using **Gradio**.  
The project focuses on correct model selection, stable inference, and real-world deployment.

🔗 **Live Demo (Hugging Face Spaces):**  
https://huggingface.co/spaces/Vyaahruthi/flan-t5-instruction-assistant

---

## 📌 Overview

This project demonstrates how to build and deploy an **instruction-tuned language model** for tasks such as:

- Question answering  
- Explanations  
- Simple recommendations  
- General knowledge responses  

Unlike purely conversational models, **FLAN-T5 is instruction-tuned**, making it more suitable for clear and direct answers.

---

## 🧠 Model Choice

- **Model:** `google/flan-t5-base`
- **Architecture:** Encoder–Decoder (Seq2Seq Transformer)
- **Why FLAN-T5?**
  - Instruction-tuned (follows commands better)
  - Stable inference on CPU
  - Suitable for deployment on free hosting platforms

Earlier experimentation with conversational models highlighted their limitations for instruction-based tasks, leading to the final choice of FLAN-T5.

---

## ⚙️ How It Works

1. The user enters a question through a chat interface.
2. The input is converted into an instruction-style prompt.
3. The prompt is tokenized and passed to the FLAN-T5 model.
4. The model generates a response using deterministic decoding.
5. The response is displayed in the web interface.

The system uses **stateless inference**, which ensures stability across multiple user queries.

---

## 🚀 Features

- Instruction-following responses
- Stable multi-query usage
- Lightweight and CPU-friendly
- Clean chat interface
- Defensive error handling
- Deployed and publicly accessible

---

## 🛠️ Tech Stack

- Python  
- Hugging Face Transformers  
- PyTorch  
- Gradio  

---

