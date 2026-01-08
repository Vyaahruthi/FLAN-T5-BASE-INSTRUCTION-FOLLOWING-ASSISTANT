import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-base"


class InstructionEngine:
    def __init__(self, model_name: str):
        self.device = "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def build_prompt(self, user_message: str) -> str:
        return (
            "Answer the following question clearly and concisely:\n\n"
            f"{user_message}"
        )

    def generate_reply(self, user_message: str) -> str:
        prompt = self.build_prompt(user_message)

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=256
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=120,
                do_sample=False
            )

        return self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()


engine = InstructionEngine(MODEL_NAME)


def chat_function(message, history):
    try:
        return engine.generate_reply(message)
    except Exception as e:
        print("Inference error:", e)
        return "Sorry, something went wrong. Please try again."


demo = gr.ChatInterface(
    fn=chat_function,
    title="FLAN-T5 Instruction-Following Assistant",
    description="Instruction-following AI assistant powered by Google's FLAN-T5 model.",
    examples=[
        "Which color will look good to wear on Diwali?",
        "Tell me a fun fact about space.",
        "Explain what machine learning is."
    ]
)

if __name__ == "__main__":
    demo.launch()
