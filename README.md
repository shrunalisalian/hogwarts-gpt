# 🧙‍♂️ **GPT for Harry Potter: Generating Magic with Transformers**  

**Skills:** Natural Language Processing (NLP), Transformers, GPT, Deep Learning  

---

## 🚀 **Project Overview**  
What if we could **generate new Harry Potter stories** using AI? This project explores the power of **transformers and autoregressive models** to generate text in the **style of J.K. Rowling's Harry Potter series**.  

Using **GPT-style models**, this project demonstrates:  
✅ **Language modeling using character-level and word-level embeddings**  
✅ **Building a transformer-based text generator from scratch**  
✅ **Fine-tuning models to capture the writing style of Harry Potter**  
✅ **Experimenting with different architectures, including the Bigram model**  

This project is perfect for **machine learning and NLP roles**, showcasing deep learning expertise in **language modeling, sequence generation, and transformer-based architectures**.  

---

## 🎯 **Key Objectives**  
✔ **Understand and implement GPT-style text generation**  
✔ **Experiment with different transformer-based architectures**  
✔ **Analyze and improve model performance with training techniques**  
✔ **Generate coherent, Harry Potter-style text using AI**  

---

## 📖 **Dataset & Preprocessing**  
This project uses **text from Harry Potter books** as training data. The text is tokenized into sequences and fed into a model that predicts the next word (or character).  

🔹 **Tokenization** – Converting text into a numerical format  
🔹 **Vocabulary Creation** – Learning word/character embeddings  
🔹 **Contextual Sequences** – Training the model on short sequences  

✅ **Example: Tokenizing Text for Training**  
```python
import torch
from transformers import GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

text = "Harry Potter and the Sorcerer's Stone"
tokens = tokenizer.encode(text, return_tensors="pt")

print(tokens)
```
💡 **Why GPT-2?**  
- **Pretrained on large-scale data** → Faster training, better context retention  
- **Autoregressive** → Predicts text step-by-step like a real writer  

---

## 🔥 **Building the Model: GPT for Harry Potter**  
This project explores multiple **text generation models**, including:  

| **Model** | **Description** |
|-----------|----------------|
| **Bigram Model** | Predicts next word based on the last one (simple baseline) |
| **GPT-Style Transformer** | Uses attention mechanisms for long-range text coherence |
| **Fine-Tuned GPT-2** | Trained on Harry Potter text for better style emulation |

✅ **Example: Implementing a Simple Bigram Model**  
```python
import torch.nn as nn

class BigramModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, x):
        return self.embedding(x)

model = BigramModel(vocab_size=5000)
```
💡 **Key Learning:** **The Bigram Model produces random, incoherent text** because it lacks long-term memory.  

---

## 🎭 **Experimenting with Transformer-Based Models**  
🔹 **Self-Attention Mechanisms** – Helps retain long-range dependencies  
🔹 **Positional Encodings** – Adds order-awareness to sequences  
🔹 **Training with GPU Acceleration** – Uses `CUDA` for faster processing  

✅ **Example: Training a Transformer Model**  
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

input_text = "Hogwarts is a place where"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids, max_length=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
💡 **Why Transformers?**  
✔ **Handles long-range dependencies**  
✔ **Can be fine-tuned for domain-specific text (e.g., Harry Potter books)**  

---

## 📊 **Model Evaluation & Results**  
### **Challenges & Key Learnings:**  
✔ **Bigram model generates nonsense** because it lacks context  
✔ **Vanilla GPT-2 generates readable but generic text**  
✔ **Fine-tuned GPT-2 starts capturing the Harry Potter style**  

✅ **Example: Generating a Harry Potter-style Story**  
```python
input_text = "Harry walked into the Forbidden Forest and"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids, max_length=100)
print(tokenizer.decode(output[0], skip_special_tokens=True))
```
💡 **Expected Output:**  
"Harry walked into the Forbidden Forest and saw a strange glow ahead. He stepped forward cautiously, his wand raised. Suddenly, a voice whispered from the darkness, 'You shouldn't be here, Potter.' Harry turned, his heart pounding..."  

---

## 🔮 **Future Enhancements**  
🔹 **Train a fine-tuned GPT-2 on all Harry Potter books** for better coherence  
🔹 **Use Reinforcement Learning (RLHF) to improve output quality**  
🔹 **Experiment with GPT-3 or LLaMA for more advanced generation**  
🔹 **Create an interactive chatbot where users can roleplay as wizards**  

---

## 🎯 **Why This Project Stands Out for NLP & ML Roles**  
✔ **Demonstrates deep learning & NLP expertise**  
✔ **Applies cutting-edge transformer models for text generation**  
✔ **Showcases hands-on model fine-tuning and evaluation**  
✔ **Highlights AI’s creative potential for storytelling & entertainment**  

---

## 🛠 **How to Run This Project**  
1️⃣ Clone the repo:  
   ```bash
   git clone https://github.com/shrunalisalian/gpt-harry-potter.git
   ```
2️⃣ Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```
3️⃣ Run the Jupyter Notebook:  
   ```bash
   jupyter notebook HarryPotter.ipynb
   ```

---

## 📌 **Connect with Me**  
- **LinkedIn:** [Shrunali Salian](https://www.linkedin.com/in/shrunali-salian/)  
- **Portfolio:** [Your Portfolio Link](#)  
- **Email:** [Your Email](#)  

---

References: https://www.youtube.com/watch?v=kCc8FmEb1nY&t=3344s
