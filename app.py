print("FAQ CHATBOT STARTED")
import tkinter as tk
from tkinter import scrolledtext
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# FAQ questions and answers
faqs = [
    {
        "question": "What is a laptop?",
        "answer": "A laptop is a portable computer with a screen, keyboard, battery, and processor."
    },
    {
        "question": "What is RAM?",
        "answer": "RAM stands for Random Access Memory. It temporarily stores data used by running programs."
    },
    {
        "question": "What is an SSD?",
        "answer": "SSD stands for Solid State Drive. It is a fast storage device used to store files and applications."
    },
    {
        "question": "What is a CPU?",
        "answer": "CPU stands for Central Processing Unit. It executes instructions and performs calculations."
    },
    {
        "question": "What is a GPU?",
        "answer": "GPU stands for Graphics Processing Unit. It processes graphics and can accelerate AI tasks."
    },
    {
        "question": "What is Python?",
        "answer": "Python is a high-level programming language used for AI, automation, web development, and data science."
    },
    {
        "question": "What is artificial intelligence?",
        "answer": "Artificial Intelligence is a field of computer science that enables machines to perform tasks requiring human-like intelligence."
    },
    {
        "question": "What is machine learning?",
        "answer": "Machine learning is a branch of AI that enables computers to learn patterns from data."
    },
    {
        "question": "What is an operating system?",
        "answer": "An operating system manages computer hardware and software. Examples include Windows, Linux, and macOS."
    },
    {
        "question": "How can I improve laptop battery life?",
        "answer": "Reduce screen brightness, close unnecessary applications, enable battery saver, and disconnect unused devices."
    },
    {
        "question": "What is a motherboard?",
        "answer": "A motherboard is the main circuit board that connects the CPU, RAM, storage, and other computer components."
    },
    {
        "question": "What is cloud computing?",
        "answer": "Cloud computing provides computing resources such as servers, storage, and software over the internet."
    },
    {
        "question": "What is an algorithm?",
        "answer": "An algorithm is a step-by-step procedure for solving a problem or completing a task."
    },
    {
        "question": "What is programming?",
        "answer": "Programming is the process of writing instructions that tell a computer how to perform tasks."
    },
    {
        "question": "How much RAM is good for programming?",
        "answer": "8 GB is suitable for basic programming, while 16 GB is a good starting point for most development tasks."
    }
]


# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Prepare FAQ questions
questions = [preprocess(faq["question"]) for faq in faqs]

# Convert questions into numerical vectors using TF-IDF
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
faq_vectors = vectorizer.fit_transform(questions)


# Find the best matching answer
def get_answer(user_question):
    cleaned_question = preprocess(user_question)

    if not cleaned_question:
        return "Please enter a question."

    user_vector = vectorizer.transform([cleaned_question])

    similarities = cosine_similarity(
        user_vector, faq_vectors
    )[0]

    best_index = similarities.argmax()
    best_score = similarities[best_index]

    if best_score < 0.15:
        return (
            "Sorry, I could not find a suitable answer. "
            "Please ask about computers, programming, or AI."
        )

    return faqs[best_index]["answer"]


# Display messages in the chat window

# Display messages in the chat window
def display_message(sender, message):
    chat_area.config(state=tk.NORMAL)
    chat_area.insert(tk.END, f"{sender}: {message}\n\n")
    chat_area.config(state=tk.DISABLED)
    chat_area.see(tk.END)


# Handle user input
def send_message(event=None):
    question = input_box.get().strip()

    if not question:
        return "break"

    input_box.delete(0, tk.END)
    display_message("You", question)

    if question.lower() in ["exit", "quit", "bye"]:
        display_message("Bot", "Goodbye! Thanks for chatting.")
        input_box.config(state=tk.DISABLED)
        send_button.config(state=tk.DISABLED)
        return "break"

    answer = get_answer(question)
    display_message("Bot", answer)
    input_box.focus_set()

    return "break"


# Create the application window
root = tk.Tk()
root.title("FAQ Chatbot Using NLP")
root.geometry("700x550")
root.minsize(500, 400)

# Heading
heading = tk.Label(
    root,
    text="FAQ CHATBOT",
    font=("Arial", 20, "bold")
)
heading.pack(pady=10)

subtitle = tk.Label(
    root,
    text="Ask questions about computers, programming, and AI",
    font=("Arial", 10)
)
subtitle.pack(pady=5)

# Chat display
chat_area = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    font=("Arial", 11),
    height=15,
    state=tk.DISABLED
)
chat_area.pack(
    padx=12,
    pady=10,
    fill=tk.BOTH,
    expand=True
)

# Input area - keep this visible at the bottom
bottom_frame = tk.Frame(root)
bottom_frame.pack(
    side=tk.BOTTOM,
    fill=tk.X,
    padx=12,
    pady=12
)

input_box = tk.Entry(
    bottom_frame,
    font=("Arial", 12)
)
input_box.pack(
    side=tk.LEFT,
    fill=tk.X,
    expand=True,
    ipady=8,
    padx=(0, 10)
)

send_button = tk.Button(
    bottom_frame,
    text="Send",
    font=("Arial", 11, "bold"),
    command=send_message
)
send_button.pack(side=tk.RIGHT)

# Press Enter to send a message
input_box.bind("<Return>", send_message)

# Welcome message
display_message(
    "Bot",
    "Hello! Welcome to the FAQ chatbot. Ask me about computers, programming, or AI."
)

# Start with the input box focused
input_box.focus_set()

root.mainloop()
