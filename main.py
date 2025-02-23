import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, pipeline
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFacePipeline
import gradio as gr

# Disable symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# --- Knowledge Distillation Section ---
def load_teacher_data(file_path="teacher_data.json"):
    """
    Load teacher model outputs from a JSON file
    :param file_path: Path to teacher data JSON
    :return: Dictionary of question-answer pairs
    """
    with open(file_path, "r", encoding="utf-8") as f:
        teacher_responses = json.load(f)
    return teacher_responses

def create_distillation_data(teacher_responses):
    """
    Create a dataset from teacher responses for student model training
    :param teacher_responses: Dictionary of teacher Q&A pairs
    :return: Hugging Face Dataset
    """
    data = [{"input": q, "output": a} for q, a in teacher_responses.items()]
    return Dataset.from_list(data)

# Load student model
model_name = "distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, load_in_8bit=True, device_map="auto")

# Add LoRA for fine-tuning
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

def preprocess_function(examples):
    """
    Encode inputs and outputs for model training
    :param examples: Data samples
    :return: Tokenized data
    """
    inputs = tokenizer(examples["input"], truncation=True, max_length=128, padding="max_length")
    targets = tokenizer(examples["output"], truncation=True, max_length=128, padding="max_length")
    inputs["labels"] = targets["input_ids"]
    return inputs

# Load and process teacher data
teacher_responses = load_teacher_data("teacher_data.json")
dataset = create_distillation_data(teacher_responses)
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# Training arguments with reduced epochs to avoid overfitting
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,  # Reduced from 10 to prevent overfitting on small dataset
    per_device_train_batch_size=2,
    logging_steps=50,
    save_steps=50,
    eval_strategy="no",
    fp16=True,
    disable_tqdm=True,
)

trainer = Trainer(model=model, args=training_args, train_dataset=tokenized_dataset)
trainer.train()
print("Knowledge distillation completed: Student model fine-tuned with teacher data.")

# --- LangChain Knowledge Base ---
loader = TextLoader("knowledge.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.from_documents(texts, embeddings)

# Create a HuggingFacePipeline for LangChain
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,  # Increased to allow complete answers
    do_sample=False,  # Use greedy search for deterministic, focused responses
    temperature=0.01,  # Very low temperature for minimal randomness
    top_p=0.9
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# Define QA chain with student model
def generate_response(question):
    """
    Generate a concise, relevant response using the fine-tuned model and external knowledge
    :param question: User input question
    :return: Generated answer only, without context or noise
    """
    # Check if question is complete (ends with ? or has minimum length)
    if not question or len(question.strip()) < 3 or "?" not in question:
        return "Please provide a complete question, e.g., 'Who created Python?'"

    # Try to directly match teacher data first (case-insensitive, strict match)
    question_clean = question.lower().strip()
    for q, a in teacher_responses.items():
        if question_clean == q.lower() or question_clean in q.lower():
            return a

    # Use RetrievalQA for external knowledge if no direct match, with strict filtering
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(k=1),  # Retrieve only the most relevant document
        return_source_documents=False,
        output_key="answer"
    )
    result = qa_chain.run(question)

    # Clean and filter the result to remove noise
    result = result.strip()
    if "Answer:" in result:
        result = result.split("Answer:")[-1].strip()
    elif ":" in result:
        result = result.split(":")[-1].strip()
    result = result.replace("\n", " ").replace("Question:", "").strip()

    # Remove any repetitive or irrelevant content
    words = result.split()
    if len(words) > 10:  # Limit length to prevent long, repetitive outputs
        result = " ".join(words[:10])
    if any(word.lower() in ["question", "is", "a", "that"] for word in words) and len(result) < 10:
        return "Sorry, I couldn’t find a relevant answer."

    return result if result and len(result) > 5 else "Sorry, I couldn’t find a relevant answer."

# --- Gradio Interface ---
interface = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(lines=2, placeholder="Ask me anything!", label="Question"),
    outputs="text",
    title="DistillMind: Smart Q&A Chatbot",
    description="A distilled LLM chatbot with external knowledge integration."
)

interface.launch()