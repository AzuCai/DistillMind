# DistillMind: Smart Q&A Chatbot

## Project Overview
DistillMind is a cutting-edge, lightweight chatbot that leverages **Knowledge Distillation** and **LangChain** for intelligent, knowledge-enhanced question answering. It distills a large language model (LLM) into DistilGPT-2, fine-tuned with LoRA on a small dataset, and integrates external knowledge via LangChain for accurate responses. Optimized for 6GB GPUs using 8-bit quantization, it features a Gradio UI for interactive testing. Ideal for showcasing advanced NLP, model optimization, and deployment skills in interviews.

## Features
- **Knowledge Distillation**: Transfers knowledge from a large LLM (simulated via teacher data) to DistilGPT-2 for efficient, low-resource operation.
- **External Knowledge Integration**: Uses LangChain to retrieve and leverage external knowledge from a text file, enhancing answer accuracy.
- **Lightweight Design**: Runs on 6GB GPUs with 8-bit quantization and LoRA fine-tuning.
- **Interactive Interface**: Gradio-based UI for real-time Q&A, with concise, relevant outputs.
- **Minimal Logging**: Training uses sparse logs and no progress bar for clean output.

## LLM-Related Knowledge Points
- **Knowledge Distillation**: Trains a small student model (DistilGPT-2) to mimic a larger teacher's outputs, using hard-label distillation on curated Q&A pairs.
- **LangChain**: Manages external knowledge retrieval with FAISS and embeddings, and chains it with LLM for contextual responses.
- **Quantization**: Applies 8-bit quantization via `bitsandbytes` to reduce memory usage on limited hardware.
- **PEFT (LoRA)**: Uses Low-Rank Adaptation for parameter-efficient fine-tuning, minimizing GPU load.
- **Prompt Engineering**: Structures inputs with natural questions to guide response generation.
- **Tokenization & Generation**: Employs Hugging Face’s tokenizer and pipeline for text processing and generation.

## Environment Setup and Deployment

### Prerequisites
- **Hardware**: GPU with at least 6GB VRAM (e.g., NVIDIA GTX 1660).
- **Operating System**: Windows, Linux, or macOS (tested on Windows).
- **Python**: 3.9 or higher.

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/DistillMind.git
   cd DistillMind
   ```
2. **Create a Conda Environment (optional but recommended)**:
    ```bash
   conda create -n DistillMind python=3.10
   conda activate DistillMind
   ```
3. **Install Dependencies**:
   ```bash
   pip install transformers torch datasets gradio bitsandbytes accelerate peft langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu
   ```

4. **Prepare Data Files**:
  Create teacher_data.json with teacher model Q&A pairs (e.g., {"What is AI?": "Artificial Intelligence (AI) refers to..."}).
  Create knowledge.txt with domain knowledge (e.g., paragraphs on AI, Python, physics).

### Running the Project
1. **Train and Launch**:
   ```bash
   pip install transformers torch datasets gradio bitsandbytes accelerate peft langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu
   ```
2. **Interact**:
   In the Gradio interface, input questions like “What is AI?” or “Who created Python?” to get concise, relevant answers.

## Acknowledgments

  Built with Hugging Face Transformers, PEFT, and LangChain.
  
  Dataset: Custom teacher_data.json and knowledge.txt based on open-domain knowledge.
  
  Enjoy exploring DistillMind!

