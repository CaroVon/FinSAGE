# FinSAGE

FinSAGE is a comprehensive solution for building Retrieval-Augmented Generation (RAG) systems and performing LoRA (Low-Rank Adaptation) fine-tuning with the **Meta-Llama-3.1-8B-Instruct** model.

## Features

- **RAG Implementation**: Build intelligent question-answering systems by retrieving relevant information from PDF documents
- **LoRA Fine-tuning**: Efficiently fine-tune the Llama 3.1 8B model with minimal computational resources
- **Integration**: Seamlessly combine RAG and LoRA for enhanced model performance
- **Web Interface**: Interactive Streamlit web application for PDF upload and chat
- **Model Management**: Easy model deployment and management utilities

## Repository Structure

```
FinSAGE/
├── LLM-RAG-Lora/
│   ├── Deploy-Llama-3/       # Model downloading and deployment
│   ├── RAG-Llama-3/          # RAG implementation with PDF support
│   ├── Lora-Llama-3/         # LoRA fine-tuning module
│   └── requirements.txt       # Dependencies
└── README.md                  # Chinese documentation
```

## Environment Setup

### Prerequisites

```
Python 3.12
CUDA 12.1
PyTorch 2.3.0
```

### Installation

1. Clone the repository:
```bash
git clone https://github.com/CaroVon/FinSAGE.git
cd FinSAGE
```

2. Install dependencies:
```bash
cd LLM-RAG-Lora
pip install -r requirements.txt
```

## Getting Started

### 1. Deploy Llama 3.1 8B Model

First, download the required models (Meta-Llama-3.1-8B-Instruct and mxbai-embed-large-v1):

```bash
cd Deploy-Llama-3
python model_download.py
```

Test the model with interactive Q&A:
```bash
python test_QA_initial.py
```

### 2. RAG (Retrieval-Augmented Generation)

The RAG module allows you to upload PDF documents and ask questions about their content.

Start the Streamlit web application:
```bash
cd ../RAG-Llama-3
streamlit run app.py
```

Then access the web interface in your browser to:
- Upload PDF documents
- Ask questions about the document content
- Adjust retrieval settings (number of results `k` and similarity threshold)

### 3. LoRA Fine-tuning

The LoRA module enables efficient fine-tuning of the Llama 3.1 model with your custom data.

#### Training

Prepare your training data in the required format, then run:
```bash
cd ../Lora-Llama-3
python train.py
```

The trained LoRA parameters will be saved in `output/llama3_1_instruct_lora/`.

#### Testing

Three testing options are available:

1. **Interactive Q&A without LoRA**:
   ```bash
   python test_QA_initial.py
   ```

2. **Interactive Q&A with LoRA**:
   ```bash
   python test_QA.py
   ```

3. **Non-interactive testing with LoRA**:
   ```bash
   python test.py
   ```

#### Merging LoRA Weights

After training, merge the LoRA weights into the base model to create a new HuggingFace-compatible model:

```bash
python merge_lora.py
```

The merged model will be saved and ready for deployment.

### 4. Combining RAG with Fine-tuned LoRA Model

After merging your LoRA weights with the base model, you can use the enhanced model with RAG:

1. Modify `RAG-Llama-3/rag.py`:
```python
# Change the LLM path from:
llm_path: str = "/path/to/Meta-Llama-3.1-8B-Instruct"

# To your merged LoRA model path:
llm_path: str = "/path/to/Meta-Llama-3.1-8B-Instruct_Lora"
```

2. Run the RAG application with your fine-tuned model:
```bash
cd ../RAG-Llama-3
streamlit run app.py
```

## Configuration

### RAG Configuration (rag.py)

Key parameters you can customize:

- **LLM Model**: Path to the Llama 3.1 model
- **Embedding Model**: `mxbai-embed-large-v1` for document embeddings
- **Chunk Size**: Document splitting size (default: 1024)
- **Chunk Overlap**: Overlap between chunks (default: 100)
- **Retrieval K**: Number of results to retrieve (adjustable in web UI)
- **Similarity Threshold**: Minimum similarity score for results (adjustable in web UI)

### LoRA Configuration (train.py)

Customize training parameters such as:

- Learning rate
- Number of epochs
- Batch size
- LoRA rank and alpha values

Refer to the training script for detailed parameter descriptions.

## Data Format for LoRA Fine-tuning

Training data should be in JSON format:

```json
[
    {
        "instruction": "User query or prompt",
        "input": "Optional context or input",
        "output": "Expected model response"
    },
    {
        "instruction": "Another query",
        "input": "",
        "output": "Model response"
    }
]
```

## Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**:
   - Ensure CUDA 12.1 is properly installed
   - Check PyTorch is built for your CUDA version

2. **Model Download Failures**:
   - Verify internet connection
   - Check sufficient disk space (Llama 3.1 8B is ~16GB)
   - For slow downloads, consider using a model mirror

3. **Memory Errors**:
   - Reduce batch size in training
   - Reduce chunk size in RAG
   - Ensure sufficient VRAM available

4. **Vector Store Errors**:
   - Delete the vector store directory and restart
   - Clear cache: `rm -rf chroma_db/`

## Project References

This project builds upon the excellent work from:

- [huanhuan-chat](https://github.com/KMnO4-zx/huanhuan-chat.git) - LoRA fine-tuning methodology
- [chatpdf-rag-deepseek-r1](https://github.com/paquino11/chatpdf-rag-deepseek-r1) - RAG implementation
- [self-llm](https://github.com/datawhalechina/self-llm) - LLM training resources

## Requirements

See `LLM-RAG-Lora/requirements.txt` for complete dependencies. Main packages include:

- transformers
- torch
- peft (for LoRA)
- langchain
- streamlit
- chromadb
- pypdf

## License

This project is open source and available under the MIT License.

## Support

For issues, questions, or contributions, please open an issue or submit a pull request on GitHub.

## Acknowledgments

- Meta for the Llama 3.1 model
- LangChain for RAG framework
- Streamlit for the web interface
- The open-source community for various tools and resources