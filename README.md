# 📚🎤 Study Mate – AI-Powered PDF Voice Assistant

<div align="center">

![Study Mate Banner](https://via.placeholder.com/800x200/667eea/ffffff?text=Study+Mate+AI)

**Your Personal AI Study Buddy!**

Transform how you interact with academic content. Upload PDFs, ask questions in natural language, and get instant answers in both text and voice.

[![Made with Python](https://img.shields.io/badge/Made%20with-Python-blue.svg)](https://python.org)
[![Powered by Gradio](https://img.shields.io/badge/Interface-Gradio-orange.svg)](https://gradio.app)
[![IBM Granite](https://img.shields.io/badge/LLM-IBM%20Granite-red.svg)](https://huggingface.co/ibm-granite)
[![Microsoft SpeechT5](https://img.shields.io/badge/TTS-Microsoft%20SpeechT5-green.svg)](https://huggingface.co/microsoft/speecht5_tts)

[🚀 Demo](#demo-preview) • [⚡ Quick Start](#quick-start) • [📖 Documentation](#documentation) • [🤝 Contributing](#contributing)

</div>

## 🌟 Why Study Mate?

Tired of scrolling through endless PDFs looking for answers? Study Mate revolutionizes how you study by combining cutting-edge AI with an intuitive interface:

- **📄 Smart PDF Processing** - Upload any academic document
- **🧠 AI-Powered Q&A** - Ask questions in plain English  
- **🎤 Voice Responses** - Listen to answers with natural speech synthesis
- **💬 Interactive Chat** - Conversational interface for better learning
- **⚡ Lightning Fast** - Get answers in 2-5 seconds

## 🚀 Features

### Core Functionality
- **📚 PDF Text Extraction** - Intelligent text parsing with chunking for optimal processing
- **🤖 Context-Aware Q&A** - Powered by IBM Granite 3.3B for accurate, relevant answers
- **🎵 Text-to-Speech** - Microsoft SpeechT5 converts answers to natural-sounding speech
- **🎨 Modern UI** - Clean, responsive Gradio interface with gradient themes

### Advanced Features
- **🔍 Smart Content Retrieval** - Finds most relevant sections for each question
- **🎧 Audio Playback** - Auto-playing responses with waveform visualization
- **📝 Chat History** - Persistent conversation tracking
- **💡 Example Questions** - Built-in prompts to get started quickly
- **🗑️ Quick Actions** - Clear chat, sample queries, and more

## 🛠️ Tech Stack

<table>
<tr>
<td><strong>Frontend</strong></td>
<td><code>Gradio</code> - Modern AI app interface</td>
</tr>
<tr>
<td><strong>Language Model</strong></td>
<td><code>IBM Granite 3.3B</code> - Advanced text generation</td>
</tr>
<tr>
<td><strong>Text-to-Speech</strong></td>
<td><code>Microsoft SpeechT5</code> - Neural voice synthesis</td>
</tr>
<tr>
<td><strong>PDF Processing</strong></td>
<td><code>PyPDF2</code> - Document text extraction</td>
</tr>
<tr>
<td><strong>ML Framework</strong></td>
<td><code>PyTorch</code> - Deep learning operations</td>
</tr>
<tr>
<td><strong>Audio Processing</strong></td>
<td><code>SoundFile</code> - Audio file handling</td>
</tr>
</table>

## ⚡ Quick Start

### Prerequisites
- Python 3.8 or higher
- 4GB+ RAM (8GB recommended for better performance)
- GPU optional but recommended for faster inference

### Installation

1️⃣ **Clone the repository**
```bash
git clone https://github.com/your-username/study-mate.git
cd study-mate
```

2️⃣ **Install dependencies**
```bash
pip install gradio transformers torch PyPDF2 soundfile datasets accelerate

# For GPU support (optional but recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

3️⃣ **Run the application**
```bash
python app.py
```

4️⃣ **Open your browser**
- Navigate to the displayed local URL (typically `http://127.0.0.1:7860`)
- Or use the public Gradio link for remote access

### Alternative: One-Command Setup
```bash
# Install and run in one go
pip install gradio transformers torch PyPDF2 soundfile datasets accelerate && python app.py
```

## 🎯 How to Use

### Step 1: Upload Your PDF
- Click "Choose PDF File" in the left panel
- Select any academic PDF (textbooks, papers, notes)
- Click "📖 Load PDF" to process the document

### Step 2: Ask Questions
- Type your question in natural language
- Examples: "What is machine learning?", "Explain the main concepts", "Summarize chapter 2"
- Click "🚀 Ask" or press Enter

### Step 3: Get Answers
- Read the text response in the chat
- Listen to the AI-generated audio automatically
- Continue the conversation with follow-up questions

## 📸 Demo Preview

<div align="center">

### 🎬 **Live Demo**
![Demo GIF](https://via.placeholder.com/600x400/f0f0f0/333333?text=Demo+Video+Here)

### 📱 **Interface Screenshots**

| PDF Upload | Chat Interface | Voice Response |
|------------|----------------|----------------|
| ![Upload](https://via.placeholder.com/250x200/667eea/ffffff?text=PDF+Upload) | ![Chat](https://via.placeholder.com/250x200/764ba2/ffffff?text=Chat+Q%26A) | ![Audio](https://via.placeholder.com/250x200/45b7d1/ffffff?text=Voice+Player) |

</div>

## 📋 Requirements

### System Requirements
- **OS**: Windows 10+, macOS 10.15+, or Linux
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 2GB free space for models
- **Network**: Internet connection for initial model downloads

### Python Dependencies
```txt
gradio>=4.0.0
transformers>=4.35.0
torch>=2.0.0
PyPDF2>=3.0.0
soundfile>=0.12.0
datasets>=2.14.0
accelerate>=0.24.0
numpy>=1.21.0
```

## 🔧 Configuration

### Model Selection
The app uses these models by default:
- **Text Generation**: `ibm-granite/granite-3.3-2b-instruct`
- **Text-to-Speech**: `microsoft/speecht5_tts`
- **Vocoder**: `microsoft/speecht5_hifigan`

### Performance Tuning
- **GPU Usage**: Automatically detected and utilized
- **Model Precision**: FP16 on GPU, FP32 on CPU
- **Chunk Size**: Optimized for 400-character segments
- **Audio Quality**: 16kHz sample rate with 0.5s sentence pauses

## 🚨 Troubleshooting

### Common Issues

**Q: Models fail to download?**
```bash
# Clear cache and retry
pip install --upgrade transformers datasets
huggingface-cli login  # If using private models
```

**Q: Audio not working?**
```bash
# Install audio dependencies
pip install --upgrade soundfile librosa
# On Linux: sudo apt-get install libsndfile1
```

**Q: Memory errors?**
```bash
# Force CPU usage
export CUDA_VISIBLE_DEVICES=""
python app.py
```

**Q: Port already in use?**
```python
# Change port in app.py
app.launch(server_port=7861)  # Change from 7860
```

### Performance Tips
- Close other applications to free up RAM
- Use GPU when available for faster responses
- Restart the app if models become unresponsive

## 📚 Documentation

### API Reference
- **PDF Processing**: Handles multi-page documents with intelligent text extraction
- **Question Answering**: Context-aware responses using retrieval-augmented generation
- **Speech Synthesis**: Sentence-by-sentence audio generation with natural pauses

### Supported File Types
- ✅ PDF documents (.pdf)
- ❌ Images, Word docs (coming soon)

### Audio Features
- **Format**: WAV, 16kHz mono
- **Synthesis**: Neural text-to-speech with natural prosody
- **Playback**: Auto-play with visual waveform feedback

## 🤝 Contributing

We welcome contributions! Here's how you can help:

### Ways to Contribute
- 🐛 **Report Bugs** - Create detailed issue reports
- 💡 **Suggest Features** - Share your ideas for improvements
- 🔧 **Submit PRs** - Fix bugs or add new features
- 📖 **Improve Docs** - Help make documentation clearer

### Development Setup
```bash
# Fork and clone the repo
git clone https://github.com/your-username/study-mate.git
cd study-mate

# Create a development branch
git checkout -b feature/your-feature-name

# Make your changes and test
python app.py

# Submit a pull request
```

### Code Style
- Follow PEP 8 guidelines
- Add docstrings to new functions
- Include error handling for robustness
- Test with different PDF types

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **IBM Research** for the Granite language models
- **Microsoft** for SpeechT5 text-to-speech
- **Hugging Face** for the transformers library
- **Gradio Team** for the amazing UI framework

## ⭐ Support

If you find Study Mate helpful:

- 🌟 **Star this repository** on GitHub
- 📢 **Share with friends** and colleagues
- 💬 **Join discussions** in our Issues section
- 🐛 **Report bugs** to help us improve

---

<div align="center">

**⚡ Study faster. Learn smarter. With Study Mate 🎓🤖**

[⬆️ Back to Top](#-study-mate--ai-powered-pdf-voice-assistant)

Made with ❤️ by [Your Name](https://github.com/your-username)

</div>
