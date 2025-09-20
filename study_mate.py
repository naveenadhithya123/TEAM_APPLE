import gradio as gr
import PyPDF2
from transformers import pipeline, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
from io import BytesIO
import re
import time
import os
import soundfile as sf
import numpy as np
import tempfile

class PDFChatbotWithTTS:
    def __init__(self, model_name="ibm-granite/granite-3.3-2b-instruct"):
        print(f"🚀 Loading Granite model: {model_name}...")
        try:
            device = 0 if torch.cuda.is_available() else -1
            self.pipe = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device=device
            )
            print("✅ Granite model loaded successfully!")
        except Exception as e:
            print(f"❌ Granite model loading error: {e}")
            self.pipe = None

        # Initialize Text-to-Speech models
        print("🎤 Loading Text-to-Speech models...")
        try:
            self.tts_processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
            self.tts_model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
            self.vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

            # Load speaker embeddings with error handling
            try:
                embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
                self.speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)
                print("✅ TTS models and embeddings loaded successfully!")
            except Exception as embed_error:
                print(f"⚠️ Using default speaker embeddings due to: {embed_error}")
                # Create default speaker embeddings as fallback
                self.speaker_embeddings = torch.randn(1, 512)

            self.tts_loaded = True
        except Exception as e:
            print(f"❌ TTS model loading error: {e}")
            self.tts_processor = None
            self.tts_model = None
            self.vocoder = None
            self.tts_loaded = False

        self.pdf_content = ""
        self.pdf_filename = ""
        self.chunks = []

    def extract_pdf_text(self, pdf_file):
        """Extract text from PDF file"""
        try:
            if pdf_file is None:
                return "❌ No PDF file selected.", ""

            print(f"📄 Processing PDF: {pdf_file}")

            # Handle file path
            if isinstance(pdf_file, str) and os.path.exists(pdf_file):
                with open(pdf_file, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    text_content = ""

                    for page_num, page in enumerate(pdf_reader.pages):
                        page_text = page.extract_text()
                        text_content += f"\n--- Page {page_num + 1} ---\n{page_text}\n"

                self.pdf_filename = os.path.basename(pdf_file)

            else:
                return "❌ Invalid file format. Please upload a valid PDF.", ""

            # Clean and process text
            text_content = re.sub(r'\s+', ' ', text_content).strip()

            if not text_content:
                return "❌ No text found in PDF. The file might be image-based or corrupted.", ""

            # Store content and create chunks
            self.pdf_content = text_content
            self.chunks = self.create_chunks(text_content)

            # Create preview
            preview = text_content[:500] + "..." if len(text_content) > 500 else text_content

            success_msg = f"""✅ Successfully loaded: {self.pdf_filename}
📊 Total pages: {len(pdf_reader.pages)}
📝 Text length: {len(text_content)} characters
🔍 Created {len(self.chunks)} chunks for fast processing

📄 Preview:
{preview}"""

            return success_msg, text_content

        except Exception as e:
            error_msg = f"❌ Error processing PDF: {str(e)}"
            print(error_msg)
            return error_msg, ""

    def create_chunks(self, text, chunk_size=400):
        """Create text chunks for faster processing"""
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1

            if current_length >= chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_length = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def find_relevant_content(self, question, max_length=800):
        """Find relevant content based on question"""
        if not self.chunks:
            return self.pdf_content[:max_length] if self.pdf_content else ""

        question_words = set(question.lower().split())
        chunk_scores = []

        # Score chunks based on word overlap
        for i, chunk in enumerate(self.chunks):
            chunk_words = set(chunk.lower().split())
            score = len(question_words.intersection(chunk_words))
            if score > 0:
                chunk_scores.append((score, chunk))

        if chunk_scores:
            # Sort by score and take top chunks
            chunk_scores.sort(reverse=True)
            relevant_content = ""
            for score, chunk in chunk_scores[:2]:  # Top 2 chunks
                relevant_content += chunk + " "
                if len(relevant_content) > max_length:
                    break
            return relevant_content[:max_length]

        # If no relevant chunks, use first chunk
        return self.chunks[0][:max_length]

    def clean_text_for_tts(self, text):
        """Clean text for better TTS pronunciation"""
        # Remove markdown formatting and special characters
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'🤖|⏱️|✨|📚|⚡|Answer:|Response time:', '', text)  # Remove emojis and labels
        text = re.sub(r'Response time:.*?s\*', '', text)  # Remove timing info
        text = re.sub(r'\*.*?\*', '', text)  # Remove italic text

        # Clean up extra spaces and newlines
        text = re.sub(r'\s+', ' ', text).strip()

        # Remove any remaining special characters that might cause issues
        text = re.sub(r'[^\w\s\.\,\!\?\-\(\)]', '', text)

        # Ensure text is not too long for TTS (optimal length for SpeechT5)
        if len(text) > 150:
            sentences = re.split(r'[.!?]+', text)
            cleaned_text = ""
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence and len(cleaned_text + sentence) < 150:
                    cleaned_text += sentence + ". "
                else:
                    break
            text = cleaned_text.strip()

        # Ensure minimum length
        if len(text) < 3:
            return "No valid text to convert to speech."

        # Add proper ending if missing
        if not text.endswith(('.', '!', '?')):
            text += "."

        return text

    def generate_speech(self, text):
        """Generate speech from text using SpeechT5"""
        try:
            if not self.tts_loaded or not all([self.tts_processor, self.tts_model, self.vocoder]):
                print("❌ TTS models not loaded properly")
                return None

            # Clean text for TTS
            clean_text = self.clean_text_for_tts(text)

            if not clean_text or len(clean_text.strip()) < 3:
                print("❌ Text too short or empty after cleaning")
                return None

            print(f"🎤 Generating speech for: '{clean_text[:300]}...'")

            # Process text with error handling
            try:
                inputs = self.tts_processor(text=clean_text, return_tensors="pt")

                # Check if inputs are valid
                if "input_ids" not in inputs or inputs["input_ids"].numel() == 0:
                    print("❌ Invalid input processing")
                    return None

            except Exception as proc_error:
                print(f"❌ Text processing error: {proc_error}")
                return None

            # Generate speech with error handling
            try:
                with torch.no_grad():
                    speech = self.tts_model.generate_speech(
                        inputs["input_ids"],
                        self.speaker_embeddings,
                        vocoder=self.vocoder
                    )

                # Check if speech was generated
                if speech is None or speech.numel() == 0:
                    print("❌ No speech output generated")
                    return None

            except Exception as gen_error:
                print(f"❌ Speech generation error: {gen_error}")
                return None

            # Save to temporary file
            try:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')

                # Convert to numpy and ensure proper format
                speech_numpy = speech.detach().cpu().numpy()

                # Normalize audio if needed
                if speech_numpy.max() > 1.0 or speech_numpy.min() < -1.0:
                    speech_numpy = speech_numpy / np.max(np.abs(speech_numpy))

                sf.write(temp_file.name, speech_numpy, samplerate=16000)
                temp_file.close()

                print(f"✅ Speech saved to: {temp_file.name}")
                return temp_file.name

            except Exception as save_error:
                print(f"❌ Audio saving error: {save_error}")
                return None

        except Exception as e:
            print(f"❌ General TTS error: {e}")
            return None

    def generate_answer_with_audio(self, question):
        """Generate answer with both text and audio"""
        if not self.pipe:
            return "❌ Model not loaded. Please restart the application.", None

        if not self.pdf_content:
            return "⚠️ Please upload and load a PDF file first!", None

        try:
            start_time = time.time()

            # Get relevant content
            relevant_content = self.find_relevant_content(question)

            # Create focused prompt
            prompt = f"""Based on this document content, answer the question concisely:

Content: {relevant_content}

Question: {question}
Answer:"""

            # Generate text response
            response = self.pipe(
                prompt,
                max_new_tokens=60,
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.pipe.tokenizer.eos_token_id,
                return_full_text=False
            )

            # Extract answer
            answer = response[0]['generated_text'].strip()

            if len(answer) > 300:
                answer = answer[:300] + "..."

            processing_time = time.time() - start_time

            # Format text response
            text_response = f"🤖 **Answer:** {answer}\n\n⏱️ *Response time: {processing_time:.1f}s*"

            # Generate audio
            audio_file = self.generate_speech(answer)

            if audio_file:
                text_response += "\n\n🎵 *Audio generated successfully!*"
            else:
                text_response += "\n\n⚠️ *Audio generation failed*"

            return text_response, audio_file

        except Exception as e:
            return f"❌ Error generating answer: {str(e)}", None

# Create global chatbot instance
chatbot = PDFChatbotWithTTS()

def handle_pdf_upload(pdf_file):
    """Handle PDF file upload"""
    return chatbot.extract_pdf_text(pdf_file)

def handle_question_with_audio(question, history):
    """Handle user question with audio response"""
    if not question.strip():
        return "", history, None

    text_answer, audio_file = chatbot.generate_answer_with_audio(question)
    history.append([question, text_answer])
    return "", history, audio_file

def clear_chat():
    """Clear chat history"""
    return []

def create_waveform_visualization():
    """Create a beautiful waveform visualization component"""
    return gr.HTML("""
    <div style="text-align: center; padding: 20px;">
        <div id="waveform" style="
            background: linear-gradient(45deg, #667eea, #764ba2);
            border-radius: 15px;
            padding: 20px;
            color: white;
            font-size: 18px;
            margin: 10px 0;
        ">
            🎵 Audio will play here when generated! 🎵
        </div>
    </div>
    """)

# Create Gradio interface
def create_interface():
    with gr.Blocks(
        theme=gr.themes.Soft(),
        title="📚 PDF Chatbot with Voice Assistant",
        css="""
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            text-align: center;
            padding: 2rem;
            border-radius: 15px;
            margin-bottom: 1.5rem;
        }
        .audio-section {
            background: linear-gradient(45deg, #764ba2, #667eea);
            border-radius: 15px;
            padding: 1.5rem;
            margin: 1rem 0;
        }
        .upload-box {
            border: 2px dashed #667eea;
            border-radius: 10px;
            padding: 1.5rem;
            background: rgba(102, 126, 234, 0.05);
        }
        .chat-box {
            border-radius: 10px;
            background: #f8f9fa;
            padding: 1rem;
        }
        """
    ) as interface:

        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>📚🎤 PDF Voice Assistant</h1>
            <h3>Powered by IBM Granite + Microsoft SpeechT5</h3>
            <p>Upload your PDF, ask questions, and hear the answers! ⚡🎵</p>
        </div>
        """)

        with gr.Row():
            # Left Column - Upload
            with gr.Column(scale=1):
                with gr.Group():
                    gr.Markdown("### 📤 **Upload PDF**")

                    pdf_input = gr.File(
                        label="Choose PDF File",
                        file_types=[".pdf"],
                        type="filepath"
                    )

                    load_btn = gr.Button("📖 Load PDF", variant="primary")

                    status_output = gr.Textbox(
                        label="📋 Status",
                        placeholder="Upload a PDF file to get started...",
                        lines=8,
                        max_lines=12,
                        interactive=False
                    )

                # Audio Section
                with gr.Group():
                    gr.HTML('<div class="audio-section">')
                    gr.Markdown("### 🎵 **Voice Response**", elem_classes="white-text")

                    audio_output = gr.Audio(
                        label="🎤 Listen to Answer",
                        type="filepath",
                        autoplay=True
                    )

                    gr.HTML('</div>')

                # Quick Actions
                gr.Markdown("### ⚡ **Quick Actions**")
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

            # Right Column - Chat
            with gr.Column(scale=2):
                with gr.Group():
                    gr.Markdown("### 💬 **Ask Questions (Text + Voice)**")

                    chatbot_interface = gr.Chatbot(
                        label="Chat with your PDF",
                        height=400,
                        placeholder="Upload a PDF and start asking questions! You'll get both text and voice answers! 🚀🎤"
                    )

                    with gr.Row():
                        question_input = gr.Textbox(
                            label="Your Question",
                            placeholder="Ask anything about your PDF...",
                            lines=2,
                            scale=4
                        )
                        ask_btn = gr.Button("🚀 Ask", variant="primary", scale=1)

        # Examples
        gr.Examples(
            examples=[
                ["What is the main topic of this document?"],
                ["Summarize the key points"],
                ["Subject of this document?"],
                ["Important Questions??"],
                ["What are the important definitions?"]
            ],
            inputs=question_input,
            label="💡 Example Questions"
        )

        # Footer
        gr.HTML("""
        <div style="text-align: center; margin-top: 2rem; padding: 1rem; background: #f8f9fa; border-radius: 10px;">
            <p><strong>🎯 Features:</strong> PDF Text Extraction • Smart Q&A • Text-to-Speech • Audio Playback</p>
            <p><strong>🔧 Models:</strong> IBM Granite 3.3B (Text Generation) • Microsoft SpeechT5 (Text-to-Speech)</p>
        </div>
        """)

        # Hidden state for content
        pdf_content = gr.State("")

        # Event handlers
        load_btn.click(
            handle_pdf_upload,
            inputs=[pdf_input],
            outputs=[status_output, pdf_content]
        )

        ask_btn.click(
            handle_question_with_audio,
            inputs=[question_input, chatbot_interface],
            outputs=[question_input, chatbot_interface, audio_output]
        )

        question_input.submit(
            handle_question_with_audio,
            inputs=[question_input, chatbot_interface],
            outputs=[question_input, chatbot_interface, audio_output]
        )

        clear_btn.click(
            clear_chat,
            outputs=[chatbot_interface]
        )

    return interface

# Launch the application
if __name__ == "__main__":
    print("🚀 Starting PDF Voice Assistant...")
    print("📋 Required packages: gradio, transformers, torch, PyPDF2, soundfile, datasets")
    print("🎤 Features: PDF Processing + Text Generation + Text-to-Speech")

    app = create_interface()
    app.launch(
        share=True,
        server_name="0.0.0.0",
        show_error=True
    )
