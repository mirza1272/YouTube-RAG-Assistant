# YouTube Transcript RAG Chatbot 🎥🤖

Welcome to the **YouTube Transcript RAG Chatbot**! This is a modern, full-stack application that allows you to instantly chat with any YouTube video. Just paste a YouTube URL, and the app will download the transcript, build a local knowledge base, and let you ask questions about the video's content using the Llama 3 LLM (via Groq) and Gemini Embeddings with automatic fallback when the Gemini quota is exhausted.

## ✨ Features
- **Instant Video Processing:** Paste a URL and chat with the video in seconds.
- **Gemini Embeddings with Fallback:** Uses Gemini for embeddings first, then falls back to a local offline embedding strategy if the API quota is exhausted.
- **Fast Inference:** Powered by Groq's insanely fast inference engine running `Llama-3.3-70b-versatile`.
- **Beautiful UI/UX:** A sleek, ChatGPT-like interface featuring Glassmorphism, a dark mode default, and fluid animations.
- **Markdown Support:** Renders Markdown and code blocks with syntax highlighting directly in the chat. Automatically Support the LLM Markdown Styling into Attractive UI.
- **Dynamic Fallbacks:** If English transcripts aren't available, the app intelligently prompts you to select from available languages.

## 🛠️ Tech Stack
- **Frontend:** Vanilla HTML5, CSS3 (Glassmorphism), Vanilla JavaScript (Fetch API)
- **Backend:** Python, Flask
- **AI/ML:** LangChain, FAISS (Vector Store), Groq (LLM), Gemini (Embeddings), local offline embedding fallback

## 🚀 Getting Started

### Prerequisites
Make sure you have Python 3.8+ installed on your machine.

### Installation

1. **Clone the repository** (or download the source code):
   ```bash
   git clone https://github.com/mirza1272/YouTube-RAG-Assistant.git
   cd youtube-rag-chatbot
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt # install all required packages
   ```

3. **Set up Environment Variables:**
   Create a `.env` file in the root directory and add your API keys:
   ```env
   GROK_API_KEY=your_groq_api_key_here
   GEMINI_API_KEY=your_gemini_api_key_here
   # Optional: provide multiple Gemini keys separated by commas for rotation
   GEMINI_API_KEYS=your_first_gemini_key,your_second_gemini_key
   ```

4. **Run the Application:**
   ```bash
   python app.py
   ```
   *(Note: If Gemini hits a quota limit, the app automatically falls back to a deterministic local embedding mode, so no extra model download is required.)*

5. **Open your browser:**
   Navigate to `http://127.0.0.1:5000` to start chatting!

---

## 🤝 Contributing (We need your help!)

We heartily welcome contributors of all skill levels! Whether you're a frontend wizard, a backend guru, or an AI enthusiast, there is a place for you here. 

We are actively looking for help to scale this project. If you're interested, please feel free to fork the repository and submit a Pull Request.

### 🎯 Open Issues / Upcoming Features

We are currently prioritizing the following features. If you want to contribute, these are perfect places to start:

1. **🧠 Chat Session History (Memory)**
   - *Issue:* Currently, the LLM answers each question independently without context of previous messages.
   - *Goal:* Implement LangChain memory (e.g., `ConversationBufferMemory`) so the bot remembers the ongoing conversation context.
   
2. **🔐 Account Login & Authentication**
   - *Issue:* The app is currently single-user and local.
   - *Goal:* Add an authentication system (JWT, Flask-Login, or OAuth) so users can create accounts, log in, and save their favorite videos and past chats securely.

3. **🎨 UI Enhancements & Polish**
   - *Issue:* While the UI is modern, there's always room for improvement!
   - *Goal:* Improve mobile responsiveness, add light/dark mode toggles, enhance accessibility (a11y), and add smooth transitions for the sidebar menus.

### How to Contribute
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Let's build the ultimate YouTube Chatbot together! Happy coding! 🎉
