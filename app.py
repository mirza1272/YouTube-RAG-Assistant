# YouTube Transcript RAG Chatbot Backend Code

from urllib.parse import urlparse, parse_qs
import os
import uuid

from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", os.urandom(24))

ACTIVE_SESSIONS = {}


# Extract video ID from YouTube URL
def extract_video_id(youtube_url):
    parsed_url = urlparse(youtube_url)

    if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
        return parse_qs(parsed_url.query).get("v", [None])[0]

    if parsed_url.hostname == "youtu.be":
        return parsed_url.path.lstrip("/")

    return None


# Get transcript from YouTube
def get_transcript(youtube_url, language="en"):
    video_id = extract_video_id(youtube_url)

    if not video_id:
        raise ValueError("Invalid YouTube URL")

    # Check if cookies file exists to bypass YouTube Cloud IP block
    cookies_file = "cookies.txt"
    use_cookies = os.path.exists(cookies_file)
    
    http_client = None
    if use_cookies:
        import requests
        from http.cookiejar import MozillaCookieJar
        
        cookie_jar = MozillaCookieJar(cookies_file)
        # Only try to load if the file exists and is not empty
        try:
            cookie_jar.load(ignore_discard=True, ignore_expires=True)
            http_client = requests.Session()
            http_client.cookies = cookie_jar
        except Exception:
            pass

    ytt_api = YouTubeTranscriptApi(http_client=http_client) if http_client else YouTubeTranscriptApi()

    try:
        transcript_list = ytt_api.fetch(video_id, languages=[language])
        transcript = " ".join(snippet.text for snippet in transcript_list)
        return transcript

    except NoTranscriptFound:
        available_transcripts = ytt_api.list(video_id)
        available_languages = []

        for transcript_item in available_transcripts:
            available_languages.append({
                "language": transcript_item.language,
                "language_code": transcript_item.language_code,
                "is_generated": transcript_item.is_generated
            })

        return {
            "error": "Transcript not found for selected language.",
            "available_transcripts": available_languages
        }

    except TranscriptsDisabled:
        return {
            "error": "Transcripts are disabled for this video.",
            "available_transcripts": []
        }


# Create chunks from transcript
def create_chunks(transcript):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=300
    )

    chunks = splitter.create_documents([transcript])
    return chunks


# Create FAISS vector store
def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-2",
        google_api_key=os.environ["GEMINI_API_KEY"]
    )

    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


# Format retrieved documents into context
def format_docs(retrieved_docs):
    return "\n\n".join(doc.page_content for doc in retrieved_docs)


# Create RAG chain
def create_rag_chain(vector_store):
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 20}
    )

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.2,
        api_key=os.getenv("GROK_API_KEY")
    )

    prompt = PromptTemplate(
        template="""
You are a helpful assistant.

Answer ONLY from the provided YouTube transcript context.

If the context is insufficient, say:
"I don't know from the video transcript."

Context:
{context}

Question:
{question}

Answer:
""",
        input_variables=["context", "question"]
    )

    parallel_chain = RunnableParallel(
        context=retriever | RunnableLambda(format_docs),
        question=RunnablePassthrough()
    )

    main_chain = (
        parallel_chain
        | prompt
        | llm
        | StrOutputParser()
    )

    return main_chain


# Build complete RAG chatbot from YouTube URL
def build_youtube_rag(youtube_url, language="en"):
    transcript_result = get_transcript(youtube_url, language)

    if isinstance(transcript_result, dict) and "error" in transcript_result:
        return transcript_result

    chunks = create_chunks(transcript_result)
    vector_store = create_vector_store(chunks)
    rag_chain = create_rag_chain(vector_store)

    return rag_chain


# Ask question from RAG chatbot
def ask_question(rag_chain, question):
    answer = rag_chain.invoke(question)
    return answer


# ------------------- FLASK ROUTES -------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/load_video", methods=["POST"])
def load_video():
    try:
        data = request.json
        youtube_url = data.get("youtube_url")
        language = data.get("language", "en")

        if not youtube_url:
            return jsonify({"error": "YouTube URL is required"}), 400

        rag_chain_or_error = build_youtube_rag(youtube_url, language)

        if isinstance(rag_chain_or_error, dict) and "error" in rag_chain_or_error:
            return jsonify(rag_chain_or_error), 400

        session_id = str(uuid.uuid4())
        ACTIVE_SESSIONS[session_id] = rag_chain_or_error

        # Try to extract chunks count from FAISS vectorstore if possible
        chunks_count = "N/A"
        try:
            # Assuming typical chain structure: parallel_chain -> context -> retriever -> vectorstore
            vector_store = rag_chain_or_error.first.steps['context'].first.vectorstore
            chunks_count = vector_store.index.ntotal
        except Exception:
            pass

        return jsonify({
            "status": "Ready",
            "session_id": session_id,
            "chunks_created": chunks_count,
            "embedding_model": "Gemini Embedding 2 (API)",
            "llm": "Grok",
            "search_type": "MMR"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Backend Error: {str(e)}"}), 500


@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    session_id = data.get("session_id")
    question = data.get("question")

    if not session_id or session_id not in ACTIVE_SESSIONS:
        return jsonify({"error": "Session expired or invalid. Please load the video again."}), 401

    if not question:
        return jsonify({"error": "Question is required."}), 400

    rag_chain = ACTIVE_SESSIONS[session_id]

    try:
        answer = ask_question(rag_chain, question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)