document.addEventListener('DOMContentLoaded', () => {
    // DOM Elements
    const setupScreen = document.getElementById('setup-screen');
    const chatScreen = document.getElementById('chat-screen');
    const sidebar = document.getElementById('sidebar');
    
    const youtubeUrlInput = document.getElementById('youtube-url');
    const transcriptLanguageSelect = document.getElementById('transcript-language');
    const btnLoadVideo = document.getElementById('btn-load-video');
    const loadingStatus = document.getElementById('loading-status');
    const loadingText = document.getElementById('loading-text');
    const setupAlert = document.getElementById('setup-alert');
    
    const chatMessages = document.getElementById('chat-messages');
    const chatInput = document.getElementById('chat-input');
    const btnSendMessage = document.getElementById('btn-send-message');
    const btnStopGeneration = document.getElementById('btn-stop-generation');
    const btnToggleSidebar = document.getElementById('btn-toggle-sidebar');
    
    const infoVideo = document.getElementById('info-video');
    const infoLanguage = document.getElementById('info-language');
    const infoChunks = document.getElementById('info-chunks');
    const infoEmbedding = document.getElementById('info-embedding');
    const infoLlm = document.getElementById('info-llm');
    const infoSearch = document.getElementById('info-search');
    
    const btnClearChat = document.getElementById('btn-clear-chat');
    const btnNewVideo = document.getElementById('btn-new-video');

    // State
    let sessionId = null;
    let abortController = null;
    let isGenerating = false;

    // Configure Marked.js with Highlight.js
    marked.setOptions({
        highlight: function(code, lang) {
            const language = hljs.getLanguage(lang) ? lang : 'plaintext';
            return hljs.highlight(code, { language }).value;
        },
        langPrefix: 'hljs language-',
        breaks: true
    });

    // --- Setup Screen Logic ---
    
    btnLoadVideo.addEventListener('click', async () => {
        const url = youtubeUrlInput.value.trim();
        const language = transcriptLanguageSelect.value;
        
        if (!url) {
            showSetupAlert('Please enter a valid YouTube URL.', 'error');
            return;
        }

        // UI Loading State
        setLoadingState(true);
        hideSetupAlert();
        loadingText.textContent = 'Fetching transcript & building knowledge base...';

        try {
            const response = await fetch('/load_video', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ youtube_url: url, language: language })
            });
            
            const data = await response.json();

            if (!response.ok) {
                // Check if it's a transcript missing error with fallbacks
                if (data.available_transcripts && data.available_transcripts.length > 0) {
                    populateLanguageDropdown(data.available_transcripts);
                    showSetupAlert(`${data.error} Please select an available language from the dropdown and try again.`, 'warning');
                } else {
                    showSetupAlert(data.error || 'Failed to load video.', 'error');
                }
                setLoadingState(false);
                return;
            }

            // Success! Transition to chat
            sessionId = data.session_id;
            
            // Update Sidebar Info
            infoVideo.textContent = url;
            infoVideo.title = url;
            infoLanguage.textContent = transcriptLanguageSelect.options[transcriptLanguageSelect.selectedIndex].text;
            infoChunks.textContent = data.chunks_created;
            infoEmbedding.textContent = data.embedding_model;
            infoLlm.textContent = data.llm;
            infoSearch.textContent = data.search_type;

            transitionToChat();
            
        } catch (error) {
            console.error('Error loading video:', error);
            showSetupAlert('Network error occurred while loading video.', 'error');
        } finally {
            setLoadingState(false);
        }
    });

    function setLoadingState(isLoading) {
        btnLoadVideo.disabled = isLoading;
        const btnText = btnLoadVideo.querySelector('.btn-text');
        const spinner = btnLoadVideo.querySelector('.spinner');
        
        if (isLoading) {
            btnText.classList.add('hidden');
            spinner.classList.remove('hidden');
            loadingStatus.classList.remove('hidden');
        } else {
            btnText.classList.remove('hidden');
            spinner.classList.add('hidden');
            loadingStatus.classList.add('hidden');
        }
    }

    function showSetupAlert(message, type) {
        setupAlert.textContent = message;
        setupAlert.className = `alert alert-${type}`;
        setupAlert.classList.remove('hidden');
    }

    function hideSetupAlert() {
        setupAlert.classList.add('hidden');
    }

    function populateLanguageDropdown(transcripts) {
        transcriptLanguageSelect.innerHTML = '';
        transcripts.forEach(t => {
            const option = document.createElement('option');
            option.value = t.language_code;
            option.textContent = t.language + (t.is_generated ? ' (Auto)' : '');
            transcriptLanguageSelect.appendChild(option);
        });
    }

    function transitionToChat() {
        setupScreen.classList.add('hidden');
        chatScreen.classList.remove('hidden');
        sidebar.classList.remove('hidden');
        chatInput.focus();
    }

    // --- Chat Screen Logic ---

    // Auto-resize textarea
    chatInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        btnSendMessage.disabled = this.value.trim() === '';
    });

    chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            if (!btnSendMessage.disabled && !isGenerating) {
                sendMessage();
            }
        }
    });

    btnSendMessage.addEventListener('click', () => {
        if (!btnSendMessage.disabled && !isGenerating) {
            sendMessage();
        }
    });

    btnStopGeneration.addEventListener('click', () => {
        if (abortController) {
            abortController.abort();
            finishGeneration();
            appendMessage('ai', 'Generation stopped by user.');
        }
    });

    btnToggleSidebar.addEventListener('click', () => {
        sidebar.classList.toggle('open');
    });

    btnClearChat.addEventListener('click', () => {
        // Keep only the welcome message
        const welcome = chatMessages.querySelector('.welcome-message');
        chatMessages.innerHTML = '';
        if (welcome) chatMessages.appendChild(welcome);
    });

    btnNewVideo.addEventListener('click', () => {
        // Reset state
        sessionId = null;
        chatScreen.classList.add('hidden');
        sidebar.classList.add('hidden');
        sidebar.classList.remove('open');
        setupScreen.classList.remove('hidden');
        youtubeUrlInput.value = '';
        hideSetupAlert();
        btnClearChat.click();
    });

    async function sendMessage() {
        const text = chatInput.value.trim();
        if (!text || !sessionId) return;

        // Add user message
        appendMessage('user', text);
        
        // Reset input
        chatInput.value = '';
        chatInput.style.height = 'auto';
        btnSendMessage.disabled = true;

        // Add loading indicator for AI
        const typingId = appendTypingIndicator();
        
        isGenerating = true;
        abortController = new AbortController();
        btnStopGeneration.classList.remove('hidden');

        try {
            const response = await fetch('/ask', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    session_id: sessionId,
                    question: text
                }),
                signal: abortController.signal
            });

            const data = await response.json();
            
            removeElement(typingId);

            if (!response.ok) {
                appendMessage('ai', `**Error:** ${data.error || 'Failed to get answer.'}`);
            } else {
                appendMessage('ai', data.answer);
            }

        } catch (error) {
            removeElement(typingId);
            if (error.name === 'AbortError') {
                console.log('Fetch aborted');
            } else {
                console.error('Error asking question:', error);
                appendMessage('ai', '**Network Error:** Could not connect to server.');
            }
        } finally {
            finishGeneration();
        }
    }

    function finishGeneration() {
        isGenerating = false;
        abortController = null;
        btnStopGeneration.classList.add('hidden');
        chatInput.focus();
    }

    function appendMessage(role, text) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${role}-message`;
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content';
        
        if (role === 'ai') {
            // Parse Markdown for AI responses
            contentDiv.innerHTML = marked.parse(text);
        } else {
            // Escape HTML for user messages
            contentDiv.textContent = text;
        }
        
        msgDiv.appendChild(contentDiv);
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
    }

    function appendTypingIndicator() {
        const id = 'typing-' + Date.now();
        const msgDiv = document.createElement('div');
        msgDiv.id = id;
        msgDiv.className = 'message ai-message';
        
        const contentDiv = document.createElement('div');
        contentDiv.className = 'message-content typing-indicator';
        
        for (let i = 0; i < 3; i++) {
            const dot = document.createElement('div');
            dot.className = 'typing-dot';
            contentDiv.appendChild(dot);
        }
        
        msgDiv.appendChild(contentDiv);
        chatMessages.appendChild(msgDiv);
        scrollToBottom();
        return id;
    }

    function removeElement(id) {
        const el = document.getElementById(id);
        if (el) el.remove();
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }
});
