const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const preview = document.getElementById('preview');
const placeholder = document.getElementById('placeholder');
const predictBtn = document.getElementById('predict-btn');
const btnText = document.querySelector('.btn-text');
const loader = document.getElementById('loader');

const resultCard = document.getElementById('result-card');
const initialMessage = document.getElementById('initial-message');
const flowerText = document.getElementById('predicted-flower');
const confidenceText = document.getElementById('confidence-text');
const descriptionText = document.getElementById('flower-description');
const searchLink = document.getElementById('search-link');

let currentFile = null;

// Automatically detect if we should use absolute or relative URLs
const API_BASE = window.location.protocol === 'file:' ? 'http://localhost:8000' : '';

dropZone.onclick = () => fileInput.click();

fileInput.onchange = (e) => {
    const file = e.target.files[0];
    if (file) handleFile(file);
};

dropZone.ondragover = (e) => { e.preventDefault(); dropZone.style.borderColor = 'var(--primary)'; };
dropZone.ondragleave = () => { dropZone.style.borderColor = 'var(--glass-border)'; };
dropZone.ondrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
};

function handleFile(file) {
    currentFile = file;
    const reader = new FileReader();
    reader.onload = (e) => {
        preview.src = e.target.result;
        preview.style.display = 'block';
        placeholder.style.display = 'none';
        predictBtn.disabled = false;
        resultCard.classList.remove('active');
        initialMessage.style.display = 'block';
    };
    reader.readAsDataURL(file);
}

async function fetchDescription(name) {
    try {
        // Fetch summary from Wikipedia REST API
        const url = `https://en.wikipedia.org/api/rest_v1/page/summary/${encodeURIComponent(name.toLowerCase())}`;
        const response = await fetch(url);
        if (!response.ok) return "No description available for this species on Wikipedia.";
        const data = await response.json();
        return data.extract || "General description not found.";
    } catch (err) {
        return "Failed to fetch description. Please check your internet connection.";
    }
}

predictBtn.onclick = async () => {
    if (!currentFile) return;

    // Loading UI state
    predictBtn.disabled = true;
    btnText.style.display = 'none';
    loader.style.display = 'block';

    const formData = new FormData();
    formData.append('file', currentFile);

    try {
        const response = await fetch(`${API_BASE}/predict`, { method: 'POST', body: formData });
        if (!response.ok) throw new Error('Network error');

        const data = await response.json();
        const name = data.flower_name;

        // Update UI Components
        initialMessage.style.display = 'none';
        resultCard.classList.add('active');
        flowerText.innerText = name.toUpperCase();
        confidenceText.innerText = (data.confidence * 100).toFixed(1) + '% Confidence';

        // Set Up Google Search Link
        searchLink.href = `https://www.google.com/search?q=${encodeURIComponent(name + ' flower')}`;

        // Fetch Species Info
        descriptionText.innerText = "Searching details...";
        const desc = await fetchDescription(name);
        descriptionText.innerText = desc;

    } catch (error) {
        alert('Connection error: Is the backend server running?');
        console.error(error);
    } finally {
        predictBtn.disabled = false;
        btnText.style.display = 'block';
        loader.style.display = 'none';
    }
};

// ═══════════ BloomBot Premium Chat Logic ═══════════
const chatFab = document.getElementById('chatFab');
const chatWindow = document.getElementById('chatWindow');
const chatClose = document.getElementById('chatClose');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const sendBtn = document.getElementById('sendBtn');
const typingIndicator = document.getElementById('typingIndicator');
const quickActions = document.getElementById('quickActions');

function getTimeStamp() {
    return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function openChat() {
    chatWindow.classList.add('open');
    chatFab.classList.add('active');
    chatFab.textContent = '✕';
    chatInput.focus();
    setTimeout(() => chatMessages.scrollTop = chatMessages.scrollHeight, 120);
}
function closeChat() {
    chatWindow.classList.remove('open');
    chatFab.classList.remove('active');
    chatFab.textContent = '🌸';
}

chatFab.addEventListener('click', () => {
    chatWindow.classList.contains('open') ? closeChat() : openChat();
});
chatClose.addEventListener('click', closeChat);

function appendMessage(text, sender = 'user') {
    const bubble = document.createElement('div');
    bubble.className = 'bubble ' + sender;
    bubble.innerHTML = text + '<span class="bubble-time">' + getTimeStamp() + '</span>';
    // Insert before typing indicator
    chatMessages.insertBefore(bubble, typingIndicator);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTyping() {
    typingIndicator.classList.add('visible');
    chatMessages.scrollTop = chatMessages.scrollHeight;
}
function hideTyping() {
    typingIndicator.classList.remove('visible');
}

async function botReply(userText) {
    showTyping();
    // Hide quick actions after first interaction (if present)
    if (quickActions) quickActions.style.display = 'none';

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question: userText })
        });

        if (!response.ok) throw new Error('Server error ' + response.status);

        const data = await response.json();
        let replyHTML = data.answer;

        // Append source links if Google/Wikipedia search was used
        if (data.web_search_used && data.sources && data.sources.length > 0) {
            replyHTML += '<div class="chat-sources"><span>🔗 Sources:</span>';
            data.sources.forEach(src => {
                replyHTML += `<a href="${src.link}" target="_blank" rel="noopener noreferrer">${src.title}</a>`;
            });
            replyHTML += '</div>';
        }

        hideTyping();
        appendMessage(replyHTML, 'bot');

    } catch (err) {
        console.error('Chat error:', err);
        hideTyping();
        appendMessage('🌸 Sorry, I could not reach the server. Make sure the backend is running on port 8000!', 'bot');
    }
}

function sendMessage(text) {
    const msg = text || chatInput.value.trim();
    if (!msg) return;
    appendMessage(msg, 'user');
    chatInput.value = '';
    botReply(msg);
}

sendBtn.addEventListener('click', () => sendMessage());
chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { e.preventDefault(); sendMessage(); }
});

// Quick-action chip clicks
document.querySelectorAll('.chip').forEach(chip => {
    chip.addEventListener('click', () => {
        const msg = chip.getAttribute('data-msg');
        sendMessage(msg);
    });
});
