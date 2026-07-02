// Application State Controller
const state = {
    user: null,
    provider: 'gemini',
    apiKey: '',
    theme: 'light',
    activeCategory: null,
    stats: {
        queries: 0,
        feedbacks: 0
    },
    // Keep track of active query-response pairs for feedback
    activeMessageForFeedback: null
};

// DOM Cache
const dom = {
    authScreen: document.getElementById('auth-screen'),
    authForm: document.getElementById('auth-form'),
    authEmail: document.getElementById('auth-email'),
    authRole: document.getElementById('auth-role'),
    
    appContainer: document.getElementById('app-container'),
    sidebarLeft: document.querySelector('.sidebar-left'),
    sidebarToggleClose: document.getElementById('sidebar-toggle-close'),
    sidebarToggleOpen: document.getElementById('sidebar-toggle-open'),
    logoutBtn: document.getElementById('logout-btn'),
    
    clearChatBtn: document.getElementById('clear-chat-btn'),
    themeToggleBtn: document.getElementById('theme-toggle-btn'),
    themeIcon: document.getElementById('theme-icon'),
    settingsToggleBtn: document.getElementById('settings-toggle-btn'),
    connectionStatus: document.getElementById('connection-status'),
    connectionDot: document.getElementById('connection-dot'),
    
    chatViewport: document.getElementById('chat-viewport'),
    emptyState: document.getElementById('empty-state'),
    messagesContainer: document.getElementById('messages-container'),
    
    chatForm: document.getElementById('chat-form'),
    chatInput: document.getElementById('chat-input'),
    activeCategoryBadge: document.getElementById('active-category-badge'),
    activeCategoryName: document.getElementById('active-category-name'),
    clearCategoryBadge: document.getElementById('clear-category-badge'),
    btnSend: document.getElementById('btn-send'),
    
    sidebarRight: document.getElementById('sidebar-right'),
    sidebarRightClose: document.getElementById('sidebar-right-close'),
    profileAvatar: document.getElementById('profile-avatar'),
    profileEmail: document.getElementById('profile-email'),
    profileRole: document.getElementById('profile-role'),
    statQueries: document.getElementById('stat-queries'),
    statFeedback: document.getElementById('stat-feedback'),
    
    customApiKey: document.getElementById('custom-api-key'),
    
    feedbackModal: document.getElementById('feedback-modal'),
    feedbackModalClose: document.getElementById('feedback-modal-close'),
    feedbackForm: document.getElementById('feedback-form'),
    feedbackRatingBadge: document.getElementById('feedback-rating-badge'),
    feedbackRatingText: document.getElementById('feedback-rating-text'),
    feedbackComment: document.getElementById('feedback-comment'),
    feedbackCancelBtn: document.getElementById('feedback-cancel-btn'),
    
    sidebarSearch: document.getElementById('sidebar-search'),
    clearSidebarSearch: document.getElementById('clear-sidebar-search'),
    toastContainer: document.getElementById('toast-container')
};

// FAQ categories mappings for rendering badges
const categoryNames = {
    "1": "Technical Issues",
    "2": "Candidate Support",
    "3": "Interviewer Queries",
    "4": "Recruiter Help",
    "5": "Platform Overview",
    "6": "Pricing & Plans"
};

// Initialize Application
function init() {
    // Configure markdown breaks
    marked.use({
        breaks: true,
        gfm: true
    });
    
    loadSession();
    initTheme();
    setupEventListeners();
    lucide.createIcons();
}

// Load configurations and sessions
function loadSession() {
    // 1. Auth session
    const storedUser = localStorage.getItem('fc_user');
    if (storedUser) {
        try {
            state.user = JSON.parse(storedUser);
            showMainApp();
        } catch (e) {
            localStorage.removeItem('fc_user');
        }
    } else {
        showAuthScreen();
    }
    
    // 2. Settings
    state.provider = localStorage.getItem('fc_provider') || 'gemini';
    const providerRadio = document.querySelector(`input[name="provider"][value="${state.provider}"]`);
    if (providerRadio) providerRadio.checked = true;
    
    state.apiKey = localStorage.getItem('fc_api_key') || '';
    dom.customApiKey.value = state.apiKey;
    
    // 3. Stats
    state.stats.queries = parseInt(localStorage.getItem('fc_queries') || '0', 10);
    state.stats.feedbacks = parseInt(localStorage.getItem('fc_feedbacks') || '0', 10);
    updateStatsUI();
}

function showAuthScreen() {
    dom.authScreen.classList.remove('hidden');
    dom.appContainer.classList.add('hidden');
}

function showMainApp() {
    dom.authScreen.classList.add('hidden');
    dom.appContainer.classList.remove('hidden');
    
    // Fill profile info
    if (state.user) {
        dom.profileEmail.textContent = state.user.email;
        dom.profileRole.textContent = state.user.role;
        
        // Initials avatar
        const parts = state.user.email.split('@')[0].split(/[._-]/);
        const initials = parts.map(p => p[0] || '').join('').substring(0, 2).toUpperCase() || 'FC';
        dom.profileAvatar.textContent = initials;
    }
}

// Theme Handling
function initTheme() {
    state.theme = localStorage.getItem('fc_theme') || 'light';
    document.documentElement.setAttribute('data-theme', state.theme);
    updateThemeIcon();
}

function toggleTheme() {
    state.theme = state.theme === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', state.theme);
    localStorage.setItem('fc_theme', state.theme);
    updateThemeIcon();
}

function updateThemeIcon() {
    if (state.theme === 'dark') {
        dom.themeIcon.setAttribute('data-lucide', 'moon');
    } else {
        dom.themeIcon.setAttribute('data-lucide', 'sun');
    }
    lucide.createIcons();
}

// Update local statistic displays
function updateStatsUI() {
    dom.statQueries.textContent = state.stats.queries;
    dom.statFeedback.textContent = state.stats.feedbacks;
}

function incrementQueryStat() {
    state.stats.queries += 1;
    localStorage.setItem('fc_queries', state.stats.queries.toString());
    updateStatsUI();
}

function incrementFeedbackStat() {
    state.stats.feedbacks += 1;
    localStorage.setItem('fc_feedbacks', state.stats.feedbacks.toString());
    updateStatsUI();
}

// UI Accordion logic for categories
function setupAccordion() {
    document.querySelectorAll('.category-trigger').forEach(trigger => {
        trigger.addEventListener('click', (e) => {
            const item = trigger.closest('.category-item');
            const isExpanded = item.classList.contains('expanded');
            
            // Close all others
            document.querySelectorAll('.category-item').forEach(other => {
                if (other !== item) {
                    other.classList.remove('expanded');
                }
            });
            
            // Toggle current
            if (isExpanded) {
                item.classList.remove('expanded');
            } else {
                item.classList.add('expanded');
                // Set active category context when expanded
                const catId = item.getAttribute('data-category-id');
                setActiveCategory(catId);
            }
        });
    });
}

function setActiveCategory(catId) {
    state.activeCategory = catId;
    
    // Highlight sidebar item border
    document.querySelectorAll('.category-item').forEach(item => {
        if (item.getAttribute('data-category-id') === catId) {
            item.classList.add('active');
        } else {
            item.classList.remove('active');
        }
    });

    if (catId && categoryNames[catId]) {
        dom.activeCategoryName.textContent = categoryNames[catId];
        dom.activeCategoryBadge.classList.remove('hidden');
    } else {
        dom.activeCategoryBadge.classList.add('hidden');
    }
}

// Event Listeners setup
function setupEventListeners() {
    // Auth Form
    dom.authForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const email = dom.authEmail.value.trim();
        const role = dom.authRole.value;
        if (email && role) {
            const session = { email, role };
            localStorage.setItem('fc_user', JSON.stringify(session));
            state.user = session;
            showMainApp();
        }
    });
    
    // Sign Out
    dom.logoutBtn.addEventListener('click', () => {
        localStorage.removeItem('fc_user');
        state.user = null;
        showAuthScreen();
    });

    // Theme Toggle
    dom.themeToggleBtn.addEventListener('click', toggleTheme);
    
    // Left Sidebar Navigation (Mobile)
    dom.sidebarToggleOpen.addEventListener('click', () => {
        dom.sidebarLeft.classList.add('open');
    });
    dom.sidebarToggleClose.addEventListener('click', () => {
        dom.sidebarLeft.classList.remove('open');
    });
    
    // Config panel toggle
    dom.settingsToggleBtn.addEventListener('click', () => {
        dom.sidebarRight.classList.toggle('hidden');
        dom.sidebarRight.classList.toggle('open');
    });
    dom.sidebarRightClose.addEventListener('click', () => {
        dom.sidebarRight.classList.add('hidden');
        dom.sidebarRight.classList.remove('open');
    });
    
    // Model Provider toggles
    document.querySelectorAll('input[name="provider"]').forEach(radio => {
        radio.addEventListener('change', (e) => {
            state.provider = e.target.value;
            localStorage.setItem('fc_provider', state.provider);
        });
    });
    
    // Custom API key changes
    dom.customApiKey.addEventListener('input', (e) => {
        state.apiKey = e.target.value.trim();
        localStorage.setItem('fc_api_key', state.apiKey);
    });

    // Clear Category badge
    dom.clearCategoryBadge.addEventListener('click', () => {
        setActiveCategory(null);
    });

    // Accordions
    setupAccordion();

    // Guided questions click
    document.addEventListener('click', (e) => {
        const questionBtn = e.target.closest('.faq-question') || e.target.closest('.chip-btn');
        if (questionBtn) {
            const query = questionBtn.getAttribute('data-query');
            if (query) {
                // If it is a sidebar question, check its parent category to sync badge
                const parentCat = questionBtn.closest('.category-item');
                if (parentCat) {
                    const catId = parentCat.getAttribute('data-category-id');
                    setActiveCategory(catId);
                }
                submitQuery(query);
            }
        }
    });

    // Clear chat history
    dom.clearChatBtn.addEventListener('click', () => {
        if (confirm("Are you sure you want to clear the conversation history?")) {
            dom.messagesContainer.innerHTML = '';
            dom.emptyState.classList.remove('hidden');
            showToast('Chat history cleared', 'info');
        }
    });

    // Textarea Enter key submit (Shift+Enter for new line)
    dom.chatInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            dom.chatForm.requestSubmit();
        }
    });

    // Chat submit
    dom.chatForm.addEventListener('submit', (e) => {
        e.preventDefault();
        const text = dom.chatInput.value.trim();
        if (text) {
            submitQuery(text);
        }
    });

    // Close Feedback Modal
    dom.feedbackModalClose.addEventListener('click', closeFeedbackModal);
    dom.feedbackCancelBtn.addEventListener('click', closeFeedbackModal);
    
    // Submit Feedback
    dom.feedbackForm.addEventListener('submit', submitFeedbackData);

    // Sidebar search FAQ filtering
    dom.sidebarSearch.addEventListener('input', (e) => {
        const query = e.target.value.toLowerCase();
        filterSidebarFAQs(query);
    });

    dom.clearSidebarSearch.addEventListener('click', () => {
        dom.sidebarSearch.value = '';
        dom.clearSidebarSearch.classList.add('hidden');
        filterSidebarFAQs('');
    });
}

// Append a message bubble to viewport
function appendMessage(sender, text, messageId = null) {
    dom.emptyState.classList.add('hidden');
    
    const messageDiv = document.createElement('div');
    messageDiv.classList.add('message', sender === 'user' ? 'user-turn' : 'assistant-turn');
    if (messageId) {
        messageDiv.setAttribute('id', messageId);
    }
    
    const headerDiv = document.createElement('div');
    headerDiv.classList.add('message-header');
    
    const avatarDiv = document.createElement('div');
    avatarDiv.classList.add('message-avatar', sender);
    avatarDiv.textContent = sender === 'user' ? 'U' : 'FC';
    
    const senderName = document.createElement('span');
    senderName.classList.add('message-sender-name');
    senderName.textContent = sender === 'user' ? 'You' : 'FloCareer Assistant';
    
    headerDiv.appendChild(avatarDiv);
    headerDiv.appendChild(senderName);
    
    const bubbleDiv = document.createElement('div');
    bubbleDiv.classList.add('message-bubble');
    
    if (sender === 'user') {
        bubbleDiv.textContent = text;
    } else {
        bubbleDiv.innerHTML = marked.parse(formatNumberedLists(text));
    }
    
    messageDiv.appendChild(headerDiv);
    messageDiv.appendChild(bubbleDiv);
    
    // Add interaction footer for assistant messages
    if (sender === 'assistant' && messageId) {
        const footerDiv = document.createElement('div');
        footerDiv.classList.add('message-footer');
        
        // Copy text button
        const copyBtn = document.createElement('button');
        copyBtn.classList.add('action-icon-btn');
        copyBtn.setAttribute('title', 'Copy response text');
        copyBtn.innerHTML = '<i data-lucide="copy"></i>';
        copyBtn.addEventListener('click', () => {
            navigator.clipboard.writeText(text);
            copyBtn.innerHTML = '<i data-lucide="check"></i>';
            setTimeout(() => {
                copyBtn.innerHTML = '<i data-lucide="copy"></i>';
                lucide.createIcons({ attrs: { class: 'action-icon-btn i' } });
            }, 2000);
            lucide.createIcons();
        });
        
        // Thumbs Up button
        const upBtn = document.createElement('button');
        upBtn.classList.add('action-icon-btn');
        upBtn.setAttribute('title', 'Thumbs up (helpful)');
        upBtn.innerHTML = '<i data-lucide="thumbs-up"></i>';
        upBtn.addEventListener('click', () => {
            handleRatingClick(messageId, 1);
        });
        
        // Thumbs Down button
        const downBtn = document.createElement('button');
        downBtn.classList.add('action-icon-btn');
        downBtn.setAttribute('title', 'Thumbs down (needs improvement)');
        downBtn.innerHTML = '<i data-lucide="thumbs-down"></i>';
        downBtn.addEventListener('click', () => {
            handleRatingClick(messageId, -1);
        });
        
        footerDiv.appendChild(copyBtn);
        footerDiv.appendChild(upBtn);
        footerDiv.appendChild(downBtn);
        messageDiv.appendChild(footerDiv);
    }
    
    dom.messagesContainer.appendChild(messageDiv);
    scrollToBottom();
    lucide.createIcons();
    
    return bubbleDiv;
}

function scrollToBottom() {
    dom.chatViewport.scrollTo({
        top: dom.chatViewport.scrollHeight,
        behavior: 'smooth'
    });
}

// Add typing visualizer
function addTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.classList.add('typing-indicator');
    indicator.id = 'typing-indicator';
    
    for (let i = 0; i < 3; i++) {
        const dot = document.createElement('div');
        dot.classList.add('typing-dot');
        indicator.appendChild(dot);
    }
    
    dom.messagesContainer.appendChild(indicator);
    scrollToBottom();
}

function removeTypingIndicator() {
    const indicator = document.getElementById('typing-indicator');
    if (indicator) {
        indicator.remove();
    }
}

// Core Chat submit and SSE Stream reader
async function submitQuery(queryText) {
    dom.chatInput.value = '';
    setActiveButtonState(false);
    
    // 1. Render User message
    appendMessage('user', queryText);
    incrementQueryStat();
    
    // 2. Add loader
    addTypingIndicator();
    
    // 3. Create unique IDs for assistant feedback tracking
    const messageId = 'msg-' + Date.now();
    let assistantText = '';
    let displayedText = '';
    let textQueue = '';
    let isPrinting = false;
    let printTimer = null;
    let responseBubble = null;

    function animateText() {
        if (textQueue.length > 0) {
            const size = Math.min(3, textQueue.length);
            displayedText += textQueue.substring(0, size);
            textQueue = textQueue.substring(size);
            responseBubble.innerHTML = marked.parse(formatNumberedLists(displayedText));
            scrollToBottom();
            printTimer = setTimeout(animateText, 15);
        } else {
            isPrinting = false;
        }
    }

    function appendToPrintQueue(newText) {
        textQueue += newText;
        if (!isPrinting && textQueue.length > 0) {
            isPrinting = true;
            animateText();
        }
    }
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                message: queryText,
                provider: state.provider,
                api_key: state.apiKey || null,
                active_category: state.activeCategory
            })
        });
        
        removeTypingIndicator();
        
        if (!response.ok) {
            throw new Error(`Server returned HTTP ${response.status}`);
        }
        
        // Setup initial response bubble
        responseBubble = appendMessage('assistant', '', messageId);
        
        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            
            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Hold partial line
            
            for (const line of lines) {
                const cleaned = line.trim();
                if (cleaned.startsWith('data: ')) {
                    const content = cleaned.slice(6);
                    if (content === '[DONE]') {
                        break;
                    }
                    
                    try {
                        const parsed = JSON.parse(content);
                        if (parsed.text) {
                            assistantText += parsed.text;
                            appendToPrintQueue(parsed.text);
                        }
                        if (parsed.active_category) {
                            // If backend inferred category during stream, highlight it!
                            setActiveCategory(parsed.active_category);
                        }
                    } catch (err) {
                        console.error('Error parsing SSE segment:', err);
                    }
                }
            }
        }
        
        // Cache this pair in DOM element data attributes for feedback submissions after printing finishes
        function finalizeResponse() {
            if (isPrinting || textQueue.length > 0) {
                setTimeout(finalizeResponse, 50);
            } else {
                const messageNode = document.getElementById(messageId);
                if (messageNode) {
                    messageNode.setAttribute('data-query', queryText);
                    messageNode.setAttribute('data-response', assistantText);
                }
            }
        }
        finalizeResponse();
        
    } catch (e) {
        removeTypingIndicator();
        setConnectionState(false, e.message);
        appendMessage('assistant', `⚠️ **Connection Error**: Failed to fetch response. Make sure the FastAPI local server is running on port 8000.\n\n*Error details: ${e.message}*`, 'err-' + Date.now());
    } finally {
        setActiveButtonState(true);
    }
}

function setActiveButtonState(enabled) {
    dom.btnSend.disabled = !enabled;
    dom.chatInput.disabled = !enabled;
}

function setConnectionState(online, errMessage = '') {
    if (online) {
        dom.connectionDot.className = 'dot dot-online';
        dom.connectionStatus.textContent = 'Connected to FloCareer Agent';
    } else {
        dom.connectionDot.className = 'dot dot-offline';
        dom.connectionStatus.textContent = `Server Offline: ${errMessage}`;
    }
}

// User Feedback handlers
function handleRatingClick(messageId, ratingVal) {
    const messageNode = document.getElementById(messageId);
    if (!messageNode) return;
    
    // Highlight selected thumbs icon in message footer
    const footer = messageNode.querySelector('.message-footer');
    const buttons = footer.querySelectorAll('.action-icon-btn');
    
    // Reset buttons
    buttons[1].classList.remove('active'); // Thumbs up
    buttons[2].classList.remove('active'); // Thumbs down
    
    if (ratingVal === 1) {
        buttons[1].classList.add('active');
    } else {
        buttons[2].classList.add('active');
    }
    
    // Prepare feedback state & show modal
    state.activeMessageForFeedback = {
        messageId: messageId,
        query: messageNode.getAttribute('data-query') || '',
        response: messageNode.getAttribute('data-response') || '',
        rating: ratingVal
    };
    
    openFeedbackModal(ratingVal);
}

function openFeedbackModal(ratingVal) {
    dom.feedbackComment.value = '';
    
    if (ratingVal === 1) {
        dom.feedbackRatingBadge.className = 'badge-positive';
        dom.feedbackRatingBadge.innerHTML = '<i data-lucide="thumbs-up" class="badge-icon"></i> <span>Positive</span>';
        dom.feedbackRatingText.textContent = 'Positive';
    } else {
        dom.feedbackRatingBadge.className = 'badge-negative';
        dom.feedbackRatingBadge.innerHTML = '<i data-lucide="thumbs-down" class="badge-icon"></i> <span>Needs Improvement</span>';
        dom.feedbackRatingText.textContent = 'Needs Improvement';
    }
    
    dom.feedbackModal.classList.remove('hidden');
    lucide.createIcons();
}

function closeFeedbackModal() {
    dom.feedbackModal.classList.add('hidden');
    state.activeMessageForFeedback = null;
}

async function submitFeedbackData(e) {
    e.preventDefault();
    if (!state.activeMessageForFeedback) return;
    
    const commentText = dom.feedbackComment.value.trim();
    const feedbackPayload = {
        query: state.activeMessageForFeedback.query,
        response: state.activeMessageForFeedback.response,
        rating: state.activeMessageForFeedback.rating,
        comment: commentText || null,
        category: state.activeCategory || null,
        user_email: state.user ? state.user.email : 'anonymous',
        user_role: state.user ? state.user.role : 'anonymous'
    };
    
    try {
        const response = await fetch('/api/feedback', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(feedbackPayload)
        });
        
        if (!response.ok) {
            throw new Error('Feedback submission failed');
        }
        
        incrementFeedbackStat();
        showToast('Thank you for helping us improve FloCareer AI!', 'success');
        
    } catch (err) {
        console.error('Failed to submit feedback:', err);
        showToast('Could not save feedback. Make sure backend is running.', 'error');
    } finally {
        closeFeedbackModal();
    }
}

// Auto-expand textarea height during typing
dom.chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = (this.scrollHeight) + 'px';
});

// Run Init
window.addEventListener('DOMContentLoaded', init);

// Filter Guided FAQs in Sidebar
function filterSidebarFAQs(query) {
    const trimmedQuery = query.trim().toLowerCase();
    if (trimmedQuery) {
        dom.clearSidebarSearch.classList.remove('hidden');
    } else {
        dom.clearSidebarSearch.classList.add('hidden');
    }

    const categories = document.querySelectorAll('.category-item');
    categories.forEach(category => {
        const questions = category.querySelectorAll('.faq-question');
        let matchCount = 0;

        questions.forEach(q => {
            const text = q.textContent.toLowerCase();
            const queryAttr = (q.getAttribute('data-query') || '').toLowerCase();
            if (text.includes(trimmedQuery) || queryAttr.includes(trimmedQuery)) {
                q.classList.remove('hidden');
                matchCount++;
            } else {
                q.classList.add('hidden');
            }
        });

        if (trimmedQuery === '') {
            // Restore original state - remove active expanded class if not active category
            category.classList.remove('hidden');
            const catId = category.getAttribute('data-category-id');
            if (state.activeCategory !== catId) {
                category.classList.remove('expanded');
            } else {
                category.classList.add('expanded');
            }
        } else {
            if (matchCount > 0) {
                category.classList.remove('hidden');
                category.classList.add('expanded');
            } else {
                category.classList.add('hidden');
                category.classList.remove('expanded');
            }
        }
    });
}

// Custom Toast Notification System
function showToast(message, type = 'success', duration = 4000) {
    if (!dom.toastContainer) return;

    const toast = document.createElement('div');
    toast.className = `toast ${type}`;

    // Map type to icons
    let iconName = 'info';
    if (type === 'success') iconName = 'check-circle';
    else if (type === 'error') iconName = 'alert-triangle';
    else if (type === 'warning') iconName = 'alert-circle';

    const titleText = type.charAt(0).toUpperCase() + type.slice(1);

    toast.innerHTML = `
        <i data-lucide="${iconName}" class="toast-icon"></i>
        <div class="toast-body">
            <div class="toast-title">${titleText}</div>
            <div class="toast-message">${message}</div>
        </div>
        <button type="button" class="toast-close" title="Close">
            <i data-lucide="x"></i>
        </button>
    `;

    dom.toastContainer.appendChild(toast);
    lucide.createIcons();

    // Trigger animation
    setTimeout(() => {
        toast.classList.add('show');
    }, 10);

    // Auto dismiss
    const dismissTimer = setTimeout(() => {
        dismissToast(toast);
    }, duration);

    // Manual close
    const closeBtn = toast.querySelector('.toast-close');
    closeBtn.addEventListener('click', () => {
        clearTimeout(dismissTimer);
        dismissToast(toast);
    });
}

function dismissToast(toast) {
    toast.classList.remove('show');
    toast.addEventListener('transitionend', () => {
        toast.remove();
    });
}

// Format flat inline lists to structured newlines
function formatNumberedLists(text) {
    if (!text) return text;
    return text.replace(/ (?=\d+\.\s)/g, '\n');
}
