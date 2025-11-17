// ===========================
// 전역 변수
// ===========================

const emotionEmojis = {
    'happy': '😊',
    'depressed': '😢',
    'surprised': '😮',
    'angry': '😠',
    'neutral': '😐'
};

const emotionNames = {
    'happy': 'Happy',
    'depressed': 'Depressed',
    'surprised': 'Surprised',
    'angry': 'Angry',
    'neutral': 'Neutral'
};

let eventSource = null;
let isConnected = false;
let recognition = null;
let isListening = false;
let silenceTimer = null;
let interimTranscript = '';
let finalTranscript = '';

// ===========================
// DOM 요소
// ===========================

const statusDot = document.getElementById('status-dot');
const statusText = document.getElementById('status-text');
const primaryEmotionEmoji = document.getElementById('primary-emotion-emoji');
const primaryEmotionName = document.getElementById('primary-emotion-name');
const primaryEmotionConfidence = document.getElementById('primary-emotion-confidence');
const lastUpdateTime = document.getElementById('last-update-time');
const modalitiesDisplay = document.getElementById('modalities-display');
const chatMessages = document.getElementById('chat-messages');
const chatInput = document.getElementById('chat-input');
const sendButton = document.getElementById('send-button');
const voiceToggle = document.getElementById('voice-toggle');
const voiceStatus = document.getElementById('voice-status');
const voiceStatusText = document.getElementById('voice-status-text');
const interimTextDiv = document.getElementById('interim-text');
const clearChatButton = document.getElementById('clear-chat');
const loadingOverlay = document.getElementById('loading-overlay');

// ===========================
// 초기화
// ===========================

document.addEventListener('DOMContentLoaded', () => {
    console.log('🚀 앱 초기화 시작');
    
    // 이벤트 리스너 등록
    sendButton.addEventListener('click', sendMessage);
    voiceToggle.addEventListener('click', toggleVoiceRecognition);
    chatInput.addEventListener('keydown', handleKeyPress);
    clearChatButton.addEventListener('click', clearChat);
    
    // 실시간 감정 스트림 연결
    connectEmotionStream();
    
    // 초기 감정 데이터 로드
    fetchCurrentEmotions();
    
    // 실시간 음성 인식 초기화 및 자동 시작
    initSpeechRecognition();
    
    console.log('✅ 앱 초기화 완료');
});

// ===========================
// 실시간 감정 스트림 (SSE)
// ===========================

function connectEmotionStream() {
    console.log('📡 SSE 연결 시도...');
    
    // 기존 연결 종료
    if (eventSource) {
        eventSource.close();
    }
    
    // 새 연결 생성
    eventSource = new EventSource('/api/emotions/stream');
    
    eventSource.onopen = () => {
        console.log('✅ SSE 연결 성공');
        isConnected = true;
        updateConnectionStatus(true);
    };
    
    eventSource.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            console.log('📊 감정 데이터 수신:', data);
            updateEmotionDisplay(data);
        } catch (error) {
            console.error('데이터 파싱 에러:', error);
        }
    };
    
    eventSource.onerror = (error) => {
        console.error('❌ SSE 에러:', error);
        isConnected = false;
        updateConnectionStatus(false);
        
        // 5초 후 재연결 시도
        setTimeout(() => {
            console.log('🔄 재연결 시도...');
            connectEmotionStream();
        }, 5000);
    };
}

// ===========================
// 연결 상태 업데이트
// ===========================

function updateConnectionStatus(connected) {
    if (connected) {
        statusDot.className = 'dot connected';
        statusText.textContent = '실시간 연결됨';
    } else {
        statusDot.className = 'dot error';
        statusText.textContent = '연결 끊김';
    }
}

// ===========================
// 감정 표시 업데이트
// ===========================

function updateEmotionDisplay(data) {
    const emotions = data.emotions;
    const primaryEmotion = data.primary_emotion;
    
    // 주요 감정 업데이트
    primaryEmotionEmoji.textContent = emotionEmojis[primaryEmotion] || '😐';
    primaryEmotionName.textContent = emotionNames[primaryEmotion] || primaryEmotion;
    primaryEmotionConfidence.textContent = `${(emotions[primaryEmotion] * 100).toFixed(1)}%`;
    
    // 5가지 감정 바 업데이트
    for (const [emotion, probability] of Object.entries(emotions)) {
        const percentage = (probability * 100).toFixed(1);
        const bar = document.getElementById(`bar-${emotion}`);
        const pct = document.getElementById(`pct-${emotion}`);
        
        if (bar && pct) {
            bar.style.width = `${percentage}%`;
            pct.textContent = `${percentage}%`;
        }
    }
    
    // 모달리티 정보 업데이트
    if (data.modalities) {
        updateModalitiesDisplay(data.modalities);
    }
    
    // 마지막 업데이트 시간
    const timestamp = new Date(data.timestamp);
    lastUpdateTime.textContent = timestamp.toLocaleTimeString('ko-KR');
}

function updateModalitiesDisplay(modalities) {
    const tags = [];
    
    if (modalities.face) {
        tags.push(`<span class="tag">👤 얼굴 (${modalities.face})</span>`);
    }
    if (modalities.voice) {
        tags.push(`<span class="tag">🎤 음성 (${modalities.voice})</span>`);
    }
    if (modalities.text) {
        tags.push(`<span class="tag">📝 텍스트 (${modalities.text})</span>`);
    }
    
    if (tags.length > 0) {
        modalitiesDisplay.innerHTML = tags.join('');
    }
}

// ===========================
// 초기 감정 데이터 가져오기
// ===========================

async function fetchCurrentEmotions() {
    try {
        const response = await fetch('/api/emotions');
        const data = await response.json();
        updateEmotionDisplay(data);
    } catch (error) {
        console.error('감정 데이터 가져오기 실패:', error);
    }
}

// ===========================
// 실시간 음성 인식 기능
// ===========================

function initSpeechRecognition() {
    // Web Speech API 지원 확인
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    
    if (!SpeechRecognition) {
        console.error('이 브라우저는 음성 인식을 지원하지 않습니다.');
        voiceToggle.disabled = true;
        voiceToggle.querySelector('.status-text').textContent = '음성 인식 미지원';
        return;
    }
    
    recognition = new SpeechRecognition();
    recognition.continuous = true;  // 계속 듣기
    recognition.interimResults = true;  // 중간 결과 표시
    recognition.lang = 'ko-KR';  // 한국어
    recognition.maxAlternatives = 1;
    
    // 음성 인식 결과 처리
    recognition.onresult = (event) => {
        interimTranscript = '';
        finalTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }
        
        // 중간 결과 표시
        if (interimTranscript) {
            interimTextDiv.textContent = interimTranscript;
            voiceStatusText.textContent = '듣고 있습니다... 🎤';
        }
        
        // 최종 결과 처리
        if (finalTranscript) {
            console.log('🎤 인식된 음성:', finalTranscript);
            interimTextDiv.textContent = '';
            
            // 침묵 타이머 초기화
            clearTimeout(silenceTimer);
            
            // 1초 침묵 후 전송
            silenceTimer = setTimeout(() => {
                if (finalTranscript.trim()) {
                    sendVoiceMessage(finalTranscript.trim());
                    finalTranscript = '';
                }
            }, 1000);
        }
    };
    
    // 음성 인식 시작
    recognition.onstart = () => {
        console.log('🎤 실시간 음성 인식 시작');
        isListening = true;
        voiceStatusText.textContent = '듣고 있습니다...';
    };
    
    // 음성 인식 종료
    recognition.onend = () => {
        console.log('🎤 음성 인식 종료');
        
        // 자동 재시작 (토글이 활성화된 경우)
        if (isListening && voiceToggle.classList.contains('active')) {
            recognition.start();
        }
    };
    
    // 에러 처리
    recognition.onerror = (event) => {
        console.error('음성 인식 에러:', event.error);
        
        if (event.error === 'no-speech') {
            // 음성이 없을 때는 무시
            return;
        }
        
        if (event.error === 'not-allowed') {
            alert('마이크 권한이 필요합니다. 브라우저 설정에서 마이크 권한을 허용해주세요.');
            stopVoiceRecognition();
        }
    };
    
    // 자동 시작
    startVoiceRecognition();
}

function toggleVoiceRecognition() {
    if (isListening) {
        stopVoiceRecognition();
    } else {
        startVoiceRecognition();
    }
}

function startVoiceRecognition() {
    if (!recognition) {
        console.error('음성 인식이 초기화되지 않았습니다.');
        return;
    }
    
    try {
        recognition.start();
        isListening = true;
        
        // UI 업데이트
        voiceToggle.classList.add('active');
        voiceToggle.querySelector('.status-text').textContent = '실시간 음성 인식 중...';
        
        console.log('✅ 실시간 음성 인식 활성화');
    } catch (error) {
        console.error('음성 인식 시작 실패:', error);
    }
}

function stopVoiceRecognition() {
    if (!recognition) return;
    
    recognition.stop();
    isListening = false;
    
    // UI 업데이트
    voiceToggle.classList.remove('active');
    voiceToggle.querySelector('.status-text').textContent = '음성 인식 중지됨 (클릭하여 시작)';
    voiceStatusText.textContent = '음성 인식이 중지되었습니다.';
    interimTextDiv.textContent = '';
    
    console.log('⏸️ 실시간 음성 인식 중지');
}

async function sendVoiceMessage(text) {
    if (!text || text.trim().length === 0) return;
    
    // 타이핑 중 메시지 추가
    const typingMessageId = addTypingIndicator();
    voiceStatusText.textContent = 'AI가 응답 생성 중...';
    
    try {
        // ChatGPT에 전송 (음성 입력 플래그)
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                message: text,
                is_voice: true  // 음성 입력 플래그 (터미널에만 로그)
            })
        });
        
        if (!response.ok) {
            throw new Error('서버 응답 오류');
        }
        
        const data = await response.json();
        
        // 타이핑 인디케이터 제거
        removeTypingIndicator(typingMessageId);
        
        // AI 응답을 타이핑 효과로 표시
        await addMessageWithTyping('ai', data.message);
        
        if (data.detected_emotions) {
            console.log('📝 텍스트 감정 분석:', data.detected_emotions);
        }
        
        voiceStatusText.textContent = '듣고 있습니다...';
        
    } catch (error) {
        console.error('음성 메시지 전송 실패:', error);
        removeTypingIndicator(typingMessageId);
        addMessageToChat('ai', '죄송합니다. 음성 처리 중 오류가 발생했습니다. 😔');
        voiceStatusText.textContent = '오류 발생 - 다시 말씀해주세요.';
    }
}

// ===========================
// 채팅 기능
// ===========================

function handleKeyPress(event) {
    // Shift + Enter로 전송
    if (event.key === 'Enter' && event.shiftKey) {
        event.preventDefault();
        sendMessage();
    }
}

async function sendMessage() {
    const message = chatInput.value.trim();
    
    if (!message) {
        return;
    }
    
    // 사용자 메시지 표시
    addMessageToChat('user', message);
    
    // 입력창 초기화
    chatInput.value = '';
    
    // 전송 버튼 비활성화
    sendButton.disabled = true;
    
    // 타이핑 중 메시지 추가
    const typingMessageId = addTypingIndicator();
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ 
                message: message,
                is_voice: false  // 텍스트 입력
            })
        });
        
        if (!response.ok) {
            throw new Error('서버 응답 오류');
        }
        
        const data = await response.json();
        
        // 타이핑 인디케이터 제거
        removeTypingIndicator(typingMessageId);
        
        // AI 응답을 타이핑 효과로 표시
        await addMessageWithTyping('ai', data.message);
        
        // 텍스트 감정 분석 결과 있으면 업데이트
        if (data.detected_emotions) {
            console.log('📝 텍스트 감정 분석:', data.detected_emotions);
        }
        
    } catch (error) {
        console.error('메시지 전송 실패:', error);
        removeTypingIndicator(typingMessageId);
        addMessageToChat('ai', '죄송합니다. 일시적인 오류가 발생했습니다. 다시 시도해주세요. 😔');
    } finally {
        // 전송 버튼 활성화
        sendButton.disabled = false;
        
        // 포커스 복귀
        chatInput.focus();
    }
}

function addMessageToChat(sender, message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const senderName = sender === 'user' ? '나' : 'AI 상담사';
    
    contentDiv.innerHTML = `
        <strong>${senderName}</strong>
        <p>${escapeHtml(message)}</p>
    `;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // 스크롤을 최하단으로
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

// 타이핑 인디케이터 추가
function addTypingIndicator() {
    const messageDiv = document.createElement('div');
    const typingId = 'typing-' + Date.now();
    messageDiv.id = typingId;
    messageDiv.className = 'message ai-message typing-indicator-message';
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    contentDiv.innerHTML = `
        <strong>AI 상담사</strong>
        <div class="typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // 스크롤을 최하단으로
    chatMessages.scrollTop = chatMessages.scrollHeight;
    
    return typingId;
}

// 타이핑 인디케이터 제거
function removeTypingIndicator(typingId) {
    const typingElement = document.getElementById(typingId);
    if (typingElement) {
        typingElement.remove();
    }
}

// 타이핑 효과로 메시지 추가
async function addMessageWithTyping(sender, message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;
    
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    const senderName = sender === 'user' ? '나' : 'AI 상담사';
    
    contentDiv.innerHTML = `
        <strong>${senderName}</strong>
        <p class="typing-text"></p>
    `;
    
    messageDiv.appendChild(contentDiv);
    chatMessages.appendChild(messageDiv);
    
    // 타이핑 효과
    const textElement = contentDiv.querySelector('.typing-text');
    const text = escapeHtml(message);
    let index = 0;
    
    return new Promise((resolve) => {
        const typingSpeed = 30; // 30ms per character
        
        const typeInterval = setInterval(() => {
            if (index < text.length) {
                textElement.textContent += text.charAt(index);
                index++;
                
                // 스크롤을 최하단으로
                chatMessages.scrollTop = chatMessages.scrollHeight;
            } else {
                clearInterval(typeInterval);
                // 타이핑이 끝나면 깜빡이는 커서를 제거하기 위해 클래스 제거
                textElement.classList.remove('typing-text');
                resolve();
            }
        }, typingSpeed);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

async function clearChat() {
    if (!confirm('대화 내용을 모두 삭제하시겠습니까?')) {
        return;
    }
    
    try {
        const response = await fetch('/api/chat/clear', {
            method: 'POST'
        });
        
        if (response.ok) {
            // 채팅 화면 초기화
            chatMessages.innerHTML = `
                <div class="message ai-message">
                    <div class="message-content">
                        <strong>AI 상담사</strong>
                        <p>안녕하세요! 저는 당신의 감정을 이해하고 함께 나누고 싶은 AI 상담사입니다. 
                           편안하게 이야기해주세요. 어떤 일이든 함께 나눌 수 있습니다. 😊</p>
                    </div>
                </div>
            `;
            
            console.log('✅ 채팅 히스토리 초기화 완료');
        }
    } catch (error) {
        console.error('채팅 초기화 실패:', error);
        alert('채팅 초기화 중 오류가 발생했습니다.');
    }
}

// ===========================
// UI 헬퍼 함수
// ===========================

function showLoading(show) {
    loadingOverlay.style.display = show ? 'flex' : 'none';
}

// ===========================
// 페이지 종료 시 정리
// ===========================

window.addEventListener('beforeunload', () => {
    if (eventSource) {
        eventSource.close();
        console.log('🔌 SSE 연결 종료');
    }
});

// ===========================
// 에러 핸들링
// ===========================

window.addEventListener('error', (event) => {
    console.error('전역 에러:', event.error);
});

// ===========================
// 디버그 헬퍼
// ===========================

if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
    console.log('🛠️ 개발 모드 활성화');
    
    // 전역 디버그 함수
    window.debugEmotions = () => {
        fetch('/api/emotions')
            .then(res => res.json())
            .then(data => console.table(data.emotions));
    };
    
    window.debugService = () => {
        fetch('/api/service/status')
            .then(res => res.json())
            .then(data => console.log('서비스 상태:', data));
    };
}

console.log('✅ app.js 로드 완료');
