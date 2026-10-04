import { useEffect, useRef } from "react";

import ChatMessage from "./ChatMessage";
import ChatInput from "./ChatInput";

function ChatPanel({ messages, inputValue, onInputChange, onSendMessage, loading, error }) {

    const messagesEndRef = useRef(null);

    // Make Chat Panel scroll down to last message
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({
            behavior: "smooth",
        });
    }, [messages]);

    return (
        <section className="chat-panel">
            <div className="panel-header">
                CONVERSATION {/*{fullName ? ` - ${fullName}` : ""}*/}
            </div>

            <div className="chat-panel__messages">
                {messages.length === 0 && (<p>Start a conversation with Medbot.</p>)}

                {/*Chat history*/}
                {messages.map((message) => (
                    <ChatMessage key={message.id} role={message.role} sender={message.role === "user" ? "You" : "Medbot"}
                    text={message.content} sources={message.sources}/>
                ))}

                {loading && (<p> Medbot is thinking...</p>)}

                {error && (<p className="error-message">{error}</p>)}

                <div ref={messagesEndRef} />
            </div>
            
            {/*Chat textbox*/}
            <ChatInput value={inputValue} onChange={onInputChange} onSubmit={onSendMessage} disabled={loading}/>

        </section>
    )
}

export default ChatPanel
