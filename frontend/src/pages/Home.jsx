import { useState, useEffect, useRef } from "react";

import NameGate from "../components/NameGate";
import ChatPanel from "../components/ChatPanel";
import { createSession, sendMessage, deleteSession, deleteSessionOnPageExit } from "../services/chatbotService";

import "../styles/Home.css";
import "../styles/Chat.css";

function Home() {
    const [sessionId, setSessionId] = useState(null);
    const [fullName, setFullName] = useState("");
    const [messages, setMessages] = useState([]);
    const [inputValue, setInputValue] = useState("");
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    // Effects below run once, so they cannot read sessionId from state - the
    // ref mirrors it so cleanup always sees the current value.
    const sessionIdRef = useRef(null);

    // Cleanup: Delete session
    useEffect(() => {
        return () => {
            if (sessionIdRef.current !== null) {

                deleteSession(sessionIdRef.current).catch((error) => {
                    console.error(
                        "Could not delete session:",
                        error
                    );
                });

            }
        };

    }, []);

    useEffect(() => {

        function handlePageHide() {
            if (sessionIdRef.current !== null) {
                deleteSessionOnPageExit(
                    sessionIdRef.current
                );
            }
        }

        window.addEventListener(
            "pagehide",
            handlePageHide
        );

        return () => {
            window.removeEventListener(
                "pagehide",
                handlePageHide
            );
        };

    }, []);

    // Verify the patient by full name and open a session.
    // Nothing is written to localStorage/sessionStorage on purpose: a refresh
    // resets this component's state, so the patient has to verify again.
    async function handleVerify(name) {
        setLoading(true);
        setError("");

        try {
            const result = await createSession(name);

            setSessionId(result.session_id);
            sessionIdRef.current = result.session_id;
            setFullName(result.full_name);

        } catch (error) {
            console.error(error);
            setError(error.message || "Could not verify name.");

        } finally {
            setLoading(false);
        }
    }

    // Send text messages to backend
    async function handleSendMessage() {
        const text = inputValue.trim();

        if (!text || loading || sessionId === null) { return; }

        // Show the patient's own message straight away, then clear the box.
        const userMessage = {id: `user-${Date.now()}`, role: "user", content: text};
        setMessages((currentMessages) => [...currentMessages, userMessage]);
        setInputValue("");

        setLoading(true);
        setError("");

        try {
            const result = await sendMessage(sessionId, text);

            const botMessage = {id: result.message_id, role: "bot", content: result.response, sources: result.sources};

            setMessages((currentMessages) => [...currentMessages, botMessage]);

        } catch (error) {
            console.error(error);
            setError(error.message || "Could not send message.");

        } finally {
            setLoading(false);
        }
    }

    return (
        <main className="home-page">

            {sessionId === null ? (
                <NameGate onVerify={handleVerify} loading={loading} error={error}/>
            ) : (
                <ChatPanel messages={messages} fullName={fullName} inputValue={inputValue} onInputChange={setInputValue}
                onSendMessage={handleSendMessage} loading={loading} error={error}/>
            )}

        </main>
    );
}

export default Home;
