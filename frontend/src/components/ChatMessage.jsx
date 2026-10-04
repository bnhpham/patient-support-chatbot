import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkBreaks from "remark-breaks";

function ChatMessage({ role, sender, text, sources }) {

    const hasSources = Array.isArray(sources) && sources.length > 0;

    return (
        <div className={`chat-message chat-message--${role}`}>
            <div className="chat-message__sender">
                {sender}
            </div>

            <div className="chat-message__bubble">
                {/*Chat message - Markdown only, no rehype-raw: retrieved/generated content can be
                adversarial in this app, so raw HTML from the LLM is never rendered, only Markdown syntax*/}
                <div className="chat-message__markdown">
                    <ReactMarkdown remarkPlugins={[remarkGfm, remarkBreaks]}>{text}</ReactMarkdown>
                </div>
            </div>

            {/*Which retrieved chunks provided this answer - useful when comparing
            retrieval behaviour before and after guardrails are added*/}
            {hasSources && (
                <details className="chat-sources">
                    <summary>{sources.length} source{sources.length > 1 ? "s" : ""} (For Debugging: Which retrieved chunks provided this answer?)</summary>

                    <ul>
                        {sources.map((source, index) => (
                            <li key={`${source.source_record_id}-${source.section}-${index}`}>
                                {source.section} &mdash; {source.source_record_id}
                            </li>
                        ))}
                    </ul>
                </details>
            )}

        </div>
    )
}

export default ChatMessage
