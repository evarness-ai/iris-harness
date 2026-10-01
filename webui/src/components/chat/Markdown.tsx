import ReactMarkdown, { type Components } from "react-markdown";
import { Link } from "react-router-dom";

// Token-styled renderers for each markdown element — no inline style, no raw
// hex. react-markdown is safe by default (raw HTML is NOT rendered), so there's
// no XSS surface and no sanitizer dependency needed.
// Exported so another screen (the stored digest) can render with the same look
// and override one element.
export const markdownComponents: Components = {
  p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
  // An app-relative link ("/actions", from an approval halt) stays in the app: in a
  // home-screen web app a new tab would throw the owner out to the browser. Anything
  // else opens in a new tab as before.
  a: ({ href, children }) =>
    href && href.startsWith("/") && !href.startsWith("//") ? (
      <Link to={href} className="text-primary underline underline-offset-2">
        {children}
      </Link>
    ) : (
      <a
        href={href}
        target="_blank"
        rel="noreferrer"
        className="text-primary underline underline-offset-2"
      >
        {children}
      </a>
    ),
  ul: ({ children }) => <ul className="mb-2 list-disc pl-5 last:mb-0">{children}</ul>,
  ol: ({ children }) => <ol className="mb-2 list-decimal pl-5 last:mb-0">{children}</ol>,
  li: ({ children }) => <li className="mb-0.5">{children}</li>,
  h1: ({ children }) => <h3 className="mb-1 mt-2 text-sm font-semibold first:mt-0">{children}</h3>,
  h2: ({ children }) => <h3 className="mb-1 mt-2 text-sm font-semibold first:mt-0">{children}</h3>,
  h3: ({ children }) => (
    <h4 className="mb-1 mt-2 text-[13px] font-semibold first:mt-0">{children}</h4>
  ),
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  hr: () => <hr className="my-3 border-border" />,
  blockquote: ({ children }) => (
    <blockquote className="my-2 border-l-2 border-border pl-3 text-fg-muted">{children}</blockquote>
  ),
  code: ({ children }) => (
    <code className="rounded bg-bg px-1 py-0.5 font-mono text-[12px]">{children}</code>
  ),
  // Block code: the <pre> owns the container; reset the nested <code> so it
  // doesn't double up the inline-code background/padding.
  pre: ({ children }) => (
    <pre className="mb-2 overflow-x-auto rounded-lg bg-bg p-3 font-mono text-[12px] last:mb-0 [&_code]:bg-transparent [&_code]:p-0">
      {children}
    </pre>
  ),
};

/** Render assistant markdown with the design tokens. */
export function Markdown({ text }: { text: string }) {
  return (
    <div className="text-sm leading-relaxed [word-break:break-word]">
      <ReactMarkdown components={markdownComponents}>{text}</ReactMarkdown>
    </div>
  );
}
