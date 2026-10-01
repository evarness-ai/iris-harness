/* The stored digest, full (loop-proof plan PR 2, graph §7).
 *
 * Telegram gets the digest in chunks and the phone a three-line headline; this
 * is the one rendering with everything, and where the push notification's tap
 * lands (/digest/<id>). /digest alone shows the newest.
 *
 * The body is the markdown the harness rendered, unchanged. The morning digest
 * is grouped (digest v5): `## ☀️ Today` group headings (shown as small uppercase
 * accent labels), `### Section` headings, one italic line per group that folds
 * its empty sections (muted here), a `⚠ couldn't build` line per group with a
 * failed section (a warning strip), and the footer after a `---` rule. Older,
 * flat digests use `## Section` headings and render with the same styles.
 *
 * Two more things differ from chat's renderer: an `iris:not-useful/<sender>` link
 * becomes a 👎 button (every other channel drops it), and ordinary https links
 * stay links. */
import { useState, type ReactNode } from "react";
import { Link, useParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import ReactMarkdown, { defaultUrlTransform, type Components } from "react-markdown";
import { toast } from "sonner";
import { Section } from "@/components/layout";
import { Tag } from "@/components/Tag";
import { markdownComponents } from "@/components/chat/Markdown";
import { ApiUnavailable, fmtDateTime } from "@/components/control/parts";
import {
  DigestNotFound,
  NOT_USEFUL_SCHEME,
  fetchDigest,
  fetchDigestList,
  markNotUseful,
  notUsefulSender,
  type StoredDigest,
} from "@/lib/digest";

/** react-markdown blanks any URL scheme it does not know; keep our one action. */
function urlTransform(url: string): string {
  return url.startsWith(NOT_USEFUL_SCHEME) ? url : defaultUrlTransform(url);
}

/* The 👎 is a 44px tap target that does not grow its line: negative vertical margins
 * take back the height it adds to the line box (an inline-block's margin box is what
 * the line holds), so a Focus row is as tall as a news row with its links. No border:
 * the box is taller than the line, a border would overlap the rows around it. */
const NOT_USEFUL_CLASS =
  "ml-1 -my-3 inline-flex h-11 min-w-11 items-center justify-center rounded px-1 align-middle text-[14px] leading-none hover:opacity-70 disabled:opacity-50";

function NotUsefulButton({ sender, children }: { sender: string; children: ReactNode }) {
  const [state, setState] = useState<"idle" | "pending" | "done">("idle");
  if (state === "done") {
    return (
      <span className="ml-1 text-[11px] text-fg-subtle" data-testid="not-useful-done">
        hidden from tomorrow's digest
      </span>
    );
  }
  return (
    <button
      type="button"
      data-testid="not-useful"
      disabled={state === "pending"}
      title={`Not useful: hide ${sender} from tomorrow's digest`}
      aria-label={`Not useful: hide ${sender} from tomorrow's digest`}
      className={NOT_USEFUL_CLASS}
      onClick={async () => {
        setState("pending");
        try {
          await markNotUseful(sender);
          setState("done");
        } catch (e) {
          setState("idle");
          toast.error(e instanceof Error ? e.message : "could not save that");
        }
      }}
    >
      {children}
    </button>
  );
}

/* Links are tap targets here: the digest is read on a phone, so every link and
 * button is at least 44px tall (the viewport smoke holds this screen to zero
 * small targets). The link stays INLINE and gets its height from vertical padding:
 * padding on an inline box widens the tap area without moving the line, so a news
 * row keeps its normal height (an inline-flex min-height pushed every row apart).
 * In-app paths stay in the app; anything else opens a new tab. */
const LINK_CLASS = "py-[15px] text-primary underline underline-offset-2";

function DigestLink({ href, children }: { href?: string; children?: ReactNode }) {
  const sender = notUsefulSender(href);
  if (sender) return <NotUsefulButton sender={sender}>{children}</NotUsefulButton>;
  if (href && href.startsWith("/") && !href.startsWith("//")) {
    return (
      <Link to={href} className={LINK_CLASS}>
        {children}
      </Link>
    );
  }
  return (
    <a href={href} target="_blank" rel="noreferrer" className={LINK_CLASS}>
      {children}
    </a>
  );
}

/* What a paragraph is, read from its markdown node (not the rendered children,
 * whose elements are this screen's own components). */
type MdNode = { type: string; tagName?: string; value?: string; children?: MdNode[] };

/** Its children that are not blank text. */
function meaningful(node?: MdNode): MdNode[] {
  return (node?.children ?? []).filter((c) => !(c.type === "text" && !c.value?.trim()));
}

/** One italic run: the group's folded "nothing" line. */
function isQuietLine(node?: MdNode): boolean {
  const parts = meaningful(node);
  return parts.length === 1 && parts[0].type === "element" && parts[0].tagName === "em";
}

/** Starts with the ⚠ of a "couldn't build" line. */
function isFailureLine(node?: MdNode): boolean {
  const first = meaningful(node)[0];
  return first?.type === "text" && (first.value ?? "").trimStart().startsWith("⚠");
}

const digestComponents: Components = {
  ...markdownComponents,
  a: ({ href, children }) => <DigestLink href={href}>{children}</DigestLink>,
  // A group: a small uppercase label in the accent colour over a rule.
  h2: ({ children }) => (
    <h3
      data-testid="digest-group"
      className="mb-1.5 mt-5 border-b border-border pb-1 text-[12px] font-semibold uppercase tracking-[0.08em] text-primary first:mt-0"
    >
      {children}
    </h3>
  ),
  // A section inside a group.
  h3: ({ children }) => (
    <h4 data-testid="digest-section" className="mb-0.5 mt-2.5 text-[14px] font-semibold text-fg">
      {children}
    </h4>
  ),
  p: ({ node, children }) => {
    if (isQuietLine(node as MdNode | undefined)) {
      return (
        <p data-testid="digest-quiet" className="mb-2 text-[13px] text-fg-muted [&_em]:not-italic">
          {children}
        </p>
      );
    }
    if (isFailureLine(node as MdNode | undefined)) {
      return (
        <p
          data-testid="digest-failed"
          className="my-2 rounded-r border-l-[3px] border-warning bg-warning/10 px-2.5 py-1.5 text-[13px]"
        >
          {children}
        </p>
      );
    }
    return <p className="mb-2 last:mb-0">{children}</p>;
  },
  // The footer (learned yesterday) sits under a dashed rule, muted.
  hr: () => <hr className="mb-1.5 mt-3 border-dashed border-border" />,
};

function DigestBody({ digest }: { digest: StoredDigest }) {
  const failed = digest.failed_sections?.length ?? 0;
  return (
    <article className="rounded-lg border border-border bg-surface p-4" data-testid="digest-body">
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <h2 className="text-sm font-semibold text-fg">{digest.subject || "Digest"}</h2>
        {failed > 0 && (
          <Tag kind="warn">
            partial · {failed} section{failed === 1 ? "" : "s"} failed
          </Tag>
        )}
        <span className="ml-auto font-mono text-[11px] text-fg-subtle">
          {fmtDateTime(digest.created_at)}
        </span>
      </div>
      <div className="text-sm leading-relaxed [word-break:break-word] [&_hr~p]:text-[13px] [&_hr~p]:text-fg-muted">
        <ReactMarkdown components={digestComponents} urlTransform={urlTransform}>
          {digest.body}
        </ReactMarkdown>
      </div>
    </article>
  );
}

function History({ current }: { current?: string }) {
  const { data } = useQuery({ queryKey: ["digests"], queryFn: () => fetchDigestList() });
  if (!data || data.length < 2) return null;
  return (
    <Section title="Earlier">
      <ul className="space-y-1 text-[13px]">
        {data.map((d) => (
          <li key={d.id} className="flex min-h-11 flex-wrap items-center gap-x-2">
            {d.id === current ? (
              <span className="text-fg">{fmtDateTime(d.created_at)}</span>
            ) : (
              <Link to={`/digest/${d.id}`} className={LINK_CLASS}>
                {fmtDateTime(d.created_at)}
              </Link>
            )}
            <span className="text-fg-subtle">{d.subject}</span>
            {d.failed_sections > 0 && <Tag kind="warn">partial</Tag>}
          </li>
        ))}
      </ul>
    </Section>
  );
}

export function DigestScreen() {
  const { digestId } = useParams();
  const { data, isLoading, error } = useQuery({
    queryKey: ["digest", digestId ?? "latest"],
    queryFn: () => fetchDigest(digestId),
    retry: (count, err) => !(err instanceof DigestNotFound) && count < 2,
  });

  let content: ReactNode;
  if (isLoading) content = <p className="text-sm text-fg-muted">Loading…</p>;
  else if (error instanceof DigestNotFound)
    content = (
      <p className="text-sm text-fg-muted">
        {digestId
          ? "That digest is no longer stored. "
          : "No digest has been sent yet. The morning digest appears here once it runs. "}
        {digestId && (
          <Link to="/digest" className={LINK_CLASS}>
            Show the latest
          </Link>
        )}
      </p>
    );
  else if (error || !data) content = <ApiUnavailable />;
  else content = <DigestBody digest={data} />;

  return (
    <div className="space-y-6">
      <Section>{content}</Section>
      <History current={data?.id} />
    </div>
  );
}
