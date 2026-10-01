import { useState } from "react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/Switch";
import { Tag } from "@/components/Tag";
import { Section } from "@/components/layout";
import { QueryState } from "@/components/control/parts";
import {
  useDigestConfig,
  usePatchDigestConfig,
  useResetDigestConfig,
} from "@/lib/queries";
import type { DigestConfigView, DigestFields } from "@/lib/control";

/** "bills_due" -> "Bills due": the section's tool-slot name, readable. */
function sectionLabel(name: string): string {
  const words = name.replace(/_/g, " ");
  return words.charAt(0).toUpperCase() + words.slice(1);
}

/** A section's name as the owner reads it: its own heading where it has one (a news
 * group: "News: Local — St. Louis"), else its slot name made readable. */
function sectionName(view: DigestConfigView, name: string): string {
  const title = view.section_titles?.[name];
  return title ? `News: ${title}` : sectionLabel(name);
}

function useSave() {
  const patch = usePatchDigestConfig();
  const save = async (changes: Partial<DigestFields>, done: string) => {
    try {
      await patch.mutateAsync(changes);
      toast.success(`${done} · tomorrow's digest uses this`);
      return true;
    } catch (e) {
      toast.error(e instanceof Error ? e.message : "save failed");
      return false;
    }
  };
  return { save, pending: patch.isPending };
}

function Changed({ on }: { on: boolean }) {
  return on ? <Tag kind="info">changed</Tag> : null;
}

function DeliveryCard({ view, canWrite }: { view: DigestConfigView; canWrite: boolean }) {
  const { save, pending } = useSave();
  const f = view.fields;
  const [time, setTime] = useState<string | null>(null);
  const channels = ["all", "telegram", "web"];
  if (!channels.includes(f.channel)) channels.push(f.channel);
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2 rounded-lg border border-border bg-surface p-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 text-sm text-fg">
            Send the morning digest <Changed on={view.changed.includes("enabled")} />
          </div>
          <p className="text-xs text-fg-muted">Off: no scheduled digest; ask for a brief any time.</p>
        </div>
        <Switch
          checked={f.enabled}
          label="Send the morning digest"
          disabled={!canWrite || pending}
          onChange={(next) => void save({ enabled: next }, `Digest ${next ? "on" : "off"}`)}
        />
      </div>
      <div className="rounded-lg border border-border bg-surface p-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm text-fg">Delivery time</span>
          <Changed on={view.changed.includes("time")} />
          <span className="ml-auto font-mono text-[11px] text-fg-subtle">
            {view.timezone} (IRIS_TZ)
          </span>
        </div>
        <p className="mt-1 text-xs text-fg-muted">
          When the digest arrives; IRIS starts building it a few minutes before. While the email
          judge is on, it first sweeps and judges new mail, so the digest shows the inbox as of now.
        </p>
        <form
          className="mt-2 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            if (time === null) return;
            void save({ time }, `Digest at ${time}`).then((ok) => ok && setTime(null));
          }}
        >
          <Input
            type="time"
            aria-label="Delivery time"
            value={time ?? f.time}
            disabled={!canWrite}
            onChange={(e) => setTime(e.target.value)}
            className="h-11 w-32 text-xs sm:h-8"
          />
          {canWrite && (
            <Button type="submit" size="sm" variant="outline" disabled={time === null || pending}>
              Save
            </Button>
          )}
        </form>
      </div>
      <div className="rounded-lg border border-border bg-surface p-3">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm text-fg">Channel</span>
          <Changed on={view.changed.includes("channel")} />
        </div>
        <p className="mt-1 text-xs text-fg-muted">
          All: full on the web (kept to re-read), chunked on Telegram, a headline push.
        </p>
        <select
          aria-label="Digest channel"
          className="mt-2 h-11 rounded-md border border-border bg-bg px-2 text-xs text-fg sm:h-8"
          value={f.channel}
          disabled={!canWrite || pending}
          onChange={(e) => void save({ channel: e.target.value }, `Channel: ${e.target.value}`)}
        >
          {channels.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
      </div>
    </div>
  );
}

function LineCap({
  name,
  view,
  canWrite,
  onClose,
}: {
  name: string;
  view: DigestConfigView;
  canWrite: boolean;
  onClose: () => void;
}) {
  const { save, pending } = useSave();
  const config = view.fields.section_config;
  const cap = config[name]?.line_cap;
  const [draft, setDraft] = useState(cap === undefined ? "" : String(cap));
  const write = async (value: number | null) => {
    const knobs = { ...(config[name] ?? {}) };
    if (value === null) delete knobs.line_cap;
    else knobs.line_cap = value;
    const next = { ...config, [name]: knobs };
    // An empty mapping clears the file's knobs for a section the file sets; for one it
    // does not, the section is simply left out.
    if (Object.keys(knobs).length === 0 && !view.file.section_config[name]) delete next[name];
    const label = sectionName(view, name);
    const done = value === null ? `${label}: every line` : `${label}: up to ${value} lines`;
    if (await save({ section_config: next }, done)) onClose();
  };
  return (
    <form
      className="flex flex-wrap items-center gap-2 pb-2 pl-1 text-xs text-fg-muted"
      onSubmit={(e) => {
        e.preventDefault();
        void write(draft.trim() === "" ? null : Number(draft));
      }}
    >
      <span>Show at most</span>
      <Input
        type="number"
        aria-label={`Lines for ${sectionName(view, name)}`}
        min={1}
        max={50}
        placeholder="all"
        value={draft}
        disabled={!canWrite}
        onChange={(e) => setDraft(e.target.value)}
        className="h-11 w-20 text-xs sm:h-8"
      />
      <span>lines</span>
      {canWrite && (
        <Button type="submit" size="sm" variant="outline" disabled={pending}>
          Save
        </Button>
      )}
    </form>
  );
}

function SectionsCard({ view, canWrite }: { view: DigestConfigView; canWrite: boolean }) {
  const { save, pending } = useSave();
  const [editing, setEditing] = useState<string | null>(null);
  const locked = view.locked_sections;
  const on = new Set(view.fields.sections);
  const order = view.all_sections.filter((s) => !locked.includes(s));
  const commit = (next: string[], nextOn: Set<string>, done: string) =>
    void save(
      {
        sections: [...next.filter((s) => nextOn.has(s)), ...locked],
        sections_off: next.filter((s) => !nextOn.has(s)),
      },
      done,
    );
  const move = (i: number, by: -1 | 1) => {
    const next = [...order];
    [next[i], next[i + by]] = [next[i + by], next[i]];
    commit(next, on, `Moved ${sectionName(view, order[i])}`);
  };
  const toggle = (name: string, value: boolean) => {
    const nextOn = new Set(on);
    if (value) nextOn.add(name);
    else nextOn.delete(name);
    commit(order, nextOn, `${sectionName(view, name)} ${value ? "on" : "off"}`);
  };
  const groups = view.groups ?? [];
  const groupOf = (name: string) => groups.find((g) => g.sections.includes(name));
  const changed =
    view.changed.includes("sections") ||
    view.changed.includes("sections_off") ||
    view.changed.includes("section_config");
  const capText = (name: string) => {
    const cap = view.fields.section_config[name]?.line_cap;
    return cap === undefined ? "all lines" : `≤ ${String(cap)} lines`;
  };
  const row = (name: string, i: number | null) => (
    <li key={name} className="py-1.5">
      <div className="flex items-center gap-2 text-sm">
        <div className="min-w-0 flex-1">
          <div className={`truncate ${on.has(name) ? "text-fg" : "text-fg-subtle line-through"}`}>
            {groupOf(name)?.icon && (
              <span aria-hidden="true" title={groupOf(name)?.title} className="mr-1">
                {groupOf(name)?.icon}
              </span>
            )}
            {sectionName(view, name)} {i === null && <Tag kind="info">always last</Tag>}
          </div>
          <button
            type="button"
            className="-my-3 inline-flex min-h-[44px] items-center text-xs text-fg-muted underline-offset-2 hover:underline sm:my-0 sm:min-h-0"
            aria-label={`Lines for ${sectionName(view, name)}: ${capText(name)}`}
            onClick={() => setEditing(editing === name ? null : name)}
          >
            {capText(name)}
          </button>
        </div>
        {canWrite && i !== null && (
          <>
            <Button
              type="button"
              size="sm"
              variant="ghost"
              aria-label={`Move ${sectionName(view, name)} up`}
              disabled={i === 0 || pending}
              onClick={() => move(i, -1)}
            >
              ▲
            </Button>
            <Button
              type="button"
              size="sm"
              variant="ghost"
              aria-label={`Move ${sectionName(view, name)} down`}
              disabled={i === order.length - 1 || pending}
              onClick={() => move(i, 1)}
            >
              ▼
            </Button>
          </>
        )}
        <Switch
          checked={i === null || on.has(name)}
          label={sectionName(view, name)}
          disabled={i === null || !canWrite || pending}
          onChange={(next) => toggle(name, next)}
        />
      </div>
      {editing === name && (
        <LineCap name={name} view={view} canWrite={canWrite} onClose={() => setEditing(null)} />
      )}
    </li>
  );
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-fg">Sections</span>
        <Changed on={changed} />
      </div>
      <p className="mt-1 text-xs text-fg-muted">
        In the order they appear within their group. Chat (&quot;add dues to my digest&quot;)
        changes the same list.
      </p>
      {groups.length > 0 && (
        <p className="mt-1 text-xs text-fg-subtle" aria-label="Digest groups">
          Groups (digest.yaml):{" "}
          {groups.map((g) => `${g.icon} ${g.title}`.trim()).join(" · ")} · More
        </p>
      )}
      <ol className="mt-2 divide-y divide-border">
        {order.map((name, i) => row(name, i))}
        {locked.map((name) => row(name, null))}
      </ol>
    </div>
  );
}

function ChipList({
  field,
  title,
  help,
  placeholder,
  view,
  canWrite,
  mark,
}: {
  field: "news_sources" | "focus_categories";
  title: string;
  help: string;
  placeholder: string;
  view: DigestConfigView;
  canWrite: boolean;
  mark?: string;
}) {
  const { save, pending } = useSave();
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-fg">{title}</span>
        <Changed on={view.changed.includes(field)} />
      </div>
      <p className="mt-1 text-xs text-fg-muted">{help}</p>
      <Chips
        title={title}
        values={view.fields[field]}
        placeholder={placeholder}
        canWrite={canWrite}
        pending={pending}
        mark={mark}
        onChange={(next, done) => save({ [field]: next }, done)}
      />
    </div>
  );
}

/** Values as removable chips, and a box to add one; `onChange` saves the new list. */
function Chips({
  title,
  values,
  placeholder,
  canWrite,
  pending,
  mark,
  onChange,
}: {
  title: string;
  values: string[];
  placeholder: string;
  canWrite: boolean;
  pending: boolean;
  mark?: string;
  onChange: (next: string[], done: string) => Promise<boolean>;
}) {
  const [draft, setDraft] = useState("");
  const add = async () => {
    const value = draft.trim();
    if (!value) return;
    if (await onChange([...values, value], `${title}: added ${value}`)) setDraft("");
  };
  return (
    <>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {values.length === 0 && <span className="text-xs text-fg-subtle">None.</span>}
        {values.map((v) => (
          <span
            key={v}
            className="inline-flex items-center gap-1 rounded-full border border-border bg-bg px-2.5 py-0.5 text-xs text-fg"
          >
            {mark && `${mark} `}
            {v}
            {canWrite && (
              <button
                type="button"
                aria-label={`Remove ${v} from ${title}`}
                className="-my-3 -mr-2 inline-flex min-h-[44px] min-w-[44px] items-center justify-center text-fg-subtle hover:text-fg sm:my-0 sm:mr-0 sm:min-h-0 sm:min-w-0"
                disabled={pending}
                onClick={() =>
                  void onChange(
                    values.filter((x) => x !== v),
                    `${title}: removed ${v}`,
                  )
                }
              >
                ✕
              </button>
            )}
          </span>
        ))}
      </div>
      {canWrite && (
        <form
          className="mt-2 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            void add();
          }}
        >
          <Input
            aria-label={`Add to ${title}`}
            placeholder={placeholder}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            className="h-11 text-xs sm:h-8"
          />
          <Button type="submit" size="sm" variant="outline" disabled={!draft.trim() || pending}>
            Add
          </Button>
        </form>
      )}
    </>
  );
}

/** Each news group's topics (digest.yaml news_groups): one research query per topic. */
function NewsGroupsCard({ view, canWrite }: { view: DigestConfigView; canWrite: boolean }) {
  const { save, pending } = useSave();
  const groups = view.fields.news_groups ?? {};
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-fg">News topics</span>
        <Changed on={view.changed.includes("news_groups")} />
      </div>
      <p className="mt-1 text-xs text-fg-muted">
        Each topic becomes one research query in its news section. {"{news_local_area}"} is the
        local area below.
      </p>
      <div className="mt-2 space-y-3">
        {Object.entries(groups).map(([slot, group]) => {
          const title = view.section_titles?.[slot] ?? sectionLabel(slot);
          return (
            <div key={slot}>
              <div className="text-xs font-medium text-fg">{title}</div>
              <Chips
                title={`${title} topics`}
                values={group.topics}
                placeholder="Add a topic (e.g. India markets)"
                canWrite={canWrite}
                pending={pending}
                onChange={(next, done) => save({ news_groups: { [slot]: { ...group, topics: next } } }, done)}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function NewsLocalAreaRow({ view, canWrite }: { view: DigestConfigView; canWrite: boolean }) {
  const { save, pending } = useSave();
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-fg">Local news area</span>
        <Changed on={view.changed.includes("news_local_area")} />
        <span className="ml-auto font-mono text-[11px] text-fg-subtle">
          file: {view.file.news_local_area}
        </span>
      </div>
      <p className="mt-1 text-xs text-fg-muted">The city or area the local news is about.</p>
      <form
        className="mt-2 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (draft === null) return;
          const next = draft.trim();
          void save({ news_local_area: next }, `Local news: ${next}`).then(
            (ok) => ok && setDraft(null),
          );
        }}
      >
        <Input
          aria-label="Local news area"
          maxLength={80}
          value={draft ?? view.fields.news_local_area}
          disabled={!canWrite}
          onChange={(e) => setDraft(e.target.value)}
          className="h-11 text-xs sm:h-8"
        />
        {canWrite && (
          <Button type="submit" size="sm" variant="outline" disabled={draft === null || pending}>
            Save
          </Button>
        )}
      </form>
    </div>
  );
}

function NewsLanguageRow({ view, canWrite }: { view: DigestConfigView; canWrite: boolean }) {
  const { save, pending } = useSave();
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-fg">News language</span>
        <Changed on={view.changed.includes("news_language")} />
        <span className="ml-auto font-mono text-[11px] text-fg-subtle">
          file: {view.file.news_language}
        </span>
      </div>
      <p className="mt-1 text-xs text-fg-muted">
        Two-letter code (en, de, ja) or &quot;any&quot;. Headlines in another script are left out.
      </p>
      <form
        className="mt-2 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (draft === null) return;
          const next = draft.trim().toLowerCase();
          void save({ news_language: next }, `News in: ${next}`).then(
            (ok) => ok && setDraft(null),
          );
        }}
      >
        <Input
          aria-label="News language"
          maxLength={3}
          autoCapitalize="none"
          value={draft ?? view.fields.news_language}
          disabled={!canWrite}
          onChange={(e) => setDraft(e.target.value)}
          className="h-11 w-24 text-xs sm:h-8"
        />
        {canWrite && (
          <Button type="submit" size="sm" variant="outline" disabled={draft === null || pending}>
            Save
          </Button>
        )}
      </form>
    </div>
  );
}

/** A Focus count (Settings → Digest): `focus_limit` in all, `focus_per_account` per inbox. */
function FocusNumberRow({
  view,
  canWrite,
  field,
  label,
  saved,
}: {
  view: DigestConfigView;
  canWrite: boolean;
  field: "focus_limit" | "focus_per_account";
  label: string;
  saved: (n: string) => string;
}) {
  const { save, pending } = useSave();
  const [draft, setDraft] = useState<string | null>(null);
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-sm text-fg">{label}</span>
        <Changed on={view.changed.includes(field)} />
        <span className="ml-auto font-mono text-[11px] text-fg-subtle">
          file: {view.file[field]}
        </span>
      </div>
      <form
        className="mt-2 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          if (draft === null) return;
          void save({ [field]: Number(draft) }, saved(draft)).then((ok) => ok && setDraft(null));
        }}
      >
        <Input
          type="number"
          aria-label={label}
          min={1}
          max={20}
          value={draft ?? String(view.fields[field])}
          disabled={!canWrite}
          onChange={(e) => setDraft(e.target.value)}
          className="h-11 w-24 text-xs sm:h-8"
        />
        {canWrite && (
          <Button type="submit" size="sm" variant="outline" disabled={draft === null || pending}>
            Save
          </Button>
        )}
      </form>
    </div>
  );
}

/** Settings → Digest: config/digest.yaml, editable (ADR-0120, loop-proof plan D4). */
export function DigestPanel({ canWrite }: { canWrite: boolean }) {
  const { data, isLoading, isError } = useDigestConfig();
  const reset = useResetDigestConfig();
  const migration = data?.migration;
  return (
    <Section title="Morning digest">
      <p className="-mt-2 mb-3 text-xs text-fg-subtle">
        Applies to the next digest; nothing to restart or re-approve.
        {migration?.status === "migrated" &&
          ` Your brief preferences moved here on ${migration.at.slice(0, 10)}` +
            (migration.dropped.length ? ` (dropped: ${migration.dropped.join(", ")}).` : ".")}
      </p>
      <QueryState loading={isLoading} error={isError} empty={!data} emptyText="—">
        {data && (
          <div className="space-y-4">
            <DeliveryCard view={data} canWrite={canWrite} />
            <SectionsCard view={data} canWrite={canWrite} />
            <NewsGroupsCard view={data} canWrite={canWrite} />
            <NewsLocalAreaRow view={data} canWrite={canWrite} />
            <ChipList
              field="news_sources"
              title="News sources"
              help="Preferred, never strict, for every news section: these rank first when they have a story; the best others still show."
              placeholder="Add a domain (e.g. bbc.com)"
              view={data}
              canWrite={canWrite}
              mark="★"
            />
            <NewsLanguageRow view={data} canWrite={canWrite} />
            <ChipList
              field="focus_categories"
              title="Focus categories"
              help="The newest emails in these triage categories (a category includes everything under it)."
              placeholder="Add a category path (e.g. email/family)"
              view={data}
              canWrite={canWrite}
            />
            <FocusNumberRow
              view={data}
              canWrite={canWrite}
              field="focus_limit"
              label="Focus: how many emails"
              saved={(n) => `Focus: newest ${n}`}
            />
            <FocusNumberRow
              view={data}
              canWrite={canWrite}
              field="focus_per_account"
              label="Focus: how many per inbox"
              saved={(n) => `Focus: newest ${n} per inbox`}
            />
            {canWrite && data.changed.length > 0 && (
              <Button
                type="button"
                size="sm"
                variant="ghost"
                onClick={() =>
                  reset.mutateAsync(undefined).then(() => toast.success("Back to digest.yaml"))
                }
              >
                Reset all to the file
              </Button>
            )}
          </div>
        )}
      </QueryState>
    </Section>
  );
}
