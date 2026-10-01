/* Removing from the Memory Map (ADR-0119, Map cleanup plan PR 4).
 *
 * The server is faked against the removal contract PR 3 serves: /memory/graph nodes
 * carry `ref` and `previously_removed`, and /memory/removed lists, previews, removes,
 * restores and deletes. The fake keeps state, so a removal really leaves the graph and
 * a restore really brings it back — the UI is tested against what it is told, not
 * against its own optimistic guess. Runs on the iPhone WebKit project at 390px. */
import { test, expect, type Page } from "@playwright/test";
import { fixtureFor } from "./fixtures";

interface Node {
  id: string;
  kind: string;
  label: string;
  confirmed: boolean;
  meta: Record<string, unknown>;
  ref?: { kind: string; id: string } | null;
  previously_removed?: boolean;
}

const NODES: Node[] = [
  { id: "you", kind: "you", label: "you", confirmed: true, meta: {} },
  {
    id: "ent-northwind",
    kind: "entity",
    label: "Northwind Bank",
    confirmed: true,
    meta: { class: "fin:FinancialInstitution" },
    ref: { kind: "entity", id: "ent-northwind" },
  },
  {
    id: "entity:walgreens",
    kind: "entity",
    label: "Walgreens",
    confirmed: true,
    meta: {},
    ref: { kind: "name", id: "Walgreens" },
  },
  {
    id: "entity:barclays",
    kind: "entity",
    label: "Barclays",
    confirmed: true,
    meta: {},
    ref: { kind: "entity", id: "ent-barclays" },
    previously_removed: true,
  },
  {
    id: "behavior:reminders",
    kind: "lesson",
    label: "reminders",
    confirmed: true,
    meta: { summary: "how to set reminders" },
  },
  {
    id: "pattern:digest",
    kind: "pattern",
    label: "asks for the digest before 9am",
    confirmed: true,
    meta: { file: "~/.iris/memory/episodic.md" },
  },
];

interface Removed {
  id: string;
  kind: string;
  label: string;
  removed_at: string;
  cascade: { statement_id: string; text: string }[];
  permanent: boolean;
  nodeId: string;
}

interface Fake {
  removed: Removed[];
  removeBodies: unknown[];
  deleteBodies: unknown[];
  writes: boolean;
}

const CASCADE: Record<string, { statement_id: string; text: string }[]> = {
  "ent-northwind": [{ statement_id: "st-1", text: "you bank with Northwind Bank" }],
};

async function fakeServer(page: Page, opts: { writes?: boolean } = {}): Promise<Fake> {
  const fake: Fake = { removed: [], removeBodies: [], deleteBodies: [], writes: opts.writes ?? true };
  const nodeFor = (t: { kind: string; id: string }) =>
    NODES.find((n) => n.ref?.kind === t.kind && n.ref?.id === t.id)!;

  await page.route("**/*", async (route) => {
    const req = route.request();
    const type = req.resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const { pathname } = new URL(req.url());
    const json = (body: unknown) =>
      route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
    const body = () => JSON.parse(req.postData() ?? "{}");

    if (pathname === "/capabilities") {
      return json({ ...(fixtureFor(pathname) as object), writes_enabled: fake.writes });
    }
    if (pathname === "/memory/graph") {
      const gone = new Set(fake.removed.map((r) => r.nodeId));
      const nodes = NODES.filter((n) => !gone.has(n.id));
      return json({
        nodes,
        edges: nodes
          .filter((n) => n.id !== "you")
          .map((n) => ({ id: `e-${n.id}`, source: "you", target: n.id, label: "" })),
        stats: { total_nodes: nodes.length, shown: nodes.length, by_kind: {} },
        focus: "you",
      });
    }
    if (pathname === "/memory/removed" && req.method() === "GET") {
      return json({ items: fake.removed });
    }
    if (pathname === "/memory/removed/preview") {
      const { targets } = body();
      return json({
        effects: targets.map((t: { kind: string; id: string }) => ({
          target: t,
          label: nodeFor(t).label,
          lines:
            t.kind === "entity"
              ? (CASCADE[t.id] ?? []).map((c) => `forgets 1 fact: ${c.text}`)
              : [`suppresses the name “${t.id.toLowerCase()}” in every summary`],
        })),
      });
    }
    if (pathname === "/memory/removed" && req.method() === "POST") {
      const b = body();
      fake.removeBodies.push(b);
      const items = b.targets.map((t: { kind: string; id: string }) => {
        const n = nodeFor(t);
        const item: Removed = {
          id: `rm-${n.id}`,
          kind: t.kind,
          label: n.label,
          removed_at: "2026-09-22T20:00:00Z",
          cascade: t.kind === "entity" ? (CASCADE[t.id] ?? []) : [],
          permanent: false,
          nodeId: n.id,
        };
        fake.removed.unshift(item);
        return item;
      });
      return json({ items });
    }
    const restore = pathname.match(/^\/memory\/removed\/([^/]+)\/restore$/);
    if (restore) {
      const id = decodeURIComponent(restore[1]);
      const item = fake.removed.find((r) => r.id === id)!;
      fake.removed = fake.removed.filter((r) => r.id !== id);
      return json({ restored: item });
    }
    if (pathname === "/memory/removed/delete") {
      const b = body();
      fake.deleteBodies.push(b);
      const deleted: string[] = [];
      const refused: { id: string; reason: string }[] = [];
      for (const id of b.ids as string[]) {
        const item = fake.removed.find((r) => r.id === id)!;
        if (item.kind === "entity") {
          refused.push({ id, reason: "a statement about it still holds" });
        } else {
          item.permanent = true;
          deleted.push(id);
        }
      }
      return json({ deleted, refused });
    }
    return json(fixtureFor(pathname));
  });
  return fake;
}

async function openMap(page: Page): Promise<void> {
  await page.goto("/memory");
  await page.getByRole("button", { name: "Map", exact: true }).click();
  await expect(page.locator(".react-flow__node", { hasText: "Northwind Bank" })).toBeVisible();
}

const node = (page: Page, label: string) => page.locator(".react-flow__node", { hasText: label });

test("removing an entity lists the fact that goes with it, and restore brings it back", async ({
  page,
}) => {
  const fake = await fakeServer(page);
  await openMap(page);

  await node(page, "Northwind Bank").click();
  await page.getByRole("button", { name: "Remove from memory…" }).click();

  const dialog = page.getByRole("dialog");
  await expect(dialog.getByText("Remove “Northwind Bank”?")).toBeVisible();
  await expect(dialog.getByText("forgets 1 fact: you bank with Northwind Bank")).toBeVisible();
  await dialog.getByRole("button", { name: "Remove", exact: true }).click();

  await expect(node(page, "Northwind Bank")).toHaveCount(0);
  expect(fake.removeBodies).toEqual([{ targets: [{ kind: "entity", id: "ent-northwind" }] }]);

  await page.getByRole("button", { name: "Removed", exact: true }).click();
  const list = page.getByTestId("removed-list");
  await expect(list.getByText("Northwind Bank", { exact: true })).toBeVisible();
  await expect(list.getByText("you bank with Northwind Bank")).toBeVisible();
  await list.getByRole("button", { name: "Restore" }).click();
  await expect(page.getByText("Nothing removed.")).toBeVisible();

  await page.getByRole("button", { name: "Map", exact: true }).click();
  await expect(node(page, "Northwind Bank")).toBeVisible();
});

test("shift-click picks several nodes for one confirmation; lessons cannot be picked", async ({
  page,
}) => {
  const fake = await fakeServer(page);
  await openMap(page);

  await node(page, "Northwind Bank").click({ modifiers: ["Shift"] });
  await node(page, "Walgreens").click({ modifiers: ["Shift"] });
  await node(page, "reminders").click({ modifiers: ["Shift"] });

  const bar = page.getByTestId("map-selection");
  await expect(bar).toContainText("2 selected");
  await bar.getByRole("button", { name: "Remove selected…" }).click();

  const dialog = page.getByRole("dialog");
  await expect(dialog.getByText("Remove 2 items?")).toBeVisible();
  await expect(dialog.getByText(/suppresses the name “walgreens”/)).toBeVisible();
  await dialog.getByRole("button", { name: "Remove", exact: true }).click();

  await expect(node(page, "Walgreens")).toHaveCount(0);
  await expect(node(page, "Northwind Bank")).toHaveCount(0);
  await expect(bar).toHaveCount(0);
  expect(fake.removeBodies).toEqual([
    {
      targets: [
        { kind: "entity", id: "ent-northwind" },
        { kind: "name", id: "Walgreens" },
      ],
    },
  ]);
});

test("lessons and patterns have no Remove and name their file; a previously removed node says so", async ({ page }) => {
  await fakeServer(page);
  await openMap(page);

  await node(page, "reminders").click();
  await expect(page.getByText(/Lessons are files you curate/)).toContainText("edit the file");
  await expect(page.getByRole("button", { name: "Remove from memory…" })).toHaveCount(0);

  // With `meta.file` (PR 3 adds it) the note names the file instead.
  await node(page, "asks for the digest").click();
  const note = page.getByText(/Patterns are files you curate/);
  await expect(note.locator("code")).toHaveText("~/.iris/memory/episodic.md");
  await expect(page.getByRole("button", { name: "Remove from memory…" })).toHaveCount(0);

  await node(page, "Barclays").click();
  await expect(page.getByText(/Previously removed — a confirmed fact named it again/)).toBeVisible();
});

test("delete permanently needs the typed word, and a refusal is shown with its reason", async ({
  page,
}) => {
  const fake = await fakeServer(page);
  await openMap(page);
  await node(page, "Northwind Bank").click({ modifiers: ["Shift"] });
  await node(page, "Walgreens").click({ modifiers: ["Shift"] });
  await page.getByRole("button", { name: "Remove selected…" }).click();
  await page.getByRole("dialog").getByRole("button", { name: "Remove", exact: true }).click();
  await expect(node(page, "Walgreens")).toHaveCount(0);

  await page.getByRole("button", { name: "Removed", exact: true }).click();
  await page.getByLabel("Select Northwind Bank").check();
  await page.getByLabel("Select Walgreens").check();
  await page.getByRole("button", { name: "Delete permanently…" }).click();

  const dialog = page.getByRole("dialog");
  const go = dialog.getByRole("button", { name: "Delete permanently" });
  await expect(go).toBeDisabled();
  await dialog.getByLabel("Type delete to confirm").fill("delete");
  await go.click();

  await expect.poll(() => fake.deleteBodies.length).toBe(1);
  const sent = fake.deleteBodies[0] as { ids: string[]; confirm: string };
  expect(sent.confirm).toBe("delete");
  expect([...sent.ids].sort()).toEqual(["rm-ent-northwind", "rm-entity:walgreens"]);
  await expect(page.getByText(/Not deleted — Northwind Bank: a statement about it still holds/)).toBeVisible();
  await expect(page.getByText("deleted · stays suppressed")).toBeVisible();
  await expect(page.getByLabel("Select Walgreens")).toBeDisabled();
});

test("with writes off there is no Remove, only the read-only note", async ({ page }) => {
  await fakeServer(page, { writes: false });
  await openMap(page);
  await node(page, "Northwind Bank").click();
  await expect(page.getByRole("button", { name: "Remove from memory…" })).toHaveCount(0);
  await expect(page.getByText(/IRIS_WEBUI_ALLOW_WRITES=1/)).toBeVisible();
});
