/* New-sender cards in the Action Center (ADR-0121 PR 3, prototype signed off 2026-09-24).
 *
 * The server is faked against the contract the API serves: GET /actions carries choice
 * cards (choices, options, card), POST /actions/{id}/invoke takes {choice, option}, and
 * /finance/senders lists what was not proposed and opens a card on "Add anyway". The fake
 * keeps state, so an answered card really leaves and an added sender really appears.
 * Runs on the iPhone WebKit project at 390px. */
import { test, expect, type Page } from "@playwright/test";
import { fixtureFor } from "./fixtures";

const TYPES = [
  { value: "bank", label: "Bank" },
  { value: "card", label: "Card" },
  { value: "insurance", label: "Insurance" },
];

function senderCard(id: string, name: string, domain: string, kind: string | null) {
  return {
    id,
    origin: "task",
    source_kind: "finance-statements",
    title: `Is ${name} one of your accounts or bills?`,
    description: "statement mail for the user's account",
    created_at: "2026-09-24T12:00:00Z",
    feedback_ref: null,
    action: {
      kind: "execute",
      label: "Answer",
      command: null,
      target_id: `sender:${domain}`,
      safe: true,
      choices: [
        { value: "confirm", label: "Yes, it's mine", primary: true, needs_option: true },
        { value: "ignore", label: "Ignore", primary: false, needs_option: false },
      ],
      options: { name: "type", label: "Type", values: TYPES, default: kind },
      card: {
        tag: "new sender",
        facts: [
          { label: "IRIS read", value: name },
          { label: "Country", value: "IN · INR" },
        ],
        evidence: [{ when: "2026-09-20", text: "E-account statement for your account(s)." }],
        evidence_label: "Why IRIS asks",
        note: `If it's yours, IRIS trusts ${domain} and reads its 6 email(s).`,
      },
    },
  };
}

interface Fake {
  cards: ReturnType<typeof senderCard>[];
  answers: { id: string; body: unknown }[];
  skipped: { domain: string; status: string; verdict: string; reason: string; email_count: number; latest_at: string }[];
}

async function fakeServer(page: Page): Promise<Fake> {
  const fake: Fake = {
    cards: [
      senderCard("t-woodgrove", "Woodgrove Bank", "woodgrovebank.test", "bank"),
      senderCard("t-eq", "Equitas Small Finance Bank", "equitas.bank.in", null),
    ],
    answers: [],
    skipped: [
      {
        domain: "alerts.fabrikambank.test",
        status: "not_proposed",
        verdict: "marketing",
        reason: "promotional offers and travel perks",
        email_count: 41,
        latest_at: "2026-09-24T00:00:00Z",
      },
    ],
  };
  await page.route("**/*", async (route) => {
    const req = route.request();
    const type = req.resourceType();
    if (type !== "fetch" && type !== "xhr") return route.continue();
    const { pathname } = new URL(req.url());
    const json = (body: unknown, status = 200) =>
      route.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });

    if (pathname === "/capabilities") {
      return json({ ...(fixtureFor(pathname) as object), writes_enabled: true });
    }
    if (pathname === "/actions") return json({ count: fake.cards.length, actions: fake.cards });
    if (pathname === "/governance/approvals") return json({ approvals: [] });
    const invoke = pathname.match(/^\/actions\/([^/]+)\/invoke$/);
    if (invoke) {
      const id = decodeURIComponent(invoke[1]);
      const body = JSON.parse(req.postData() ?? "{}");
      fake.answers.push({ id, body });
      const card = fake.cards.find((c) => c.id === id)!;
      fake.cards = fake.cards.filter((c) => c.id !== id);
      if (body.choice === "ignore") {
        const domain = card.action.target_id.slice("sender:".length);
        fake.skipped.unshift({ ...fake.skipped[0], domain, status: "ignored", reason: "" });
        return json({ id, result: `Ignored ${domain}.` });
      }
      return json({ id, result: `Added ${card.action.card.facts[0].value} as your ${body.option} account.` });
    }
    if (pathname === "/finance/senders") {
      return json({ open: fake.cards.length, waiting: 2, not_proposed: fake.skipped });
    }
    const propose = pathname.match(/^\/finance\/senders\/([^/]+)\/propose$/);
    if (propose) {
      const domain = decodeURIComponent(propose[1]);
      fake.skipped = fake.skipped.filter((s) => s.domain !== domain);
      fake.cards.push(senderCard(`t-${domain}`, "Fabrikam Meridian Bank", domain, null));
      return json({ domain, result: `${domain} is up for your answer in the Action Center.` });
    }
    return json(fixtureFor(pathname));
  });
  return fake;
}

const card = (page: Page, name: string) =>
  page.locator("div.rounded-xl", { hasText: `Is ${name} one of your accounts or bills?` });

test("a sender card shows what IRIS read and answers with the picked type", async ({ page }) => {
  const fake = await fakeServer(page);
  await page.goto("/actions");

  const woodgrove = card(page, "Woodgrove Bank");
  await expect(woodgrove.getByText("new sender")).toBeVisible();
  await expect(woodgrove.getByText("IN · INR")).toBeVisible();
  await expect(woodgrove.getByText("E-account statement for your account(s).")).toBeVisible();
  await expect(woodgrove.getByRole("button", { name: "Bank" })).toHaveAttribute("aria-pressed", "true");

  // The owner changes IRIS's guess before confirming.
  await woodgrove.getByRole("button", { name: "Card" }).click();
  await woodgrove.getByRole("button", { name: "Yes, it's mine" }).click();

  await expect(page.getByText("Added Woodgrove Bank as your card account.")).toBeVisible();
  await expect(card(page, "Woodgrove Bank")).toHaveCount(0);
  expect(fake.answers).toEqual([{ id: "t-woodgrove", body: { choice: "confirm", option: "card" } }]);
});

test("a card with no guessed type cannot be confirmed until one is picked", async ({ page }) => {
  const fake = await fakeServer(page);
  await page.goto("/actions");

  const equitas = card(page, "Equitas Small Finance Bank");
  await expect(equitas.getByRole("button", { name: "Yes, it's mine" })).toBeDisabled();
  await equitas.getByRole("button", { name: "Bank" }).click();
  await equitas.getByRole("button", { name: "Yes, it's mine" }).click();
  expect(fake.answers[0]).toEqual({ id: "t-eq", body: { choice: "confirm", option: "bank" } });
});

test("ignore moves the sender to the list, where it can be undone", async ({ page }) => {
  const fake = await fakeServer(page);
  await page.goto("/actions");

  await card(page, "Woodgrove Bank").getByRole("button", { name: "Ignore" }).click();
  expect(fake.answers[0]).toEqual({ id: "t-woodgrove", body: { choice: "ignore", option: null } });

  await page.getByText("Senders IRIS saw in your mail but didn't ask about").click();
  const row = page.locator("details li", { hasText: "woodgrovebank.test" });
  await expect(row.getByText("you chose Ignore")).toBeVisible();
  await expect(row.getByRole("button", { name: "Undo ignore" })).toBeVisible();
});

test("add anyway opens a card for a sender IRIS skipped", async ({ page }) => {
  await fakeServer(page);
  await page.goto("/actions");

  await page.getByText("Senders IRIS saw in your mail but didn't ask about").click();
  const fabrikam = page.locator("details li", { hasText: "alerts.fabrikambank.test" });
  await expect(fabrikam.getByText("41 emails · promotional offers and travel perks")).toBeVisible();
  await fabrikam.getByRole("button", { name: "Add anyway" }).click();

  await expect(card(page, "Fabrikam Meridian Bank")).toBeVisible();
  // The list empties (the toast is an <li> too, so look inside the list only).
  await expect(page.locator("details li", { hasText: "alerts.fabrikambank.test" })).toHaveCount(0);
});

test("the card fits the phone", async ({ page }) => {
  await fakeServer(page);
  await page.goto("/actions");
  await expect(card(page, "Woodgrove Bank")).toBeVisible();
  const overflow = await page.evaluate(() => document.documentElement.scrollWidth - window.innerWidth);
  expect(overflow).toBeLessThanOrEqual(0);
});
