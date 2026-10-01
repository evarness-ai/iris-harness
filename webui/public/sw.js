/* IRIS service worker.
 *
 * It exists so the console can be added to the Home Screen and, from Track 2b
 * PR 9, receive Web Push — iOS only delivers push to a home-screen web app, and
 * only through a service worker.
 *
 * It deliberately does NOT cache the app shell. A cache-first shell would
 * eventually serve an index.html pointing at content-hashed assets that no
 * longer exist, which is the exact failure `static_ui.py` keeps the shell on
 * `no-cache` to avoid. The only cached thing is the offline page.
 *
 * So: navigations go to the network, and fall back to the offline page when
 * the tailnet is unreachable. Everything else is untouched — no interception,
 * no stale data, no surprises in a request path that carries a bearer cookie.
 *
 * PR 9 added `push` and `notificationclick` below. */

const CACHE = "iris-offline-v1";
const OFFLINE_URL = "/offline.html";

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE)
      .then((cache) => cache.add(new Request(OFFLINE_URL, { cache: "reload" })))
      // Take over without waiting for every old tab to close: there is no
      // shared cached state to corrupt, so an older worker has nothing to
      // hand over carefully.
      .then(() => self.skipWaiting()),
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim()),
  );
});

self.addEventListener("fetch", (event) => {
  // Only navigations. Data calls carry credentials and must never be served
  // from a cache, and assets are immutable and already cached by the browser.
  if (event.request.mode !== "navigate") return;
  event.respondWith(
    fetch(event.request).catch(async () => {
      const cache = await caches.open(CACHE);
      const offline = await cache.match(OFFLINE_URL);
      return (
        offline ??
        new Response("Can't reach IRIS.", {
          status: 503,
          headers: { "content-type": "text/plain; charset=utf-8" },
        })
      );
    }),
  );
});


/* ── Web Push (Track 2b PR 9; reminder buttons loop-proof PR 3b) ─────────────
 *
 * On iOS this is the only way a notification arrives: no APNs, no app store,
 * no Apple Developer Program — but the app must have been added to the Home
 * Screen, and the push must be delivered to a worker. This is that worker.
 *
 * The payload is decrypted by the browser before it gets here, so it is the
 * plaintext JSON the connector sent: { title, body, url, data, tag?, renotify?,
 * actions? }. A reminder's `data` carries `reminder_id`, and its `actions` are
 * Done and Snooze 1h. Chrome, Android and desktop show those as buttons; iOS
 * shows no buttons, and a tap opens the reminder sheet at `data.url` instead. */

/** The reminder API each notification button answers with. A bill's reminder
 * (loop-proof PR 4) carries Paid (its Done) and Tomorrow, 1 hour or — on a "Did you
 * pay?" question — Not yet, which the server records as seen, not snoozed. */
const REMINDER_ACTIONS = {
  done: { path: "done", body: { source: "push" } },
  snooze_1h: { path: "snooze", body: { for: "1h", source: "push" } },
  paid: { path: "done", body: { source: "push" } },
  not_yet: { path: "snooze", body: { for: "not_yet", source: "push" } },
  snooze_tomorrow: { path: "snooze", body: { for: "tomorrow_9am", source: "push" } },
};

/** What the notification says once a button answered: the action decides it. */
function confirmation(action, answer) {
  const reminder = answer.reminder || {};
  const bill = reminder.bill || null;
  if (action === "done") return { title: "Done ✓", body: reminder.text || "" };
  if (action === "paid") {
    return { title: "Paid ✓", body: bill ? `${bill.entity}: no more reminders for this bill.` : "" };
  }
  if (action === "not_yet") {
    const next = answer.next && answer.next.remind_at_local;
    return {
      title: "Noted — not paid yet",
      body: next
        ? `I'll ask again ${next}.`
        : "That was the last ask: it stays in the digest.",
    };
  }
  return {
    title: `Snoozed until ${clockOf(reminder.remind_at_local) || "later"}`,
    body: (bill && bill.headline) || reminder.text || "",
  };
}

/** Buttons only where the platform has room for them. Safari has no
 * `maxActions`, and there the reminder sheet does the job. */
function notificationActions(actions) {
  const max = (self.Notification && self.Notification.maxActions) || 0;
  if (!Array.isArray(actions) || max <= 0) return undefined;
  return actions.slice(0, max);
}

/** Never let a notification be lost to an option this browser rejects: iOS
 * revokes the push permission of a web app that receives a push and displays
 * nothing, so a refusal is retried as the plain notification. */
function show(title, options) {
  return self.registration.showNotification(title, options).catch(() => {
    const { actions: _dropped, ...plain } = options;
    return self.registration.showNotification(title, plain);
  });
}

self.addEventListener("push", (event) => {
  let payload = {};
  try {
    payload = event.data ? event.data.json() : {};
  } catch {
    // A push with no body, or one this version does not understand, must
    // still show something (see `show`).
    payload = {};
  }

  const title = payload.title || "IRIS";
  const data = payload.data && typeof payload.data === "object" ? payload.data : {};
  event.waitUntil(
    show(title, {
      body: payload.body || "Something needs you.",
      icon: "/icon-192.png",
      badge: "/icon-192.png",
      // Collapse repeats of the same subject rather than stacking: the
      // health watch can re-notify about one incident several times.
      // Reminders carry their own tag (`reminder:<id>`), so two reminders never
      // replace each other; a snoozed one returns under its tag with
      // `renotify`, so it alerts again instead of swapping in silently.
      tag: payload.tag || title,
      renotify: Boolean(payload.tag && payload.renotify),
      data: { url: data.url || payload.url || "/", reminder_id: data.reminder_id || null },
      actions: notificationActions(payload.actions),
    }),
  );
});

/** "9:03 AM" out of the server's "Mon Sep 28, 9:03 AM". */
function clockOf(local) {
  const parts = String(local || "").split(", ");
  return parts[parts.length - 1] || "";
}

/** Answer a reminder from its notification button, then say what happened.
 *
 * The confirmation replaces the reminder's notification (same tag), so the
 * owner sees the result where they tapped. A failure says so and keeps the
 * sheet one tap away: a button that silently did nothing would leave the
 * owner believing a reminder was handled when it was not. */
async function answerReminder(id, action, url) {
  const spec = REMINDER_ACTIONS[action];
  const tag = `reminder:${id}`;
  try {
    const r = await fetch(`/api/v1/reminders/${encodeURIComponent(id)}/${spec.path}`, {
      method: "POST",
      credentials: "include",
      headers: { "content-type": "application/json", accept: "application/json" },
      body: JSON.stringify(spec.body),
    });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const answer = await r.json().catch(() => ({}));
    const { title, body } = confirmation(action, answer);
    return show(title, {
      body,
      icon: "/icon-192.png",
      badge: "/icon-192.png",
      tag,
      data: { url, reminder_id: id },
    });
  } catch {
    return show("Couldn't update the reminder", {
      body: "Tap to open it in IRIS.",
      icon: "/icon-192.png",
      badge: "/icon-192.png",
      tag,
      data: { url, reminder_id: id },
    });
  }
}

/** Focus a tab that is already open rather than piling up new ones — on a
 * home-screen web app there is only ever one, and opening a second is
 * visibly wrong. */
function openApp(target) {
  return self.clients.matchAll({ type: "window", includeUncontrolled: true }).then((clients) => {
    for (const client of clients) {
      if ("focus" in client) {
        if ("navigate" in client && client.url !== target) client.navigate(target);
        return client.focus();
      }
    }
    return self.clients.openWindow(target);
  });
}

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const data = event.notification.data || {};
  const path = data.url || (data.reminder_id ? `/reminders/${data.reminder_id}` : "/");

  // A button (never on iOS): answer the reminder without opening the app.
  if (data.reminder_id && REMINDER_ACTIONS[event.action]) {
    event.waitUntil(answerReminder(data.reminder_id, event.action, path));
    return;
  }

  // A tap on the body (iOS's only gesture): open where it points — for a
  // reminder, its sheet, where Done and Snooze live.
  event.waitUntil(openApp(new URL(path, self.location.origin).href));
});
