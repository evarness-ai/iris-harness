/* Subscribing this browser to notifications (Track 2b PR 9).
 *
 * The whole flow in one place, because each step fails differently and the
 * owner needs to be told which one:
 *
 *   1. the browser must support push at all
 *   2. on iOS, the app must have been ADDED TO THE HOME SCREEN — Safari
 *      proper cannot subscribe, and the API is simply absent there
 *   3. the user must grant permission, from a real tap
 *   4. the subscription goes to the harness, which stores it
 *
 * Step 2 is the one that surprises people: everything looks present in
 * Safari, `Notification` exists, and `requestPermission` resolves "denied"
 * without a prompt. Detecting standalone mode up front turns that into a
 * sentence rather than a mystery.
 */
import { apiFetch } from "./http";

export type PushSupport =
  | { ok: true }
  | { ok: false; reason: "unsupported" | "needs-home-screen" | "denied" };

/** iOS only exposes push to a home-screen web app. */
export function isStandalone(): boolean {
  // `navigator.standalone` is the iOS-specific signal; the media query is the
  // standard one every other platform honours.
  const iosStandalone = (navigator as { standalone?: boolean }).standalone === true;
  return iosStandalone || window.matchMedia("(display-mode: standalone)").matches;
}

const isApple = () => /iP(hone|ad|od)/.test(navigator.userAgent);

export function pushSupport(): PushSupport {
  if (!("serviceWorker" in navigator) || !("PushManager" in window)) {
    // On an iPhone this is what "opened in Safari, not installed" looks like.
    return { ok: false, reason: isApple() && !isStandalone() ? "needs-home-screen" : "unsupported" };
  }
  if (isApple() && !isStandalone()) return { ok: false, reason: "needs-home-screen" };
  if (Notification.permission === "denied") return { ok: false, reason: "denied" };
  return { ok: true };
}

function urlBase64ToBytes(base64: string): ArrayBuffer {
  // The VAPID key travels as unpadded base64url; PushManager wants raw bytes.
  // Returned as ArrayBuffer, not Uint8Array: TS types applicationServerKey as
  // ArrayBufferView<ArrayBuffer>, which a Uint8Array<ArrayBufferLike> is not.
  const padded = base64.padEnd(base64.length + ((4 - (base64.length % 4)) % 4), "=");
  const raw = atob(padded.replace(/-/g, "+").replace(/_/g, "/"));
  const bytes = new Uint8Array(raw.length);
  for (let i = 0; i < raw.length; i += 1) bytes[i] = raw.charCodeAt(i);
  return bytes.buffer;
}

export interface PushKeyInfo {
  public_key: string;
  subject: string;
  subscriptions: number;
}

export async function getPushKey(): Promise<PushKeyInfo> {
  const resp = await apiFetch("/api/v1/push/key");
  if (!resp.ok) throw new Error(`could not read the push key (HTTP ${resp.status})`);
  return (await resp.json()) as PushKeyInfo;
}

/** Subscribe this browser. Must be called from a user gesture — iOS requires it. */
export async function subscribeToPush(label: string): Promise<void> {
  const support = pushSupport();
  if (!support.ok) throw new Error(support.reason);

  const permission = await Notification.requestPermission();
  if (permission !== "granted") throw new Error("denied");

  const { public_key: publicKey } = await getPushKey();
  const registration = await navigator.serviceWorker.ready;
  const existing = await registration.pushManager.getSubscription();
  // Re-subscribing with a different VAPID key silently fails to deliver, so
  // drop any subscription made under an older key before making a new one.
  if (existing) await existing.unsubscribe();

  const subscription = await registration.pushManager.subscribe({
    userVisibleOnly: true, // required by every browser, and by iOS strictly
    applicationServerKey: urlBase64ToBytes(publicKey),
  });

  const json = subscription.toJSON();
  const resp = await apiFetch("/api/v1/push/subscribe", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({
      endpoint: subscription.endpoint,
      p256dh: json.keys?.p256dh,
      auth: json.keys?.auth,
      label,
    }),
  });
  if (!resp.ok) {
    // Do not leave the browser subscribed to a harness that does not know
    // about it: that is a notification nobody can ever send.
    await subscription.unsubscribe();
    throw new Error(`the harness refused the subscription (HTTP ${resp.status})`);
  }
}

/** Stop notifications on this browser, both sides. */
export async function unsubscribeFromPush(): Promise<void> {
  const registration = await navigator.serviceWorker.ready;
  const subscription = await registration.pushManager.getSubscription();
  if (!subscription) return;
  await apiFetch("/api/v1/push/subscribe", {
    method: "DELETE",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ endpoint: subscription.endpoint }),
  });
  await subscription.unsubscribe();
}

export async function isSubscribed(): Promise<boolean> {
  if (!("serviceWorker" in navigator) || !("PushManager" in window)) return false;
  const registration = await navigator.serviceWorker.ready;
  return (await registration.pushManager.getSubscription()) !== null;
}
