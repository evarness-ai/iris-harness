"""Web Push: notifications to a home-screen web app, no APNs and no app store.

Track 2b (mobile + cloud UI plan, decision 37). iOS 16.4+ delivers standard
W3C Web Push to a web app added to the Home Screen, with no Apple Developer
Program membership — which is what let the owner's approval-banner
requirement survive the free-signing choice in decision 12, and it works on
Android unchanged.

Four pieces: ``encryption`` (RFC 8291, verified against the RFC's own test
vector), ``vapid`` (RFC 8292, proving who is sending), ``store`` (who agreed
to be notified) and ``connector`` (an ordinary IChannelConnector, so the
health watch's existing broadcast reaches it unmodified).
"""

from iris_harness.services.channels.web_push.connector import WebPushConnector
from iris_harness.services.channels.web_push.store import PushSubscription, PushSubscriptionStore

__all__ = ["PushSubscription", "PushSubscriptionStore", "WebPushConnector"]
