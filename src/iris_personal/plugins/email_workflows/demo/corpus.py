"""The demo mailbox's corpus: 200 synthetic emails, generated from a fixed seed.

``python -m iris_personal.plugins.email_workflows.demo.corpus`` rewrites ``corpus.json``
beside this module; a test fails when the checked-in file and this generator disagree,
so the data is always reproducible from the code.

Everything is invented. People are at ``example.com`` / ``example.org`` /
``example.net``; companies are made up and live under the reserved ``.test`` TLD. No
real person, brand or address appears.

A message's times and dates are relative: ``minutes_ago`` is measured back from the
moment the demo mailbox is first opened, and a body carries date tokens the provider
fills in then -- ``{date+9}`` (an ISO date nine days after), ``{when+2@15:30}`` (an ISO
date and time) and ``{nice+9}`` ("Fri Oct 09"). So the mailbox reads as current on
every machine, and re-runs of the demo read the same.

``kind`` is what the generator meant each email to be (``bill``, ``event``,
``needs_reply``, ``fyi``, ``automated_ask``, ``unsure``, ``promo``, ``social``,
``sent``). It is data for tests and the docs; the provider never shows it to the judge.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SEED = 20260930
CORPUS_PATH = Path(__file__).with_name("corpus.json")

OWNER_NAME = "Sam Rivera"
OWNER_ADDRESS = "sam.rivera@example.com"
OWNER = f"{OWNER_NAME} <{OWNER_ADDRESS}>"

# Everything arrives within the last ~70 hours: the judge queues mail it received in the
# last 72 (judge.yaml ``queue_hours``), so the whole demo inbox is judged on first run.
MAX_MINUTES_AGO = 70 * 60

PROMO_CATEGORY = "email/promotions"
SOCIAL_CATEGORY = "email/social"


@dataclass
class _Mail:
    kind: str
    sender: str
    subject: str
    body: str
    minutes_ago: int
    labels: list[str] = field(default_factory=lambda: ["INBOX"])
    to: list[str] = field(default_factory=lambda: [OWNER_ADDRESS])
    cc: list[str] = field(default_factory=list)
    thread: str | None = None
    attachments: list[dict[str, Any]] = field(default_factory=list)
    vendor_category: str | None = None


def _pdf(name: str, size: int) -> dict[str, Any]:
    return {"filename": name, "mime_type": "application/pdf", "size_bytes": size}


def _ics(name: str) -> dict[str, Any]:
    return {"filename": name, "mime_type": "text/calendar", "size_bytes": 812}


# -- the people and companies (all invented) ------------------------------------------

PEOPLE = {
    "petra": "Petra Raman <petra.raman@example.com>",
    "marcus": "Marcus Chen <marcus.chen@example.org>",
    "lena": "Lena Okafor <lena.okafor@example.org>",
    "diego": "Diego Alvarez <diego.alvarez@example.com>",
    "hannah": "Hannah Brooks <hannah.brooks@example.net>",
    "tomas": "Tomas Varga <tomas.varga@example.net>",
    "aisha": "Aisha Patel <aisha.patel@example.com>",
    "owen": "Owen Fitzgerald <owen.fitzgerald@example.org>",
    "mei": "Mei Lin <mei.lin@example.com>",
    "remy": "Remy Kumar <remy.kumar@example.net>",
    "nora": "Nora Svensson <nora.svensson@example.org>",
    "jonah": "Jonah Weiss <jonah.weiss@example.com>",
}

BILLERS = [
    # (sender, what, amount range, pdf name)
    ("Brightwater Utilities <billing@brightwater-utilities.test>", "electricity", (64, 188)),
    ("Tidewell Mobile <billing@tidewell-mobile.test>", "mobile plan", (38, 92)),
    ("Maplebrook Water District <billing@maplebrook-water.test>", "water and sewer", (41, 97)),
    ("Clearpath Internet <billing@clearpath-net.test>", "home internet", (55, 80)),
    ("Evergreen Mutual Insurance <policy@evergreen-mutual.test>", "home insurance", (88, 240)),
    ("Oakridge Property Management <rent@oakridge-pm.test>", "rent", (1450, 1980)),
    ("Summit Auto Finance <loans@summit-autofinance.test>", "car loan", (289, 415)),
]
CARD = "Harborstone Card Services <statements@harborstone-card.test>"
CARD_ALERTS = "Harborstone Card Services <alerts@harborstone-card.test>"

NEWSLETTERS = [
    ("The Morning Ledger <newsletter@morningledger.test>", "The Morning Ledger"),
    ("Gardeners' Weekly <digest@gardenersweekly.test>", "Gardeners' Weekly"),
    ("Code & Coffee <newsletter@codeandcoffee.test>", "Code & Coffee"),
    ("City of Maplebrook <news@cityofmaplebrook.test>", "Maplebrook City Bulletin"),
    ("Trailhead Running Club <news@trailheadrunners.test>", "Trailhead Notes"),
]
NEWSLETTER_TOPICS = [
    "five charts on the housing market",
    "what the new transit plan means for commuters",
    "the quiet comeback of the paper notebook",
    "a field guide to fall bulbs",
    "why your tomatoes split, and how to stop it",
    "a gentle intro to property-based testing",
    "reading a flame graph in ten minutes",
    "the week's library events and road closures",
    "leaf collection starts on the east side",
    "hill repeats, explained",
    "recovery weeks are training too",
    "a history of the neighborhood farmers market",
    "three recipes for the last of the zucchini",
    "the case for boring technology",
]

STORES = [
    ("Northfield Books <orders@northfield-books.test>", "Northfield Books"),
    ("Greenleaf Grocers <receipts@greenleaf-grocers.test>", "Greenleaf Grocers"),
    ("Lanternfish Hardware <orders@lanternfish-hardware.test>", "Lanternfish Hardware"),
]
SHIPPER = "Parcelwise <tracking@parcelwise.test>"
PROMO_SENDERS = [
    ("Lumina Home Goods <deals@luminahome.test>", "Lumina Home"),
    ("Peak Outfitters <offers@peakoutfitters.test>", "Peak Outfitters"),
    ("Saltmarsh Coffee Roasters <hello@saltmarshcoffee.test>", "Saltmarsh Coffee"),
    ("Brambleberry Kids <sale@brambleberrykids.test>", "Brambleberry Kids"),
    ("Voyagely Travel <deals@voyagely.test>", "Voyagely"),
    ("Northfield Books <offers@northfield-books.test>", "Northfield Books"),
]
PROMO_SUBJECTS = [
    "{brand}: 30% off everything this weekend",
    "Last chance: free shipping ends tonight",
    "New arrivals picked for you",
    "Your cart misses you",
    "Members save an extra 15% today",
    "The fall collection is here",
    "Flash sale: 6 hours only",
    "Earn double points this week",
    "{brand} gift guide: under $25",
    "We saved your favorites",
]
SOCIAL_SENDERS = [
    "Circlely <notify@circlely.test>",
    "Pinloop <updates@pinloop.test>",
    "Trailhead Running Club <community@trailheadrunners.test>",
]
SOCIAL_SUBJECTS = [
    "Petra Raman commented on your photo",
    "You have 3 new followers",
    "Marcus Chen shared a post with you",
    "Your weekly activity summary",
    "Nora Svensson tagged you in a photo",
    "5 people viewed your profile",
    "New event near you: Saturday trail cleanup",
]


def _money(rng: random.Random, low: int, high: int) -> str:
    return f"{rng.uniform(low, high):.2f}"


# -- the generator ----------------------------------------------------------------------


def _bills(rng: random.Random, out: list[_Mail]) -> None:
    for index, (sender, what, (low, high)) in enumerate(BILLERS):
        amount = _money(rng, low, high)
        due_in = rng.randint(4, 18)
        attach = [_pdf(f"{what.replace(' ', '-')}-bill.pdf", rng.randint(38_000, 96_000))]
        out.append(
            _Mail(
                kind="bill",
                sender=sender,
                subject=f"Your {what} bill is ready",
                body=(
                    f"Hello {OWNER_NAME.split()[0]},\n\nYour {what} bill for this period is "
                    f"ready.\n\nAmount due: ${amount}\nDue date: {{date+{due_in}}}\n\n"
                    "The full bill is attached. Pay online or set up autopay from your "
                    "account page.\n\nThank you for being a customer."
                ),
                minutes_ago=rng.randint(30, MAX_MINUTES_AGO),
                attachments=attach if index % 2 == 0 else [],
            )
        )
    # Card statement: a balance AND a minimum, kept apart.
    balance = _money(rng, 900, 2400)
    minimum = f"{max(35.0, float(balance) * 0.02):.2f}"
    out.append(
        _Mail(
            kind="bill",
            sender=CARD,
            subject="Your Harborstone statement is available",
            body=(
                "Your monthly statement for the card ending 4417 is available.\n\n"
                f"Statement balance: ${balance}\nMinimum payment: ${minimum}\n"
                "Payment due: {date+21}\n\nSign in to view the statement or schedule a "
                "payment."
            ),
            minutes_ago=rng.randint(60, MAX_MINUTES_AGO),
            attachments=[_pdf("statement-4417.pdf", 142_311)],
        )
    )
    # Payment confirmations: "just paid" is a bill too (judge.yaml).
    for sender, what, (low, high) in rng.sample(BILLERS, 5):
        out.append(
            _Mail(
                kind="bill",
                sender=sender,
                subject=f"Payment received for your {what} account",
                body=(
                    f"Thanks! Payment received: ${_money(rng, low, high)} on "
                    "{nice-1}.\nNo action is needed. A receipt is in your account history."
                ),
                minutes_ago=rng.randint(30, MAX_MINUTES_AGO),
            )
        )
    # Three more bills due, from the busiest billers.
    for sender, what, (low, high) in rng.sample(BILLERS[:4], 3):
        out.append(
            _Mail(
                kind="bill",
                sender=sender,
                subject=f"Reminder: {what} payment due soon",
                body=(
                    f"This is a friendly reminder about your {what} account.\n\n"
                    f"Amount due: ${_money(rng, low, high)}\nDue date: "
                    f"{{date+{rng.randint(2, 6)}}}\n\nIf you already paid, thank you."
                ),
                minutes_ago=rng.randint(10, MAX_MINUTES_AGO),
            )
        )


def _events(rng: random.Random, out: list[_Mail]) -> None:
    events = [
        (
            "Orchard Lane Dental <appointments@orchardlane-dental.test>",
            "Appointment confirmed: cleaning and check-up",
            "Your appointment with Dr. Hale is confirmed.",
            "10:30",
            [_ics("orchard-lane-appointment.ics")],
        ),
        (
            "Kestrel Airways <itinerary@kestrel-air.test>",
            "Your trip is booked: Maplebrook to Port Juniper",
            "Booking reference KX7Q2M. Flight KA 214 departs Maplebrook Regional.",
            "07:45",
            [_pdf("itinerary-KX7Q2M.pdf", 58_204), _ics("KA214.ics")],
        ),
        (
            "Willow Creek Elementary <office@willowcreek-school.test>",
            "Family science night in the gym",
            "Families are invited to science night; students will show their projects.",
            "18:00",
            [_pdf("science-night-flyer.pdf", 211_880)],
        ),
        (
            "The Copper Pot Bistro <reservations@copperpot-bistro.test>",
            "Reservation confirmed for 4 guests",
            "We look forward to seeing you. Your table for 4 is held for 15 minutes.",
            "19:30",
            [],
        ),
        (
            "Riverside Physio <frontdesk@riverside-physio.test>",
            "Your physiotherapy session is booked",
            "Please arrive 10 minutes early for your follow-up session with Jordan.",
            "08:15",
            [],
        ),
        (
            "Willow Creek Elementary <office@willowcreek-school.test>",
            "Parent-teacher conference scheduled",
            "Your conference with Ms. Duarte is scheduled in room 12.",
            "16:10",
            [],
        ),
        (
            "Maplebrook Public Library <events@maplebrook-library.test>",
            "You're registered: Saturday storytime",
            "Your family is registered for storytime in the children's room.",
            "11:00",
            [],
        ),
        (
            "Brightline Vet Clinic <care@brightline-vet.test>",
            "Appointment confirmed for Juniper (annual exam)",
            "Juniper's annual exam and vaccines are booked with Dr. Moss.",
            "09:00",
            [_ics("juniper-vet.ics")],
        ),
    ]
    for sender, subject, text, at, attachments in events:
        day = rng.randint(1, 20)
        out.append(
            _Mail(
                kind="event",
                sender=sender,
                subject=subject,
                body=f"{text}\n\nWhen: {{when+{day}@{at}}}\nDate: {{nice+{day}}}\n\n"
                "Need to change it? Use the link in your account.",
                minutes_ago=rng.randint(15, MAX_MINUTES_AGO),
                attachments=attachments,
            )
        )
    # Calendar invitations from colleagues: invited to be somewhere, at a time.
    invites = [
        ("lena", "Invitation: Q4 planning review", "Q4 planning review", "14:00"),
        ("marcus", "Invitation: vendor review walkthrough", "vendor review walkthrough", "11:30"),
        ("nora", "Invitation: design crit", "design crit for the onboarding flow", "15:00"),
        ("jonah", "Invitation: team lunch", "team lunch at the Copper Pot", "12:30"),
    ]
    for who, subject, what, at in invites:
        day = rng.randint(1, 9)
        out.append(
            _Mail(
                kind="event",
                sender=PEOPLE[who],
                subject=subject,
                body=f"You have been invited to the {what}.\n\nWhen: {{when+{day}@{at}}}\n"
                "Where: 3rd floor, Harbor room\n\nOrganizer: "
                f"{PEOPLE[who].split(' <')[0]}",
                minutes_ago=rng.randint(15, MAX_MINUTES_AGO),
                attachments=[_ics("invite.ics")],
            )
        )


def _people(rng: random.Random, out: list[_Mail]) -> None:
    """Mail a real person wrote to the owner, some in threads the owner answered."""

    def ago(low: int, high: int) -> int:
        return rng.randint(low, high)

    # Thread: dinner. Petra asks, Sam replies, Petra follows up (the follow-up waits).
    t1 = ago(2400, 3600)
    out += [
        _Mail(
            "needs_reply",
            PEOPLE["petra"],
            "Dinner next Saturday?",
            "Hi Sam,\n\nWe are hosting a small dinner next Saturday. Are you free around 7? "
            "Let me know if Alex can come too.\n\nPriya",
            t1,
            thread="t-dinner",
        ),
        _Mail(
            "sent",
            OWNER,
            "Re: Dinner next Saturday?",
            "We'd love to! Both of us will be there. Sent from my laptop.",
            t1 - 90,
            labels=["SENT"],
            to=[PEOPLE["petra"]],
            thread="t-dinner",
        ),
        _Mail(
            "needs_reply",
            PEOPLE["petra"],
            "Re: Dinner next Saturday?",
            "Wonderful. Could you bring a dessert? Anything without nuts, please.\n\nP.",
            t1 - 400,
            thread="t-dinner",
        ),
    ]
    # Thread: client question. Owen asks, Sam answers, Owen asks one more thing.
    t2 = ago(3000, 4000)
    out += [
        _Mail(
            "needs_reply",
            PEOPLE["owen"],
            "Rollout plan for the Harbor account",
            "Hi Sam,\n\nThanks for the call. Could you send the rollout plan by "
            "{date+3}? Our board meets the day after.\n\nBest,\nOwen",
            t2,
            thread="t-rollout",
            attachments=[
                {
                    "filename": "harbor-rollout-draft.xlsx",
                    "mime_type": "application/vnd.openxmlformats-officedocument."
                    "spreadsheetml.sheet",
                    "size_bytes": 48_551,
                }
            ],
        ),
        _Mail(
            "sent",
            OWNER,
            "Re: Rollout plan for the Harbor account",
            "Attached is the plan. Happy to walk through it.",
            t2 - 200,
            labels=["SENT"],
            to=[PEOPLE["owen"]],
            thread="t-rollout",
        ),
        _Mail(
            "needs_reply",
            PEOPLE["owen"],
            "Re: Rollout plan for the Harbor account",
            "Got it, thank you. One more question: can you confirm the go-live date is "
            "still {nice+12}?\n\nOwen",
            t2 - 900,
            thread="t-rollout",
        ),
    ]
    # Thread: family. Hannah writes to Sam and Diego; Hannah follows up.
    t3 = ago(1500, 2600)
    out += [
        _Mail(
            "needs_reply",
            PEOPLE["hannah"],
            "Mom's birthday plans",
            "Hi both,\n\nMom turns 70 next month. What do you think about a picnic at "
            "Lakeside Park instead of a restaurant?\n\nH",
            t3,
            cc=[PEOPLE["diego"]],
            thread="t-birthday",
        ),
        _Mail(
            "needs_reply",
            PEOPLE["hannah"],
            "Re: Mom's birthday plans",
            "Diego is in for the picnic. Sam, would you handle the cake? Let me know if "
            "that is too much.\n\nH",
            t3 - 300,
            cc=[PEOPLE["diego"]],
            thread="t-birthday",
        ),
    ]
    # Thread: the landlord, answered already.
    t4 = ago(2000, 3000)
    out += [
        _Mail(
            "needs_reply",
            PEOPLE["tomas"],
            "Boiler inspection",
            "Hello Sam,\n\nThe annual boiler inspection is due. Can you let me in on a "
            "weekday morning next week?\n\nTomas",
            t4,
            thread="t-boiler",
        ),
        _Mail(
            "sent",
            OWNER,
            "Re: Boiler inspection",
            "Tuesday at 9 works. See you then.",
            t4 - 60,
            labels=["SENT"],
            to=[PEOPLE["tomas"]],
            thread="t-boiler",
        ),
    ]
    singles = [
        (
            "marcus",
            "Draft vendor review",
            "Hi Sam,\n\nThe vendor review draft is attached. Could you look it over before "
            "{nice+2}? Mostly sections 3 and 4.\n\nThanks,\nMarcus",
            [
                {
                    "filename": "vendor-review-draft.docx",
                    "mime_type": "application/vnd.openxmlformats-officedocument."
                    "wordprocessingml.document",
                    "size_bytes": 88_020,
                }
            ],
        ),
        (
            "lena",
            "Friday standup",
            "Sam, I am out on Friday. Can you run the standup? The notes template is in "
            "the team folder.\n\nLena",
            [],
        ),
        (
            "aisha",
            "Carpool swap this week",
            "Hi! Would you swap carpool days with me this week, Wednesday for Thursday? "
            "Our schedule changed.\n\nAisha",
            [],
        ),
        (
            "mei",
            "Book club pick",
            "Hi Sam,\n\nFor next month I am torn between two novels. What do you think: the "
            "lighthouse one or the one about the orchestra?\n\nMei",
            [],
        ),
        (
            "remy",
            "Coffee next week?",
            "Hi Sam,\n\nIt has been a while. Are you free for coffee next week? I would "
            "love to hear about the new role.\n\nRavi",
            [],
        ),
        (
            "diego",
            "Keeping an eye on our place",
            "Hey Sam, we are away from {nice+5} to {nice+9}. Could you water the plants "
            "once or twice? The spare key is where it always is.\n\nDiego",
            [],
        ),
        (
            "nora",
            "Onboarding copy review",
            "Hi Sam,\n\nCould you review the new onboarding copy by {date+4}? I left "
            "comments on the three screens that changed.\n\nNora",
            [],
        ),
        (
            "jonah",
            "Reference letter",
            "Hi Sam,\n\nI am applying for the fellowship. Would you be willing to write a "
            "short reference letter? The deadline is {nice+14}.\n\nThanks so much,\nJonah",
            [],
        ),
        (
            "petra",
            "Borrowing your ladder",
            "Hi! Can you lend us your tall ladder this weekend? We are finally painting the "
            "hallway.\n\nPriya",
            [],
        ),
        (
            "lena",
            "Headcount for Q1",
            "Sam, can you send me your Q1 headcount estimate by {date+6}? A range is fine "
            "for now.\n\nLena",
            [],
        ),
        (
            "marcus",
            "Conference talk proposal",
            "Would you co-present the migration story at the spring conference? The "
            "proposal is due soon, so let me know if you are in.\n\nMarcus",
            [],
        ),
        (
            "aisha",
            "Bake sale volunteers",
            "Hi Sam, could you help at the bake sale table for an hour on Friday? We are "
            "two people short.\n\nAisha",
            [],
        ),
        (
            "owen",
            "Quarterly check-in",
            "Hi Sam,\n\nCan you suggest two times for our quarterly check-in next " "week?\n\nOwen",
            [],
        ),
        (
            "tomas",
            "Lease renewal",
            "Hello Sam,\n\nYour lease renews in two months. Would you like to renew for "
            "another year? Let me know if you have questions.\n\nTomas",
            [],
        ),
        (
            "hannah",
            "Photos from the lake",
            "I finally sorted the photos from the lake trip. Can you send me the ones from "
            "your phone too?\n\nH",
            [],
        ),
    ]
    for who, subject, body, attachments in singles:
        out.append(
            _Mail(
                "needs_reply",
                PEOPLE[who],
                subject,
                body,
                ago(20, MAX_MINUTES_AGO),
                attachments=attachments,
            )
        )
    # Two notes Sam sent on their own.
    for to, subject, body in (
        ("marcus", "Notes from today", "Sharing my notes from the planning session."),
        ("diego", "Your package", "A package for you landed on our porch; I'll drop it by."),
    ):
        out.append(
            _Mail(
                "sent",
                OWNER,
                subject,
                body,
                ago(60, MAX_MINUTES_AGO),
                labels=["SENT"],
                to=[PEOPLE[to]],
            )
        )


def _fyi(rng: random.Random, out: list[_Mail]) -> None:
    topics = list(NEWSLETTER_TOPICS)
    rng.shuffle(topics)
    for index in range(28):
        sender, name = NEWSLETTERS[index % len(NEWSLETTERS)]
        topic = topics[index % len(topics)]
        out.append(
            _Mail(
                "fyi",
                sender,
                f"{name}: {topic}",
                f"This week in {name}: {topic}. Plus reader mail, a short quiz and the "
                "calendar of upcoming talks.\n\nYou are receiving this because you "
                "subscribed. Unsubscribe any time.",
                rng.randint(10, MAX_MINUTES_AGO),
            )
        )
    for index in range(12):
        sender, name = STORES[index % len(STORES)]
        order = rng.randint(100_000, 999_999)
        total = _money(rng, 12, 160)
        out.append(
            _Mail(
                "fyi",
                sender,
                f"Your {name} receipt (order {order})",
                f"Thanks for shopping with {name}. Order {order}, total ${total}, paid by "
                "card. Keep this email for your records.",
                rng.randint(10, MAX_MINUTES_AGO),
                attachments=(
                    [_pdf(f"receipt-{order}.pdf", rng.randint(20_000, 40_000))]
                    if index % 3 == 0
                    else []
                ),
            )
        )
    for _ in range(10):
        parcel = f"PW{rng.randint(10_000_000, 99_999_999)}"
        status = rng.choice(
            [
                "has shipped",
                "is out for delivery",
                "was delivered to your front door",
                "is arriving {nice+2}",
            ]
        )
        out.append(
            _Mail(
                "fyi",
                SHIPPER,
                f"Your parcel {parcel} {status.split(' {')[0]}",
                f"Parcel {parcel} {status}. Track it any time on the Parcelwise site.",
                rng.randint(10, MAX_MINUTES_AGO),
            )
        )
    notices = [
        (CARD_ALERTS, "Transaction alert", "A charge of ${amt} at Greenleaf Grocers was made."),
        (CARD_ALERTS, "Transaction alert", "A charge of ${amt} at Lanternfish Hardware was made."),
        (CARD_ALERTS, "Your credit limit was reviewed", "Good news: your limit is unchanged."),
        (
            "Cloudnest Storage <security@cloudnest.test>",
            "New sign-in to your account",
            "We noticed a new sign-in from a Mac in Maplebrook. If this was you, no action "
            "is needed.",
        ),
        (
            "Cloudnest Storage <security@cloudnest.test>",
            "Your password was changed",
            "The password for your account was changed. If this was you, you can ignore "
            "this message.",
        ),
        (
            "Brightwater Utilities <updates@brightwater-utilities.test>",
            "Planned maintenance in your area",
            "Crews will upgrade lines in your area on {nice+6}. Short outages are possible.",
        ),
        (
            "Tidewell Mobile <updates@tidewell-mobile.test>",
            "You used 80% of your data",
            "You have used 80% of this month's data. It resets on {nice+11}.",
        ),
        (
            "Clearpath Internet <news@clearpath-net.test>",
            "Service update: faster upload speeds",
            "We raised upload speeds on your plan at no cost. Nothing to do on your side.",
        ),
        (
            "City of Maplebrook <alerts@cityofmaplebrook.test>",
            "Street sweeping on your block",
            "Street sweeping is scheduled on your block. Please move cars from the east side.",
        ),
        (
            "Evergreen Mutual Insurance <policy@evergreen-mutual.test>",
            "Your policy documents are available",
            "Your updated policy documents are in your account. Coverage is unchanged.",
        ),
    ]
    for sender, subject, body in notices:
        out.append(
            _Mail(
                "fyi",
                sender,
                subject,
                body.replace("{amt}", _money(rng, 8, 140)),
                rng.randint(10, MAX_MINUTES_AGO),
            )
        )
    for _ in range(6):
        role = rng.choice(
            ["Staff Engineer", "Product Designer", "Data Analyst", "Engineering Manager"]
        )
        out.append(
            _Mail(
                "fyi",
                "Hireline <jobs@hireline.test>",
                f"New jobs for you: {role}",
                f"12 new {role} roles match your saved search. See them on Hireline.",
                rng.randint(10, MAX_MINUTES_AGO),
            )
        )


def _automated_asks(rng: random.Random, out: list[_Mail]) -> None:
    """System mail that sounds like a person asking: the judge's person-only rule turns a
    needs-reply verdict on these into FYI."""
    asks = [
        (
            "Cloudnest Storage <noreply@cloudnest.test>",
            "Action required: confirm your email preferences",
            "Could you take a moment to confirm how often we may email you?",
        ),
        (
            "Northfield Books <feedback@northfield-books.test>",
            "How was your order?",
            "Would you rate your recent order? It takes 30 seconds.",
        ),
        (
            "Tidewell Mobile <survey@tidewell-mobile.test>",
            "Tell us about your service",
            "Can you spare two minutes for our customer survey?",
        ),
    ]
    for sender, subject, body in asks:
        out.append(_Mail("automated_ask", sender, subject, body, rng.randint(10, MAX_MINUTES_AGO)))


def _unsure(rng: random.Random, out: list[_Mail]) -> None:
    """Too little text to tell (judge.yaml's ``unsure``)."""
    for who, subject in (
        ("jonah", "Fwd:"),
        ("remy", "hey"),
        ("nora", "?"),
        ("mei", "this"),
        ("diego", "Fwd: fwd"),
    ):
        out.append(
            _Mail(
                "unsure",
                PEOPLE[who],
                subject,
                "Sent from my phone",
                rng.randint(10, MAX_MINUTES_AGO),
            )
        )


def _promotions(rng: random.Random, out: list[_Mail]) -> None:
    for index in range(58):
        sender, brand = PROMO_SENDERS[index % len(PROMO_SENDERS)]
        subject = rng.choice(PROMO_SUBJECTS).replace("{brand}", brand)
        out.append(
            _Mail(
                "promo",
                sender,
                subject,
                f"{subject}. Shop the collection at {brand}. Offer ends {{nice+3}}.",
                rng.randint(10, MAX_MINUTES_AGO),
                labels=["INBOX", "CATEGORY_PROMOTIONS"],
                vendor_category=PROMO_CATEGORY,
            )
        )


def _social(rng: random.Random, out: list[_Mail]) -> None:
    for index in range(13):
        subject = SOCIAL_SUBJECTS[index % len(SOCIAL_SUBJECTS)]
        out.append(
            _Mail(
                "social",
                SOCIAL_SENDERS[index % len(SOCIAL_SENDERS)],
                subject,
                f"{subject}. Open the app to see more.",
                rng.randint(10, MAX_MINUTES_AGO),
                labels=["INBOX", "CATEGORY_SOCIAL"],
                vendor_category=SOCIAL_CATEGORY,
            )
        )


_SECTIONS: tuple[Callable[[random.Random, list[_Mail]], None], ...] = (
    _bills,
    _events,
    _people,
    _fyi,
    _automated_asks,
    _unsure,
    _promotions,
    _social,
)


def generate(seed: int = SEED) -> list[dict[str, Any]]:
    """The corpus as JSON-ready rows, oldest first, ids ``demo-0001``..."""
    rng = random.Random(seed)  # noqa: S311 - reproducible synthetic data, not security
    mails: list[_Mail] = []
    for section in _SECTIONS:
        section(rng, mails)
    mails.sort(key=lambda m: (-m.minutes_ago, m.subject))
    rows: list[dict[str, Any]] = []
    for number, mail in enumerate(mails, start=1):
        message_id = f"demo-{number:04d}"
        attachments = [
            {**a, "attachment_id": f"{message_id}-att-{i}"}
            for i, a in enumerate(mail.attachments, start=1)
        ]
        rows.append(
            {
                "id": message_id,
                "thread_id": mail.thread or f"t-{message_id}",
                "kind": mail.kind,
                "from": mail.sender,
                "to": mail.to,
                "cc": mail.cc,
                "subject": mail.subject,
                "minutes_ago": mail.minutes_ago,
                "body": mail.body,
                "labels": mail.labels,
                "vendor_category": mail.vendor_category,
                "attachments": attachments,
            }
        )
    return rows


def render(rows: list[dict[str, Any]]) -> str:
    return json.dumps({"seed": SEED, "messages": rows}, indent=1, ensure_ascii=False) + "\n"


def load_corpus(path: Path = CORPUS_PATH) -> list[dict[str, Any]]:
    """The checked-in corpus rows."""
    data = json.loads(path.read_text(encoding="utf-8"))
    messages = data.get("messages") if isinstance(data, dict) else None
    if not isinstance(messages, list):
        raise ValueError(f"{path}: not a demo corpus")
    return messages


def kind_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(r["kind"]) for r in rows).items()))


def main() -> None:
    rows = generate()
    CORPUS_PATH.write_text(render(rows), encoding="utf-8")
    print(f"wrote {len(rows)} messages to {CORPUS_PATH}: {kind_counts(rows)}")  # noqa: T201


if __name__ == "__main__":
    main()
