"""The benchmark's UI scenes: web, desktop and mobile, each simple / medium / complex, as HTML.

Every scored element carries data-gt="<kind>" and a unique data-name; bench/render.py reads their boxes, colours and text from the DOM.
Kinds: panel (a container: header, sidebar, card, list, dialog), button, input, text, icon, avatar, chip, toggle, divider, image.

    python -m bench.scenes        writes data/scenes/<scene>.html
"""

from __future__ import annotations

import html
import random
from pathlib import Path

SCENES_DIR = Path(__file__).resolve().parent.parent / "data" / "scenes"
ICONS = {
    "search": '<circle cx="9" cy="9" r="6" fill="none" stroke="currentColor" stroke-width="2"/><path d="M14 14l5 5" stroke="currentColor" stroke-width="2"/>',
    "home": '<path d="M3 10l7-6 7 6v8h-5v-5H8v5H3z" fill="currentColor"/>',
    "bell": '<path d="M10 3a5 5 0 0 1 5 5v4l2 3H3l2-3V8a5 5 0 0 1 5-5zM8 17h4a2 2 0 0 1-4 0z" fill="currentColor"/>',
    "gear": '<circle cx="10" cy="10" r="3" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="10" cy="10" r="7" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="3 2"/>',
    "folder": '<path d="M2 5h6l2 2h8v9H2z" fill="currentColor"/>',
    "file": '<path d="M5 2h7l4 4v12H5z" fill="none" stroke="currentColor" stroke-width="2"/>',
    "plus": '<path d="M10 4v12M4 10h12" stroke="currentColor" stroke-width="2"/>',
    "star": '<path d="M10 2l2.4 5 5.6.6-4.2 3.8 1.2 5.6L10 14l-5 3 1.2-5.6L2 7.6 7.6 7z" fill="currentColor"/>',
    "user": '<circle cx="10" cy="7" r="4" fill="currentColor"/><path d="M3 18a7 7 0 0 1 14 0z" fill="currentColor"/>',
    "cart": '<path d="M2 3h3l2 9h9l2-6H6" fill="none" stroke="currentColor" stroke-width="2"/><circle cx="8" cy="16" r="1.6" fill="currentColor"/><circle cx="15" cy="16" r="1.6" fill="currentColor"/>',
    "mail": '<rect x="2" y="4" width="16" height="12" rx="2" fill="none" stroke="currentColor" stroke-width="2"/><path d="M2 5l8 6 8-6" fill="none" stroke="currentColor" stroke-width="2"/>',
    "chart": '<path d="M3 17V9M8 17V4M13 17v-6M18 17V7" stroke="currentColor" stroke-width="3"/>',
}
AVATAR_COLORS = ["#f97316", "#8b5cf6", "#14b8a6", "#ec4899", "#eab308", "#06b6d4", "#84cc16"]
IMAGE_COLORS = ["#fca5a5", "#93c5fd", "#86efac", "#fcd34d", "#c4b5fd", "#f9a8d4"]


class Scene:
    def __init__(self, name: str, width: int, height: int) -> None:
        self.name, self.width, self.height = name, width, height
        self.used: set[str] = set()

    def nm(self, name: str) -> str:
        base, i = name, 2
        while name in self.used:
            name, i = f"{base}-{i}", i + 1
        self.used.add(name)
        return name

    # each helper returns HTML for one scored element
    def el(self, kind: str, name: str, inner: str = "", cls: str = "", style: str = "", tag: str = "div") -> str:
        return f'<{tag} data-gt="{kind}" data-name="{self.nm(name)}" class="{cls}" style="{style}">{inner}</{tag}>'

    def text(self, name: str, words: str, style: str = "", cls: str = "") -> str:
        return self.el("text", name, html.escape(words), cls, style, tag="span")

    def icon(self, name: str, which: str, style: str = "") -> str:
        return f'<svg data-gt="icon" data-name="{self.nm(name)}" class="icon" viewBox="0 0 20 20" style="{style}">{ICONS[which]}</svg>'

    def button(self, name: str, label: str, cls: str = "btn", style: str = "") -> str:
        return self.el("button", name, self.text(name + "-label", label), cls, style)

    def input(self, name: str, placeholder: str, style: str = "") -> str:
        return self.el("input", name, self.text(name + "-placeholder", placeholder), "input", style)

    def chip(self, name: str, label: str) -> str:
        return self.el("chip", name, self.text(name + "-label", label), "chip")

    def avatar(self, name: str, color: str, size: int = 36, style: str = "") -> str:
        return self.el("avatar", name, "", "avatar", f"background:{color};width:{size}px;height:{size}px;{style}")

    def toggle(self, name: str, on: bool = True) -> str:
        return self.el("toggle", name, "", "toggle" + ("" if on else " off"))

    def image(self, name: str, color: str, w: int, h: int, radius: int = 8) -> str:
        return self.el("image", name, "", "", f"width:{w}px;height:{h}px;border-radius:{radius}px;background:{color};flex:none")

    def page(self, body: str) -> str:
        return (
            f'<!doctype html><html><head><meta charset="utf-8"><title>{self.name}</title><link rel="stylesheet" href="theme.css">'
            f"<style>body{{width:{self.width}px;height:{self.height}px}}</style></head><body>{body}</body></html>"
        )


def web_simple() -> Scene:
    s = Scene("web-simple", 1440, 900)
    nav = s.el("panel", "navbar", "".join([
        s.icon("logo", "star", "width:28px;height:28px;color:var(--accent)"),
        s.text("brand", "Northwind", "font-size:18px;font-weight:700;margin-left:10px"),
        '<div style="flex:1"></div>',
        s.text("nav-pricing", "Pricing", "margin-right:28px"),
        s.text("nav-docs", "Docs", "margin-right:28px"),
        s.button("nav-signin", "Sign in", "btn secondary"),
    ]), "row", "height:64px;padding:0 48px;background:var(--surface);border-bottom:1px solid var(--border)")
    hero = '<div class="col" style="align-items:center;margin-top:140px">' + "".join([
        s.text("hero-title", "Ship dashboards your team will read", "font-size:44px;font-weight:800"),
        s.text("hero-subtitle", "Connect a data source, pick a template, publish in minutes.", "font-size:18px;margin-top:16px", "muted"),
        '<div class="row" style="gap:12px;margin-top:36px">' + s.button("cta-primary", "Start free trial", "btn", "height:44px;padding:0 24px") + s.button("cta-secondary", "Book a demo", "btn secondary", "height:44px;padding:0 24px") + "</div>",
    ]) + "</div>"
    s.html = s.page(nav + hero)
    return s


def web_medium() -> Scene:
    s = Scene("web-medium", 1440, 900)
    side_items = [("Overview", "home"), ("Reports", "chart"), ("Inbox", "mail"), ("Settings", "gear")]
    side = s.el("panel", "sidebar", "".join([
        s.text("side-brand", "Acme Analytics", "font-size:16px;font-weight:700;color:var(--sidebar-text);margin:20px 20px 28px"),
        *[f'<div class="row" style="height:40px;padding:0 20px;gap:12px;color:var(--sidebar-text)">{s.icon("side-icon-" + w.lower(), ic)}{s.text("side-" + w.lower(), w)}</div>' for w, ic in side_items],
    ]), "col", "position:absolute;left:0;top:0;width:240px;height:900px;background:var(--sidebar)")
    header = s.el("panel", "topbar", "".join([
        s.text("page-title", "Overview", "font-size:22px;font-weight:700"),
        '<div style="flex:1"></div>',
        s.input("search", "Search reports", "width:280px;margin-right:16px"),
        s.icon("bell", "bell", "margin-right:16px;color:var(--muted)"),
        s.avatar("me", "#f97316"),
    ]), "row", "position:absolute;left:240px;top:0;width:1200px;height:72px;padding:0 32px;background:var(--surface);border-bottom:1px solid var(--border)")
    stats = [("Revenue", "$48,920"), ("Orders", "1,284"), ("Customers", "932"), ("Refunds", "17")]
    cards = '<div class="row" style="position:absolute;left:272px;top:104px;gap:24px">' + "".join(
        s.el("panel", f"stat-{k.lower()}", s.text(f"stat-{k.lower()}-label", k, "", "muted") + s.text(f"stat-{k.lower()}-value", v, "font-size:28px;font-weight:700;margin-top:8px"), "card col", "width:270px;height:120px;padding:20px")
        for k, v in stats) + "</div>"
    rows = []
    for i, (who, what, st) in enumerate([("Maria Lopez", "Order #1042", "Paid"), ("Chen Wei", "Order #1041", "Pending"), ("Sam Patel", "Order #1040", "Paid"), ("Ana Silva", "Order #1039", "Refunded"), ("Tom Berg", "Order #1038", "Paid")]):
        rows.append(f'<div class="row" style="height:56px;padding:0 20px;gap:14px;border-top:1px solid var(--border)">{s.avatar(f"row{i}-avatar", AVATAR_COLORS[i])}{s.text(f"row{i}-name", who, "width:200px")}{s.text(f"row{i}-order", what, "width:180px", "muted")}<div style="flex:1"></div>{s.chip(f"row{i}-status", st)}</div>')
    table = s.el("panel", "orders", f'<div class="row" style="height:56px;padding:0 20px">{s.text("orders-title", "Recent orders", "font-size:16px;font-weight:700")}<div style="flex:1"></div>{s.button("orders-export", "Export", "btn secondary sm")}</div>' + "".join(rows), "card col", "position:absolute;left:272px;top:256px;width:1152px;height:340px;overflow:hidden")
    s.html = s.page(side + header + cards + table)
    return s


def web_complex() -> Scene:
    s = Scene("web-complex", 1440, 900)
    rnd = random.Random(7)
    header = s.el("panel", "topbar", "".join([
        s.icon("logo", "star", "color:var(--accent);width:24px;height:24px"),
        s.text("brand", "Forge", "font-size:17px;font-weight:700;margin:0 32px 0 10px"),
        *[s.text(f"tab-{t.lower()}", t, "margin-right:24px" + (";color:var(--accent);font-weight:600" if t == "Settings" else ""), "" if t == "Settings" else "muted") for t in ["Projects", "Builds", "Members", "Settings"]],
        '<div style="flex:1"></div>',
        s.input("search", "Search", "width:220px;margin-right:12px"),
        s.button("new", "New project", "btn sm"),
        s.avatar("me", "#8b5cf6", 32, "margin-left:12px"),
    ]), "row", "height:56px;padding:0 24px;background:var(--surface);border-bottom:1px solid var(--border)")
    nav = s.el("panel", "settings-nav", "".join(
        f'<div class="row" style="height:36px;padding:0 12px;border-radius:8px;{"background:var(--surface-2);" if i == 1 else ""}">{s.text("snav-" + w.lower().replace(" ", "-"), w)}</div>'
        for i, w in enumerate(["General", "Notifications", "Security", "Billing", "Integrations", "API keys", "Audit log"])
    ), "col", "position:absolute;left:24px;top:80px;width:220px;gap:4px")
    form_rows = []
    for i, (label, desc, on) in enumerate([("Email digests", "A summary of activity every morning", True), ("Build failures", "Notify me when a build on main fails", True), ("Mentions", "When someone mentions you in a comment", False), ("Deploys", "When a deploy finishes or rolls back", True), ("Weekly report", "Usage and spend for the last 7 days", False)]):
        form_rows.append(f'<div class="row" style="height:64px;border-top:1px solid var(--border);padding:0 20px"><div class="col">{s.text(f"pref{i}-label", label, "font-weight:600")}{s.text(f"pref{i}-desc", desc, "font-size:12px;margin-top:4px", "muted")}</div><div style="flex:1"></div>{s.toggle(f"pref{i}-toggle", on)}</div>')
    prefs = s.el("panel", "prefs-card", f'<div style="padding:18px 20px">{s.text("prefs-title", "Notification preferences", "font-size:16px;font-weight:700")}</div>' + "".join(form_rows), "card col", "position:absolute;left:268px;top:80px;width:720px;height:386px;overflow:hidden")
    fields = "".join(
        f'<div class="col" style="gap:6px">{s.text(f"field{i}-label", lab, "font-size:12px;font-weight:600")}{s.input(f"field{i}", ph, "width:316px")}</div>'
        for i, (lab, ph) in enumerate([("Slack webhook", "https://hooks.slack.com/…"), ("Email address", "ops@forge.dev"), ("Quiet hours from", "22:00"), ("Quiet hours to", "07:00")])
    )
    channels = s.el("panel", "channels-card", f'<div style="padding:18px 20px 0">{s.text("channels-title", "Channels", "font-size:16px;font-weight:700")}</div><div style="display:grid;grid-template-columns:316px 316px;gap:16px 48px;padding:16px 20px">{fields}</div><div class="row" style="padding:4px 20px 20px;gap:10px;justify-content:flex-end">{s.button("cancel", "Cancel", "btn secondary sm")}{s.button("save", "Save changes", "btn sm")}</div>', "card col", "position:absolute;left:268px;top:490px;width:720px;height:270px")
    act = []
    for i in range(7):
        who = rnd.choice(["Maria", "Chen", "Sam", "Ana", "Tom", "Iris", "Omar"])
        what = rnd.choice(["deployed web-app", "rotated an API key", "invited a member", "changed billing", "merged #%d" % rnd.randint(100, 999)])
        act.append(f'<div class="row" style="gap:10px;height:52px;padding:0 16px;border-top:1px solid var(--border)">{s.avatar(f"act{i}-avatar", AVATAR_COLORS[i % 7], 28)}<div class="col">{s.text(f"act{i}-who", who + " " + what, "font-size:13px")}{s.text(f"act{i}-when", f"{i + 2} min ago", "font-size:11px", "muted")}</div></div>')
    activity = s.el("panel", "activity-card", f'<div class="row" style="padding:16px">{s.text("activity-title", "Activity", "font-size:16px;font-weight:700")}<div style="flex:1"></div>{s.chip("activity-live", "Live")}</div>' + "".join(act), "card col", "position:absolute;left:1012px;top:80px;width:404px;height:440px;overflow:hidden")
    usage = s.el("panel", "usage-card", s.text("usage-title", "Usage this month", "font-size:16px;font-weight:700") + '<div class="row" style="gap:14px;margin-top:18px;align-items:flex-end">' + "".join(s.image(f"bar{i}", IMAGE_COLORS[i % 6], 34, 40 + 17 * ((i * 5) % 7), 4) for i in range(8)) + "</div>", "card col", "position:absolute;left:1012px;top:544px;width:404px;height:216px;padding:18px 20px")
    s.html = s.page(header + nav + prefs + channels + activity + usage)
    return s


def desktop_simple() -> Scene:
    s = Scene("desktop-simple", 1280, 800)
    dialog = s.el("panel", "dialog", "".join([
        s.icon("dialog-icon", "bell", "width:32px;height:32px;color:var(--warning)"),
        s.text("dialog-title", "Unsaved changes", "font-size:18px;font-weight:700;margin-top:16px"),
        s.text("dialog-body", "Save the changes to report.xlsx before closing?", "margin-top:8px", "muted"),
        '<div class="row" style="gap:10px;margin-top:28px;justify-content:flex-end;width:100%">' + s.button("dont-save", "Don't save", "btn secondary") + s.button("cancel", "Cancel", "btn secondary") + s.button("save", "Save", "btn") + "</div>",
    ]), "card col", "position:absolute;left:400px;top:260px;width:480px;height:236px;padding:28px")
    s.html = s.page(dialog)
    return s


def desktop_medium() -> Scene:
    s = Scene("desktop-medium", 1280, 800)
    toolbar = s.el("panel", "toolbar", "".join([
        s.button("compose", "Compose", "btn sm"),
        *[s.icon(f"tb-{w}", w, "margin-left:18px;color:var(--muted)") for w in ["folder", "star", "bell"]],
        '<div style="flex:1"></div>',
        s.input("search", "Search mail", "width:300px;height:30px"),
    ]), "row", "height:52px;padding:0 16px;background:var(--surface-2);border-bottom:1px solid var(--border)")
    folders = s.el("panel", "folders", "".join(
        f'<div class="row" style="height:34px;padding:0 14px;gap:10px;{"background:var(--chip);" if i == 0 else ""}border-radius:6px">{s.icon("f-icon-" + n.lower(), ic)}{s.text("f-" + n.lower(), n)}<div style="flex:1"></div>{s.chip("f-count-" + n.lower(), str(c)) if c else ""}</div>'
        for i, (n, ic, c) in enumerate([("Inbox", "mail", 12), ("Starred", "star", 0), ("Sent", "folder", 0), ("Drafts", "file", 3), ("Archive", "folder", 0)])
    ), "col", "position:absolute;left:0;top:52px;width:220px;height:748px;padding:10px;gap:2px;background:var(--surface);border-right:1px solid var(--border)")
    mails = []
    for i, (who, subj) in enumerate([("GitHub", "[forge] Build #812 failed on main"), ("Lena Fischer", "Q3 planning — draft agenda"), ("Stripe", "Your payout of €4,210 is on the way"), ("Ravi Kumar", "Re: API rate limits"), ("Notion", "3 updates in Engineering wiki"), ("Mia Wong", "Lunch on Thursday?")]):
        mails.append(f'<div class="row" style="height:64px;padding:0 14px;gap:12px;border-bottom:1px solid var(--border);{"background:var(--surface-2);" if i == 1 else ""}">{s.avatar(f"m{i}-avatar", AVATAR_COLORS[i])}<div class="col">{s.text(f"m{i}-from", who, "font-weight:600")}{s.text(f"m{i}-subject", subj, "font-size:12px;margin-top:2px", "muted")}</div></div>')
    listing = s.el("panel", "mail-list", "".join(mails), "col", "position:absolute;left:220px;top:52px;width:380px;height:748px;background:var(--surface);border-right:1px solid var(--border)")
    reader = s.el("panel", "reader", "".join([
        s.text("r-subject", "Q3 planning — draft agenda", "font-size:20px;font-weight:700"),
        f'<div class="row" style="gap:10px;margin-top:16px">{s.avatar("r-avatar", AVATAR_COLORS[1])}<div class="col">{s.text("r-from", "Lena Fischer", "font-weight:600")}{s.text("r-to", "to me, Ravi, Mia", "font-size:12px", "muted")}</div></div>',
        s.el("divider", "r-divider", "", "divider", "margin:20px 0"),
        s.text("r-line1", "Hi all — here is a first cut of the agenda for Monday.", ""),
        s.text("r-line2", "Please add anything I missed before Friday.", "margin-top:8px"),
        f'<div class="row" style="gap:10px;margin-top:28px">{s.button("reply", "Reply", "btn sm")}{s.button("forward", "Forward", "btn secondary sm")}</div>',
    ]), "col", "position:absolute;left:600px;top:52px;width:680px;height:748px;padding:28px 32px;background:var(--bg)")
    s.html = s.page(toolbar + folders + listing + reader)
    return s


def desktop_complex() -> Scene:
    s = Scene("desktop-complex", 1280, 800)
    title = s.el("panel", "titlebar", "".join([
        *[f'<div style="width:12px;height:12px;border-radius:6px;background:{c};margin-right:8px"></div>' for c in ["#ff5f57", "#febc2e", "#28c840"]],
        s.text("window-title", "forge — main.ts", "margin-left:auto;margin-right:auto;font-size:12px", "muted"),
    ]), "row", "height:32px;padding:0 12px;background:var(--surface-2);border-bottom:1px solid var(--border)")
    activity = s.el("panel", "activitybar", "".join(s.icon(f"ab-{w}", w, "color:var(--muted);margin:12px 0") for w in ["file", "search", "folder", "chart", "gear"]), "col", "position:absolute;left:0;top:32px;width:48px;height:742px;align-items:center;background:var(--surface-2);border-right:1px solid var(--border)")
    tree_items = [("src", 0, "folder"), ("main.ts", 1, "file"), ("app.ts", 1, "file"), ("routes", 1, "folder"), ("users.ts", 2, "file"), ("builds.ts", 2, "file"), ("util", 1, "folder"), ("log.ts", 2, "file"), ("tests", 0, "folder"), ("main.test.ts", 1, "file"), ("package.json", 0, "file"), ("README.md", 0, "file")]
    tree = s.el("panel", "explorer", s.text("explorer-title", "EXPLORER", "font-size:11px;font-weight:700;margin:10px 14px", "muted") + "".join(
        f'<div class="row" style="height:26px;padding-left:{14 + 16 * d}px;gap:8px;{"background:var(--chip);" if n == "main.ts" else ""}">{s.icon("t-icon-" + n, ic, "width:16px;height:16px;color:var(--muted)")}{s.text("t-" + n, n, "font-size:13px")}</div>'
        for n, d, ic in tree_items
    ), "col", "position:absolute;left:48px;top:32px;width:240px;height:742px;background:var(--surface);border-right:1px solid var(--border)")
    tabs = s.el("panel", "tabs", "".join(
        f'<div class="row" style="height:36px;padding:0 14px;gap:8px;border-right:1px solid var(--border);{"background:var(--bg);" if i == 0 else ""}">{s.icon("tab-icon-" + n, "file", "width:14px;height:14px;color:var(--muted)")}{s.text("tab-" + n, n, "font-size:12px")}</div>'
        for i, n in enumerate(["main.ts", "app.ts", "users.ts"])
    ), "row", "position:absolute;left:288px;top:32px;width:992px;height:36px;background:var(--surface-2);border-bottom:1px solid var(--border)")
    code_lines = ["import { createApp } from './app';", "import { log } from './util/log';", "", "const port = Number(process.env.PORT ?? 8080);", "const app = createApp({ cors: true });", "", "app.listen(port, () => {", "  log.info(`listening on ${port}`);", "});"]
    code = s.el("panel", "editor", "".join(
        f'<div class="row" style="height:22px">{s.text(f"ln{i + 1}", str(i + 1), "width:36px;text-align:right;margin-right:20px;font:12px Menlo,monospace", "muted")}{s.text(f"code{i + 1}", line, "font:13px Menlo,monospace;white-space:pre") if line else ""}</div>'
        for i, line in enumerate(code_lines)
    ), "col", "position:absolute;left:288px;top:68px;width:992px;height:420px;padding-top:12px;background:var(--bg)")
    term = s.el("panel", "terminal", "".join([
        f'<div class="row" style="height:34px;gap:18px;padding:0 14px;border-bottom:1px solid var(--border)">{s.text("term-tab-terminal", "TERMINAL", "font-size:11px;font-weight:700")}{s.text("term-tab-problems", "PROBLEMS", "font-size:11px", "muted")}{s.chip("problems-count", "2")}<div style="flex:1"></div>{s.icon("term-plus", "plus", "width:16px;height:16px;color:var(--muted)")}</div>',
        *[s.text(f"term{i}", t, "font:12px Menlo,monospace;margin:6px 14px 0;display:block;white-space:pre") for i, t in enumerate(["$ npm test", "PASS  tests/main.test.ts (4 tests)", "Tests: 4 passed, 4 total", "$ "])],
    ]), "col", "position:absolute;left:288px;top:488px;width:992px;height:286px;background:var(--surface);border-top:1px solid var(--border)")
    status = s.el("panel", "statusbar", "".join([
        s.text("st-branch", "main", "font-size:11px;color:#fff;margin-right:16px"),
        s.text("st-errors", "0 errors, 2 warnings", "font-size:11px;color:#fff"),
        '<div style="flex:1"></div>',
        s.text("st-pos", "Ln 8, Col 31", "font-size:11px;color:#fff;margin-right:16px"),
        s.text("st-lang", "TypeScript", "font-size:11px;color:#fff"),
    ]), "row", "position:absolute;left:0;top:774px;width:1280px;height:26px;padding:0 12px;background:var(--accent)")
    s.html = s.page(title + activity + tree + tabs + code + term + status)
    return s


def mobile_simple() -> Scene:
    s = Scene("mobile-simple", 390, 844)
    body = '<div class="col" style="padding:96px 24px 0">' + "".join([
        s.icon("logo", "star", "width:48px;height:48px;color:var(--accent)"),
        s.text("title", "Welcome back", "font-size:28px;font-weight:800;margin-top:24px"),
        s.text("subtitle", "Sign in to continue", "margin-top:6px", "muted"),
        s.input("email", "Email", "margin-top:36px;height:48px;width:342px"),
        s.input("password", "Password", "margin-top:14px;height:48px;width:342px"),
        s.text("forgot", "Forgot password?", "margin-top:14px;align-self:flex-end;color:var(--accent);font-weight:600"),
        s.button("signin", "Sign in", "btn", "margin-top:28px;height:50px;width:342px;border-radius:12px"),
        s.text("signup", "No account? Create one", "margin-top:24px;align-self:center", "muted"),
    ]) + "</div>"
    s.html = s.page(body)
    return s


def mobile_medium() -> Scene:
    s = Scene("mobile-medium", 390, 844)
    top = s.el("panel", "appbar", s.text("app-title", "Feed", "font-size:22px;font-weight:800") + '<div style="flex:1"></div>' + s.icon("app-search", "search") + s.icon("app-bell", "bell", "margin-left:18px"), "row", "height:60px;padding:0 20px;background:var(--surface);border-bottom:1px solid var(--border)")
    posts = []
    for i, (who, txt, col) in enumerate([("Nora", "Sunrise over the bay this morning.", IMAGE_COLORS[1]), ("Diego", "New ceramics batch is out of the kiln!", IMAGE_COLORS[0])]):
        posts.append(s.el("panel", f"post{i}", "".join([
            f'<div class="row" style="gap:10px">{s.avatar(f"p{i}-avatar", AVATAR_COLORS[i])}<div class="col">{s.text(f"p{i}-name", who, "font-weight:700")}{s.text(f"p{i}-time", f"{i + 1}h", "font-size:12px", "muted")}</div></div>',
            s.text(f"p{i}-text", txt, "margin-top:12px;display:block"),
            s.image(f"p{i}-photo", col, 326, 150, 10).replace("flex:none", "flex:none;margin-top:12px"),
            f'<div class="row" style="gap:18px;margin-top:12px">{s.icon(f"p{i}-like", "star", "color:var(--muted)")}{s.icon(f"p{i}-share", "mail", "color:var(--muted)")}</div>',
        ]), "card col", f"margin:{16 if i == 0 else 12}px 16px 0;padding:16px;width:358px"))
    nav = s.el("panel", "tabbar", "".join(s.icon(f"tab-{w}", w, "color:" + ("var(--accent)" if w == "home" else "var(--muted)")) for w in ["home", "search", "plus", "bell", "user"]), "row", "position:absolute;left:0;top:780px;width:390px;height:64px;justify-content:space-around;background:var(--surface);border-top:1px solid var(--border)")
    s.html = s.page(top + "".join(posts) + nav)
    return s


def mobile_complex() -> Scene:
    s = Scene("mobile-complex", 390, 844)
    top = s.el("panel", "appbar", s.icon("back", "home") + s.text("app-title", "Cart", "font-size:20px;font-weight:800;margin-left:14px") + '<div style="flex:1"></div>' + s.chip("cart-count", "4 items"), "row", "height:56px;padding:0 16px;background:var(--surface);border-bottom:1px solid var(--border)")
    chips = '<div class="row" style="gap:8px;padding:14px 16px 4px">' + "".join(s.chip(f"filter-{c.lower()}", c) for c in ["All", "Produce", "Dairy", "Bakery"]) + "</div>"
    items = []
    for i, (name, price, qty) in enumerate([("Organic apples", "€3.40", 2), ("Greek yogurt", "€2.15", 1), ("Sourdough loaf", "€4.80", 1), ("Free-range eggs", "€3.95", 1)]):
        items.append(s.el("panel", f"item{i}", "".join([
            s.image(f"i{i}-thumb", IMAGE_COLORS[(i + 2) % 6], 56, 56, 10),
            f'<div class="col" style="margin-left:12px">{s.text(f"i{i}-name", name, "font-weight:600")}{s.text(f"i{i}-price", price, "font-size:12px;margin-top:4px", "muted")}</div>',
            '<div style="flex:1"></div>',
            s.button(f"i{i}-minus", "−", "btn secondary sm", "width:28px;padding:0"),
            s.text(f"i{i}-qty", str(qty), "width:28px;text-align:center"),
            s.button(f"i{i}-plus", "+", "btn sm", "width:28px;padding:0"),
        ]), "card row", "margin:10px 16px 0;padding:12px;height:80px"))
    promo = s.el("panel", "promo", s.input("promo-code", "Promo code", "flex:1;height:40px") + s.button("promo-apply", "Apply", "btn secondary", "height:40px;margin-left:10px"), "row", "margin:16px 16px 0")
    summary = s.el("panel", "summary", "".join(
        f'<div class="row" style="height:26px">{s.text(f"sum-{k.lower()}", k, "", "muted" if k != "Total" else "")}<div style="flex:1"></div>{s.text(f"sum-{k.lower()}-value", v, "font-weight:700" if k == "Total" else "")}</div>'
        for k, v in [("Subtotal", "€17.70"), ("Delivery", "€2.00"), ("Total", "€19.70")]
    ), "card col", "margin:16px 16px 0;padding:12px 16px")
    checkout = s.button("checkout", "Checkout · €19.70", "btn", "position:absolute;left:16px;top:716px;width:358px;height:52px;border-radius:14px")
    nav = s.el("panel", "tabbar", "".join(s.icon(f"tab-{w}", w, "color:" + ("var(--accent)" if w == "cart" else "var(--muted)")) for w in ["home", "search", "cart", "user"]), "row", "position:absolute;left:0;top:784px;width:390px;height:60px;justify-content:space-around;background:var(--surface);border-top:1px solid var(--border)")
    s.html = s.page(top + chips + "".join(items) + promo + summary + checkout + nav)
    return s


SCENES = [web_simple, web_medium, web_complex, desktop_simple, desktop_medium, desktop_complex, mobile_simple, mobile_medium, mobile_complex]

if __name__ == "__main__":
    for make in SCENES:
        scene = make()
        (SCENES_DIR / f"{scene.name}.html").write_text(scene.html, encoding="utf-8")
        print(scene.name, scene.width, scene.height, len(scene.used), "scored elements")
