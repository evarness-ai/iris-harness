#!/usr/bin/env python3
"""Generate the presentation export ``docs/architecture/iris-architecture.drawio``
from a code-defined model that mirrors the canonical Mermaid in
``docs/architecture/ARCHITECTURE.md``.

The Mermaid blocks remain the diffable source of truth; this script makes the draw.io
export *reproducible* — re-run it after the Mermaid changes instead of hand-editing
mxGraph XML. Output is plain (uncompressed) mxGraph so it stays diff-friendly and the
diagrams CI check (`scripts/check_diagrams.py`) validates it.

    poetry run python scripts/gen_architecture_drawio.py

History: the prior hand-authored 16-page export is archived as
``iris-architecture.v0.drawio``.
"""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parents[1] / "docs" / "architecture" / "iris-architecture.drawio"

# group -> (fill, stroke)
PALETTE = {
    "io": ("#dae8fc", "#6c8ebf"),
    "core": ("#d5e8d4", "#82b366"),
    "gov": ("#ffe6cc", "#d79b00"),
    "cap": ("#fff2cc", "#d6b656"),
    "cog": ("#e1d5e7", "#9673a6"),
    "store": ("#f8cecc", "#b85450"),
    "aux": ("#f5f5f5", "#666666"),
}


class Page:
    def __init__(self, name: str, pid: str, w: int = 1100, h: int = 760) -> None:
        self.name, self.pid, self.w, self.h = name, pid, w, h
        self.cells: list[str] = []

    def node(self, nid, label, x, y, w=220, h=50, group="aux", shape="rounded"):
        fill, stroke = PALETTE[group]
        style = (
            f"rounded={'1' if shape == 'rounded' else '0'};whiteSpace=wrap;html=1;"
            f"fillColor={fill};strokeColor={stroke};"
        )
        if shape == "cylinder":
            style = f"shape=cylinder3;whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
        if shape == "title":
            style = "text;html=1;fontSize=16;fontStyle=1;align=left;verticalAlign=middle;"
        self.cells.append(
            f'<mxCell id="{nid}" value="{escape(label)}" style="{style}" vertex="1" parent="1">'
            f'<mxGeometry x="{x}" y="{y}" width="{w}" height="{h}" as="geometry"/></mxCell>'
        )
        return nid

    def edge(self, src, dst, label="", dashed=False):
        style = "edgeStyle=orthogonalEdgeStyle;rounded=0;html=1;endArrow=block;"
        if dashed:
            style += "dashed=1;"
        eid = f"e_{src}_{dst}_{len(self.cells)}"
        self.cells.append(
            f'<mxCell id="{eid}" value="{escape(label)}" style="{style}" edge="1" parent="1" '
            f'source="{src}" target="{dst}"><mxGeometry relative="1" as="geometry"/></mxCell>'
        )

    def xml(self) -> str:
        body = "\n        ".join(self.cells)
        return (
            f'  <diagram name="{escape(self.name)}" id="{self.pid}">\n'
            f'    <mxGraphModel dx="1100" dy="760" grid="1" gridSize="10" guides="1" '
            f'tooltips="1" connect="1" arrows="1" fold="1" page="1" pageWidth="{self.w}" '
            f'pageHeight="{self.h}" math="0" shadow="0">\n'
            f"      <root>\n"
            f'        <mxCell id="0"/>\n'
            f'        <mxCell id="1" parent="0"/>\n'
            f"        {body}\n"
            f"      </root>\n"
            f"    </mxGraphModel>\n"
            f"  </diagram>\n"
        )


pages: list[Page] = []


# ── 00 High-level overview (ARCHITECTURE.md §0) ──────────────────────────────
p = Page("00 - High-level overview", "hl0")
p.node("t", "IRIS — high-level architecture", 40, 10, 600, 30, "aux", "title")
p.node(
    "ui",
    "Channels & services: Web UI · CLI · Telegram · IRIS API :8003 · Governor :8080",
    40,
    60,
    680,
    40,
    "io",
)
p.node("rt", "IrisRuntime (composition root, bootstrap.py)", 40, 130, 320, 40, "core")
p.node(
    "core",
    "Agentic core: IntentRouter → TaskPlanner → ReAct → AgentExecutor → ResponseCurator",
    40,
    200,
    680,
    40,
    "core",
)
p.node(
    "gov",
    "Governance kernel — mandatory passage: classify · egress · vault · hooks · evaluator · approvals · audit",
    40,
    270,
    680,
    50,
    "gov",
)
p.node(
    "agents",
    "9 agents: email · finance · planner · system · calendar · rag · filemanager · coding · code_exec",
    40,
    350,
    420,
    50,
    "cap",
)
p.node("tools", "Tools · Sandbox (Docker / gVisor)", 490, 350, 230, 50, "cap")
p.node("cog", "Cognition: Memory · Wiki · Learning (+ replay-eval)", 40, 430, 320, 40, "cog")
p.node("tiers", "TierRouter → local (Ollama / LM Studio) + cloud", 390, 430, 330, 40, "cog")
p.node("stores", "Data stores: SQLite · ChromaDB · Fernet vault", 40, 510, 320, 40, "store")
p.node("obs", "Observability: OpenTelemetry (OTLP export) · session log", 390, 510, 330, 40, "aux")
p.edge("ui", "rt")
p.edge("rt", "core")
p.edge("core", "agents")
p.edge("agents", "tools")
p.edge("core", "cog")
p.edge("core", "tiers")
p.edge("core", "gov", "every LLM/tool call", dashed=True)
p.edge("agents", "gov", "governed", dashed=True)
p.edge("cog", "stores")
p.edge("agents", "stores")
p.edge("gov", "stores")
p.edge("rt", "obs", "spans", dashed=True)
pages.append(p)


# ── 00.1 Harness topology (§0.1) ─────────────────────────────────────────────
p = Page("00.1 - Harness topology", "hl1")
p.node("t", "Harness topology (per-turn path + autonomy)", 40, 10, 600, 30, "aux", "title")
p.node("gw", "Channel Gateway", 40, 60, 150, 40, "io")
p.node("api", "IRIS API", 210, 60, 120, 40, "io")
p.node("rt", "Runtime", 350, 60, 120, 40, "core")
p.node("ir", "Intent Router", 490, 60, 130, 40, "core")
p.node("mem", "Memory Retriever", 40, 140, 160, 40, "cog")
p.node("tp", "Task Planner", 220, 140, 130, 40, "core")
p.node("ae", "Agent Executor", 370, 140, 140, 40, "core")
p.node(
    "agent",
    "Agent: email · finance · planner · system · calendar · rag · filemanager · coding · code_exec",
    40,
    220,
    470,
    50,
    "cap",
)
p.node("tool", "Tool: web · memory · wiki · rag · MCP", 540, 220, 240, 50, "cap")
p.node("rc", "Response Curator", 40, 300, 160, 40, "core")
p.node("gov", "Governance (guard at every LLM/tool call)", 850, 130, 240, 40, "gov")
p.node("llm", "LLM Call (tier-routed)", 850, 60, 240, 40, "cog")
p.node("hb", "HeartbeatScheduler (22)", 40, 380, 200, 40, "aux")
p.node("bus", "EventBus", 270, 380, 130, 40, "aux")
p.node("learn", "LearningEngine · MissionEngine · SystemHealth", 430, 380, 350, 40, "aux")
p.node("pg", "Prompt Guard 2 (G1/G2, local)", 850, 300, 240, 40, "gov")
p.node("lg", "Llama Guard 3 + curator judges (G3)", 850, 360, 240, 40, "gov")
p.edge("gw", "api")
p.edge("api", "rt")
p.edge("rt", "ir")
p.edge("ir", "gov", "guard", dashed=True)
p.edge("gov", "llm", "control", dashed=True)
p.edge("ir", "mem")
p.edge("mem", "tp")
p.edge("tp", "ae")
p.edge("ae", "agent")
p.edge("agent", "gov", "guard", dashed=True)
p.edge("gov", "tool", "control", dashed=True)
p.edge("agent", "rc")
p.edge("rc", "api")
p.edge("hb", "agent", "triggers", dashed=True)
p.edge("agent", "bus", "events", dashed=True)
p.edge("learn", "ae", "observe", dashed=True)
p.edge("ir", "pg", "inbound screen", dashed=True)
p.edge("rc", "lg", "output safety", dashed=True)
pages.append(p)


# ── 00.2 Harness primitives (§0.2) ───────────────────────────────────────────
p = Page("00.2 - Harness primitives", "hl2")
p.node("t", "Harness primitives (anatomy)", 40, 10, 600, 30, "aux", "title")
p.node(
    "ctx",
    "Context — MemoryContext per turn (identity · facts · recent turns · provenance)",
    40,
    60,
    360,
    50,
    "cog",
)
p.node(
    "mem",
    "Memory — SemanticIndex (5 ChromaDB collections) + retriever · compaction · wiki · episodic",
    40,
    130,
    360,
    50,
    "cog",
)
p.node(
    "loop",
    "The loop — AgenticCore ReAct: IntentRouter → TaskPlanner → ReAct → ResponseCurator",
    440,
    90,
    360,
    60,
    "core",
)
p.node(
    "tools",
    "Tools — built-in/core · skill tools · MCP (one flat list, see 02.5)",
    440,
    200,
    360,
    50,
    "cap",
)
p.node(
    "skills",
    "Skills — SkillRegistry + SemanticSkillRouter (cosine >= 0.45); brief auto-wrap",
    40,
    280,
    360,
    50,
    "cap",
)
p.node(
    "gov",
    "Governance — mandatory passage: classify · egress · vault · approvals · audit · threat guards",
    440,
    290,
    360,
    60,
    "gov",
)
p.node("hb", "Heartbeats (22) + EventBus + Missions", 40, 380, 360, 50, "aux")
p.node(
    "learn",
    "Learning — signals → crystallizer → replay-eval → experiments",
    40,
    450,
    360,
    50,
    "aux",
)
p.node("obs", "Observability — OpenTelemetry (OTLP export) · session log", 440, 420, 360, 50, "aux")
p.edge("mem", "ctx")
p.edge("ctx", "loop")
p.edge("loop", "tools")
p.edge("skills", "tools", "surfaced AS tools")
p.edge("loop", "gov", "every LLM/tool call", dashed=True)
p.edge("hb", "loop", "triggers", dashed=True)
p.edge("loop", "learn")
p.edge("learn", "mem")
p.edge("loop", "obs", "spans", dashed=True)
pages.append(p)


# ── 01 System context (§1) ───────────────────────────────────────────────────
p = Page("01 - System context", "ctx")
p.node("t", "System context (L1)", 40, 10, 400, 30, "aux", "title")
p.node("user", "User", 60, 70, 120, 40, "io")
p.node("ch", "Channels: Web UI · CLI · Telegram · Console", 240, 70, 320, 40, "io")
p.node("rt", "IrisRuntime (composition root, bootstrap.py)", 240, 150, 320, 50, "core")
p.node("local", "Local LLMs: Ollama / LM Studio (Tier 1/2/3 + router)", 40, 250, 320, 50, "cog")
p.node(
    "cloud",
    "Cloud LLMs (egress-gated): GitHub Models · OpenRouter · Anthropic · Copilot",
    400,
    250,
    360,
    50,
    "cog",
)
p.node(
    "ext",
    "External: Gmail · Google Drive/Calendar · MCP · iCloud/Photos (read)",
    40,
    340,
    360,
    50,
    "cap",
)
p.node("stores", "Local stores: SQLite · ChromaDB · Fernet vault", 440, 340, 320, 50, "store")
p.node("obs", "Observability: OpenTelemetry (OTLP export) · session log", 240, 430, 320, 40, "aux")
p.edge("user", "ch")
p.edge("ch", "rt")
p.edge("rt", "local", "private tiers first")
p.edge("rt", "cloud", "non-private, egress-gated")
p.edge("rt", "ext")
p.edge("rt", "stores")
p.edge("rt", "obs", "spans", dashed=True)
pages.append(p)


# ── 02 Containers / subsystems (§2) ──────────────────────────────────────────
p = Page("02 - Containers / subsystems", "cont")
p.node("t", "Containers / subsystems (L2)", 40, 10, 500, 30, "aux", "title")
p.node(
    "ch",
    "Channels / CLI / services (iris · iris-code · Governor :8080 · IRIS API :8003)",
    40,
    60,
    720,
    40,
    "io",
)
p.node(
    "short",
    "Deterministic short-circuits: reminder · time/date · routine-mgmt · routine-authoring · calendar writes",
    40,
    120,
    720,
    40,
    "core",
)
p.node(
    "core",
    "Agentic core (5 stages): IntentRouter → TaskPlanner → ReAct → AgentExecutor → ResponseCurator (in-process multi-headed judge)",
    40,
    180,
    720,
    50,
    "core",
)
p.node(
    "gov",
    "Governance kernel (mandatory passage): hooks (PreClassify/PreLLMCall/PreToolUse/PostToolUse/PostStep) · classifier+egress · vault+broker · evaluator · approvals (in-chat + Action Center) · audit+archive · threat G1/G2/G3 · MCP signing",
    40,
    250,
    720,
    80,
    "gov",
)
p.node(
    "tiers",
    "TierRouter (1/2/3 + router) + OllamaArbiter · ResourceGovernor",
    40,
    350,
    360,
    50,
    "cog",
)
p.node(
    "agents",
    "Agents: email · finance · planner · system · calendar (registered) · coding (13-node) · code_exec · rag/filemanager (skills/heartbeats)",
    40,
    420,
    720,
    60,
    "cap",
)
p.node(
    "cog",
    "Memory & knowledge: MemoryRetriever+SemanticIndex · WikiEngine · LearningEngine/ExperimentLoop · SkillCrystallizer + replay-eval pre-flight · escalation judge · ProvenanceLedger",
    40,
    500,
    720,
    70,
    "cog",
)
p.node(
    "proactive",
    "Proactive autonomy: HeartbeatScheduler (22) · MissionEngine · ChannelGateway · SystemHealth",
    40,
    590,
    720,
    40,
    "aux",
)
p.node("stores", "Data stores (see page 03)", 40, 650, 360, 40, "store")
p.edge("ch", "short")
p.edge("ch", "core")
p.edge("core", "tiers")
p.edge("core", "gov", "every LLM/tool call", dashed=True)
p.edge("core", "agents")
p.edge("agents", "cog")
p.edge("proactive", "core")
p.edge("gov", "stores")
pages.append(p)


# ── 02.1 Agents & capabilities (§2.1) ────────────────────────────────────────
p = Page("02.1 - Agents & capabilities", "agents")
p.node(
    "t", "Agents & capabilities (how each of the 7+2 is realized)", 40, 10, 600, 30, "aux", "title"
)
p.node("intent", "IntentRouter → AgentExecutor", 40, 60, 280, 40, "core")
p.node(
    "sc",
    "Deterministic short-circuits (reminder · time/date · routine · calendar writes)",
    360,
    60,
    400,
    40,
    "core",
)
# registered
p.node("reg", "AgentExecutor-registered handlers", 40, 130, 350, 30, "aux", "title")
p.node("a_email", "email", 40, 170, 100, 40, "cap")
p.node("a_fin", "finance", 150, 170, 100, 40, "cap")
p.node("a_plan", "planner", 260, 170, 100, 40, "cap")
p.node("a_sys", "system", 40, 220, 100, 40, "cap")
p.node("a_cal", "calendar (reads)", 150, 220, 140, 40, "cap")
p.node("a_code", "coding_agent", 300, 220, 120, 40, "cap")
p.node("a_exec", "code_exec", 40, 270, 120, 40, "cap")
# non-registered
p.node(
    "nonreg", "Realized via deterministic / skills / heartbeats", 440, 130, 320, 30, "aux", "title"
)
p.node("a_calw", "calendar/reminder writes (short-circuit → calendar.db)", 440, 170, 320, 40, "cap")
p.node("a_rag", "rag (docs-search · grounded-qa)", 440, 220, 320, 40, "cap")
p.node("a_fm", "filemanager (custodian + catalog)", 440, 270, 320, 40, "cap")
p.node("stores", "per-agent stores (see page 03)", 40, 350, 300, 40, "store")
p.edge("intent", "a_email")
p.edge("sc", "a_calw")
p.edge("a_sys", "a_rag", "tool calls", dashed=True)
p.edge("a_plan", "a_email", "reads", dashed=True)
p.edge("a_plan", "a_cal", "reads", dashed=True)
p.edge("a_plan", "a_fin", "reads", dashed=True)
p.edge("a_code", "a_exec")
p.edge("a_email", "stores")
p.edge("a_calw", "stores")
pages.append(p)


# ── 02.2 Event bus & heartbeats (§2.2) ───────────────────────────────────────
p = Page("02.2 - Event bus & heartbeats", "bus")
p.node(
    "t", "Event-driven interconnections (EventBus + 21 heartbeats)", 40, 10, 620, 30, "aux", "title"
)
p.node(
    "hb",
    "Heartbeats (22): email_sweep · finance_monitor · finance_ingest · reminder_tick · notification_reminder_tick · calendar_mirror · meeting_prep · filemanager_* · wiki_lint · learning_tick · crystallize_tick · sandbox_preflight_tick · learning_analysis_tick · experiment_remeasure_tick · escalation_priors_tick · routine_tick · routine_reflection_tick · pressure_tick · health_tick · mission_proposal_tick",
    40,
    60,
    720,
    110,
    "aux",
)
p.node("email", "email", 40, 200, 120, 40, "cap")
p.node("finance", "finance", 180, 200, 120, 40, "cap")
p.node("reminders", "calendar / reminders", 320, 200, 180, 40, "cap")
p.node("fm", "FileManager", 520, 200, 140, 40, "cap")
p.node("bus", "EventBus", 300, 280, 160, 50, "aux")
p.node("triage", "EmailTriage", 40, 360, 140, 40, "cap")
p.node("wiki", "WikiEngine", 200, 360, 140, 40, "cog")
p.node("brief", "briefing skill", 360, 360, 140, 40, "cap")
p.node("chan", "ChannelGateway → Console/Telegram", 520, 360, 240, 40, "io")
p.node("learn", "LearningEngine · SystemHealth", 40, 440, 280, 40, "aux")
p.edge("email", "bus", "email.new_arrived")
p.edge("finance", "bus", "finance.alert")
p.edge("reminders", "bus", "reminder.fired")
p.edge("bus", "triage", "→ triage")
p.edge("bus", "wiki", "→ wiki ingest")
p.edge("bus", "brief", "→ morning brief")
p.edge("bus", "chan", "→ channels")
p.edge("hb", "learn", "learning/health ticks", dashed=True)
pages.append(p)


# ── 02.5 Tools & skills (§2.5) ───────────────────────────────────────────────
p = Page("02.5 - Tools & skills (capability surface)", "toolskill")
p.node(
    "t",
    "Tools & skills — the capability surface (3 sources, one flat tool list)",
    40,
    10,
    760,
    30,
    "aux",
    "title",
)
p.node("msg", "user message", 40, 70, 180, 40, "io")
p.node("loop", "ReAct loop (offered one flat tool list)", 40, 150, 300, 50, "core")
# source 1
p.node("s1", "1 · Built-in / core tools (always available)", 380, 60, 380, 30, "aux", "title")
p.node(
    "bt",
    "web_search · code_exec · wiki_search · memory_search · retrieval · stock_quote · iris_doc · agents · system_health · pending_actions · learning_*",
    380,
    95,
    380,
    70,
    "cap",
)
# source 2
p.node("s2", "2 · Skill tools (config/skills/*)", 380, 185, 380, 30, "aux", "title")
p.node(
    "reg",
    "SkillRegistry.discover() — manifest.yaml + tools.py + agent.md",
    380,
    220,
    380,
    40,
    "cap",
)
p.node(
    "router",
    "SemanticSkillRouter — cosine >= 0.45 vs manifest (ONNX MiniLM)",
    380,
    270,
    380,
    40,
    "cap",
)
p.node("brief", "kind: brief → auto-wrapped callable tool (slot-filling)", 380, 320, 380, 40, "cap")
p.node("nonbrief", "non-brief → bound BaseTool invoked", 380, 370, 380, 40, "cap")
# source 3
p.node("s3", "3 · MCP tools (external servers)", 380, 430, 380, 30, "aux", "title")
p.node("bridge", "MCPBridge (mcp-servers.yaml)", 380, 465, 380, 40, "cap")
p.node("sign", "Ed25519 signing + per-(persona,server,tool) allowlist", 380, 515, 380, 40, "cap")
# harness
p.node(
    "gov",
    "Governance — PreToolUse/PostToolUse (allowlist · persona surface · signing · threat-G2)",
    40,
    280,
    300,
    70,
    "gov",
)
p.node("prov", "ProvenanceLedger → grounding judge", 40, 380, 300, 50, "cog")
p.node(
    "direct",
    "Answer directly — no tool (math · definitions · well-known facts · conversation; issue 0027)",
    40,
    460,
    300,
    60,
    "core",
)
p.edge("msg", "loop")
p.edge("loop", "direct", "knows it")
p.edge("bt", "loop")
p.edge("msg", "router", "semantic match", dashed=True)
p.edge("reg", "router")
p.edge("router", "brief")
p.edge("router", "nonbrief")
p.edge("brief", "loop")
p.edge("nonbrief", "loop")
p.edge("bridge", "sign")
p.edge("sign", "loop")
p.edge("loop", "gov", "every tool call", dashed=True)
p.edge("loop", "prov", "retrieval-class output", dashed=True)
pages.append(p)


# ── 03 Data stores (§4) ──────────────────────────────────────────────────────
p = Page("03 - Data stores (ownership)", "stores")
p.node("t", "Data stores (ownership)", 40, 10, 400, 30, "aux", "title")
p.node(
    "g",
    "Governance (~/.local/share/iris · ~/.config/iris · IRIS_HOME-relocatable)",
    40,
    60,
    700,
    30,
    "aux",
    "title",
)
p.node("a2", "audit.db — kernel governance ledger", 40, 100, 230, 50, "store", "cylinder")
p.node("a1", "data/audit.db — governor-guard + router", 290, 100, 230, 50, "store", "cylinder")
p.node("v", "vault.db — Fernet-over-SQLite", 540, 100, 200, 50, "store", "cylinder")
p.node("arch", "Parquet + zstd — cold audit archive", 40, 170, 230, 50, "store", "cylinder")
p.node(
    "gx",
    "checkpoints · cost-ledger · approvals · side_effects (created when features run)",
    290,
    170,
    450,
    50,
    "store",
    "cylinder",
)
p.node("c", "Cognition (ChromaDB + SQLite)", 40, 250, 400, 30, "aux", "title")
p.node(
    "chroma",
    "SemanticIndex — facts · signals · turns · wiki · episodic",
    40,
    290,
    330,
    50,
    "store",
    "cylinder",
)
p.node("docs", "iris_documents — data/chroma_docs (RAG)", 390, 290, 250, 50, "store", "cylinder")
p.node("learn", "data/learning.db", 40, 360, 200, 50, "store", "cylinder")
p.node("d", "Domain (SQLite)", 40, 440, 400, 30, "aux", "title")
p.node(
    "cal",
    "data/calendar.db — events + reminders (single writer, ADR-0075)",
    40,
    480,
    380,
    50,
    "store",
    "cylinder",
)
p.node("email", "data/email.db", 440, 480, 160, 50, "store", "cylinder")
p.node("fin", "data/finance.db", 620, 480, 150, 50, "store", "cylinder")
p.node("rag", "data/rag.db", 40, 550, 150, 50, "store", "cylinder")
p.node("fmc", "data/filemanager_catalog.db", 210, 550, 230, 50, "store", "cylinder")
p.node(
    "tasks",
    "data/tasks.db — goals · tasks · notification_reminders",
    460,
    550,
    310,
    50,
    "store",
    "cylinder",
)
p.node("mem", "MemoryStore · MissionStore · RoutineStore", 40, 620, 330, 50, "store", "cylinder")
pages.append(p)


# ── 04 Security guards & judges (§5) ─────────────────────────────────────────
p = Page("04 - Security guards & judges", "sec")
p.node("t", "Security guards & judges (model-driven, local)", 40, 10, 560, 30, "aux", "title")
p.node("user", "user turn", 40, 70, 140, 40, "io")
p.node("toolout", "retrieved / tool content", 40, 140, 200, 40, "cap")
p.node("g1", "G1 inbound guard — PromptGuardInboundHook (PreClassify)", 280, 70, 320, 40, "gov")
p.node(
    "g2", "G2 retrieved guard — PromptGuardRetrievedHook (PostToolUse)", 280, 140, 320, 40, "gov"
)
p.node(
    "detector",
    "ThreatDetector / ThreatClassifier (swappable · NullClassifier fallback)",
    280,
    210,
    320,
    40,
    "gov",
)
p.node("pg", "Llama Prompt Guard 2 (86M, transformers/ONNX, local)", 280, 280, 320, 40, "cog")
p.node("cur", "ResponseCurator — multi-headed judge (in-process)", 40, 360, 380, 30, "aux", "title")
p.node("det", "safety · schema · consistency (deterministic)", 40, 400, 330, 40, "core")
p.node("faith", "faithfulness (LLM judge)", 40, 450, 230, 40, "cog")
p.node("leak", "leak (LLM judge, default-on)", 290, 450, 230, 40, "cog")
p.node("ground", "grounding (LLM judge + provenance)", 40, 500, 280, 40, "cog")
p.node("g3", "G3 output_safety", 340, 500, 180, 40, "gov")
p.node("lg", "Llama Guard 3 (llama-guard3:1b, Ollama, local)", 540, 500, 240, 50, "cog")
p.node("mcp", "MCP server signing — Ed25519 + trust store", 640, 70, 240, 50, "gov")
p.node("br", "MCP bridge", 640, 150, 160, 40, "cap")
p.edge("user", "g1")
p.edge("toolout", "g2")
p.edge("g1", "detector")
p.edge("g2", "detector")
p.edge("detector", "pg")
p.edge("g3", "lg")
p.edge("faith", "ground", "tier-routed local judge", dashed=True)
p.edge("mcp", "br", "gates launch", dashed=True)
pages.append(p)


# ── Per-agent component pages (mirror agent-components.md) ───────────────────
# Each page: internal components in the left column, harness touchpoints (EXEC /
# GOV / BUS / HB / store) in the right column. Faithful to agent-components.md;
# the calendar page reflects the newer single-write authority (ADR-0075).


def agent_page(name, pid, title, internal, harness, edges):
    """internal/harness: list of (id, label, group). edges: (src, dst[, label[, dashed]])."""
    pg = Page(name, pid)
    pg.node("t", title, 40, 10, 760, 30, "aux", "title")
    y = 60
    for nid, label, group in internal:
        pg.node(nid, label, 40, y, 380, 44, group)
        y += 62
    y2 = 60
    for nid, label, group in harness:
        pg.node(nid, label, 640, y2, 240, 44, group)
        y2 += 62
    for e in edges:
        src, dst = e[0], e[1]
        label = e[2] if len(e) > 2 else ""
        dashed = e[3] if len(e) > 3 else False
        pg.edge(src, dst, label, dashed)
    pages.append(pg)


agent_page(
    "05 - email",
    "ag_email",
    "email agent (src/iris_personal/email/)",
    [
        ("fetch", "gmail_fetch / attachments", "cap"),
        ("store", "EmailStore (email.db)", "store"),
        ("sweep", "EmailSweepHandler", "cap"),
        ("triage", "EmailTriageClassifier", "cap"),
        ("knn", "KnnGateRunner + CategoryCentroid", "cap"),
        ("digest", "InboxDigest", "cap"),
        ("followup", "FollowupDetector", "cap"),
        ("ingest", "wiki_ingestion subscriber", "cap"),
    ],
    [
        ("EXEC", "EXEC (AgentExecutor)", "core"),
        ("HB", "HB: email_sweep", "aux"),
        ("BUS", "BUS (EventBus)", "aux"),
        ("WIKI", "WikiEngine", "cog"),
        ("GOV", "GOV (kernel)", "gov"),
    ],
    [
        ("EXEC", "digest"),
        ("HB", "sweep"),
        ("sweep", "store"),
        ("fetch", "store"),
        ("sweep", "BUS", "email.new_arrived"),
        ("BUS", "triage"),
        ("triage", "knn"),
        ("triage", "BUS", "email.classified"),
        ("BUS", "ingest"),
        ("ingest", "WIKI"),
        ("BUS", "followup"),
        ("digest", "GOV", "governed LLM", True),
    ],
)

agent_page(
    "06 - finance",
    "ag_fin",
    "finance agent (src/iris_personal/finance/) — extraction Tier-3 LOCAL only",
    [
        ("store", "FinanceStore (finance.db)", "store"),
        ("ingest", "ingest (PDF unlock)", "cap"),
        ("extract", "Tier3LocalExtractor", "cap"),
        ("cat", "FinanceKnnCategorizer", "cap"),
        ("monitor", "FinanceMonitorHandler", "cap"),
        ("bills", "bills / accounts / source_of_truth", "cap"),
        ("custody", "custody_migration → FileManager", "cap"),
    ],
    [
        ("EXEC", "EXEC (AgentExecutor)", "core"),
        ("HB", "HB: finance_monitor / finance_ingest", "aux"),
        ("BUS", "BUS (EventBus)", "aux"),
        ("FM", "FileManager catalog", "cap"),
        ("GOV", "GOV (kernel)", "gov"),
    ],
    [
        ("EXEC", "store"),
        ("ingest", "extract"),
        ("extract", "GOV", "Tier-3 LOCAL", True),
        ("extract", "store"),
        ("store", "cat"),
        ("HB", "monitor"),
        ("monitor", "bills"),
        ("monitor", "BUS", "finance.alert"),
        ("custody", "FM", "managed bytes", True),
    ],
)

agent_page(
    "07 - planner",
    "ag_plan",
    "planner agent (src/iris_harness/planner/) — read-only across other stores",
    [
        ("handler", "build_planner_handler", "cap"),
        ("compose", "plan.py (assemble)", "cap"),
        ("dp", "DailyPlan (models)", "cap"),
    ],
    [
        ("EXEC", "EXEC (AgentExecutor)", "core"),
        ("EM", "EmailStore", "store"),
        ("CAL", "calendar.db / reminders", "store"),
        ("FIN", "FinanceStore", "store"),
        ("GOV", "GOV (kernel)", "gov"),
    ],
    [
        ("EXEC", "handler"),
        ("handler", "compose"),
        ("compose", "dp"),
        ("compose", "EM", "reads", True),
        ("compose", "CAL", "reads", True),
        ("compose", "FIN", "reads", True),
        ("compose", "GOV", "governed narrative LLM", True),
    ],
)

agent_page(
    "08 - calendar / reminders",
    "ag_cal",
    "calendar / reminders — registered reads + deterministic writes (ADR-0075)",
    [
        ("digest", "build_calendar_digest (reads, registered)", "cap"),
        ("svc", "CalendarService", "cap"),
        ("wgov", "CalendarWriteGovernor (R2/R3 + 600s undo)", "cap"),
        ("store", "CalendarStore (calendar.db — events + reminders)", "store"),
        ("providers", "ICS / Google / Apple providers (write-back)", "cap"),
        ("remstore", "ReminderStore (tasks.db notification_reminders)", "store"),
        ("tick", "reminder_tick / notification_reminder_tick", "cap"),
    ],
    [
        ("EXEC", "EXEC (reads)", "core"),
        ("SC", "Deterministic write short-circuit", "core"),
        ("HB", "HB: calendar_mirror / meeting_prep / reminder ticks", "aux"),
        ("BUS", "BUS (EventBus)", "aux"),
        ("GW", "ChannelGateway → Console/Telegram", "io"),
        ("GOV", "GOV (kernel)", "gov"),
    ],
    [
        ("EXEC", "digest", "reads"),
        ("SC", "svc", "create event/invite/appt/reminder"),
        ("svc", "wgov"),
        ("wgov", "store"),
        ("svc", "providers", "Google/Apple write-back", True),
        ("HB", "tick"),
        ("tick", "remstore"),
        ("tick", "BUS", "reminder.fired"),
        ("BUS", "GW"),
        ("digest", "GOV", "governed", True),
    ],
)

agent_page(
    "09 - rag",
    "ag_rag",
    "rag agent (src/iris_harness/rag/) — skills via the ReAct tool loop",
    [
        ("loaders", "loaders + ocr (PDF/image)", "cap"),
        ("chunker", "chunker", "cap"),
        ("store", "DocumentStore (rag.db)", "store"),
        ("index", "DocumentIndex (iris_documents)", "store"),
        ("retrieve", "retrieve (Citation/RetrievedChunk)", "cap"),
        ("qa", "qa (GroundedAnswer)", "cap"),
        ("obs", "obsidian connector", "cap"),
    ],
    [
        ("SKILL", "skills: docs-search / grounded-qa (tool loop)", "cap"),
        ("PROV", "ProvenanceLedger → grounding judge", "cog"),
        ("GOV", "GOV (kernel)", "gov"),
    ],
    [
        ("loaders", "chunker"),
        ("chunker", "store"),
        ("store", "index"),
        ("SKILL", "retrieve"),
        ("retrieve", "index"),
        ("retrieve", "qa"),
        ("retrieve", "PROV", "retrieval-class output", True),
        ("SKILL", "GOV", "PreToolUse", True),
    ],
)

agent_page(
    "10 - filemanager",
    "ag_fm",
    "filemanager (src/iris_personal/filemanager/) — single custodian + catalog",
    [
        ("svc", "FileManagerService", "cap"),
        ("custodian", "FileCustodian (managed bytes)", "cap"),
        ("catalog", "FileCatalog (filemanager_catalog.db)", "store"),
        ("wgov", "FileWriteGovernor", "cap"),
        ("indexer", "FileIndexer (os_local/icloud)", "cap"),
        ("photos", "PhotosLibrary", "cap"),
        ("audit", "FileAuditLog", "store"),
        ("undo", "FileUndoStore", "store"),
    ],
    [
        ("EXEC", "EXEC (skills)", "core"),
        ("HB", "HB: retention / index / photos", "aux"),
        ("GOV", "GOV (kernel)", "gov"),
    ],
    [
        ("EXEC", "svc", "skills", True),
        ("svc", "custodian"),
        ("custodian", "catalog"),
        ("custodian", "wgov"),
        ("HB", "indexer"),
        ("indexer", "catalog"),
        ("HB", "photos"),
        ("photos", "catalog"),
        ("svc", "audit"),
        ("wgov", "GOV", "governed writes", True),
    ],
)

agent_page(
    "11 - system / general",
    "ag_sys",
    "system / general agent (src/iris_harness/agent/) — the agentic core pipeline",
    [
        ("intent", "IntentRouter", "core"),
        ("planner", "TaskPlanner", "core"),
        ("react", "ReAct loop", "core"),
        ("tools", "tools: web/memory/wiki · code_exec · skills", "cap"),
        ("curator", "ResponseCurator (in-process judges)", "core"),
    ],
    [
        ("EXEC", "EXEC (AgentExecutor)", "core"),
        ("MEM", "MemoryRetriever + wiki", "cog"),
        ("PROV", "ProvenanceLedger", "cog"),
        ("GOV", "GOV (kernel)", "gov"),
    ],
    [
        ("EXEC", "intent"),
        ("intent", "planner"),
        ("planner", "react"),
        ("react", "tools"),
        ("react", "curator"),
        ("intent", "GOV", "PreClassify", True),
        ("tools", "GOV", "Pre/PostToolUse", True),
        ("react", "GOV", "PreLLMCall (tier-routed)", True),
        ("react", "MEM"),
        ("react", "PROV"),
        ("PROV", "curator"),
    ],
)

agent_page(
    "12 - coding_agent",
    "ag_code",
    "coding_agent (src/iris_code/) — 13-node LangGraph, 6 personas",
    [
        ("agent", "CodingAgent", "cap"),
        ("pipeline", "Pipeline (13-node LangGraph)", "cap"),
        ("runner", "SubAgentRunner (6 personas)", "cap"),
        ("llm", "CodingLLMClient", "cap"),
        ("git", "git managers (dry-run/embedded/HTTP)", "cap"),
        ("mcp", "MCPBridge", "cap"),
        ("closeout", "CloseoutDecision / DocSyncReport", "cap"),
    ],
    [
        ("EXEC", "EXEC (AgentExecutor) / iris-code CLI", "core"),
        ("SAND", "Sandbox (Docker/gVisor)", "cap"),
        ("GOV", "GOV (kernel)", "gov"),
    ],
    [
        ("EXEC", "agent"),
        ("agent", "pipeline"),
        ("pipeline", "runner"),
        ("runner", "llm"),
        ("runner", "git"),
        ("runner", "GOV", "PreToolUse · persona surface · MCP signing", True),
        ("mcp", "GOV", "signed servers", True),
        ("runner", "SAND", "run_command", True),
        ("pipeline", "closeout"),
    ],
)

agent_page(
    "13 - code_exec",
    "ag_exec",
    "code_exec (tools/sandbox_tools.py, sandbox/) — from the ReAct loop",
    [
        ("host", "SandboxToolHost", "cap"),
        ("guard", "_destructive_match guard", "cap"),
        ("ws", "SessionWorkspace", "store"),
        ("rt", "runtime: DockerSandbox / GVisorSandbox", "cap"),
    ],
    [
        ("REACT", "system agent ReAct loop", "core"),
        ("EXEC", "EXEC (AgentExecutor)", "core"),
        ("GOV", "GOV (kernel)", "gov"),
    ],
    [
        ("REACT", "EXEC"),
        ("EXEC", "host"),
        ("host", "guard"),
        ("guard", "rt"),
        ("host", "ws"),
        ("rt", "GOV", "egress proxy / cap-drop / signing", True),
    ],
)


def main() -> None:
    out = '<mxfile host="app.diagrams.net" agent="iris-docs (gen_architecture_drawio.py)" version="24.0.0">\n'
    out += "".join(pg.xml() for pg in pages)
    out += "</mxfile>\n"
    OUT.write_text(out, encoding="utf-8")
    print(f"wrote {OUT.relative_to(Path.cwd())} — {len(pages)} page(s)")


if __name__ == "__main__":
    main()
