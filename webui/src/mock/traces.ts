import type { Resources, Trace, TraceNode } from '../lib/types';

// Mock traces for the MVP. Shapes + numbers are seeded from real IRIS session
// JSONL (~/.iris/logs/session-*.jsonl): the event order
//   user_message -> intent_router(llm) -> memory -> planner -> agent(llm)
//   -> response_curator -> agent_response
// and real figures (intent llm ~686ms/702tok on llama3.2:3b; agent llm on
// granite4/llama). Every node carries input/output (the stage's data flow),
// plus module/method for the verbose view. Resource values are representative
// mock numbers; GPU% is shown on LLM nodes to exercise the UI (real GPU capture
// on macOS is a documented limitation — see webui/README.md).

const RAM_TOTAL = 16.0;

function res(cpu: number, ramFree: number, gpu: number | null = null, thermal = false): Resources {
  return { cpu_percent: cpu, ram_free_gb: ramFree, ram_total_gb: RAM_TOTAL, gpu_percent: gpu, thermal_throttled: thermal };
}

// ── Trace 1: "what time is it today?" — simple system-agent path ───────────────
const timeNodes: TraceNode[] = [
  {
    id: 'gw', kind: 'gateway', label: 'channel_gateway', component_path: 'src/iris_harness/server/channel_gateway/main.py',
    module: 'iris_harness.server.channel_gateway.main', method: 'ws_proxy', status: 'ok', t_offset_ms: 0, duration_ms: 6,
    input: '{"channel":"web","session_id":"079d3fd77f7d","text":"what time is it today?"}',
    output: 'forwarded → iris_api POST /chat',
    resources: res(18, 11.2),
  },
  {
    id: 'api', kind: 'api', label: 'iris_api · /chat', component_path: 'src/iris_harness/server/iris_api/main.py',
    module: 'iris_harness.server.iris_api.main', method: 'chat', status: 'ok', t_offset_ms: 2, duration_ms: 2390,
    input: 'ChatRequest(message="what time is it today?", session_id="079d3fd77f7d", channel="web", strict=False)',
    output: 'ChatResponse(text="It\'s Tuesday, 16 June 2026, 9:41 AM (IST).", intent="system", agent_type="system", has_errors=False)',
    resources: res(22, 11.0),
  },
  {
    id: 'rt', kind: 'runtime', label: 'IrisRuntime.chat', component_path: 'src/iris_harness/runtime/bootstrap.py',
    module: 'iris_harness.runtime.bootstrap', method: 'IrisRuntime.chat', status: 'ok', t_offset_ms: 5, duration_ms: 2385,
    input: 'message="what time is it today?", session_id="079d3fd77f7d"',
    output: 'orchestrated 6 stages · 2 llm_calls · 1842 tokens · 2.40 s',
    resources: res(24, 10.9),
  },
  {
    id: 'ir', kind: 'intent_router', label: 'intent_router', component_path: 'src/iris_harness/agent/intent_router.py',
    module: 'iris_harness.agent.intent_router', method: 'IntentRouter.classify', status: 'ok', t_offset_ms: 8, duration_ms: 720,
    input: 'what time is it today?',
    output: 'IntentResult(intent="system", agent_type="system", is_multi_step=False, confidence=0.98)',
    resources: res(31, 10.7),
  },
  {
    id: 'gov-ir', kind: 'governance', label: 'pre_llm_call', component_path: 'src/iris_harness/kernel/governance/kernel.py',
    module: 'iris_harness.kernel.governance.kernel', method: 'GovernanceKernel.fire(PRE_LLM_CALL)', status: 'ok', t_offset_ms: 12, duration_ms: 2,
    input: 'hook=PRE_LLM_CALL agent=intent_router tier=tier_1 model=llama3.2:3b',
    output: 'HookDecision(outcome="allow")',
    governance: { hook: 'pre_llm_call', decision: 'allow', reason: 'tier_1 local model within policy' },
    resources: res(31, 10.7),
  },
  {
    id: 'llm-ir', kind: 'llm', label: 'classify intent', component_path: 'src/iris_harness/llm/arbiter.py',
    module: 'iris_harness.llm.arbiter', method: 'OllamaArbiter.invoke', status: 'ok', t_offset_ms: 16, duration_ms: 686,
    model: 'llama3.2:3b', provider: 'ollama', tokens: { prompt: 674, completion: 28, total: 702 },
    input: "[system] You are IRIS's request router. Classify the user message into a single JSON object (intent, agent_type, is_multi_step, confidence) — output JSON and NOTHING else.\n[user] what time is it today?",
    output: '{"intent": "system", "agent_type": "system", "is_multi_step": false, "confidence": 0.98}',
    resources: res(78, 9.4, 41),
  },
  {
    id: 'mem', kind: 'memory', label: 'memory_retriever', component_path: 'src/iris_harness/memory/retriever.py',
    module: 'iris_harness.memory.retriever', method: 'MemoryRetriever.retrieve', status: 'ok', t_offset_ms: 735, duration_ms: 18,
    input: 'session_id="079d3fd77f7d", message="what time is it today?"',
    output: 'MemoryContext(user_profile=✓, active_context=∅, episodic_patterns=0, recent_turns=2)',
    resources: res(20, 10.6),
  },
  {
    id: 'tp', kind: 'task_planner', label: 'task_planner', component_path: 'src/iris_harness/agent/task_planner.py',
    module: 'iris_harness.agent.task_planner', method: 'TaskPlanner.plan', status: 'ok', t_offset_ms: 758, duration_ms: 6,
    input: 'intent="system", is_multi_step=False',
    output: 'TaskPlan(size=1, first_agent="system", tasks=[SubTask(id="t0", agent="system")])',
    resources: res(19, 10.6),
  },
  {
    id: 'ae', kind: 'agent_executor', label: 'agent_executor', component_path: 'src/iris_harness/agent/agent_executor.py',
    module: 'iris_harness.agent.agent_executor', method: 'AgentExecutor.execute_stream', status: 'ok', t_offset_ms: 768, duration_ms: 1600,
    input: 'TaskPlan(size=1) → dispatch "system"',
    output: '[AgentResult(agent="system", ok=True, chars=42)]',
    resources: res(26, 10.4),
  },
  {
    id: 'sa', kind: 'agent', label: 'system_agent (ReAct)', component_path: 'src/iris_harness/agent/agentic_core.py',
    module: 'iris_harness.agent.agentic_core', method: 'AgenticCore.run', status: 'ok', t_offset_ms: 772, duration_ms: 1592,
    input: 'task="answer the user", tools=[get_datetime, …], iterations≤4',
    output: "final answer: It's Tuesday, 16 June 2026, 9:41 AM (IST).",
    resources: res(34, 10.2),
  },
  {
    id: 'gov-sa', kind: 'governance', label: 'pre_llm_call', component_path: 'src/iris_harness/kernel/governance/kernel.py',
    module: 'iris_harness.kernel.governance.kernel', method: 'GovernanceKernel.fire(PRE_LLM_CALL)', status: 'ok', t_offset_ms: 775, duration_ms: 2,
    input: 'hook=PRE_LLM_CALL agent=system tier=tier_1 model=granite4:latest',
    output: 'HookDecision(outcome="allow")',
    governance: { hook: 'pre_llm_call', decision: 'allow' },
    resources: res(34, 10.2),
  },
  {
    id: 'llm-sa', kind: 'llm', label: 'answer', component_path: 'src/iris_harness/llm/arbiter.py',
    module: 'iris_harness.llm.arbiter', method: 'OllamaArbiter.invoke', status: 'ok', t_offset_ms: 782, duration_ms: 1480,
    model: 'granite4:latest', provider: 'ollama', tokens: { prompt: 1100, completion: 40, total: 1140 },
    input: '[system] You are IRIS, a personal AI assistant. The current datetime is 2026-06-16T09:41:12+05:30 (Asia/Kolkata).\n[user] what time is it today?',
    output: "It's Tuesday, 16 June 2026, 9:41 AM (IST).",
    resources: res(83, 8.8, 57, false),
  },
  {
    id: 'rc', kind: 'response_curator', label: 'response_curator', component_path: 'src/iris_harness/agent/response_curator.py',
    module: 'iris_harness.agent.response_curator', method: 'ResponseCurator.curate', status: 'ok', t_offset_ms: 2270, duration_ms: 120,
    input: 'draft="It\'s Tuesday, 16 June 2026, 9:41 AM (IST)." · agents=["system"]',
    output: 'CuratedResponse(text="It\'s Tuesday, 16 June 2026, 9:41 AM (IST).", sources=[], has_errors=False)',
    resources: res(29, 10.1),
  },
];
const timeTrace: Trace = {
  session_id: '079d3fd77f7d', trace_id: '079d3fd77f7d~0', request: 'what time is it today?', started_at: '2026-06-16T09:41:12.118+05:30',
  total_duration_ms: 2400, total_tokens: 1842,
  steps: [
    { id: 's0', seq: 0, type: 'request', title: 'Request: what time is it today?', t_offset_ms: 0, status: 'ok', detail: 'what time is it today?' },
    { id: 's1', seq: 1, type: 'llm', title: 'LLM llama3.2:3b: {"intent":"system",…}', t_offset_ms: 16, duration_ms: 686, status: 'ok', fields: { model: 'llama3.2:3b', tokens: 702 }, detail: 'PROMPT: classify the request…\nOUTPUT: {"intent":"system","agent_type":"system","is_multi_step":false,"confidence":0.98}' },
    { id: 's2', seq: 2, type: 'intent', title: 'Intent → system (system, conf 0.98)', t_offset_ms: 735, status: 'ok', fields: { intent: 'system', agent: 'system', confidence: 0.98, multi_step: false } },
    { id: 's3', seq: 3, type: 'memory', title: 'Memory loaded (profile ✓, episodic 0, recent 2)', t_offset_ms: 735, status: 'ok' },
    { id: 's4', seq: 4, type: 'plan', title: 'Plan: 1 task(s), first=system', t_offset_ms: 758, status: 'ok' },
    { id: 's5', seq: 5, type: 'llm', title: "LLM granite4:latest: It's Tuesday, 16 June 2026, 9:41 AM (IST).", t_offset_ms: 782, duration_ms: 1480, status: 'ok', fields: { model: 'granite4:latest', tokens: 1140 }, detail: "PROMPT: …current datetime is 2026-06-16T09:41:12+05:30…\nOUTPUT: It's Tuesday, 16 June 2026, 9:41 AM (IST)." },
    { id: 's6', seq: 6, type: 'curator', title: 'Curator: safety=pass, faithfulness=pass', t_offset_ms: 2270, status: 'ok', fields: { safety: 'pass', faithfulness: 'pass' } },
    { id: 's7', seq: 7, type: 'response', title: 'Final answer (42 chars)', t_offset_ms: 2390, status: 'ok', detail: "It's Tuesday, 16 June 2026, 9:41 AM (IST).", fields: { tokens: 1842 } },
  ],
  nodes: timeNodes,
  edges: [
    { id: 'e1', source: 'gw', target: 'api', kind: 'data' },
    { id: 'e2', source: 'api', target: 'rt', kind: 'data' },
    { id: 'e3', source: 'rt', target: 'ir', kind: 'data' },
    { id: 'e4', source: 'ir', target: 'gov-ir', kind: 'control', label: 'guard' },
    { id: 'e5', source: 'gov-ir', target: 'llm-ir', kind: 'control' },
    { id: 'e6', source: 'llm-ir', target: 'mem', kind: 'data', label: 'intent' },
    { id: 'e7', source: 'mem', target: 'tp', kind: 'data' },
    { id: 'e8', source: 'tp', target: 'ae', kind: 'data', label: 'plan' },
    { id: 'e9', source: 'ae', target: 'sa', kind: 'data' },
    { id: 'e10', source: 'sa', target: 'gov-sa', kind: 'control', label: 'guard' },
    { id: 'e11', source: 'gov-sa', target: 'llm-sa', kind: 'control' },
    { id: 'e12', source: 'llm-sa', target: 'rc', kind: 'data', label: 'draft' },
  ],
};

// ── Trace 2: "summarize today's AI news as a PDF" — code_exec + tool + approval ──
const pdfNodes: TraceNode[] = [
  {
    id: 'gw', kind: 'gateway', label: 'channel_gateway', component_path: 'src/iris_harness/server/channel_gateway/main.py',
    module: 'iris_harness.server.channel_gateway.main', method: 'ws_proxy', status: 'ok', t_offset_ms: 0, duration_ms: 7,
    input: '{"channel":"telegram","session_id":"c41a9b2e7f30","text":"refresh AI news and give me an updated pdf with summaries"}',
    output: 'forwarded → iris_api POST /chat/stream',
    resources: res(21, 10.1),
  },
  {
    id: 'api', kind: 'api', label: 'iris_api · /chat/stream', component_path: 'src/iris_harness/server/iris_api/main.py',
    module: 'iris_harness.server.iris_api.main', method: 'chat_stream', status: 'ok', t_offset_ms: 2, duration_ms: 9120,
    input: 'ChatRequest(message="refresh AI news and give me an updated pdf with summaries", channel="telegram")',
    output: 'StreamingResponse(NDJSON) · 47 StreamEvent chunks · PDF artifact attached',
    resources: res(25, 9.9),
  },
  {
    id: 'rt', kind: 'runtime', label: 'IrisRuntime.chat_stream', component_path: 'src/iris_harness/runtime/bootstrap.py',
    module: 'iris_harness.runtime.bootstrap', method: 'IrisRuntime.chat_stream', status: 'ok', t_offset_ms: 5, duration_ms: 9112,
    input: 'message="refresh AI news and give me an updated pdf with summaries"',
    output: 'orchestrated 7 stages · 3 llm_calls · 1 tool_run · 6351 tokens · 9.13 s',
    resources: res(27, 9.8),
  },
  {
    id: 'ir', kind: 'intent_router', label: 'intent_router', component_path: 'src/iris_harness/agent/intent_router.py',
    module: 'iris_harness.agent.intent_router', method: 'IntentRouter.classify', status: 'ok', t_offset_ms: 9, duration_ms: 690,
    input: 'refresh AI news and give me an updated pdf with summaries',
    output: 'IntentResult(intent="code_exec", agent_type="code_exec", is_multi_step=True, confidence=0.93)',
    resources: res(33, 9.6),
  },
  {
    id: 'llm-ir', kind: 'llm', label: 'classify intent', component_path: 'src/iris_harness/llm/arbiter.py',
    module: 'iris_harness.llm.arbiter', method: 'OllamaArbiter.invoke', status: 'ok', t_offset_ms: 14, duration_ms: 672,
    model: 'llama3.2:3b', provider: 'ollama', tokens: { prompt: 690, completion: 31, total: 721 },
    input: "[system] You are IRIS's request router … route 'code_exec' when the user wants an artifact PRODUCED (PDF, Excel, chart) or a script RUN.\n[user] refresh AI news and give me an updated pdf with summaries",
    output: '{"intent": "code_exec", "agent_type": "code_exec", "is_multi_step": true, "confidence": 0.93}',
    resources: res(80, 8.9, 44),
  },
  {
    id: 'mem', kind: 'memory', label: 'memory_retriever', component_path: 'src/iris_harness/memory/retriever.py',
    module: 'iris_harness.memory.retriever', method: 'MemoryRetriever.retrieve', status: 'ok', t_offset_ms: 710, duration_ms: 22,
    input: 'session_id="c41a9b2e7f30"',
    output: 'MemoryContext(user_profile=✓, active_context="AI news digest task", episodic_patterns=3)',
    resources: res(22, 9.5),
  },
  {
    id: 'tp', kind: 'task_planner', label: 'task_planner', component_path: 'src/iris_harness/agent/task_planner.py',
    module: 'iris_harness.agent.task_planner', method: 'TaskPlanner.plan', status: 'ok', t_offset_ms: 738, duration_ms: 820,
    input: 'intent="code_exec", is_multi_step=True',
    output: 'TaskPlan(size=3, tasks=["fetch AI headlines","summarize each","render PDF"])',
    resources: res(30, 9.3),
  },
  {
    id: 'llm-tp', kind: 'llm', label: 'decompose tasks', component_path: 'src/iris_harness/llm/arbiter.py',
    module: 'iris_harness.llm.arbiter', method: 'OllamaArbiter.invoke', status: 'ok', t_offset_ms: 745, duration_ms: 800,
    model: 'llama3.2:3b', provider: 'ollama', tokens: { prompt: 940, completion: 110, total: 1050 },
    input: '[system] Break the request into an ordered list of concrete sub-tasks. Return a JSON array of task strings.\n[user] refresh AI news and give me an updated pdf with summaries',
    output: '["fetch AI headlines", "summarize each headline", "render summaries to PDF"]',
    resources: res(76, 8.6, 39),
  },
  {
    id: 'ae', kind: 'agent_executor', label: 'agent_executor', component_path: 'src/iris_harness/agent/agent_executor.py',
    module: 'iris_harness.agent.agent_executor', method: 'AgentExecutor.execute_stream', status: 'ok', t_offset_ms: 1565, duration_ms: 7540,
    input: 'TaskPlan(size=3) → dispatch "code_exec"',
    output: '[AgentResult(agent="code_exec", ok=True, artifacts=["ai_news_2026-06-16.pdf"])]',
    resources: res(28, 9.1),
  },
  {
    id: 'ca', kind: 'agent', label: 'code_exec_agent', component_path: 'src/iris_harness/runtime/bootstrap.py',
    module: 'iris_harness.runtime.bootstrap', method: '_make_code_exec_handler', status: 'ok', t_offset_ms: 1570, duration_ms: 7530,
    input: 'tasks=["fetch AI headlines","summarize each","render PDF"], sandbox=docker',
    output: 'wrote ai_news_2026-06-16.pdf (8 pages) + digest text',
    resources: res(38, 8.7),
  },
  {
    id: 'gov-tool', kind: 'governance', label: 'pre_tool_use', component_path: 'src/iris_harness/kernel/governance/kernel.py',
    module: 'iris_harness.kernel.governance.kernel', method: 'GovernanceKernel.fire(PRE_TOOL_USE)', status: 'ok', t_offset_ms: 1580, duration_ms: 4,
    input: 'hook=PRE_TOOL_USE tool=run_shell cmd="python render_news_pdf.py …"',
    output: 'HookDecision(outcome="require_approval") → auto-approved (demo)',
    governance: { hook: 'pre_tool_use', decision: 'require_approval', reason: 'run_shell in sandbox — human approval (auto-approved in demo)' },
    resources: res(38, 8.7),
  },
  {
    id: 'tool', kind: 'tool', label: 'run_shell (sandbox)', component_path: 'src/iris_harness/tools/sandbox_tools.py',
    module: 'iris_harness.tools.sandbox_tools', method: 'run_shell', status: 'ok', t_offset_ms: 1600, duration_ms: 4200,
    input: 'python render_news_pdf.py --out ai_news_2026-06-16.pdf',
    output: 'fetched 12 headlines\nwrote ai_news_2026-06-16.pdf (8 pages)',
    tool: { cmd: 'python render_news_pdf.py --out ai_news_2026-06-16.pdf', exit_code: 0, stdout: 'fetched 12 headlines\nwrote ai_news_2026-06-16.pdf (8 pages)', stderr: '' },
    resources: res(64, 7.9, null, true),
  },
  {
    id: 'llm-ca', kind: 'llm', label: 'summarize + narrate', component_path: 'src/iris_harness/llm/arbiter.py',
    module: 'iris_harness.llm.arbiter', method: 'OllamaArbiter.invoke', status: 'ok', t_offset_ms: 5850, duration_ms: 3120,
    model: 'qwen2.5:7b', provider: 'ollama', tokens: { prompt: 4200, completion: 380, total: 4580 },
    input: '[system] Summarize these 12 AI headlines into a concise digest with one line each, then write a short intro.\n[user] <12 headlines + article snippets …>',
    output: "Here's your AI news digest (PDF attached):\n1) … 2) … 3) …",
    resources: res(88, 6.9, 71, true),
  },
  {
    id: 'rc', kind: 'response_curator', label: 'response_curator', component_path: 'src/iris_harness/agent/response_curator.py',
    module: 'iris_harness.agent.response_curator', method: 'ResponseCurator.curate', status: 'ok', t_offset_ms: 9000, duration_ms: 110,
    input: 'draft="Here\'s your AI news digest …" · artifacts=["ai_news_2026-06-16.pdf"]',
    output: 'CuratedResponse(text="Here\'s your AI news digest (PDF attached): …", sources=[12 urls], has_errors=False)',
    resources: res(31, 8.4),
  },
];
const pdfTrace: Trace = {
  session_id: 'c41a9b2e7f30', trace_id: 'c41a9b2e7f30~0', request: 'refresh AI news and give me an updated pdf with summaries', started_at: '2026-06-16T08:02:55.400+05:30',
  total_duration_ms: 9130, total_tokens: 6351,
  steps: [
    { id: 's0', seq: 0, type: 'request', title: 'Request: refresh AI news and give me an updated pdf…', t_offset_ms: 0, status: 'ok', detail: 'refresh AI news and give me an updated pdf with summaries' },
    { id: 's1', seq: 1, type: 'intent', title: 'Intent → code_exec (multi-step, conf 0.93)', t_offset_ms: 9, status: 'ok', fields: { intent: 'code_exec', agent: 'code_exec', confidence: 0.93, multi_step: true } },
    { id: 's2', seq: 2, type: 'plan', title: 'Plan: 3 task(s) — fetch headlines, summarize, render PDF', t_offset_ms: 738, status: 'ok', detail: '["fetch AI headlines","summarize each headline","render summaries to PDF"]' },
    { id: 's3', seq: 3, type: 'tool', title: 'Tool run_shell(render_news_pdf.py) → 8-page PDF', t_offset_ms: 1600, duration_ms: 4200, status: 'ok', fields: { tool: 'run_shell', approval: 'required→granted' }, detail: 'args: python render_news_pdf.py --out ai_news_2026-06-16.pdf\n\nresult: fetched 12 headlines; wrote ai_news_2026-06-16.pdf (8 pages)' },
    { id: 's4', seq: 4, type: 'llm', title: 'LLM qwen2.5:7b: summarized 12 headlines into a digest', t_offset_ms: 5850, duration_ms: 3120, status: 'ok', fields: { model: 'qwen2.5:7b', tokens: 4580 }, detail: "PROMPT: Summarize these 12 AI headlines…\nOUTPUT: Here's your AI news digest (PDF attached): 1) … 2) …" },
    { id: 's5', seq: 5, type: 'curator', title: 'Curator: safety=pass, schema=pass', t_offset_ms: 9000, status: 'ok', fields: { safety: 'pass', schema: 'pass' } },
    { id: 's6', seq: 6, type: 'response', title: 'Final answer + PDF artifact', t_offset_ms: 9000, status: 'ok', detail: "Here's your AI news digest (PDF attached): …", fields: { tokens: 6351 } },
  ],
  nodes: pdfNodes,
  edges: [
    { id: 'e1', source: 'gw', target: 'api', kind: 'data' },
    { id: 'e2', source: 'api', target: 'rt', kind: 'data' },
    { id: 'e3', source: 'rt', target: 'ir', kind: 'data' },
    { id: 'e4', source: 'ir', target: 'llm-ir', kind: 'control', label: 'guard' },
    { id: 'e5', source: 'llm-ir', target: 'mem', kind: 'data', label: 'intent' },
    { id: 'e6', source: 'mem', target: 'tp', kind: 'data' },
    { id: 'e7', source: 'tp', target: 'llm-tp', kind: 'control' },
    { id: 'e8', source: 'llm-tp', target: 'ae', kind: 'data', label: 'plan' },
    { id: 'e9', source: 'ae', target: 'ca', kind: 'data' },
    { id: 'e10', source: 'ca', target: 'gov-tool', kind: 'control', label: 'guard' },
    { id: 'e11', source: 'gov-tool', target: 'tool', kind: 'control' },
    { id: 'e12', source: 'tool', target: 'llm-ca', kind: 'data', label: 'observation' },
    { id: 'e13', source: 'llm-ca', target: 'rc', kind: 'data', label: 'draft' },
  ],
};

// ── Trace 3: "delete all my emails" — governance DENY (refusal path) ────────────
const denyNodes: TraceNode[] = [
  {
    id: 'gw', kind: 'gateway', label: 'channel_gateway', component_path: 'src/iris_harness/server/channel_gateway/main.py',
    module: 'iris_harness.server.channel_gateway.main', method: 'ws_proxy', status: 'ok', t_offset_ms: 0, duration_ms: 6,
    input: '{"channel":"web","session_id":"a7e0c1d94b22","text":"delete all my emails"}',
    output: 'forwarded → iris_api POST /chat',
    resources: res(19, 10.8),
  },
  {
    id: 'api', kind: 'api', label: 'iris_api · /chat', component_path: 'src/iris_harness/server/iris_api/main.py',
    module: 'iris_harness.server.iris_api.main', method: 'chat', status: 'ok', t_offset_ms: 2, duration_ms: 940,
    input: 'ChatRequest(message="delete all my emails", session_id="a7e0c1d94b22")',
    output: 'ChatResponse(text="I can\'t bulk-delete all your emails …", intent="communication", has_errors=False)',
    resources: res(23, 10.6),
  },
  {
    id: 'rt', kind: 'runtime', label: 'IrisRuntime.chat', component_path: 'src/iris_harness/runtime/bootstrap.py',
    module: 'iris_harness.runtime.bootstrap', method: 'IrisRuntime.chat', status: 'ok', t_offset_ms: 5, duration_ms: 935,
    input: 'message="delete all my emails"',
    output: 'orchestrated · 1 llm_call · governance DENY on tool · refusal returned',
    resources: res(24, 10.5),
  },
  {
    id: 'ir', kind: 'intent_router', label: 'intent_router', component_path: 'src/iris_harness/agent/intent_router.py',
    module: 'iris_harness.agent.intent_router', method: 'IntentRouter.classify', status: 'ok', t_offset_ms: 8, duration_ms: 700,
    input: 'delete all my emails',
    output: 'IntentResult(intent="communication", agent_type="email", is_multi_step=False, confidence=0.90)',
    resources: res(32, 10.3),
  },
  {
    id: 'llm-ir', kind: 'llm', label: 'classify intent', component_path: 'src/iris_harness/llm/arbiter.py',
    module: 'iris_harness.llm.arbiter', method: 'OllamaArbiter.invoke', status: 'ok', t_offset_ms: 12, duration_ms: 688,
    model: 'llama3.2:3b', provider: 'ollama', tokens: { prompt: 700, completion: 30, total: 730 },
    input: "[system] You are IRIS's request router …\n[user] delete all my emails",
    output: '{"intent": "communication", "agent_type": "email", "is_multi_step": false, "confidence": 0.9}',
    resources: res(79, 9.1, 43),
  },
  {
    id: 'ea', kind: 'agent', label: 'email_agent', component_path: 'src/iris_personal/email/handler.py',
    module: 'iris_personal.email.handler', method: 'EmailHandler.handle', status: 'ok', t_offset_ms: 715, duration_ms: 150,
    input: 'intent="communication", action="bulk_delete", scope="all"',
    output: 'requested tool email.bulk_delete(scope=all) → blocked downstream',
    resources: res(27, 10.2),
  },
  {
    id: 'gov-tool', kind: 'governance', label: 'pre_tool_use', component_path: 'src/iris_harness/kernel/governance/kernel.py',
    module: 'iris_harness.kernel.governance.kernel', method: 'GovernanceKernel.fire(PRE_TOOL_USE)', status: 'error', t_offset_ms: 720, duration_ms: 5,
    input: 'hook=PRE_TOOL_USE tool=email.bulk_delete args={"scope":"all"}',
    output: 'HookDecision(outcome="deny", reason="bulk irreversible delete — deny-by-default")',
    governance: { hook: 'pre_tool_use', decision: 'deny', reason: 'bulk irreversible delete — deny-by-default (R2/R3); no scoped grant' },
    resources: res(27, 10.2),
  },
  {
    id: 'tool', kind: 'tool', label: 'email.bulk_delete', component_path: 'src/iris_personal/email/tools.py',
    module: 'iris_personal.email.tools', method: 'bulk_delete', status: 'skipped', t_offset_ms: 725, duration_ms: 0,
    input: 'email.bulk_delete(scope=all)',
    output: '⨯ not executed — blocked by governance kernel (deny)',
    tool: { cmd: 'email.bulk_delete(scope=all)', exit_code: -1, stdout: '', stderr: 'blocked by governance kernel (deny)' },
    resources: res(25, 10.3),
  },
  {
    id: 'rc', kind: 'response_curator', label: 'response_curator', component_path: 'src/iris_harness/agent/response_curator.py',
    module: 'iris_harness.agent.response_curator', method: 'ResponseCurator.curate', status: 'ok', t_offset_ms: 730, duration_ms: 200,
    input: 'governance_denied=True, action="email.bulk_delete"',
    output: "I can't bulk-delete all your emails — that's an irreversible action blocked by policy. I can archive or delete a specific set if you confirm which.",
    resources: res(28, 10.2),
  },
];
const denyTrace: Trace = {
  session_id: 'a7e0c1d94b22', trace_id: 'a7e0c1d94b22~0', request: 'delete all my emails', started_at: '2026-06-16T10:15:03.900+05:30',
  total_duration_ms: 945, total_tokens: 730,
  steps: [
    { id: 's0', seq: 0, type: 'request', title: 'Request: delete all my emails', t_offset_ms: 0, status: 'ok', detail: 'delete all my emails' },
    { id: 's1', seq: 1, type: 'llm', title: 'LLM llama3.2:3b: classified as communication/email', t_offset_ms: 12, duration_ms: 688, status: 'ok', fields: { model: 'llama3.2:3b', tokens: 730 }, detail: 'OUTPUT: {"intent":"communication","agent_type":"email","is_multi_step":false,"confidence":0.9}' },
    { id: 's2', seq: 2, type: 'intent', title: 'Intent → email (communication, conf 0.90)', t_offset_ms: 8, status: 'ok', fields: { intent: 'communication', agent: 'email', confidence: 0.9 } },
    { id: 's3', seq: 3, type: 'tool', title: 'Tool email.bulk_delete(scope=all) → BLOCKED', t_offset_ms: 725, status: 'error', fields: { tool: 'email.bulk_delete', ok: false }, detail: 'args: {"scope":"all"}\n\nresult: blocked by governance kernel (deny) — bulk irreversible delete, deny-by-default (R2/R3), no scoped grant' },
    { id: 's4', seq: 4, type: 'response', title: 'Final answer: refusal with safe alternative', t_offset_ms: 730, status: 'ok', detail: "I can't bulk-delete all your emails — that's an irreversible action blocked by policy. I can archive or delete a specific set if you confirm which.", fields: { tokens: 730 } },
  ],
  nodes: denyNodes,
  edges: [
    { id: 'e1', source: 'gw', target: 'api', kind: 'data' },
    { id: 'e2', source: 'api', target: 'rt', kind: 'data' },
    { id: 'e3', source: 'rt', target: 'ir', kind: 'data' },
    { id: 'e4', source: 'ir', target: 'llm-ir', kind: 'control', label: 'guard' },
    { id: 'e5', source: 'llm-ir', target: 'ea', kind: 'data', label: 'intent' },
    { id: 'e6', source: 'ea', target: 'gov-tool', kind: 'control', label: 'guard' },
    { id: 'e7', source: 'gov-tool', target: 'tool', kind: 'control', label: 'deny' },
    { id: 'e8', source: 'gov-tool', target: 'rc', kind: 'data', label: 'refusal' },
  ],
};

export const MOCK_TRACES: Trace[] = [timeTrace, pdfTrace, denyTrace];
