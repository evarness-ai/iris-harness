// Trace data model. Shaped from real IRIS session JSONL events
// (~/.iris/logs/session-<id>.jsonl) plus the resource metrics that
// session_log.py now captures per step (CPU/RAM; GPU reserved on macOS).

export type NodeKind =
  | 'gateway'
  | 'api'
  | 'runtime'
  | 'intent_router'
  | 'memory'
  | 'task_planner'
  | 'agent_executor'
  | 'agent'
  | 'llm'
  | 'tool'
  | 'governance'
  | 'response_curator';

export type NodeStatus = 'ok' | 'error' | 'skipped';

export interface Resources {
  cpu_percent: number;
  ram_free_gb: number;
  ram_total_gb: number;
  gpu_percent: number | null; // null = unavailable (e.g. macOS/Apple Silicon)
  thermal_throttled: boolean;
}

export interface TokenUsage {
  prompt: number;
  completion: number;
  total: number;
}

export interface GovernanceInfo {
  hook: string; // e.g. pre_llm_call, pre_tool_use, pre_response
  decision: 'allow' | 'deny' | 'transform' | 'require_approval';
  reason?: string;
}

export interface ToolInfo {
  cmd: string;
  exit_code: number;
  stdout: string;
  stderr: string;
}

export interface TraceNode {
  id: string;
  kind: NodeKind;
  label: string;
  component_path: string; // source file this maps to
  module?: string; // python module path, e.g. iris_harness.agent.intent_router (verbose)
  method?: string; // function/method invoked, e.g. IntentRouter.classify (verbose)
  status: NodeStatus;
  t_offset_ms: number; // ms after request start when this step began
  duration_ms: number;

  // input/output of the stage — present for every node so the full data flow is
  // inspectable. For LLMs these are the prompt + completion; for other stages the
  // structured payload in/out.
  input?: string;
  output?: string;

  // LLM-specific
  model?: string;
  provider?: string;
  tier?: string; // llm_tiers.yaml key, or a role label (search_synthesis, profile:<name>)
  agent?: string; // pipeline stage or agent that made the call
  tokens?: TokenUsage;

  // tool-specific
  tool?: ToolInfo;

  // governance-specific
  governance?: GovernanceInfo;

  resources?: Resources;
}

export interface TraceEdge {
  id: string;
  source: string;
  target: string;
  kind: 'control' | 'data';
  label?: string;
  // Execution order: when this edge's target started, 1-based across the turn.
  seq?: number;
}

export type StepType =
  | 'request'
  | 'intent'
  | 'memory'
  | 'plan'
  | 'llm'
  | 'tool'
  | 'curator'
  | 'stop'
  | 'response'
  | 'error';

export interface ReasoningStep {
  id: string;
  seq: number;
  type: StepType;
  title: string;
  t_offset_ms: number;
  duration_ms?: number | null;
  status: NodeStatus;
  iteration?: number | null;
  detail?: string | null;
  fields?: Record<string, unknown>;
}

export interface Trace {
  session_id: string;
  trace_id: string; // "<session_id>~<turn_index>"
  request: string;
  started_at: string; // ISO timestamp
  total_duration_ms: number;
  total_tokens: number;
  nodes: TraceNode[];
  edges: TraceEdge[];
  steps?: ReasoningStep[];
  /** Every hook decision of the turn, in ledger order (the graph folds them per host). */
  governance?: GovernanceEvent[];
}

/** One audit row of the turn: columns + the documented payload fields, reason masked. */
export interface GovernanceEvent {
  id: number;
  t_offset_ms: number;
  run_id?: string | null;
  step_id?: number | null;
  hook_point: string;
  plugin: string;
  decision: string;
  severity?: string | null;
  classification?: string | null;
  tier?: string | null;
  locality?: 'local' | 'cloud' | null;
  reason: string;
  caller?: string;
  deterministic?: boolean;
  handler?: string;
  tool_name?: string;
  capability?: string;
  method?: string;
  digest_alg?: string;
}
