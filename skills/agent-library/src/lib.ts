const supportedAgents = ["codex", "claude-code", "cursor", "generic"] as const
const supportedModes = ["full", "prompt", "checklist"] as const

const hostNoteByAgent = {
  codex: "Prefer short operational wording and search before acting when prior context may matter.",
  "claude-code": "Prefer search before coding when project context may come from prior sessions or indexed notes.",
  cursor: "Prefer search before implementation when design decisions or prior context may already exist.",
  generic: "Prefer search before acting when durable context may already be stored."
} as const

const promptByAgent = {
  codex: "When Agent Library is available, treat it as long-term retrieval memory, not a transcript dump.",
  "claude-code": "When Agent Library is available in Claude Code, treat it as long-term retrieval memory, not a transcript dump.",
  cursor: "When Agent Library is available in Cursor, treat it as long-term retrieval memory, not a transcript dump.",
  generic: "When Agent Library is available, treat it as long-term retrieval memory, not a transcript dump."
} as const

const decisionChecklist = [
  "Search first when the task may depend on prior sessions, indexed docs, recurring user preferences, or durable project context.",
  "Read the full document before citing or relying on a search hit.",
  "Write only durable, reusable knowledge that is likely to help in a future session.",
  "Inspect the library with get_library_overview before writing, and use suggest_library_location when placement is unclear.",
  "Prefer update_library_doc over add_to_library when refining an existing note or artifact.",
  "Skip memory for ephemeral chat, one-off command output, low-value summaries, and facts that are easy to re-derive from the current repo."
]

const promptBody = agent => [
  promptByAgent[agent],
  "",
  "Search first when the task may depend on prior sessions, indexed docs, recurring user preferences, or durable project context. Use search_library first, then read_from_library before relying on a result.",
  "",
  "Write only durable, reusable knowledge that is likely to help in a future session. Do not save every response, plan, debug log, or transient thought.",
  "",
  "Before writing, inspect the library with get_library_overview and use suggest_library_location if placement is unclear.",
  "",
  "Prefer update_library_doc over add_to_library when refining an existing note or artifact. Avoid creating near-duplicate documents.",
  "",
  "Skip memory entirely for ephemeral chat, one-off command output, facts that are easy to re-derive from the current repo, and low-value summaries."
].join("\n")

const renderPrompt = agent => ["```text", promptBody(agent), "```"].join("\n")

const renderChecklist = () => decisionChecklist.map(line => `- ${line}`).join("\n")

const renderFull = agent => [
  `# Agent Library Memory Policy (${agent})`,
  "",
  hostNoteByAgent[agent],
  "",
  "## Search first",
  "",
  "- Search when prior sessions, indexed docs, or recurring preferences may matter.",
  "- Use search_library first and read_from_library before acting on a hit.",
  "",
  "## Write only durable knowledge",
  "",
  "- Save information only if it is durable, user- or project-specific, and likely to help in a future session.",
  "- Do not write transient plans, raw logs, or low-value summaries.",
  "",
  "## Update instead of duplicate",
  "",
  "- Use get_library_overview before writing.",
  "- Use suggest_library_location when placement is unclear.",
  "- Prefer update_library_doc when refining an existing note or artifact.",
  "",
  "## Copy-paste prompt",
  "",
  renderPrompt(agent),
  "",
  "## Quick checklist",
  "",
  renderChecklist()
].join("\n")

const parseArgs = argv => {
  const tokens = argv[0] === "--" ? argv.slice(1) : argv
  const pairs = tokens.reduce(
    (state, token, index) =>
      token.startsWith("--") ? [...state, [token, tokens[index + 1] ?? ""]] : state,
    []
  )

  const readFlag = name => pairs.find(([flag]) => flag === name)?.[1]
  const help = tokens.includes("--help") || tokens.includes("-h")

  return {
    help,
    agent: readFlag("--agent") ?? "generic",
    mode: readFlag("--mode") ?? "full",
    out: readFlag("--out")
  }
}

const usage = [
  "Usage: agent-library [--agent codex|claude-code|cursor|generic] [--mode full|prompt|checklist] [--out path]",
  "",
  "Examples:",
  "  skills run agent-library -- --agent codex --mode prompt",
  "  skills run agent-library -- --mode checklist",
  "  skills run agent-library -- --agent claude-code --mode full --out ./agent-library.md"
].join("\n")

const validateInput = ({ agent, mode }) => {
  if (!supportedAgents.includes(agent)) {
    return `Unsupported agent '${agent}'. Expected one of: ${supportedAgents.join(", ")}.`
  }

  if (!supportedModes.includes(mode)) {
    return `Unsupported mode '${mode}'. Expected one of: ${supportedModes.join(", ")}.`
  }
}

const renderContent = ({ agent, mode }) =>
  mode === "prompt" ? renderPrompt(agent) : mode === "checklist" ? renderChecklist() : renderFull(agent)

export { parseArgs, renderContent, usage, validateInput }
