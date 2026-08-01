import { describe, expect, test } from "bun:test"

import { parseArgs, renderContent, validateInput } from "./lib"

describe("parseArgs", () => {
  test("parses explicit agent and mode", () => {
    expect(parseArgs(["--agent", "codex", "--mode", "prompt"])).toEqual({
      help: false,
      agent: "codex",
      mode: "prompt",
      out: undefined
    })
  })

  test("ignores the run separator", () => {
    expect(parseArgs(["--", "--mode", "checklist"]).mode).toBe("checklist")
  })
})

describe("validateInput", () => {
  test("accepts supported values", () => {
    expect(validateInput({ agent: "cursor", mode: "full" })).toBeUndefined()
  })

  test("rejects unsupported agent", () => {
    expect(validateInput({ agent: "windsurf", mode: "full" })).toContain("Unsupported agent")
  })
})

describe("renderContent", () => {
  test("renders a prompt block", () => {
    expect(renderContent({ agent: "codex", mode: "prompt" })).toContain("```text")
  })

  test("renders a checklist", () => {
    expect(renderContent({ agent: "generic", mode: "checklist" })).toContain("Search first")
  })
})
