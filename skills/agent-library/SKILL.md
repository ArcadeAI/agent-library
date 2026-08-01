---
name: agent-library
description: Generate an opinionated Agent Library usage policy for coding agents. Use when a user wants a ready-to-paste prompt, checklist, or memory workflow for Codex, Claude Code, Cursor, or similar agents using Agent Library as long-term retrieval memory.
version: 0.1.0
source: custom
category: Development Tools
tags:
  - agent-library
  - memory
  - codex
  - claude-code
  - cursor
  - prompt
---

# Agent Library Memory

Generate a portable memory policy for coding agents that use Agent Library.

## What it does

- explains when an agent should search memory first
- explains what belongs in long-term retrieval memory
- explains when to update an existing note instead of creating a new one
- outputs a ready-to-paste prompt snippet or a short checklist

## Command

```bash
skills run agent-library -- --agent codex --mode prompt
```

## Inputs

- `--agent <codex|claude-code|cursor|generic>`
  - optional
  - defaults to `generic`
- `--mode <full|prompt|checklist>`
  - optional
  - defaults to `full`
- `--out <path>`
  - optional
  - writes the generated output to a file instead of only printing it

## Output contract

The skill returns plain text or Markdown guidance.

- `full` outputs a compact policy with search, write, and update rules plus a prompt block
- `prompt` outputs only the ready-to-paste prompt
- `checklist` outputs only the short decision checklist

When `--out` is provided, the same content is also written to the given path.

## Failure modes

- unsupported `--agent` value
- unsupported `--mode` value
- unwritable `--out` path

## Examples

Generate a Codex-ready prompt:

```bash
skills run agent-library -- --agent codex --mode prompt
```

Generate a full guide and save it:

```bash
skills run agent-library -- --agent claude-code --mode full --out ./agent-library.md
```

Generate only the quick checklist:

```bash
skills run agent-library -- --mode checklist
```
