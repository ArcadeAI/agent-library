# Agent Build Instructions: agent-library

This folder is a portable `skills.md` custom skill. Keep it valid against the local portable skill contract.

## Contract

- Skill name: `agent-library`
- Description: Generate an opinionated Agent Library memory policy for coding agents
- Manifest files: `SKILL.md` frontmatter and `skill.json`
- Runtime entrypoint: `src/index.ts`
- User command: `skills run agent-library -- [args]`

## Build Rules

1. Keep executable logic in `src/`.
2. Keep `skill.json` in sync when inputs, commands, or version change.
3. Keep `SKILL.md` user-facing and concise. It should explain usage, outputs, and failure modes.
4. Run `bun test` from this folder when behavior changes.
5. Verify with `skills port ./skills/agent-library --name agent-library --overwrite`, then `skills validate agent-library`.
6. Smoke-test with `skills run agent-library -- --help` and at least one real prompt generation command.
7. Do not commit secrets, build output, `.env`, or generated exports.
