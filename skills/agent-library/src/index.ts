#!/usr/bin/env bun

import { mkdir } from "node:fs/promises"
import { dirname, resolve } from "node:path"

import { parseArgs, renderContent, usage, validateInput } from "./lib"

const main = async argv => {
  const input = parseArgs(argv)

  if (input.help) {
    console.log(usage)
    return
  }

  const error = validateInput(input)
  if (error) {
    console.error(error)
    process.exitCode = 1
    return
  }

  const content = renderContent(input)

  if (input.out) {
    const target = resolve(input.out)
    await mkdir(dirname(target), { recursive: true })
    await Bun.write(target, content)
    console.log(`Wrote Agent Library guidance to ${target}`)
    return
  }

  console.log(content)
}

main(process.argv.slice(2)).catch(error => {
  console.error(error instanceof Error ? error.message : String(error))
  process.exit(1)
})
