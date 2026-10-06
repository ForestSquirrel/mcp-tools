import { atom, read, update } from 'claude-code'
import type { EngineInterface, Register, Timer } from 'claude-code'

import type { Describe, DescribeParam, DescribeRunOption, Form, Job, RunEntry, SweepAxis } from '../types'

// The Python MCP server (mcp-tools/server.py) as /mcp lists it.
const SERVER = 'bifurcation-studio'
const PANE = 'bif'
const POLL_MS = 500
const STEPS = 50 // stepper steps across a parameter's [min, max]

const rhsPath = atom({ plugin: 'bif-studio', key: 'rhsPath' } as const, '')
const described = atom({ plugin: 'bif-studio', key: 'described' } as const, null as Describe | null)
const callError = atom({ plugin: 'bif-studio', key: 'callError' } as const, '')
const form = atom({ plugin: 'bif-studio', key: 'form' } as const, null as Form | null)
const showAdvanced = atom({ plugin: 'bif-studio', key: 'showAdvanced' } as const, false)
const job = atom({ plugin: 'bif-studio', key: 'job' } as const, null as Job | null)
const runs = atom({ plugin: 'bif-studio', key: 'runs' } as const, [] as RunEntry[])
const selected = atom({ plugin: 'bif-studio', key: 'selected' } as const, '')

type Cells = { columns: number; rows: number; cells: string }

// Module variables: lost on a reload, which session.start repairs (polling)
// or which only costs a refetch (raster cache).
let poller: Timer | null = null
let polling = false
const rasterCache = new Map<string, Cells>()

/** Calls a server tool and parses its JSON text result; throws on isError. */
async function callServer($: EngineInterface, tool: string, args: Record<string, unknown>): Promise<unknown> {
  const result = await $.mcp.call(SERVER, tool, args)
  const text = result.content.map(b => (b.type === 'text' ? b.text : '')).join('')
  if (result.isError) throw new Error(text || `${tool} failed`)
  return JSON.parse(text)
}

function absolute(path: string, cwd: string): string {
  return path.startsWith('/') || path.startsWith('~') ? path : `${cwd.replace(/\/$/, '')}/${path}`
}

type SystemsList = { resolve?: Record<string, string>; systems?: { name: string; where: string }[] }

/**
 * `/bif` argument → file: a path (has a slash or ends in .py) as given,
 * else a system name, looked up on the server (a draft wins over a
 * permanent system). Throws with the known names when there is none.
 */
async function resolveArg($: EngineInterface, arg: string): Promise<string> {
  if (arg.includes('/') || arg.endsWith('.py')) return absolute(arg, await $.session.cwd())
  const listed = (await callServer($, 'list_systems', {})) as SystemsList
  const hit = listed.resolve?.[arg]
  if (hit) return hit
  const names = Object.keys(listed.resolve ?? {}).sort().join(', ')
  throw new Error(`no system named "${arg}". Known: ${names || 'none'}`)
}

function fmt(v: number | string | null | undefined): string {
  if (v === null || v === undefined) return '–'
  return typeof v === 'number' ? String(Number(v.toPrecision(6))) : v
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v))
}

// ── the form ──────────────────────────────────────────────────────────────────
function optionDefault(o: DescribeRunOption, mode: string): string | number {
  return o.modeDefaults?.[mode] ?? o.default
}

function appliesTo(o: DescribeRunOption, mode: string): boolean {
  return o.name !== 'mode' && (!o.modes || o.modes.includes(mode))
}

function axisFor(p: DescribeParam): SweepAxis {
  return { param: p.name, min: p.sweep?.[0] ?? p.min ?? 0, max: p.sweep?.[1] ?? p.max ?? 1 }
}

/** Parses and checks one run option's text against its spec; null if invalid. */
function parseOption(o: DescribeRunOption, text: string): string | number | null {
  if (o.type === 'enum') return o.options?.includes(text) ? text : null
  const v = Number(text)
  if (!Number.isFinite(v) || (o.type === 'int' && !Number.isInteger(v))) return null
  if ((o.min !== undefined && v < o.min) || (o.max !== undefined && v > o.max)) return null
  return v
}

/** Controls for a described file; keeps the previous values for the same file. */
function buildForm(desc: Describe, prev: Form | null): Form | null {
  const params = desc.params ?? []
  const sweepable = params.filter(p => p.sweepable)
  const first = sweepable[0]
  if (!desc.ok || !desc.system || !first) return null
  const old = prev && prev.path === desc.path ? prev : null
  const byName = new Map(params.map(p => [p.name, p]))
  const keep = (a: SweepAxis | null | undefined) => (a && byName.get(a.param)?.sweepable ? a : null)

  const modeOpt = desc.runOptions?.find(o => o.name === 'mode')
  const mode = old && modeOpt?.options?.includes(old.mode) ? old.mode : String(modeOpt?.default ?? 'bifurcation')
  const x = keep(old?.x) ?? axisFor(first)
  const keptY = keep(old?.y)
  const otherY = sweepable.find(p => p.name !== x.param)
  const y = keptY && keptY.param !== x.param ? keptY : otherY ? axisFor(otherY) : null

  const values: Record<string, number> = {}
  for (const p of params) {
    const was = old?.params[p.name]
    values[p.name] =
      was !== undefined && p.min !== null && p.max !== null ? clamp(was, p.min, p.max) : (p.default ?? 0)
  }
  const options: Record<string, string | number> = {}
  for (const o of desc.runOptions ?? []) {
    if (o.name === 'mode') continue
    const was = old?.options[o.name]
    options[o.name] = was !== undefined && parseOption(o, String(was)) !== null ? was : optionDefault(o, mode)
  }
  return { path: desc.path, rhsHash: desc.system.rhsHash, mode, x, y, params: values, options }
}

function setMode(desc: Describe, f: Form, mode: string): Form {
  const options = { ...f.options }
  for (const o of desc.runOptions ?? []) {
    if (o.modeDefaults) options[o.name] = optionDefault(o, mode)
  }
  return { ...f, mode, options }
}

// ── server round trips ────────────────────────────────────────────────────────
/** Describes the RHS file at `path` into the pane's state. */
async function load($: EngineInterface, path: string): Promise<Describe | null> {
  await update($, rhsPath, () => path)
  try {
    const desc = (await callServer($, 'describe', { path })) as Describe
    await update($, described, () => desc)
    await update($, callError, () => '')
    if (desc.ok) {
      // Same file as the pane's form: carry it over; else the settings last
      // used for this file (kept across sessions), else defaults.
      const current = await read($, form)
      const stored = current?.path === desc.path ? null : ((await $.store.get(formKey(desc.path))) as Form | undefined)
      await update($, form, prev => buildForm(desc, stored ?? prev))
    }
    await $.store.set('lastRhs', path)
    return desc
  } catch (err) {
    await update($, callError, () => `${SERVER}: ${(err as Error).message}`)
    return null
  }
}

type Listed = { runs?: RunEntry[]; inspect?: string }

async function listRuns($: EngineInterface, last: number): Promise<Listed> {
  return (await callServer($, 'list_runs', { session: await $.session.id(), last })) as Listed
}

async function refreshRuns($: EngineInterface): Promise<void> {
  try {
    const listed = await listRuns($, 50)
    await update($, runs, () => (Array.isArray(listed.runs) ? listed.runs : []))
  } catch (err) {
    await update($, callError, () => `${SERVER}: ${(err as Error).message}`)
  }
}

function stopPolling(): void {
  poller?.cancel()
  poller = null
}

function startPolling($: EngineInterface): void {
  if (poller) return
  poller = $.clock.every(POLL_MS, () => void poll($))
}

async function poll($: EngineInterface): Promise<void> {
  if (polling) return
  polling = true
  try {
    const current = await read($, job)
    if (!current || current.state !== 'running') return stopPolling()
    const st = (await callServer($, 'job_status', { job_id: current.jobId })) as Job
    await update($, job, () => ({ ...current, state: st.state, progress: st.progress, message: st.message, error: st.error }))
    if (st.state !== 'running') {
      stopPolling()
      await refreshRuns($)
      if (st.state === 'done') await update($, selected, () => current.runId)
      if (st.state === 'error') $.ui.toast(`Run ${current.runId} failed`)
    }
  } catch (err) {
    stopPolling()
    await update($, job, j => (j ? { ...j, state: 'error' as const, error: (err as Error).message } : j))
  } finally {
    polling = false
  }
}

async function run($: EngineInterface): Promise<void> {
  const [desc, f, current] = [await read($, described), await read($, form), await read($, job)]
  if (!desc?.ok || !f || current?.state === 'running') return
  const sweep: Record<string, SweepAxis> = { x: f.x }
  if (f.mode === 'period-map') {
    if (!f.y) return void $.ui.toast('A period map needs two sweepable parameters')
    sweep.y = f.y
  }
  const runOptions: Record<string, string | number> = { mode: f.mode }
  for (const o of desc.runOptions ?? []) {
    const v = f.options[o.name]
    if (appliesTo(o, f.mode) && v !== undefined) runOptions[o.name] = v
  }
  try {
    const started = (await callServer($, 'submit_run', {
      rhs_path: desc.path,
      session: await $.session.id(),
      sweep,
      params: f.params,
      run_options: runOptions,
    })) as { jobId: string; runId: string }
    await update($, job, () => ({ ...started, state: 'running' as const, progress: 0, message: 'Queued' }))
    startPolling($)
  } catch (err) {
    await update($, job, () => ({
      jobId: '', runId: '', state: 'error' as const, progress: 0, message: 'Not started', error: (err as Error).message,
    }))
  }
}

async function cancel($: EngineInterface): Promise<void> {
  const current = await read($, job)
  if (current?.state !== 'running') return
  try {
    await callServer($, 'cancel_job', { job_id: current.jobId })
    await update($, job, j => (j ? { ...j, message: 'Cancelling…' } : j))
  } catch (err) {
    $.ui.toast(`Cancel failed: ${(err as Error).message}`)
  }
}

function formKey(path: string): string {
  return `form:${path}`
}

/** Changes the pane's form and remembers it for this file across sessions. */
async function changeForm($: EngineInterface, fn: (f: Form) => Form): Promise<void> {
  const next = await update($, form, cur => (cur ? fn(cur) : cur))
  if (next) await $.store.set(formKey(next.path), next)
}

/** Whether a desktop viewer can open (X11 or Wayland display set). */
async function hasDisplay($: EngineInterface): Promise<boolean> {
  return Boolean((await $.env.get('DISPLAY')) || (await $.env.get('WAYLAND_DISPLAY')))
}

/**
 * Opens a picture in the desktop's viewer. Detached (setsid -f): a viewer
 * left running would otherwise hold $.process.run until its timeout.
 */
async function openExternal($: EngineInterface, path: string): Promise<void> {
  try {
    const { exitCode, stderr } = await $.process.run(['setsid', '-f', 'xdg-open', path], { timeoutMs: 10_000 })
    if (exitCode !== 0) $.ui.toast(`xdg-open failed (${exitCode}): ${stderr.trim().slice(0, 200)}`)
  } catch (err) {
    $.ui.toast(`Could not open the viewer: ${(err as Error).message}`)
  }
}

/** Half-block cells for a picture at a size, from the server; cached per size. */
async function rasterFor($: EngineInterface, png: string, columns: number, rows: number): Promise<Cells | null> {
  const key = `${png}|${columns}|${rows}`
  const hit = rasterCache.get(key)
  if (hit) return hit
  try {
    const cells = (await callServer($, 'raster', { png_path: png, columns, rows })) as Cells
    if (rasterCache.size > 16) rasterCache.clear()
    rasterCache.set(key, cells)
    return cells
  } catch {
    return null
  }
}

async function isKittyTerminal($: EngineInterface): Promise<boolean> {
  const [term, program, kitty] = [await $.env.get('TERM'), await $.env.get('TERM_PROGRAM'), await $.env.get('KITTY_WINDOW_ID')]
  return Boolean(kitty) || /kitty|ghostty/i.test(`${term ?? ''} ${program ?? ''}`)
}

function runLabel(r: RunEntry): string {
  const ax = (a?: { param: string; min: number; max: number }) => (a ? `${a.param} ${fmt(a.min)}…${fmt(a.max)}` : '')
  const kind = r.mode === 'period-map' ? 'map' : '1-D'
  const axes = [ax(r.sweep?.x), ax(r.sweep?.y)].filter(Boolean).join(' × ')
  const state = r.state === 'done' ? '' : `  [${r.state}]`
  return `${r.id}  ${r.system?.name ?? ''} ${kind}  ${axes}${state}`
}

function summaryLine(r: RunEntry): string {
  const s = r.summary ?? {}
  if (r.mode === 'period-map') {
    const periods = Object.entries((s.periods ?? {}) as Record<string, number>)
      .map(([k, v]) => `${k}: ${v}`)
      .join(', ')
    return `${String(s.cells ?? '?')} cells · periods ${periods}`
  }
  return `${String(s.points ?? '?')} points · ${String(s.emptyColumns ?? 0)}/${String(s.columns ?? '?')} empty columns`
}

// ── what Claude reads ─────────────────────────────────────────────────────────
/** The context line Claude reads after it writes or edits an RHS file. */
function contractNote(desc: Describe, file: string, isLoaded: boolean): string {
  if (!desc.ok) {
    return [
      `bif-studio: ${file} breaks the RHS contract (${desc.errors.length} error(s)):`,
      ...desc.errors.map(err => `- ${err}`),
      'Fix these; the file is checked again on every Write/Edit.',
    ].join('\n')
  }
  const params = (desc.params ?? []).map(p => `${p.name}${p.sweepable ? '' : ' (fixed)'}`).join(', ')
  const where = isLoaded ? 'The /bif pane reloaded it.' : `The user can open it with /bif ${file}.`
  const warn = desc.warnings.length ? ` Warnings: ${desc.warnings.join('; ')}` : ''
  return `bif-studio: ${file} passes the RHS contract (${desc.system?.name}, ${desc.system?.kind}, params: ${params}). ${where}${warn}`
}

function sweepText(r: RunEntry): string {
  const ax = (a?: { param: string; min: number; max: number }) => (a ? `${a.param} ${fmt(a.min)}…${fmt(a.max)}` : '')
  return [ax(r.sweep?.x), ax(r.sweep?.y)].filter(Boolean).join(' × ')
}

function fixedText(r: RunEntry): string {
  const swept = new Set(Object.values(r.sweep ?? {}).map(a => a.param))
  return Object.entries(r.params ?? {})
    .filter(([k]) => !swept.has(k))
    .map(([k, v]) => `${k}=${fmt(v)}`)
    .join(' ')
}

/** The prompt the pane's "Ask Claude" field submits. */
function askText(question: string, r: RunEntry, inspect: string): string {
  return [
    `[Bifurcation Studio] Question about ${r.id} (${r.system?.name ?? '?'}, ${r.mode ?? '?'}, ${sweepText(r)}${fixedText(r) ? `, fixed ${fixedText(r)}` : ''}):`,
    question,
    '',
    `Data: ${r.paths?.npz ?? '–'}`,
    `Preview: ${r.paths?.preview ?? '–'} (view only if the numbers are not enough)`,
    `Analyse with: ${inspect} summary ${r.paths?.npz ?? '<npz>'}  (also slice / compare; never print raw arrays)`,
  ].join('\n')
}

/** Describes a just-written file and returns the note for Claude, or null. */
async function checkWrittenRhs($: EngineInterface, file: string, content: string | undefined): Promise<string | null> {
  const loaded = await read($, rhsPath)
  const isLoaded = file === loaded
  // Other files only when they look like an RHS file (a Write that defines rhs).
  if (!isLoaded && !(content !== undefined && /@cuda\.jit[\s\S]*def\s+rhs\s*\(/.test(content))) return null
  try {
    const desc = isLoaded ? await load($, file) : ((await callServer($, 'describe', { path: file })) as Describe)
    return desc ? contractNote(desc, file, isLoaded) : null
  } catch {
    return null
  }
}

// ── hooks ─────────────────────────────────────────────────────────────────────
export const register: Register = on => {
  on('session.start', async ($, e, next) => {
    const started = await next(e)
    // First, before anything that waits on the server: a call that hangs
    // (server reconnecting) must not keep /bif from existing.
    await $.command.register({
      name: 'bif',
      description: 'Open Bifurcation Studio on an RHS file',
      argumentHint: '[path/to/rhs.py]',
    })
    if ((await $.store.get('showAdvanced')) === true) await update($, showAdvanced, () => true)
    const last = await $.store.get('lastRhs')
    if (typeof last === 'string' && (await read($, rhsPath)) === '') {
      await update($, rhsPath, () => last)
    }
    // A reload drops timers: pick a running job back up.
    if ((await read($, job))?.state === 'running') startPolling($)
    // Python owns deletion; keep this session's folder whatever its age.
    // Not awaited: nothing here depends on it, and the server may be down.
    void $.session
      .id()
      .then(id => callServer($, 'cleanup', { older_than_hours: 24, keep_session: id }))
      .catch(() => {})
    return started
  })

  // Claude writes or edits the RHS file: check it against the contract and
  // tell Claude in the same turn (context on the tool result), not a new prompt.
  on('tool.call', { tool: 'Write' }, async ($, e, next) => {
    const result = await next(e)
    if (result.deny !== undefined || result.isError) return result
    const note = await checkWrittenRhs($, e.file_path, e.content)
    return note ? { ...result, context: [...(result.context ?? []), note] } : result
  }).catch(($, e, next) => next(e)) // a failed check never touches the write itself

  on('tool.call', { tool: 'Edit' }, async ($, e, next) => {
    const result = await next(e)
    if (result.deny !== undefined || result.isError) return result
    const note = await checkWrittenRhs($, e.file_path, undefined)
    return note ? { ...result, context: [...(result.context ?? []), note] } : result
  }).catch(($, e, next) => next(e))

  // Claude saved or kept a system through the server: if it is the one the
  // pane shows, follow it (a kept draft moves to the permanent folder).
  on('tool.call', { tool: new RegExp(`^mcp__${SERVER}__(save_system|keep_system)$`) }, async ($, e, next) => {
    const result = await next(e)
    if (result.deny !== undefined || result.isError || !result.text) return result
    const loaded = await read($, rhsPath)
    try {
      const out = JSON.parse(result.text) as { name?: string; path?: string }
      const draftOfLoaded = loaded.endsWith(`/drafts/${out.name}.py`)
      if (out.path && (out.path === loaded || (draftOfLoaded && e.tool.endsWith('keep_system')))) {
        await load($, out.path)
      }
    } catch {
      // Not JSON: nothing to follow.
    }
    return result
  }).catch(($, e, next) => next(e))

  on('command.run', { command: 'bif' }, async ($, e) => {
    const arg = e.args.trim()
    let path: string
    try {
      path = arg ? await resolveArg($, arg) : await read($, rhsPath)
    } catch (err) {
      return { text: `Bifurcation Studio: ${(err as Error).message}` }
    }
    let text = 'Bifurcation Studio opened. Give it a system: /bif <name> or /bif path/to/rhs.py'
    if (path) {
      const desc = await load($, path)
      if (desc) await refreshRuns($)
      text = !desc
        ? `Bifurcation Studio: could not reach ${SERVER}: ${await read($, callError)}`
        : desc.ok
          ? `Bifurcation Studio: ${desc.system?.name} loaded.`
          : `Bifurcation Studio: ${path} has ${desc.errors.length} contract error(s).`
    }
    await $.ui.open({ id: PANE, title: 'Bifurcation', focus: true })
    return { text }
  })

  on('ui.render', { component: 'Pane', requestId: PANE }, async ($, e) => {
    if (e.surface !== 'terminal') {
      const { Box, Text } = $.ui.resolve(e)
      return (
        <Box>
          <Text dimColor>Bifurcation Studio draws in the terminal for now.</Text>
        </Box>
      )
    }
    const { Box, Text, Button, Select, Input, Raster, Image } = $.ui.resolve(e)
    const path = await read($, rhsPath)
    const desc = await read($, described)
    const error = await read($, callError)
    const f = await read($, form)
    const advanced = await read($, showAdvanced)
    const j = await read($, job)
    const list = await read($, runs)
    const pick = await read($, selected)
    const width = Math.max(30, e.props.bodyColumns ?? 60)
    const pad = (s: string, n: number) => s + ' '.repeat(Math.max(1, n - s.length))
    const isRunning = j?.state === 'running'

    const shown = desc && desc.path === path ? desc : null
    const sys = shown?.system
    const params = shown?.params ?? []
    const nameWidth = Math.min(16, Math.max(4, ...params.map(p => p.label.length)) + 1)
    const barWidth = Math.max(8, Math.min(40, width - nameWidth - 26))
    const setForm = (fn: (f: Form) => Form) => void changeForm($, fn)

    const header = (
      <Box key="header" flexDirection="column">
        {sys ? (
          <Text>
            <Text bold>{sys.name}</Text>
            <Text dimColor>{`  ${sys.kind} · dim ${sys.dim} · ${sys.rhsHash.slice(7, 15)}`}</Text>
          </Text>
        ) : (
          <Text bold>Bifurcation Studio</Text>
        )}
        <Text dimColor wrap="truncate-start">{path || 'No system yet: /bif <name> or /bif path/to/rhs.py'}</Text>
        {error !== '' && <Text color="error">{error}</Text>}
      </Box>
    )

    const problems = shown && (
      <Box key="problems" flexDirection="column">
        {!shown.ok && <Text color="error" bold>{`Contract errors (${shown.errors.length})`}</Text>}
        {shown.errors.map((msg, i) => (
          <Box key={`err-${i}`}>
            <Text color="error">{`  ${msg}`}</Text>
          </Box>
        ))}
        {shown.warnings.map((msg, i) => (
          <Box key={`warn-${i}`}>
            <Text color="warning">{msg}</Text>
          </Box>
        ))}
      </Box>
    )

    // One sweep axis: which parameter, and its range.
    const axisRow = (axis: 'x' | 'y', a: SweepAxis, exclude: string) => (
      <Box key={`axis-${axis}`} flexDirection="column">
        <Select
          key={`sweep-${axis}`}
          label={`Sweep ${axis}`}
          value={a.param}
          options={params.filter(p => p.sweepable && p.name !== exclude).map(p => ({ value: p.name, label: p.label }))}
          onSelect={v => {
            const p = params.find(q => q.name === v)
            if (p) setForm(cur => ({ ...cur, [axis]: axisFor(p) }))
          }}
        />
        <Box>
          {(['min', 'max'] as const).map(end => (
            <Input
              key={`sweep-${axis}-${end}`}
              label={`${axis} ${end}`}
              value={fmt(a[end])}
              onSubmit={text => {
                const v = Number(text)
                const p = params.find(q => q.name === a.param)
                const lo = p?.min ?? -Infinity
                const hi = p?.max ?? Infinity
                const other = end === 'min' ? a.max : a.min
                const ok = Number.isFinite(v) && v >= lo && v <= hi && (end === 'min' ? v < other : v > other)
                if (!ok) return void $.ui.toast(`${axis} ${end} must be in [${fmt(lo)}, ${fmt(hi)}] and keep min < max`)
                setForm(cur => {
                  const cura = cur[axis]
                  return cura ? { ...cur, [axis]: { ...cura, [end]: v } } : cur
                })
              }}
            />
          ))}
        </Box>
      </Box>
    )

    const swept = new Set(f ? [f.x.param, ...(f.mode === 'period-map' && f.y ? [f.y.param] : [])] : [])

    const controls = shown?.ok && f && (
      <Box key="controls" flexDirection="column" marginTop={1}>
        <Select
          key="mode"
          label="Mode"
          value={f.mode}
          options={(shown.runOptions?.find(o => o.name === 'mode')?.options ?? []).map(m => ({ value: m }))}
          onSelect={m => setForm(cur => setMode(shown, cur, m))}
        />
        {axisRow('x', f.x, '')}
        {f.mode === 'period-map' && f.y && axisRow('y', f.y, f.x.param)}

        <Text bold>Parameters</Text>
        {params.map(p => {
          const v = f.params[p.name] ?? p.default ?? 0
          if (swept.has(p.name)) {
            return (
              <Box key={`param-${p.name}`}>
                <Text>
                  {`  ${pad(p.label, nameWidth)}`}
                  <Text color="suggestion">{`swept on ${f.x.param === p.name ? 'x' : 'y'}`}</Text>
                </Text>
              </Box>
            )
          }
          const lo = p.min ?? v - 1
          const hi = p.max ?? v + 1
          const step = (hi - lo) / STEPS
          const filled = Math.round(((v - lo) / (hi - lo || 1)) * barWidth)
          const nudge = (dir: number) =>
            setForm(cur => {
              const now = cur.params[p.name] ?? v
              const next = Number(clamp(now + dir * step, lo, hi).toPrecision(6))
              return { ...cur, params: { ...cur.params, [p.name]: next } }
            })
          return (
            <Box key={`param-${p.name}`}>
              <Text>{`  ${pad(p.label, nameWidth)}`}</Text>
              <Button key={`dec-${p.name}`} plain onPress={() => nudge(-1)}>
                −
              </Button>
              <Text color="suggestion">{` ${'█'.repeat(filled)}`}</Text>
              <Text dimColor>{`${'░'.repeat(Math.max(0, barWidth - filled))} `}</Text>
              <Button key={`inc-${p.name}`} plain onPress={() => nudge(1)}>
                +
              </Button>
              <Input
                key={`val-${p.name}`}
                value={fmt(v)}
                onSubmit={text => {
                  const n = Number(text)
                  if (!Number.isFinite(n) || n < lo || n > hi) return void $.ui.toast(`${p.name} must be in [${fmt(lo)}, ${fmt(hi)}]`)
                  setForm(cur => ({ ...cur, params: { ...cur.params, [p.name]: n } }))
                }}
              />
            </Box>
          )
        })}

        <Box>
          <Button
            key="advanced"
            hotkey="a"
            plain
            onPress={() => void update($, showAdvanced, s => !s).then(v => $.store.set('showAdvanced', v))}
          >
            {advanced ? '▾ Run options' : '▸ Run options'}
          </Button>
        </Box>
        {advanced &&
          (shown.runOptions ?? [])
            .filter(o => appliesTo(o, f.mode))
            .map(o =>
              o.type === 'enum' ? (
                <Select
                  key={`opt-${o.name}`}
                  label={o.name}
                  value={String(f.options[o.name] ?? o.default)}
                  options={(o.options ?? []).map(v => ({ value: v }))}
                  onSelect={v => setForm(cur => ({ ...cur, options: { ...cur.options, [o.name]: v } }))}
                />
              ) : (
                <Input
                  key={`opt-${o.name}`}
                  label={o.name}
                  value={fmt(f.options[o.name] ?? o.default)}
                  onSubmit={text => {
                    const v = parseOption(o, text.trim())
                    if (v === null) return void $.ui.toast(`${o.name}: ${o.type} in [${fmt(o.min)}, ${fmt(o.max)}]`)
                    setForm(cur => ({ ...cur, options: { ...cur.options, [o.name]: v } }))
                  }}
                />
              ),
            )}
      </Box>
    )

    const actions = (
      <Box key="actions" marginTop={1}>
        {shown?.ok && f && (
          <Button key="run" hotkey="r" variant="primary" onPress={() => void run($)}>
            {isRunning ? 'Running…' : 'Run'}
          </Button>
        )}
        {isRunning && (
          <Button key="cancel" hotkey="x" onPress={() => void cancel($)}>
            Cancel
          </Button>
        )}
        {path !== '' && (
          <Button key="reload" hotkey="d" onPress={() => void load($, path)}>
            Re-describe
          </Button>
        )}
      </Box>
    )

    const barCols = Math.max(10, Math.min(40, width - 30))
    const progress = j && (j.state === 'running' || j.state === 'error') && (
      <Box key="progress" flexDirection="column">
        {j.state === 'running' && (
          <Text>
            <Text color="suggestion">{'█'.repeat(Math.round((j.progress / 100) * barCols))}</Text>
            <Text dimColor>{'░'.repeat(barCols - Math.round((j.progress / 100) * barCols))}</Text>
            {` ${j.progress}%  ${j.message}`}
          </Text>
        )}
        {j.state === 'error' && <Text color="error">{`${j.runId || 'Run'} failed: ${j.error ?? j.message}`}</Text>}
      </Box>
    )

    // The selected run, else the newest finished one.
    const done = list.filter(r => r.state === 'done')
    const current = list.find(r => r.id === pick) ?? done[done.length - 1]
    let picture = null
    if (current?.paths?.preview) {
      const rows = Math.min(32, Math.max(8, Math.round(width * 0.3)))
      const cells = await rasterFor($, current.paths.preview, width, rows)
      if (cells) {
        picture = (await isKittyTerminal($)) ? (
          <Image
            key={`img-${current.id}`}
            source={{ file: current.paths.preview, format: 'png' }}
            columns={cells.columns}
            rows={cells.rows}
            alt={`${current.id} preview`}
          />
        ) : (
          <Raster key={`raster-${current.id}`} columns={cells.columns} rows={cells.rows} cells={cells.cells} />
        )
      }
    }

    const viewer = current?.state === 'done' && (await hasDisplay($))
    const result = current && (
      <Box key="result" flexDirection="column" marginTop={1}>
        <Text>
          <Text bold>{current.id}</Text>
          <Text dimColor>{`  ${current.state === 'done' ? summaryLine(current) : current.error ?? current.state}`}</Text>
        </Text>
        {picture}
        {current.state === 'done' && viewer && (
          <Box key="viewer">
            {current.paths?.chart && (
              <Button key="open-chart" hotkey="o" onPress={() => void openExternal($, current.paths?.chart ?? '')}>
                Open chart
              </Button>
            )}
            {current.paths?.png && (
              <Button key="open-png" hotkey="p" onPress={() => void openExternal($, current.paths?.png ?? '')}>
                Open full-res
              </Button>
            )}
          </Box>
        )}
        {current.state === 'done' && (
          <Input
            key="ask"
            label="Ask Claude"
            placeholder="e.g. where does chaos start, and is there a period-3 window?"
            submitLabel="Ask"
            onSubmit={question => {
              const q = question.trim()
              if (!q) return
              // Not awaited: it resolves when Claude's turn starts.
              void listRuns($, 1)
                .then(listed => listed.inspect ?? 'python runinspect.py')
                .catch(() => 'python runinspect.py')
                .then(inspect => $.prompt.submit({ text: askText(q, current, inspect) }))
              $.ui.toast(`Asked Claude about ${current.id}`)
            }}
          />
        )}
        {list.length > 1 && (
          <Select
            key="history"
            label="Runs"
            value={current.id}
            options={[...list].reverse().map(r => ({ value: r.id, label: runLabel(r) }))}
            onSelect={id => void update($, selected, () => id)}
          />
        )}
      </Box>
    )

    return (
      <Box flexDirection="column" width={width}>
        {header}
        {problems}
        {controls}
        {actions}
        {progress}
        {result}
      </Box>
    )
  })
}
