import { describe, expect, mock, test } from 'claude-code/testing'
import type { Engine } from 'claude-code/testing'
import type { On } from 'claude-code'

const ROSSLER = {
  contractVersion: 2,
  ok: true,
  path: '/work/rossler.py',
  system: { name: 'rossler', kind: 'ode', rhsHash: 'sha256:1739ebe89f8b1bcd', dim: 3, nParams: 3 },
  params: [
    { index: 0, name: 'c', label: 'c', default: 5.7, min: 1, max: 20, sweepable: true, sweep: [2, 8] },
    { index: 1, name: 'a', label: 'a', default: 0.2, min: 0, max: 0.5, sweepable: true, sweep: [0, 0.5] },
    { index: 2, name: 'b', label: 'b', default: 0.2, min: 0, max: 2, sweepable: false, sweep: [0, 2] },
  ],
  state: [{ index: 0, name: 'x', default: 0.1 }],
  runOptions: [
    { name: 'mode', type: 'enum', options: ['bifurcation', 'period-map'], default: 'bifurcation' },
    { name: 'param_steps', type: 'int', default: 2000, modeDefaults: { 'period-map': 300 }, min: 10, max: 20000 },
    { name: 'eps_scale', type: 'float', default: 0.01, min: 0.0001, max: 1, modes: ['period-map'] },
  ],
  errors: [],
  warnings: [],
}

const BROKEN = {
  contractVersion: 2,
  ok: false,
  path: '/work/rossler.py',
  errors: ["line 7: PARAMS[1] ('b'): missing 'max'"],
  warnings: [],
}

const DONE_RUN = {
  id: 'run-0001',
  state: 'done',
  system: { name: 'rossler' },
  mode: 'bifurcation',
  sweep: { x: { param: 'c', min: 2, max: 8 } },
  summary: { points: 34629, emptyColumns: 0, columns: 2000 },
  paths: { preview: '/cache/s1/run-0001.preview.png', png: '/cache/s1/run-0001.png' },
}

type Call = { tool: string; args: Record<string, unknown> }

/**
 * Stands in for the Python server and the rest of the engine. `server` maps a
 * tool to its JSON reply (or an Error); every call is recorded in `calls`.
 */
function world(on: On, server: Record<string, (args: Record<string, unknown>) => unknown>, env: Record<string, string> = {}) {
  const calls: Call[] = []
  on('session.cwd', async () => ({ value: '/work' }))
  on('session.id', async () => ({ value: 's1' }))
  on('ui.open', async () => ({ value: { isPlaced: true as const } }))
  mock.store(on)
  mock.env(on, env)
  on('mcp.call', async (_$, e) => {
    calls.push({ tool: e.tool, args: { ...e.args } })
    const answer = (server[e.tool] ?? (() => ({})))({ ...e.args })
    const isError = answer instanceof Error
    const text = isError ? answer.message : JSON.stringify(answer)
    return { value: { content: [{ type: 'text' as const, text }], isError } }
  })
  return calls
}

/** /bif typed at the prompt. */
function bif($: Engine, args: string) {
  return $.command.run({
    command: 'bif', args, origin: { kind: 'composer' },
    presentation: { isFullscreen: true, columns: 160 },
  })
}

function mountPane($: Engine) {
  return $.ui.mount({
    plugin: 'bif-studio',
    surface: 'terminal',
    component: 'Pane',
    requestId: 'bif',
    props: {
      title: 'Bifurcation', isFocused: true, bodyColumns: 80, placement: 'dock',
      scroll: { offset: 0, bodyRows: 40 }, view: {},
    },
  })
}

// Two half-block cells, red over blue (what studio.raster packs).
const CELLS = { columns: 2, rows: 1, cells: 'gCUAAAAA/wD/AAAAgCUAAAAA/wD/AAAA' }

describe('/bif describe', () => {
  test('describes the file through bifurcation-studio and builds controls', async ($, on) => {
    const calls = world(on, { describe: () => ROSSLER, list_runs: () => ({ runs: [] }) })
    const out = await bif($, 'rossler.py')
    expect(out.text).toContain('rossler loaded')
    expect(calls.find(c => c.tool === 'describe')?.args.path).toBe('/work/rossler.py')

    const ui = await mountPane($)
    expect((await ui.find({ key: 'param-c' }))?.text).toContain('swept on x')
    expect((await ui.find({ key: 'val-a' }))?.props.value).toBe('0.2')
    expect((await ui.find({ key: 'sweep-x' }))?.props.value).toBe('c')
    expect((await ui.find({ key: 'sweep-x-min' }))?.props.value).toBe('2')
    expect(await ui.find({ key: 'run' })).toBeDefined()
    expect(await ui.find({ key: 'axis-y' })).toBeUndefined()
  })

  test('shows contract errors with their line numbers, and no Run', async ($, on) => {
    world(on, { describe: () => BROKEN, list_runs: () => ({ runs: [] }) })
    const out = await bif($, '/work/rossler.py')
    expect(out.text).toContain('1 contract error')

    const ui = await mountPane($)
    expect((await ui.find({ key: 'err-0' }))?.text).toContain("line 7: PARAMS[1] ('b'): missing 'max'")
    expect(await ui.find({ key: 'run' })).toBeUndefined()
  })

  test('Re-describe picks up a fixed file', async ($, on) => {
    let fixed = false
    world(on, { describe: () => (fixed ? ROSSLER : BROKEN), list_runs: () => ({ runs: [] }) })
    await bif($, 'rossler.py')
    const ui = await mountPane($)
    expect(await ui.find({ key: 'err-0' })).toBeDefined()

    fixed = true
    await ui.press({ key: 'reload' })
    expect(await ui.find({ key: 'err-0' })).toBeUndefined()
    expect(await ui.find({ key: 'run' })).toBeDefined()
  })

  test('says when the server cannot be reached', async ($, on) => {
    world(on, { describe: () => new Error('server not connected') })
    const out = await bif($, 'rossler.py')
    expect(out.text).toContain('could not reach bifurcation-studio: bifurcation-studio: server not connected')
  })
})

describe('/bif controls', () => {
  test('steppers move a parameter by (max - min) / 50 and stay in range', async ($, on) => {
    world(on, { describe: () => ROSSLER, list_runs: () => ({ runs: [] }) })
    await bif($, 'rossler.py')
    const ui = await mountPane($)
    await ui.press({ key: 'inc-a' })
    expect((await ui.find({ key: 'val-a' }))?.props.value).toBe('0.21')
    for (let i = 0; i < 40; i++) await ui.press({ key: 'dec-a' })
    expect((await ui.find({ key: 'val-a' }))?.props.value).toBe('0')
  })

  test('typed values are checked against the range', async ($, on) => {
    world(on, { describe: () => ROSSLER, list_runs: () => ({ runs: [] }) })
    await bif($, 'rossler.py')
    const ui = await mountPane($)
    await ui.input({ key: 'val-b', text: '1.5' })
    expect((await ui.find({ key: 'val-b' }))?.props.value).toBe('1.5')
    await ui.input({ key: 'val-b', text: '9' })
    expect((await ui.find({ key: 'val-b' }))?.props.value).toBe('1.5')
    await ui.input({ key: 'sweep-x-max', text: '1' }) // below min: refused
    expect((await ui.find({ key: 'sweep-x-max' }))?.props.value).toBe('8')
  })

  test('period-map mode adds a y axis and its mode defaults', async ($, on) => {
    const calls = world(on, { describe: () => ROSSLER, list_runs: () => ({ runs: [] }), submit_run: () => ({ jobId: 'j1', runId: 'run-0001' }) })
    await bif($, 'rossler.py')
    const ui = await mountPane($)
    await ui.select({ key: 'mode', value: 'period-map' })
    expect((await ui.find({ key: 'sweep-y' }))?.props.value).toBe('a')
    expect((await ui.find({ key: 'param-a' }))?.text).toContain('swept on y')

    await ui.press({ key: 'run' })
    const sent = calls.find(c => c.tool === 'submit_run')?.args
    expect(sent?.sweep).toEqual({ x: { param: 'c', min: 2, max: 8 }, y: { param: 'a', min: 0, max: 0.5 } })
    expect(sent?.run_options).toEqual({ mode: 'period-map', param_steps: 300, eps_scale: 0.01 })
  })
})

describe('/bif runs', () => {
  test('Run submits, polls to done, and shows the preview as a Raster', async ($, on) => {
    const clock = mock.clock(on)
    let state = 'running'
    const calls = world(on, {
      describe: () => ROSSLER,
      list_runs: () => ({ runs: state === 'done' ? [DONE_RUN] : [] }),
      submit_run: () => ({ jobId: 'j1', runId: 'run-0001' }),
      job_status: () => ({ state, progress: state === 'done' ? 100 : 40, message: state === 'done' ? 'Done' : 'Running GPU integration' }),
      raster: () => CELLS,
    })
    await bif($, 'rossler.py')
    const ui = await mountPane($)
    await ui.press({ key: 'inc-b' })
    await ui.press({ key: 'run' })

    const sent = calls.find(c => c.tool === 'submit_run')?.args
    expect(sent?.session).toBe('s1')
    expect(sent?.sweep).toEqual({ x: { param: 'c', min: 2, max: 8 } })
    expect(sent?.params).toEqual({ c: 5.7, a: 0.2, b: 0.24 })
    expect(sent?.run_options).toEqual({ mode: 'bifurcation', param_steps: 2000 })
    expect(await ui.find({ key: 'cancel' })).toBeDefined()

    await clock.advance(500)
    expect((await ui.find({ key: 'progress' }))?.text).toContain('40%')

    state = 'done'
    await clock.advance(500)
    expect(await ui.find({ key: 'progress' })).toBeUndefined()
    expect(await ui.find({ key: 'cancel' })).toBeUndefined()
    expect((await ui.find({ key: 'result' }))?.text).toContain('34629 points')
    const raster = await ui.find({ key: 'raster-run-0001' })
    expect(raster?.props).toMatchObject(CELLS)
    expect(calls.find(c => c.tool === 'raster')?.args.png_path).toBe(DONE_RUN.paths.preview)

    // Polling stopped: no more job_status calls.
    const polls = calls.filter(c => c.tool === 'job_status').length
    await clock.advance(2000)
    expect(calls.filter(c => c.tool === 'job_status').length).toBe(polls)
  })

  test('kitty terminals get an Image instead', async ($, on) => {
    world(on, { describe: () => ROSSLER, list_runs: () => ({ runs: [DONE_RUN] }), raster: () => CELLS }, { TERM: 'xterm-kitty' })
    await bif($, 'rossler.py')
    const ui = await mountPane($)
    expect((await ui.find({ key: 'img-run-0001' }))?.props.source).toEqual({ file: DONE_RUN.paths.preview, format: 'png' })
    expect(await ui.find({ key: 'raster-run-0001' })).toBeUndefined()
  })

  test('Cancel asks the server to stop the job', async ($, on) => {
    mock.clock(on)
    const calls = world(on, {
      describe: () => ROSSLER,
      list_runs: () => ({ runs: [] }),
      submit_run: () => ({ jobId: 'j1', runId: 'run-0001' }),
      cancel_job: () => ({ jobId: 'j1', state: 'running' }),
    })
    await bif($, 'rossler.py')
    const ui = await mountPane($)
    await ui.press({ key: 'run' })
    await ui.press({ key: 'cancel' })
    expect(calls.find(c => c.tool === 'cancel_job')?.args).toEqual({ job_id: 'j1' })
    expect((await ui.find({ key: 'progress' }))?.text).toContain('Cancelling')
  })

  test('a failed run shows the server error', async ($, on) => {
    const clock = mock.clock(on)
    world(on, {
      describe: () => ROSSLER,
      list_runs: () => ({ runs: [] }),
      submit_run: () => ({ jobId: 'j1', runId: 'run-0001' }),
      job_status: () => ({ state: 'error', progress: 5, message: 'Failed', error: "TypingError: name 'undefined_name' is not defined" }),
    })
    await bif($, 'rossler.py')
    const ui = await mountPane($)
    await ui.press({ key: 'run' })
    await clock.advance(500)
    expect((await ui.find({ key: 'progress' }))?.text).toContain("run-0001 failed: TypingError: name 'undefined_name'")
  })

  test('a refused submit is shown, not thrown', async ($, on) => {
    world(on, { describe: () => ROSSLER, list_runs: () => ({ runs: [] }), submit_run: () => new Error('ValueError: sweep.x: min (3) must be < max (1)') })
    await bif($, 'rossler.py')
    const ui = await mountPane($)
    await ui.press({ key: 'run' })
    expect((await ui.find({ key: 'progress' }))?.text).toContain('min (3) must be < max (1)')
  })
})

describe('Claude integration', () => {
  /** Claude's Write/Edit, answered beneath the plugin as the engine would. */
  function fileTools(on: On) {
    on('tool.call', { tool: 'Write' }, async () => ({ result: { type: 'update' } as never }))
    on('tool.call', { tool: 'Edit' }, async () => ({ result: { filePath: 'x' } as never }))
  }

  test('a Write to the loaded RHS file comes back with its contract errors, and the pane updates', async ($, on) => {
    let broken = false
    world(on, { describe: () => (broken ? BROKEN : ROSSLER), list_runs: () => ({ runs: [] }) })
    fileTools(on)
    await bif($, 'rossler.py')
    broken = true
    const out = await $.tool.call({ tool: 'Write', file_path: '/work/rossler.py', content: 'x' } as never)
    expect(out.context?.join('\n')).toContain("breaks the RHS contract (1 error(s)):\n- line 7: PARAMS[1] ('b'): missing 'max'")
    const ui = await mountPane($)
    expect(await ui.find({ key: 'err-0' })).toBeDefined()
  })

  test('a Write of a new RHS-looking file is described without loading it', async ($, on) => {
    const calls = world(on, { describe: () => ({ ...ROSSLER, path: '/work/lorenz.py' }), list_runs: () => ({ runs: [] }) })
    fileTools(on)
    const content = 'from numba import cuda\n@cuda.jit(device=True)\ndef rhs(t, y, p, dy):\n    pass\n'
    const out = await $.tool.call({ tool: 'Write', file_path: '/work/lorenz.py', content } as never)
    expect(out.context?.join('\n')).toContain('passes the RHS contract')
    expect(out.context?.join('\n')).toContain('/bif /work/lorenz.py')
    expect(calls.find(c => c.tool === 'describe')?.args.path).toBe('/work/lorenz.py')
  })

  test('unrelated edits are left alone', async ($, on) => {
    const calls = world(on, { describe: () => ROSSLER, list_runs: () => ({ runs: [] }) })
    fileTools(on)
    const out = await $.tool.call({ tool: 'Edit', file_path: '/work/README.md', old_string: 'a', new_string: 'b' } as never)
    expect(out.context).toBeUndefined()
    expect(calls.length).toBe(0)
  })

  test('Ask Claude submits the question with the run and how to analyse it', async ($, on) => {
    world(on, { describe: () => ROSSLER, list_runs: () => ({ runs: [DONE_RUN], inspect: '/py /srv/runinspect.py' }), raster: () => CELLS })
    const asked: string[] = []
    on('prompt.submit', async (_$, e) => (asked.push(e.text), { text: e.text }))
    await bif($, 'rossler.py')
    const ui = await mountPane($)
    await ui.input({ key: 'ask', text: 'where does chaos start?' })
    expect(asked.length).toBe(1)
    expect(asked[0]).toContain('Question about run-0001 (rossler, bifurcation, c 2…8')
    expect(asked[0]).toContain('where does chaos start?')
    expect(asked[0]).toContain('/py /srv/runinspect.py summary')
  })
})

describe('named systems', () => {
  const SYSTEMS = {
    resolve: { rossler: '/srv/systems/rossler.py', chua: '/cache/drafts/chua.py' },
    systems: [{ name: 'rossler', where: 'permanent' }, { name: 'chua', where: 'draft' }],
  }

  test('/bif <name> opens the system the server resolves', async ($, on) => {
    const calls = world(on, {
      list_systems: () => SYSTEMS,
      describe: args => ({ ...ROSSLER, path: String(args.path) }),
      list_runs: () => ({ runs: [] }),
    })
    const out = await bif($, 'chua')
    expect(out.text).toContain('loaded')
    expect(calls.find(c => c.tool === 'describe')?.args.path).toBe('/cache/drafts/chua.py')
  })

  test('an unknown name lists the known ones', async ($, on) => {
    world(on, { list_systems: () => SYSTEMS })
    const out = await bif($, 'duffing')
    expect(out.text).toContain('no system named "duffing". Known: chua, rossler')
  })

  test('keeping the loaded draft moves the pane to the permanent file', async ($, on) => {
    const calls = world(on, {
      list_systems: () => SYSTEMS,
      describe: args => ({ ...ROSSLER, path: String(args.path) }),
      list_runs: () => ({ runs: [] }),
    })
    on('tool.call', { tool: /keep_system$/ }, async () => ({
      result: 'x' as never,
      text: JSON.stringify({ name: 'chua', path: '/srv/systems/chua.py', replaced: false }),
    }))
    await bif($, 'chua')
    await $.tool.call({ tool: 'mcp__bifurcation-studio__keep_system', name: 'chua' } as never)
    const described = calls.filter(c => c.tool === 'describe').map(c => c.args.path)
    expect(described).toEqual(['/cache/drafts/chua.py', '/srv/systems/chua.py'])
  })
})

describe('polish', () => {
  const RUN = { ...DONE_RUN, paths: { ...DONE_RUN.paths, chart: '/cache/s1/run-0001.chart.png' } }

  test('with a display, o opens the chart detached through xdg-open', async ($, on) => {
    world(on, { describe: () => ROSSLER, list_runs: () => ({ runs: [RUN] }), raster: () => CELLS }, { DISPLAY: ':0' })
    const ran: string[][] = []
    on('process.run', async (_$, e) => {
      ran.push([...e.argv])
      return { value: { exitCode: 0, stdout: '', stderr: '', isStdoutTruncated: false, isStderrTruncated: false } }
    })
    await bif($, 'rossler.py')
    const ui = await mountPane($)
    await ui.press({ key: 'open-chart' })
    await ui.press({ key: 'open-png' })
    expect(ran).toEqual([
      ['setsid', '-f', 'xdg-open', RUN.paths.chart],
      ['setsid', '-f', 'xdg-open', RUN.paths.png],
    ])
  })

  test('without a display there are no viewer buttons', async ($, on) => {
    world(on, { describe: () => ROSSLER, list_runs: () => ({ runs: [RUN] }), raster: () => CELLS })
    await bif($, 'rossler.py')
    const ui = await mountPane($)
    expect(await ui.find({ key: 'open-chart' })).toBeUndefined()
  })

  test('a file opens with the settings last used for it, and changes are remembered', async ($, on) => {
    world(on, {
      describe: args => {
        const path = String(args.path)
        return path.endsWith('other.py') ? { ...ROSSLER, path, system: { ...ROSSLER.system, rhsHash: 'sha256:other' } } : ROSSLER
      },
      list_runs: () => ({ runs: [] }),
    })
    await bif($, 'rossler.py')
    const ui = await mountPane($)
    await ui.input({ key: 'val-b', text: '1.25' })
    await ui.select({ key: 'mode', value: 'period-map' })

    // Another file, then back: the stored form comes back, not the defaults.
    await bif($, '/work/other.py')
    expect((await ui.find({ key: 'val-b' }))?.props.value).toBe('0.2')
    await bif($, 'rossler.py')
    expect((await ui.find({ key: 'val-b' }))?.props.value).toBe('1.25')
    expect((await ui.find({ key: 'mode' }))?.props.value).toBe('period-map')
  })
})
