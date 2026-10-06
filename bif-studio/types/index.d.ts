// What contract.describe() returns (contract v2), as the pane reads it.
export type DescribeParam = {
  index: number
  name: string
  label: string
  default: number | null
  min: number | null
  max: number | null
  sweepable: boolean
  sweep: [number, number] | null
}

export type DescribeRunOption = {
  name: string
  type: 'enum' | 'int' | 'float'
  default: string | number
  options?: string[]
  min?: number
  max?: number
  modes?: string[]
  modeDefaults?: Record<string, number>
}

export type Describe = {
  contractVersion: number
  ok: boolean
  path: string
  system?: { name: string; kind: 'ode' | 'hybrid'; rhsHash: string; dim: number; nParams: number }
  params?: DescribeParam[]
  state?: { index: number; name: string; default: number | null }[]
  runOptions?: DescribeRunOption[]
  errors: string[]
  warnings: string[]
}

/** One sweep axis as the form holds it. */
export type SweepAxis = { param: string; min: number; max: number }

/** The pane's controls, rebuilt when the RHS file's hash changes. */
export type Form = {
  /** Values carry over to a re-described file only when this path matches. */
  path: string
  rhsHash: string
  mode: string
  x: SweepAxis
  y: SweepAxis | null
  params: Record<string, number>
  options: Record<string, string | number>
}

/** The run in flight (or the last one), as job_status reports it. */
export type Job = {
  jobId: string
  runId: string
  state: 'running' | 'done' | 'error' | 'cancelled'
  progress: number
  message: string
  error?: string
}

/** One entry of the session's index.json (list_runs). */
export type RunEntry = {
  id: string
  state: string
  system?: { name: string } | null
  mode?: string
  params?: Record<string, number>
  sweep?: Record<string, { param: string; min: number; max: number }>
  summary?: Record<string, unknown> | null
  paths?: { png?: string; preview?: string; chart?: string; npz?: string; json?: string }
  error?: string | null
}

declare module 'claude-code' {
  interface PluginState {
    'bif-studio': {
      rhsPath: string
      described: Describe | null
      callError: string
      form: Form | null
      showAdvanced: boolean
      job: Job | null
      runs: RunEntry[]
      selected: string
    }
  }
}
