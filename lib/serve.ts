// The router: who serves a prompt, and what happens when the gateway refuses.
//
// quality (60% of the score) is the served model's rating over the best free model meeting the
// prompt's requirements, so the router ranks every free candidate by the brief's rating_milli and
// dials the strongest first, ties broken by id so the order never depends on anything but the brief.
// failover (10%) reads the optional x-makefun-upstream-failures trail: every refusal this server
// retried and then served must appear there, as at most 8 {"status","code","model"} rows in call
// order. A refused model never dials again this trial: one rate-limited model must not take later
// prompts down with it, and a trial's tie-break is its gateway-call count, so calls spent on a model
// that already refused are pure loss. When every candidate has refused, the refused set is dropped
// and everyone is tried once more — a last resort beats an unserved prompt.
//
// Everything here is a pure function of its arguments and the callbacks' answers: no clock, no
// randomness, so a replayed trial walks the same candidates and reports the same trail.

import { capabilitiesOf, meetsRequirements, type Json, type ModelRow, type Requirement } from './providers.ts';

/** One refused gateway call, as the trail header's rows spell it. */
export interface TrailRow {
  status: number;
  code: string;
  model: string | null;
}

/** The router's answer: one served candidate's relay, or the give-up status. */
export interface ServeOk {
  ok: true;
  model: string;
  upstream: Response;
  trail: TrailRow[];
}

export interface ServeFailed {
  ok: false;
  status: number;
  trail: TrailRow[];
}

export type ServeResult = ServeOk | ServeFailed;

/** The most trail rows a header may carry: a longer trail is read as no trail at all. */
export const TRAIL_MAX = 8;

/** A status the trail header accepts: outside 100..599 the whole header reads as no trail. */
const TRAIL_MIN_STATUS = 100;
const TRAIL_MAX_STATUS = 599;

/** An error body's code (`error.code`, or `error.type` where a provider spells it that way). */
export function errorCodeOf(text: string): string {
  let value: unknown;
  try {
    value = JSON.parse(text);
  } catch {
    return '';
  }
  if (!isJsonObject(value) || !isJsonObject(value.error)) return '';
  const error = value.error;
  if (typeof error.code === 'string' && error.code !== '') return error.code.slice(0, 64);
  if (typeof error.type === 'string' && error.type !== '') return error.type.slice(0, 64);
  return '';
}

function isJsonObject(v: unknown): v is Record<string, unknown> {
  return v !== null && typeof v === 'object' && !Array.isArray(v);
}

/**
 * The candidates that may serve one prompt, strongest first. Models that already refused this trial
 * step aside while anyone fresh is left; every row's order is a function of the brief and the
 * refusal set alone.
 */
export function candidatesFor(models: readonly ModelRow[], requires: Requirement, refused: ReadonlySet<string>): ModelRow[] {
  const eligible = models.filter((model) => meetsRequirements(requires, capabilitiesOf(model)));
  const fresh = eligible.filter((model) => !refused.has(model.id));
  const pool = fresh.length > 0 ? fresh : eligible;
  return [...pool].sort((a, b) => b.rating_milli - a.rating_milli || (a.id < b.id ? -1 : a.id > b.id ? 1 : 0));
}

/**
 * Dial the candidates in order until one answers. A refused call (a 4xx, a 5xx, an unreachable
 * gateway) marks its model, joins the trail and moves to the next candidate; the give-up answer is
 * the caller's to send. Each candidate's request body is built by the callback, so the OpenAI path
 * passes its body through and the Anthropic path translates first.
 */
export async function serveWithFailover(
  candidates: readonly ModelRow[],
  bodyFor: (model: ModelRow) => Json,
  dial: (body: Json) => Promise<Response>,
  onRefused: (id: string) => void,
): Promise<ServeResult> {
  const trail: TrailRow[] = [];
  for (const model of candidates) {
    let res: Response;
    try {
      res = await dial(bodyFor(model));
    } catch {
      onRefused(model.id);
      pushTrail(trail, { status: 599, code: 'unreachable', model: model.id });
      continue;
    }
    if (res.ok && res.body !== null) {
      return { ok: true, model: model.id, upstream: res, trail };
    }
    let code = '';
    try {
      code = errorCodeOf(await res.text());
    } catch {
      code = '';
    }
    onRefused(model.id);
    pushTrail(trail, { status: Math.min(Math.max(res.status, TRAIL_MIN_STATUS), TRAIL_MAX_STATUS), code, model: model.id });
  }
  return { ok: false, status: 502, trail };
}

/** The trail as the response header's value: none when nothing was refused. */
export function trailHeader(trail: readonly TrailRow[]): string | undefined {
  return trail.length === 0 ? undefined : JSON.stringify(trail.slice(0, TRAIL_MAX));
}

function pushTrail(trail: TrailRow[], row: TrailRow): void {
  if (trail.length < TRAIL_MAX) trail.push(row);
}
