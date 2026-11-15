// free-model-aggregator-02 — the entry server for make.fun's free-model aggregator contest.
//
// The trial's client sends six prompts (OpenAI-shaped or Anthropic-shaped, some streaming) to this
// server; each one is routed to the strongest free model the brief lists that meets the prompt's
// requirements, dialled through the grader's gateway. Everything this server emits is a pure function
// of the brief, the arriving requests and the gateway's answers: no clock, no randomness — a replay of
// the same trial must produce the same bytes.
//
// Run with: node --experimental-strip-types main.ts  (or `... main.ts --list` to print the catalog).

import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type Brief,
  type Json,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  anthropicEventsFromOpenAI,
  openAIError,
  openAIModelList,
  openAIJsonWithModel,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';
import {
  candidatesFor,
  serveWithFailover,
  trailHeader,
  type TrailRow,
} from './lib/serve.ts';

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const OPENAI_BASE = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
const OPENAI_KEY = process.env.OPENAI_API_KEY ?? '';

const MAX_BODY_BYTES = 8 * 1024 * 1024;
const DIAL_TIMEOUT_MS = 25000;
const REGISTER_TIMEOUT_MS = 5000;
const WORK_TIMEOUT_MS = 2000;
const POLL_INTERVAL_MS = 100;
const ENV_OWNED_PORT = 8080;
const TRAIL_HEADER = 'x-makefun-upstream-failures';

const brief: Brief = readBrief(PROMPT_FILE);
if (brief.error !== null) console.error(`brief: ${brief.error}`);
const catalog = buildCatalog(brief);

/** The gateway's chat-completions URL: derived from OPENAI_BASE_URL, empty when that is unset. */
const CHAT_URL =
  OPENAI_BASE === '' ? '' : `${OPENAI_BASE}${OPENAI_BASE.endsWith('/v1') ? '/chat/completions' : '/v1/chat/completions'}`;

/** The models this entry serves, in the three fields the model lists need. */
const servedModels: ServedModel[] = freeModels(brief).map((m) => ({ id: m.id, provider: m.provider, display_name: m.display_name }));

/** Which prompt row the env's next request belongs to: the trial sends them one at a time, in order. */
let promptCursor = 0;
/** Model ids that have refused a gateway call this trial; they step aside while fresh candidates exist. */
const refused = new Set<string>();

type Wire = 'openai' | 'anthropic';

function wireOfPath(path: string): Wire {
  return path.startsWith('/v1/messages') ? 'anthropic' : 'openai';
}

/** An error body in the named wire's shape. */
function wireError(wire: Wire, type: string, message: string): Json {
  return wire === 'anthropic' ? anthropicError(type, message) : openAIError(message, type, null);
}

/** Send JSON: content-type, a correct content-length, any extra headers, then the serialized body. */
function sendJson(res: ServerResponse, status: number, body: Json, extra: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, {
    'content-type': 'application/json',
    'content-length': Buffer.byteLength(text),
    ...extra,
  });
  res.end(text);
}

/** Send an error in the named wire's shape. */
function sendError(res: ServerResponse, wire: Wire, status: number, type: string, message: string, extra: Record<string, string> = {}): void {
  sendJson(res, status, wireError(wire, type, message), extra);
}

/** Read a request body as one JSON object; null when it is over the cap or not a single JSON object. */
async function readJsonBody(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let total = 0;
  try {
    for await (const chunk of req) {
      const buf = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk as Uint8Array);
      total += buf.length;
      if (total > MAX_BODY_BYTES) return null;
      chunks.push(buf);
    }
  } catch {
    return null;
  }
  let parsed: unknown;
  try {
    parsed = JSON.parse(Buffer.concat(chunks).toString('utf8'));
  } catch {
    return null;
  }
  return isRecord(parsed) ? parsed : null;
}

/** The requirements the arriving request implies, when the brief holds no prompt row. */
function requiresFromBody(body: Json): { vision?: boolean; tools?: boolean; json?: boolean } {
  const requires: { vision?: boolean; tools?: boolean; json?: boolean } = {};
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const content = raw.content;
    if (!Array.isArray(content)) continue;
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'image' || block.type === 'image_url') {
        requires.vision = true;
        break;
      }
    }
    if (requires.vision === true) break;
  }
  if (Array.isArray(body.tools) && body.tools.length > 0) requires.tools = true;
  const format = body.response_format;
  if (isRecord(format) && (format.type === 'json_object' || format.type === 'json_schema')) requires.json = true;
  return requires;
}

function trailHeadersOf(trail: readonly TrailRow[]): Record<string, string> {
  const value = trailHeader(trail);
  return value === undefined ? {} : { [TRAIL_HEADER]: value };
}

/** POST one body to the gateway and hand back the raw Response. */
function dialGateway(body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${OPENAI_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(CHAT_URL, {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    signal: AbortSignal.timeout(DIAL_TIMEOUT_MS),
  });
}

/** Serve one chat/messages request: pick a model, dial with failover, relay the answer. */
async function handleServe(req: IncomingMessage, res: ServerResponse, wire: Wire): Promise<void> {
  const body = await readJsonBody(req);
  if (body === null) {
    sendError(res, wire, 400, 'invalid_request_error', 'the request body is not one JSON object');
    return;
  }

  const row = promptCursor < brief.prompts.length ? brief.prompts[promptCursor] : undefined;
  promptCursor += 1;
  const requires = row !== undefined ? row.requires : requiresFromBody(body);
  const streaming = body.stream === true;

  const candidates = candidatesFor(freeModels(brief), requires, refused);
  if (candidates.length === 0) {
    if (wire === 'anthropic') sendError(res, wire, 503, 'overloaded_error', 'no free model meets this prompt');
    else sendError(res, wire, 503, 'no_free_model', 'no free model meets this prompt');
    return;
  }

  const bodyFor = (model: { id: string }): Json =>
    wire === 'anthropic' ? anthropicToOpenAI(body, model.id) : { ...body, model: model.id };

  const outcome = await serveWithFailover(candidates, bodyFor, dialGateway, (id) => refused.add(id));

  if (!outcome.ok) {
    const extra = trailHeadersOf(outcome.trail);
    if (wire === 'anthropic') sendError(res, wire, 502, 'api_error', 'every candidate refused', extra);
    else sendError(res, wire, 502, 'upstream_error', 'every candidate refused', extra);
    return;
  }

  const extra = trailHeadersOf(outcome.trail);

  if (streaming) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...extra,
    });
    try {
      const relay =
        wire === 'anthropic' ? anthropicEventsFromOpenAI(outcome.upstream.body!, outcome.model) : openAIStreamWithModel(outcome.upstream.body!, outcome.model);
      for await (const frame of relay) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }

  let upstream: unknown;
  try {
    upstream = JSON.parse(await outcome.upstream.text());
  } catch {
    upstream = null;
  }
  if (!isRecord(upstream)) {
    sendError(res, wire, 502, wire === 'anthropic' ? 'api_error' : 'upstream_error', 'the gateway answer was not JSON', extra);
    return;
  }
  const answer = wire === 'anthropic' ? openAIToAnthropic(upstream, outcome.model) : openAIJsonWithModel(upstream, outcome.model);
  sendJson(res, 200, answer, extra);
}

function handle(req: IncomingMessage, res: ServerResponse): void {
  const path = (req.url ?? '/').split('?')[0];

  if (req.method === 'GET' && path === '/v1/models') {
    const anthropic = req.headers['x-api-key'] !== undefined;
    sendJson(res, 200, anthropic ? anthropicModelList(servedModels) : openAIModelList(servedModels));
    return;
  }
  if (req.method === 'POST' && path === '/v1/chat/completions') {
    void handleServe(req, res, 'openai');
    return;
  }
  if (req.method === 'POST' && path === '/v1/messages') {
    void handleServe(req, res, 'anthropic');
    return;
  }
  const wire = wireOfPath(path);
  sendError(res, wire, 404, 'invalid_request_error', `no route for ${req.method ?? 'GET'} ${path}`);
}

/** POST the catalog to the env's /register and start the /work poll loop. */
async function registerAndPoll(port: number): Promise<void> {
  const reply = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: 1, port, catalog }),
    signal: AbortSignal.timeout(REGISTER_TIMEOUT_MS),
  });
  if (!reply.ok) {
    const text = await reply.text().catch(() => '');
    throw new Error(`register failed: ${reply.status} ${text.slice(0, 500)}`);
  }
  const info: unknown = await reply.json().catch(() => null);
  if (isRecord(info) && typeof info.requests === 'number') {
    console.error(`trial holds ${info.requests} prompts`);
  }
  const free = freeModels(brief);
  const jsonCapable = free.filter((model) => meetsRequirements({ json: true }, capabilitiesOf(model))).length;
  console.error(`trial free models ${free.length} jsoncapable ${jsonCapable}`);

  let polling = false;
  const timer = setInterval(async () => {
    if (polling) return;
    polling = true;
    try {
      const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(WORK_TIMEOUT_MS) });
      const work: unknown = await res.json();
      if (isRecord(work) && work.done === true) {
        clearInterval(timer);
        server.close();
        process.exit(0);
      }
    } catch {
      // Retry on the next tick.
    } finally {
      polling = false;
    }
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  const listIndex = process.argv.indexOf('--list');
  if (listIndex !== -1) {
    await new Promise<void>((resolve) => process.stdout.write(`${JSON.stringify(catalog)}\n`, () => resolve()));
    process.exit(0);
  }

  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is empty');
  if (CHAT_URL === '') throw new Error('the gateway chat URL is empty (OPENAI_BASE_URL is unset)');

  await new Promise<void>((resolve) => server.listen(0, '0.0.0.0', () => resolve()));
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no TCP port');
  const port = address.port;
  if (port === ENV_OWNED_PORT) throw new Error(`the OS handed out the env-owned port ${ENV_OWNED_PORT}`);

  await registerAndPoll(port);
  console.log(`free-model-aggregator-02 listening on port ${port}`);
}

const server = createServer(handle);

process.on('SIGTERM', () => process.exit(0));
process.on('SIGINT', () => process.exit(0));

start().catch((err) => {
  console.error((err as Error)?.stack ?? String(err));
  process.exit(1);
});
