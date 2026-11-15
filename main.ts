// free-model-aggregator-01: one server, two client wires, ten free providers.
//
// The trial gives this server a brief (the provider directory, the quality table, the workload
// manifest) and an env (the providers, the register/work routes). It probes each provider's catalog
// route, builds one catalog, registers its port with the env, and serves the client's 20 requests:
//
//   GET  /v1/models            the free model ids, OpenAI shape for Authorization: Bearer,
//                              Anthropic shape for x-api-key
//   POST /v1/chat/completions  an OpenAI-shaped client, answered by the best free model that meets
//                              the request, in JSON or SSE
//   POST /v1/messages          an Anthropic-shaped client, answered the same way
//
// Routing, per request, in rating order: every free model that meets the request's requirements
// (tools, vision, JSON mode, context length), best rating first, ties broken by provider id then
// model id so a replay takes the same path. A JSON-mode OpenAI request never routes to an
// anthropic-wire provider: that wire cannot carry response_format and its echo would report
// json_mode false. A provider call that answers 402, 429 or 5xx fails over to the next candidate;
// the provider's own error message says why it refused, and the router reads it: "budget is spent"
// or "is out" marks the provider exhausted for the rest of the trial, "refused this call" is one
// scheduled refusal and the same provider is tried once more. A call that answers 200 is the
// request's one provider call; a second success is never sought.
//
// Every translation preserves what the provider echoes and the client compares: system prompt, user
// text, image, tool names in order, max_tokens, stop sequences, JSON mode. Streaming OpenAI
// requests are forwarded with stream_options.include_usage so the usage chunk arrives; translated
// streams are collected from the provider and re-emitted in the client's event order with the
// provider's usage and stop reason. Same-wire answers pass through untouched.
//
// Determinism: no wall clock, no unseeded randomness, fixed timestamps in every listing and every
// translated answer. The server polls GET /work and exits 0 when the env says the workload is done;
// `node main.ts --list` prints the catalog document and exits 0 before any server starts.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  authHeaders,
  buildCatalog,
  capabilitiesOf,
  isRecord,
  joinBase,
  meetsRequirements,
  modelFree,
  probeProvider,
  ratingOfModel,
  readBrief,
  type CatalogReport,
  type Json,
  type Probe,
  type ProbedModel,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIToAnthropicRequest,
  parseAnthropicJson,
  parseAnthropicStream,
  parseGeminiJson,
  parseGeminiStream,
  parseOpenAIJson,
  parseOpenAIStream,
  resultToAnthropicJson,
  resultToAnthropicStream,
  resultToOpenAIJson,
  resultToOpenAIStream,
  toGeminiRequest,
  type ServedModel,
  type UpstreamResult,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** How long one provider's catalog probe may take, in milliseconds. */
const PROBE_TIMEOUT_MS = 5000;
/** How long one provider call may take, in milliseconds. */
const PROVIDER_TIMEOUT_MS = 5000;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 32 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';

const brief = readBrief(PROMPT_FILE);
let probes: Probe[] = [];
let catalog: CatalogReport = { providers: [] };
/** Providers whose error message said their budget is spent or they are out: never tried again. */
const exhausted = new Set<string>();
/** How many client API requests have arrived: the manifest is in arrival order, one request at a time. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/** Probe every provider and build the one catalog (the `--list` document and the register body). */
async function load(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  probes = await Promise.all(brief.directory.providers.map((provider) => probeProvider(ENV_URL, provider, PROBE_TIMEOUT_MS)));
  for (const probe of probes) if (!probe.ok && probe.error !== null) log(probe.error);
  catalog = buildCatalog(ENV_URL, probes, brief.ratings);
}

const byId = (a: ProbedModel, b: ProbedModel): number => (a.id < b.id ? -1 : a.id > b.id ? 1 : 0);

/** The free models across the providers that answered, each id once, in the brief's provider order. */
function servedModels(): ServedModel[] {
  const out: ServedModel[] = [];
  const seen = new Set<string>();
  for (const probe of probes) {
    if (!probe.ok) continue;
    for (const model of [...probe.models].sort(byId)) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (seen.has(model.id)) continue;
      seen.add(model.id);
      out.push({ id: model.id, provider: probe.provider.id, display_name: model.display_name });
    }
  }
  return out;
}

interface Candidate {
  probe: Probe;
  model: ProbedModel;
  rating: number;
}

/**
 * Every free model that can serve the request, best rating first, ties by provider id then model id.
 * A JSON-mode OpenAI request never takes an anthropic-wire provider: that wire cannot carry
 * response_format, so its echo would report json_mode false and the request would fail its check.
 * Exhausted providers are left out; when every candidate is exhausted the list is empty.
 */
function candidatesFor(requires: Requirement, clientWire: 'openai' | 'anthropic', jsonMode: boolean): Candidate[] {
  const out: Candidate[] = [];
  for (const probe of probes) {
    if (!probe.ok || exhausted.has(probe.provider.id)) continue;
    if (clientWire === 'openai' && jsonMode && probe.provider.wire === 'anthropic') continue;
    for (const model of probe.models) {
      const row = ratingOfModel(brief.ratings, probe.provider.id, model.id);
      if (!modelFree(model, row)) continue;
      if (!meetsRequirements(requires, capabilitiesOf(model, row))) continue;
      out.push({ probe, model, rating: row?.rating_milli ?? 0 });
    }
  }
  out.sort((a, b) => b.rating - a.rating || (a.probe.provider.id < b.probe.provider.id ? -1 : a.probe.provider.id > b.probe.provider.id ? 1 : 0) || (a.model.id < b.model.id ? -1 : a.model.id > b.model.id ? 1 : 0));
  return out;
}

/** The requirements of the request now arriving: the manifest's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const manifest = brief.workload?.manifest ?? [];
  const entry = manifest[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // The manifest is the declared source of each request's requirements; this fallback is a guess for
  // a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the manifest has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** Whether the OpenAI client asked for JSON mode. */
function jsonModeOf(clientWire: 'openai' | 'anthropic', body: Json): boolean {
  if (clientWire !== 'openai') return false;
  return isRecord(body.response_format) && (body.response_format.type === 'json_object' || body.response_format.type === 'json_schema');
}

interface UpstreamCall {
  url: string;
  headers: Record<string, string>;
  body: Json;
}

/** The provider call for one candidate: the URL, the provider's own auth header, and the body in its wire. */
function upstreamCall(candidate: Candidate, clientWire: 'openai' | 'anthropic', body: Json, jsonMode: boolean): UpstreamCall {
  const provider = candidate.probe.provider;
  const base = joinBase(ENV_URL, provider.base);
  const stream = body.stream === true;
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    ...authHeaders(provider.auth, provider.key),
  };
  if (stream) headers.accept = 'text/event-stream';
  if (provider.wire === 'openai') {
    const req =
      clientWire === 'openai'
        ? { ...body, model: candidate.model.id, ...(stream ? { stream_options: { include_usage: true } } : {}) }
        : anthropicToOpenAI(body, candidate.model.id);
    return { url: `${base}/v1/chat/completions`, headers, body: req };
  }
  if (provider.wire === 'anthropic') {
    headers['anthropic-version'] = '2023-06-01';
    const req = clientWire === 'anthropic' ? { ...body, model: candidate.model.id } : openAIToAnthropicRequest(body, candidate.model.id);
    return { url: `${base}/v1/messages`, headers, body: req };
  }
  const req = toGeminiRequest(body, clientWire, jsonMode);
  const encoded = encodeURIComponent(candidate.model.id);
  const url = stream ? `${base}/v1beta/models/${encoded}:streamGenerateContent?alt=sse` : `${base}/v1beta/models/${encoded}:generateContent`;
  return { url, headers, body: req };
}

/** The provider's error message, from its error envelope in any wire, or the raw text. */
function errorMessage(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error) && typeof value.error.message === 'string') return value.error.message;
  } catch {
    // Not JSON: the raw text is the message.
  }
  return text.slice(0, 500);
}

/** A JSON value parsed from text, or null when it is not one JSON object. */
function safeJson(text: string): unknown {
  try {
    const value: unknown = JSON.parse(text);
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** One provider's answer reduced to a result, in its own wire; null when the answer is not one. */
function parseResult(wire: 'openai' | 'anthropic' | 'gemini', stream: boolean, text: string): UpstreamResult | null {
  if (wire === 'openai') return stream ? parseOpenAIStream(text) : parseOpenAIJson(safeJson(text));
  if (wire === 'anthropic') return stream ? parseAnthropicStream(text) : parseAnthropicJson(safeJson(text));
  return stream ? parseGeminiStream(text) : parseGeminiJson(safeJson(text));
}

/** Copy a provider's event stream to the client as it arrives. */
async function pipeStream(stream: ReadableStream<Uint8Array>, res: ServerResponse): Promise<void> {
  const reader = stream.getReader();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      res.write(Buffer.from(value));
    }
  } finally {
    reader.releaseLock();
    res.end();
  }
}

/** Send one collected result in the client's wire, JSON or SSE. */
function sendResult(res: ServerResponse, clientWire: 'openai' | 'anthropic', stream: boolean, result: UpstreamResult, model: string): void {
  if (clientWire === 'openai') {
    if (stream) {
      res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
      res.end(resultToOpenAIStream(result, model));
    } else {
      sendJson(res, 200, resultToOpenAIJson(result, model));
    }
    return;
  }
  if (stream) {
    res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
    res.end(resultToAnthropicStream(result, model));
  } else {
    sendJson(res, 200, resultToAnthropicJson(result, model));
  }
}

/**
 * Serve one client request: try the candidates in rating order until one answers 200, failing over
 * on any refusal. A refusal whose message says the budget is spent or the provider is out marks it
 * exhausted; a refusal that says the provider refused this one call is retried once on the same
 * provider. A 200 from a provider on the client's own wire passes through untouched; a 200 from
 * another wire is collected and translated, and an answer that does not parse fails over like a
 * refusal. When every candidate has refused, the request ends in a 502 in the client's wire.
 */
async function serve(clientWire: 'openai' | 'anthropic', body: Json, res: ServerResponse): Promise<void> {
  const requires = requirementsFor(body);
  const jsonMode = jsonModeOf(clientWire, body);
  const candidates = candidatesFor(requires, clientWire, jsonMode);
  if (candidates.length === 0) {
    return sendError(res, clientWire, 503, clientWire === 'anthropic' ? 'overloaded_error' : 'no_free_model', 'no free model on the directory meets this request');
  }
  const stream = body.stream === true;
  const clientModel = typeof body.model === 'string' && body.model !== '' ? body.model : 'auto';
  for (const candidate of candidates) {
    if (exhausted.has(candidate.probe.provider.id)) continue;
    const call = upstreamCall(candidate, clientWire, body, jsonMode);
    let attempt = 0;
    while (attempt < 2) {
      attempt += 1;
      let upstream: Response;
      try {
        upstream = await fetch(call.url, {
          method: 'POST',
          headers: call.headers,
          body: JSON.stringify(call.body),
          signal: AbortSignal.timeout(PROVIDER_TIMEOUT_MS),
        });
      } catch {
        break; // Unreachable: try the next candidate.
      }
      if (upstream.ok) {
        if (candidate.probe.provider.wire === clientWire) {
          if (stream && upstream.body !== null) {
            res.writeHead(200, { 'content-type': 'text/event-stream', 'cache-control': 'no-cache', connection: 'keep-alive' });
            await pipeStream(upstream.body, res);
            return;
          }
          const text = await upstream.text();
          res.writeHead(200, { 'content-type': upstream.headers.get('content-type') ?? 'application/json' });
          res.end(text);
          return;
        }
        const text = await upstream.text();
        const result = parseResult(candidate.probe.provider.wire, stream, text);
        if (result === null) break; // A 200 that is not an answer: try the next candidate.
        sendResult(res, clientWire, stream, result, clientWire === 'openai' ? candidate.model.id : clientModel);
        return;
      }
      const message = errorMessage(await upstream.text().catch(() => ''));
      if (/budget is spent|is out/.test(message)) {
        exhausted.add(candidate.probe.provider.id);
        log(`${candidate.probe.provider.id} is exhausted: ${message}`);
        break;
      }
      if (/refused this call/.test(message) && attempt === 1) continue; // One scheduled refusal: try it once more.
      break;
    }
  }
  return sendError(res, clientWire, 502, clientWire === 'anthropic' ? 'api_error' : 'upstream_unreachable', 'every free model that meets this request refused it');
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text) });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('openai', body, res);
}

/** POST /v1/messages: an Anthropic-shaped client, answered by the best free model that meets it. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  await serve('anthropic', body, res);
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the workload's size. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the workload holds ${reply.requests} requests`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  await load();
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${catalog.providers.length} answered providers and ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
