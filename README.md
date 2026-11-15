# free-model-aggregator-01

One server, two client wires, ten simulated free-tier providers.

`main.ts` (Node 22, standard library only) reads the trial brief, probes each
provider's catalog route in its own wire, registers its port and catalog with
the env, and serves the client's 20 requests:

- `GET /v1/models` — the free model ids, OpenAI shape for `Authorization: Bearer`,
  Anthropic shape for `x-api-key`
- `POST /v1/chat/completions` — OpenAI wire, JSON or SSE
- `POST /v1/messages` — Anthropic wire, JSON or SSE

Routing is by the quality table's rating: every free model meeting the
request's requirements (tools, vision, JSON mode, context length), best first.
A JSON-mode OpenAI request never routes to an anthropic-wire provider. A 402,
429 or 5xx fails over to the next candidate; a refusal that says the budget is
spent marks the provider exhausted, a single scheduled refusal is retried once
on the same provider. All three upstream wires (OpenAI, Anthropic, Gemini) are
translated in both directions, preserving the system prompt, user text, image,
tool names, max_tokens, stop sequences and JSON mode the client compares, and
the provider's usage and stop reason come back exactly as reported.

`node main.ts --list` prints the catalog document the server registers,
byte for byte. The server polls the env and exits 0 when the workload is done.

## Layout

- `main.ts` — the server: brief, catalog, routes, register, poll, routing and failover
- `lib/providers.ts` — the brief's three documents and the catalog probe
- `lib/wire.ts` — the client wires and the three upstream wires, in both directions
- `LICENSE` — MIT
