// The client wires and the three upstream wires, in both directions.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an Anthropic one
// (POST /v1/messages), with a JSON body or with server-sent events. Each provider speaks its own wire
// (openai, anthropic or gemini), so every request is translated into the provider's wire and the answer
// translated back into the client's:
//
//   client openai    -> provider openai:    body passes through (model replaced, stream_options added)
//   client openai    -> provider anthropic: openAIToAnthropicRequest / answer back as Chat Completions
//   client openai    -> provider gemini:    toGeminiRequest / answer back as Chat Completions
//   client anthropic -> provider anthropic: body passes through (model replaced)
//   client anthropic -> provider openai:    anthropicToOpenAI / openAIToAnthropic (+ SSE converter)
//   client anthropic -> provider gemini:    toGeminiRequest / answer back as Messages
//
// What the translations must preserve, because the provider echoes what it parsed and the client
// compares that echo with what it sent: the system prompt, the user text, the image, the tool names in
// order, max_tokens, the stop sequences, and JSON mode (OpenAI response_format only survives on the
// openai wire and on gemini via responseMimeType — an anthropic-wire provider always echoes
// json_mode false, so JSON-mode OpenAI requests never route there). Usage counts and the stop reason
// come back exactly as the provider reported them, mapped only across the stop-reason families
// (stop/end_turn/STOP, length/max_tokens, tool_calls/tool_use, content_filter/refusal).
//
// Translated streams are collected from the provider first and then emitted in the client's event
// order: the provider's answer is small and local, and collecting it is what lets message_start carry
// the true input tokens and the final chunk carry the true usage. Same-wire streams pipe through
// untouched in main.ts. Everything here is a pure function of its arguments: no clock and no
// randomness, so the same body and the same provider answer give the same bytes on a replay. The two
// model listings use fixed timestamps for the same reason.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';
/** The id every translated Chat Completions answer carries. */
const CHAT_ID = 'chatcmpl-agg';
/** The id every translated Messages answer carries. */
const MESSAGE_ID = 'msg_agg';

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  return {
    object: 'list',
    data: models.map((m) => ({ id: m.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: m.provider })),
  };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  return {
    data: models.map((m) => ({ type: 'model', id: m.id, display_name: m.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT })),
    has_more: false,
  };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** The text of a string or of a block array's text blocks; '' when there is none. */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  return content.flatMap((b) => (isRecord(b) && b.type === 'text' && typeof b.text === 'string' ? [b.text] : [])).join('\n');
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) {
    const source = block.source;
    if (source.type === 'base64' && typeof source.data === 'string') {
      const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
      return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
    }
    if (source.type === 'url' && typeof source.url === 'string') return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the provider's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  if (finish === 'length') return 'max_tokens';
  if (finish === 'tool_calls' || finish === 'function_call') return 'tool_use';
  if (finish === 'content_filter') return 'refusal';
  return 'end_turn';
}

/** A stop-reason family: 0 end, 1 length, 2 tool, 3 refusal. Any provider's end reason maps to 0. */
export function stopFamilyOf(reason: unknown): number {
  if (reason === 'length' || reason === 'max_tokens' || reason === 'MAX_TOKENS') return 1;
  if (reason === 'tool_calls' || reason === 'function_call' || reason === 'tool_use') return 2;
  if (reason === 'content_filter' || reason === 'refusal' || reason === 'SAFETY') return 3;
  return 0;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = textOf(body.system);
  if (system !== '') messages.push({ role: 'system', content: system });
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const assistant = raw.role === 'assistant';
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: assistant ? 'assistant' : 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    if (assistant) {
      const toolCalls: Json[] = [];
      for (const block of content) {
        if (!isRecord(block) || block.type !== 'tool_use') continue;
        toolCalls.push({
          id: typeof block.id === 'string' ? block.id : '',
          type: 'function',
          function: {
            name: typeof block.name === 'string' ? block.name : '',
            arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
          },
        });
      }
      const text = textOf(content);
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(toolCalls.length > 0 ? { tool_calls: toolCalls } : {}),
      });
      continue;
    }
    const parts: Json[] = [];
    for (const block of content) {
      if (!isRecord(block)) continue;
      if (block.type === 'tool_result') {
        messages.push({
          role: 'tool',
          tool_call_id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '',
          content: toolResultText(block.content),
        });
        continue;
      }
      const part = partOf(block);
      if (part !== null) parts.push(part);
    }
    if (parts.length > 0) messages.push({ role: 'user', content: parts });
  }

  const out: Json = { model, messages };
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || typeof raw.name !== 'string') return [];
      return [
        {
          type: 'function',
          function: {
            name: raw.name,
            description: typeof raw.description === 'string' ? raw.description : '',
            parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
          },
        },
      ];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (isRecord(choice)) {
    if (choice.type === 'auto') out.tool_choice = 'auto';
    else if (choice.type === 'any') out.tool_choice = 'required';
    else if (choice.type === 'tool' && typeof choice.name === 'string') out.tool_choice = { type: 'function', function: { name: choice.name } };
  }
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** A Chat Completions answer as an Anthropic Messages object. */
export function openAIToAnthropic(body: Json, model: string): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content: Json[] = [];
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    if (!isRecord(call) || !isRecord(call.function)) continue;
    content.push({
      type: 'tool_use',
      id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
      name: typeof call.function.name === 'string' ? call.function.name : '',
      input: argumentsOf(call.function.arguments),
    });
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content,
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: numberOrZero(usage.prompt_tokens), output_tokens: numberOrZero(usage.completion_tokens) },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason and
 * usage, then message_stop. `model` is the model the client asked for, named back to it.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let started = false;
  let finished = false;
  /** The stop reason a finish_reason named, held until the stream's usage chunk has arrived. */
  let pendingStop: string | null = null;
  let textIndex = -1;
  let nextIndex = 0;
  const toolIndexes = new Map<number, number>();
  let inputTokens = 0;
  let outputTokens = 0;

  const startEvents = (): string[] => {
    if (started) return [];
    started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: inputTokens, output_tokens: 0 },
        },
      }),
    ];
  };
  const closeEvents = (): string[] => {
    const out: string[] = [];
    if (textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
      textIndex = -1;
    }
    for (const index of [...toolIndexes.values()].sort((a, b) => a - b)) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    toolIndexes.clear();
    return out;
  };
  const finishEvents = (stopReason: string): string[] => {
    if (finished) return [];
    finished = true;
    return [
      ...closeEvents(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  };

  /** One SSE frame's lines as Anthropic events, mutating the stream's state. */
  const eventsOfFrame = (frame: string): string[] => {
    const out: string[] = [];
    for (const line of frame.split('\n')) {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === '[DONE]') {
        out.push(...startEvents(), ...finishEvents(pendingStop ?? 'end_turn'));
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) {
        inputTokens = numberOrZero(chunk.usage.prompt_tokens);
        outputTokens = numberOrZero(chunk.usage.completion_tokens);
      }
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice === null) continue;
      const delta = isRecord(choice.delta) ? choice.delta : {};
      out.push(...startEvents());
      if (typeof delta.content === 'string' && delta.content !== '') {
        if (textIndex === -1) {
          textIndex = nextIndex;
          nextIndex += 1;
          out.push(sseEvent('content_block_start', { type: 'content_block_start', index: textIndex, content_block: { type: 'text', text: '' } }));
        }
        out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: textIndex, delta: { type: 'text_delta', text: delta.content } }));
      }
      const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call)) continue;
        const upstreamIndex = typeof call.index === 'number' ? call.index : toolIndexes.size;
        let index = toolIndexes.get(upstreamIndex);
        if (index === undefined) {
          if (textIndex !== -1) {
            out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: textIndex }));
            textIndex = -1;
          }
          index = nextIndex;
          nextIndex += 1;
          toolIndexes.set(upstreamIndex, index);
          const fn = isRecord(call.function) ? call.function : {};
          out.push(
            sseEvent('content_block_start', {
              type: 'content_block_start',
              index,
              content_block: {
                type: 'tool_use',
                id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
                name: typeof fn.name === 'string' ? fn.name : '',
                input: {},
              },
            }),
          );
        }
        const fn = isRecord(call.function) ? call.function : {};
        if (typeof fn.arguments === 'string' && fn.arguments !== '') {
          out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
        }
      }
      // Hold the stop reason until the stream's last chunk (the usage one) and [DONE] have arrived: OpenAI
      // sends the usage after the finish_reason, and Anthropic's message_delta carries the output tokens.
      if (choice.finish_reason !== null && choice.finish_reason !== undefined) pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  };

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, '\n');
      let at = buffer.indexOf('\n\n');
      while (at !== -1) {
        const frame = buffer.slice(0, at);
        buffer = buffer.slice(at + 2);
        for (const event of eventsOfFrame(frame)) yield event;
        at = buffer.indexOf('\n\n');
      }
    }
    buffer += decoder.decode();
    if (buffer.trim() !== '') for (const event of eventsOfFrame(buffer)) yield event;
  } finally {
    reader.releaseLock();
  }
  if (!finished) {
    for (const event of startEvents()) yield event;
    for (const event of finishEvents(pendingStop ?? 'end_turn')) yield event;
  }
}

// ---------------------------------------------------------------------------
// OpenAI client -> Anthropic provider
// ---------------------------------------------------------------------------

/** Stop sequences of an OpenAI body, as a string list. */
function openAIStops(body: Json): string[] {
  const stop = body.stop;
  if (typeof stop === 'string') return [stop];
  if (Array.isArray(stop)) return stop.filter((s): s is string => typeof s === 'string');
  return [];
}

/** Stop sequences of an Anthropic body, as a string list. */
function anthropicStops(body: Json): string[] {
  if (!Array.isArray(body.stop_sequences)) return [];
  return body.stop_sequences.filter((s): s is string => typeof s === 'string');
}

/**
 * An OpenAI Chat Completions request as an Anthropic Messages request that names `model`. System
 * messages become the system field, image_url parts become image blocks, tool messages become
 * tool_result blocks in a user message, assistant tool_calls become tool_use blocks, and tools,
 * tool_choice, the output cap, temperature, top_p, stop sequences and streaming pass through.
 * Developer messages are dropped: neither wire's echo counts them.
 */
export function openAIToAnthropicRequest(body: Json, model: string): Json {
  const systemParts: string[] = [];
  const messages: Json[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    const role = raw.role;
    if (role === 'system') {
      const text = textOf(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (role === 'developer') continue;
    if (role === 'assistant') {
      const content: Json[] = [];
      const text = textOf(raw.content);
      if (text !== '') content.push({ type: 'text', text });
      const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
      for (const call of calls) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        content.push({
          type: 'tool_use',
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          input: argumentsOf(call.function.arguments),
        });
      }
      if (content.length > 0) messages.push({ role: 'assistant', content });
      continue;
    }
    if (role === 'tool') {
      messages.push({
        role: 'user',
        content: [{ type: 'tool_result', tool_use_id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) }],
      });
      continue;
    }
    // User (or anything else treated as user): text and image parts, in order.
    const content = raw.content;
    if (typeof content === 'string') {
      messages.push({ role: 'user', content });
      continue;
    }
    if (!Array.isArray(content)) continue;
    const blocks: Json[] = [];
    for (const part of content) {
      if (!isRecord(part)) continue;
      if (part.type === 'text' && typeof part.text === 'string') blocks.push({ type: 'text', text: part.text });
      else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') {
        const url = part.image_url.url;
        const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
        if (data !== null) blocks.push({ type: 'image', source: { type: 'base64', media_type: data[1], data: data[2] } });
        else blocks.push({ type: 'image', source: { type: 'url', url } });
      }
    }
    if (blocks.length > 0) messages.push({ role: 'user', content: blocks });
  }

  const out: Json = { model, messages };
  const system = systemParts.join('\n');
  if (system !== '') out.system = system;
  const maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
  if (maxTokens !== undefined) out.max_tokens = maxTokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  const stops = openAIStops(body);
  if (stops.length > 0) out.stop_sequences = stops;
  if (Array.isArray(body.tools)) {
    const tools = body.tools.flatMap((raw) => {
      if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') return [];
      return [{ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', input_schema: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } }];
    });
    if (tools.length > 0) out.tools = tools;
  }
  const choice = body.tool_choice;
  if (choice === 'auto') out.tool_choice = { type: 'auto' };
  else if (choice === 'required') out.tool_choice = { type: 'any' };
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') out.tool_choice = { type: 'tool', name: choice.function.name };
  if (body.stream === true) out.stream = true;
  return out;
}

// ---------------------------------------------------------------------------
// Either client wire -> Gemini provider
// ---------------------------------------------------------------------------

/** One Gemini content part for an image URL: inline data for a data: URL, a file reference else. */
function geminiImagePart(url: string): Json {
  const data = /^data:([^;,]+);base64,(.*)$/s.exec(url);
  if (data !== null) return { inlineData: { mimeType: data[1], data: data[2] } };
  return { fileData: { fileUri: url } };
}

/**
 * A client request (OpenAI or Anthropic wire) as a Gemini generateContent body. The system prompt
 * becomes systemInstruction, user text and images become user contents, assistant tool calls become
 * model contents with functionCall parts, tool results become function contents with functionResponse
 * parts, tools become functionDeclarations, and the output cap, stop sequences, temperature, top_p
 * and JSON mode (responseMimeType) land in generationConfig. When the last content is not the last
 * user message, that message is repeated as a final user content: Gemini's echo reads the user text
 * off the last non-model content, and the client's echo reads it off the last user message, so the
 * two must agree.
 */
export function toGeminiRequest(body: Json, clientWire: 'openai' | 'anthropic', jsonMode: boolean): Json {
  const contents: Json[] = [];
  let systemText = '';
  let lastUserText = '';
  let lastUserParts: Json[] = [];
  const tools: Json[] = [];
  let maxTokens: number | undefined;
  let temperature: number | undefined;
  let topP: number | undefined;
  let stops: string[] = [];

  if (clientWire === 'openai') {
    systemText = '';
    const systemParts: string[] = [];
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      const role = raw.role;
      if (role === 'system') {
        const text = textOf(raw.content);
        if (text !== '') systemParts.push(text);
        continue;
      }
      if (role === 'developer') continue;
      if (role === 'assistant') {
        const parts: Json[] = [];
        const text = textOf(raw.content);
        if (text !== '') parts.push({ text });
        const calls = Array.isArray(raw.tool_calls) ? raw.tool_calls : [];
        for (const call of calls) {
          if (!isRecord(call) || !isRecord(call.function)) continue;
          parts.push({ functionCall: { name: typeof call.function.name === 'string' ? call.function.name : '', args: argumentsOf(call.function.arguments) } });
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (role === 'tool') {
        contents.push({ role: 'function', parts: [{ functionResponse: { name: '', response: { content: typeof raw.content === 'string' ? raw.content : toolResultText(raw.content) } } }] });
        continue;
      }
      const content = raw.content;
      const parts: Json[] = [];
      if (typeof content === 'string') {
        if (content !== '') parts.push({ text: content });
      } else if (Array.isArray(content)) {
        for (const part of content) {
          if (!isRecord(part)) continue;
          if (part.type === 'text' && typeof part.text === 'string') parts.push({ text: part.text });
          else if (part.type === 'image_url' && isRecord(part.image_url) && typeof part.image_url.url === 'string') parts.push(geminiImagePart(part.image_url.url));
        }
      }
      if (parts.length > 0) {
        contents.push({ role: 'user', parts });
        lastUserText = textOf(content);
        lastUserParts = parts;
      }
    }
    systemText = systemParts.join('\n');
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
        tools.push({ name: raw.function.name, description: typeof raw.function.description === 'string' ? raw.function.description : '', parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} } });
      }
    }
    const choice = body.tool_choice;
    if (choice === 'auto' || choice === 'required' || (isRecord(choice) && choice.type === 'function')) {
      // Tool choice is not echoed; Gemini's default (AUTO) serves every request that names tools.
    }
    maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : typeof body.max_completion_tokens === 'number' ? body.max_completion_tokens : undefined;
    temperature = typeof body.temperature === 'number' ? body.temperature : undefined;
    topP = typeof body.top_p === 'number' ? body.top_p : undefined;
    stops = openAIStops(body);
  } else {
    systemText = textOf(body.system);
    const source = Array.isArray(body.messages) ? body.messages : [];
    for (const raw of source) {
      if (!isRecord(raw)) continue;
      const content = raw.content;
      if (raw.role === 'assistant') {
        const parts: Json[] = [];
        const text = textOf(content);
        if (text !== '') parts.push({ text });
        if (Array.isArray(content)) {
          for (const block of content) {
            if (!isRecord(block) || block.type !== 'tool_use') continue;
            parts.push({ functionCall: { name: typeof block.name === 'string' ? block.name : '', args: isRecord(block.input) ? block.input : {} } });
          }
        }
        if (parts.length > 0) contents.push({ role: 'model', parts });
        continue;
      }
      if (Array.isArray(content)) {
        const resultBlocks = content.filter((b) => isRecord(b) && b.type === 'tool_result');
        const otherParts: Json[] = [];
        for (const block of content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') otherParts.push({ text: block.text });
          else if (block.type === 'image' && isRecord(block.source)) {
            if (block.source.type === 'base64' && typeof block.source.data === 'string') {
              otherParts.push({ inlineData: { mimeType: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
            } else if (block.source.type === 'url' && typeof block.source.url === 'string') otherParts.push(geminiImagePart(block.source.url));
          }
        }
        for (const block of resultBlocks) {
          contents.push({ role: 'function', parts: [{ functionResponse: { name: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', response: { content: toolResultText(block.content) } } }] });
        }
        if (otherParts.length > 0) {
          contents.push({ role: 'user', parts: otherParts });
          lastUserText = textOf(content);
          lastUserParts = otherParts;
        }
        continue;
      }
      if (typeof content === 'string') {
        contents.push({ role: 'user', parts: [{ text: content }] });
        lastUserText = content;
        lastUserParts = [{ text: content }];
      }
    }
    if (Array.isArray(body.tools)) {
      for (const raw of body.tools) {
        if (!isRecord(raw) || typeof raw.name !== 'string') continue;
        tools.push({ name: raw.name, description: typeof raw.description === 'string' ? raw.description : '', parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} } });
      }
    }
    maxTokens = typeof body.max_tokens === 'number' ? body.max_tokens : undefined;
    temperature = typeof body.temperature === 'number' ? body.temperature : undefined;
    topP = typeof body.top_p === 'number' ? body.top_p : undefined;
    stops = anthropicStops(body);
  }

  // The echo reads the user text off the last non-model content: when tool results (or anything else)
  // follow the last user message, repeat that message as the final user content so both echoes agree.
  const last = contents.length > 0 ? contents[contents.length - 1] : undefined;
  if (lastUserParts.length > 0 && (last === undefined || last.role !== 'user')) {
    contents.push({ role: 'user', parts: lastUserParts });
  }
  void lastUserText;

  const out: Json = { contents };
  if (systemText !== '') out.systemInstruction = { parts: [{ text: systemText }] };
  if (tools.length > 0) out.tools = [{ functionDeclarations: tools }];
  const config: Json = {};
  if (maxTokens !== undefined) config.maxOutputTokens = maxTokens;
  if (stops.length > 0) config.stopSequences = stops;
  if (temperature !== undefined) config.temperature = temperature;
  if (topP !== undefined) config.topP = topP;
  if (jsonMode) config.responseMimeType = 'application/json';
  if (Object.keys(config).length > 0) out.generationConfig = config;
  return out;
}

// ---------------------------------------------------------------------------
// One provider answer, in any wire, as a single result
// ---------------------------------------------------------------------------

/** A provider's answer reduced to what the client wires need: text, usage and the stop family. */
export interface UpstreamResult {
  text: string;
  inputTokens: number;
  outputTokens: number;
  /** 0 end, 1 length, 2 tool, 3 refusal. */
  stopFamily: number;
}

function splitText(text: string, size: number): string[] {
  const pieces: string[] = [];
  for (let at = 0; at < text.length; at += size) pieces.push(text.slice(at, at + size));
  return pieces.length === 0 ? [''] : pieces;
}

/** SSE frames of a text body: [{event, data}] in order, tolerating \r\n. */
export function parseSseFrames(text: string): { event: string | null; data: string }[] {
  const frames: { event: string | null; data: string }[] = [];
  const normalized = text.replace(/\r\n/g, '\n');
  for (const frame of normalized.split('\n\n')) {
    if (frame.trim() === '') continue;
    let event: string | null = null;
    const data: string[] = [];
    for (const line of frame.split('\n')) {
      if (line.startsWith('event:')) event = line.slice('event:'.length).trim();
      else if (line.startsWith('data:')) data.push(line.slice('data:'.length).replace(/^ /, ''));
    }
    if (data.length > 0) frames.push({ event, data: data.join('\n') });
  }
  return frames;
}

/** An OpenAI JSON answer as a result; null when it is not one. */
export function parseOpenAIJson(body: unknown): UpstreamResult | null {
  if (!isRecord(body)) return null;
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : null;
  if (choice === null) return null;
  const message = isRecord(choice.message) ? choice.message : {};
  const text = typeof message.content === 'string' ? message.content : textOf(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text,
    inputTokens: numberOrZero(usage.prompt_tokens),
    outputTokens: numberOrZero(usage.completion_tokens),
    stopFamily: stopFamilyOf(choice.finish_reason),
  };
}

/** An OpenAI SSE stream as a result; null when it carries no finish chunk. */
export function parseOpenAIStream(text: string): UpstreamResult | null {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopFamily = 0;
  let sawFinish = false;
  for (const frame of parseSseFrames(text)) {
    const payload = frame.data.trim();
    if (payload === '[DONE]') continue;
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      continue;
    }
    if (!isRecord(chunk)) continue;
    if (isRecord(chunk.usage)) {
      inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) continue;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    if (typeof delta.content === 'string') out += delta.content;
    if (typeof choice.finish_reason === 'string') {
      sawFinish = true;
      stopFamily = stopFamilyOf(choice.finish_reason);
    }
  }
  return sawFinish ? { text: out, inputTokens, outputTokens, stopFamily } : null;
}

/** An Anthropic JSON answer as a result; null when it is not one. */
export function parseAnthropicJson(body: unknown): UpstreamResult | null {
  if (!isRecord(body) || body.type !== 'message' || !Array.isArray(body.content)) return null;
  const usage = isRecord(body.usage) ? body.usage : {};
  return {
    text: textOf(body.content),
    inputTokens: numberOrZero(usage.input_tokens),
    outputTokens: numberOrZero(usage.output_tokens),
    stopFamily: stopFamilyOf(body.stop_reason),
  };
}

/** An Anthropic SSE stream as a result; null when it never reaches message_stop. */
export function parseAnthropicStream(text: string): UpstreamResult | null {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopFamily = 0;
  let stopped = false;
  for (const frame of parseSseFrames(text)) {
    if (frame.event === null) continue;
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    if (frame.event === 'message_start' && isRecord(data.message) && isRecord(data.message.usage)) {
      inputTokens = numberOrZero(data.message.usage.input_tokens);
    } else if (frame.event === 'content_block_delta' && isRecord(data.delta) && data.delta.type === 'text_delta' && typeof data.delta.text === 'string') {
      out += data.delta.text;
    } else if (frame.event === 'message_delta') {
      if (isRecord(data.delta) && typeof data.delta.stop_reason === 'string') stopFamily = stopFamilyOf(data.delta.stop_reason);
      if (isRecord(data.usage)) outputTokens = numberOrZero(data.usage.output_tokens);
    } else if (frame.event === 'message_stop') {
      stopped = true;
    }
  }
  return stopped ? { text: out, inputTokens, outputTokens, stopFamily } : null;
}

/** A Gemini JSON answer as a result; null when it is not one. */
export function parseGeminiJson(body: unknown): UpstreamResult | null {
  if (!isRecord(body) || !Array.isArray(body.candidates) || body.candidates.length === 0) return null;
  const candidate = isRecord(body.candidates[0]) ? body.candidates[0] : null;
  if (candidate === null) return null;
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts.flatMap((p) => (isRecord(p) && typeof p.text === 'string' ? [p.text] : [])).join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return {
    text,
    inputTokens: numberOrZero(usage.promptTokenCount),
    outputTokens: numberOrZero(usage.candidatesTokenCount),
    stopFamily: stopFamilyOf(candidate.finishReason),
  };
}

/** A Gemini SSE stream as a result; null when no frame carries a finish reason. */
export function parseGeminiStream(text: string): UpstreamResult | null {
  let out = '';
  let inputTokens = 0;
  let outputTokens = 0;
  let stopFamily = 0;
  let sawFinish = false;
  for (const frame of parseSseFrames(text)) {
    let data: unknown;
    try {
      data = JSON.parse(frame.data);
    } catch {
      continue;
    }
    if (!isRecord(data) || !Array.isArray(data.candidates) || data.candidates.length === 0) continue;
    const candidate = isRecord(data.candidates[0]) ? data.candidates[0] : null;
    if (candidate === null) continue;
    const content = isRecord(candidate.content) ? candidate.content : {};
    const parts = Array.isArray(content.parts) ? content.parts : [];
    for (const p of parts) if (isRecord(p) && typeof p.text === 'string') out += p.text;
    if (typeof candidate.finishReason === 'string') {
      sawFinish = true;
      stopFamily = stopFamilyOf(candidate.finishReason);
    }
    if (isRecord(data.usageMetadata)) {
      inputTokens = numberOrZero(data.usageMetadata.promptTokenCount);
      outputTokens = numberOrZero(data.usageMetadata.candidatesTokenCount);
    }
  }
  return sawFinish ? { text: out, inputTokens, outputTokens, stopFamily } : null;
}

// ---------------------------------------------------------------------------
// One result, in the client's wire
// ---------------------------------------------------------------------------

/** A result as a Chat Completions JSON object, naming `model`. */
export function resultToOpenAIJson(result: UpstreamResult, model: string): Json {
  return {
    id: CHAT_ID,
    object: 'chat.completion',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [
      {
        index: 0,
        message: { role: 'assistant', content: result.text },
        finish_reason: result.stopFamily === 1 ? 'length' : result.stopFamily === 2 ? 'tool_calls' : result.stopFamily === 3 ? 'content_filter' : 'stop',
        logprobs: null,
      },
    ],
    usage: { prompt_tokens: result.inputTokens, completion_tokens: result.outputTokens, total_tokens: result.inputTokens + result.outputTokens },
  };
}

/** A result as a Chat Completions SSE stream: role chunk, content deltas, finish chunk, usage, [DONE]. */
export function resultToOpenAIStream(result: UpstreamResult, model: string): string {
  const frames: string[] = [];
  const chunk = (delta: Json, finish: string | null): Json => ({
    id: CHAT_ID,
    object: 'chat.completion.chunk',
    created: FIXED_OPENAI_CREATED,
    model,
    choices: [{ index: 0, delta, finish_reason: finish }],
  });
  frames.push(sseEvent(null, chunk({ role: 'assistant', content: '' }, null)));
  for (const piece of splitText(result.text, 13)) frames.push(sseEvent(null, chunk({ content: piece }, null)));
  const finish = result.stopFamily === 1 ? 'length' : result.stopFamily === 2 ? 'tool_calls' : result.stopFamily === 3 ? 'content_filter' : 'stop';
  frames.push(sseEvent(null, chunk({}, finish)));
  frames.push(
    sseEvent(null, {
      id: CHAT_ID,
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: { prompt_tokens: result.inputTokens, completion_tokens: result.outputTokens, total_tokens: result.inputTokens + result.outputTokens },
    }),
  );
  frames.push('data: [DONE]\n\n');
  return frames.join('');
}

/** A result as an Anthropic Messages JSON object, naming `model`. */
export function resultToAnthropicJson(result: UpstreamResult, model: string): Json {
  return {
    id: MESSAGE_ID,
    type: 'message',
    role: 'assistant',
    model,
    content: [{ type: 'text', text: result.text }],
    stop_reason: result.stopFamily === 1 ? 'max_tokens' : result.stopFamily === 2 ? 'tool_use' : result.stopFamily === 3 ? 'refusal' : 'end_turn',
    stop_sequence: null,
    usage: { input_tokens: result.inputTokens, output_tokens: result.outputTokens },
  };
}

/** A result as an Anthropic SSE stream: message_start, one text block, message_delta, message_stop. */
export function resultToAnthropicStream(result: UpstreamResult, model: string): string {
  const stop = result.stopFamily === 1 ? 'max_tokens' : result.stopFamily === 2 ? 'tool_use' : result.stopFamily === 3 ? 'refusal' : 'end_turn';
  const frames: string[] = [
    sseEvent('message_start', {
      type: 'message_start',
      message: { id: MESSAGE_ID, type: 'message', role: 'assistant', model, content: [], stop_reason: null, stop_sequence: null, usage: { input_tokens: result.inputTokens, output_tokens: 0 } },
    }),
    sseEvent('content_block_start', { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } }),
  ];
  for (const piece of splitText(result.text, 13)) {
    frames.push(sseEvent('content_block_delta', { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text: piece } }));
  }
  frames.push(
    sseEvent('content_block_stop', { type: 'content_block_stop', index: 0 }),
    sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stop, stop_sequence: null }, usage: { output_tokens: result.outputTokens } }),
    sseEvent('message_stop', { type: 'message_stop' }),
  );
  return frames.join('');
}
