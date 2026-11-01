/**
 * remote-embed.ts - Remote OpenAI-compatible embedding backend
 *
 * Ports the LM Studio/vLLM/TEI/OpenAI `/v1/embeddings` client used by the
 * sibling project `localcrab` (opencrab/stores/openai_embedding.py +
 * resilient_embedding.py) so qmd can route embedding generation to a
 * remote server instead of a local GGUF model.
 *
 * No node-llama-cpp import here: this module only uses the global `fetch`,
 * so it lazy-loads cleanly and is trivial to unit test without a GPU.
 */

/** A single remote embedding endpoint, parsed from a `http[s]://host/v1#model-id` URI. */
export interface RemoteEmbedEndpoint {
  /** The original URI as configured, e.g. `http://localhost:1234/v1#text-embedding-3-small`. */
  raw: string;
  /** Everything before the `#` fragment, e.g. `http://localhost:1234/v1`. `/embeddings` and `/models` are appended to this. */
  apiBase: string;
  /** The percent-decoded URL fragment, used as the `model` field in request bodies. */
  modelId: string;
}

/** Detect whether a model URI string should be routed to the remote embedding backend. */
export function isRemoteEmbedModel(uri: string): boolean {
  return uri.startsWith("http://") || uri.startsWith("https://");
}

/**
 * Parse a remote embedding URI into its API base and model id.
 * The model id is carried in the URL fragment (`#model-id`) since fragments
 * are never sent over the wire, keeping the base URL and model cleanly separated
 * while flowing through the existing single-string config field.
 * Throws if the URI has no fragment, since there is no way to infer the model id otherwise.
 */
export function parseRemoteEmbedUri(uri: string): RemoteEmbedEndpoint {
  const hashIndex = uri.indexOf("#");
  const fragment = hashIndex === -1 ? "" : uri.slice(hashIndex + 1);
  if (!fragment) {
    throw new Error(`remote embed URI requires a \`#model-id\` fragment: ${uri}`);
  }
  const apiBase = uri.slice(0, hashIndex);
  const modelId = decodeURIComponent(fragment);
  return { raw: uri, apiBase, modelId };
}

export const DEFAULT_REMOTE_EMBED_TIMEOUT_MS = 30000;
export const DEFAULT_REMOTE_EMBED_HEALTH_TTL_MS = 15000;
// Soft guardrail only: remote servers have no shared tokenizer for qmd to
// truncate against locally, so this is a generous char cap (chunks are
// already ~900 tokens / ~3600 chars) that just prevents pathological inputs
// from being sent, rather than an accurate token-aware limit.
const DEFAULT_MAX_CHARS = 12000;

export type RemoteEmbedderOptions = {
  /** Sent as `Authorization: Bearer <apiKey>` when set; omitted entirely otherwise (e.g. unauthenticated LM Studio). */
  apiKey?: string;
  /** Timeout in ms for `/embeddings` requests. Default 30000. */
  timeoutMs?: number;
  /** How long (ms) an endpoint is skipped after a failure, independently per endpoint. Default 15000. */
  healthTtlMs?: number;
  /** Soft per-text char cap applied before sending (guardrail only, see DEFAULT_MAX_CHARS). */
  maxChars?: number;
  /** L2-normalize returned vectors. Default true. */
  normalize?: boolean;
  /** Max texts per `/embeddings` request; larger batches are split. Default: no limit. */
  batchSize?: number;
  /** Pause (ms) between consecutive requests of one call, to throttle load on a shared server. Default 0. */
  delayMs?: number;
};

/** Thrown when every configured endpoint is unhealthy or failed for a given call. */
export class RemoteEmbedAllFailedError extends Error {}

/**
 * The server answered but rejected the input itself (e.g. llama.cpp's
 * "input (N tokens) is too large to process"). The endpoint is healthy, so it
 * is not marked unhealthy and the other endpoints are not tried.
 */
export class RemoteEmbedInputError extends Error {}

/** HTTP statuses / messages that mean "this input", not "this server". */
function isInputRejection(status: number, body: string): boolean {
  if (status === 413) return true;
  return (status === 400 || status === 500)
    && /too large|too long|context (length|size|window)|exceeds?|maximum .*tokens|batch size/i.test(body);
}

function l2Normalize(vector: number[]): number[] {
  let sumSquares = 0;
  for (const v of vector) sumSquares += v * v;
  const norm = Math.sqrt(sumSquares);
  if (norm === 0) return vector;
  return vector.map((v) => v / norm);
}

/**
 * Validate an OpenAI-style `/embeddings` response for `count` inputs and return
 * the vectors in input order. Entries are placed by their `index` (the API does
 * not promise response order); a response without any `index` is taken
 * positionally. Throws on a wrong count, missing/duplicate/out-of-range
 * indexes, non-finite values or mixed dimensions, so a malformed response can
 * never attach a vector to the wrong chunk.
 */
export function parseEmbeddingsResponse(body: unknown, count: number): number[][] {
  const data = (body as { data?: unknown })?.data;
  if (!Array.isArray(data)) throw new Error("response has no data array");
  if (data.length !== count) throw new Error(`expected ${count} embeddings, got ${data.length}`);
  const hasIndex = data.some((item) => (item as { index?: unknown })?.index !== undefined);
  const out: (number[] | undefined)[] = new Array(count);
  let dims: number | undefined;
  data.forEach((item, position) => {
    const { index, embedding } = (item ?? {}) as { index?: unknown; embedding?: unknown };
    const slot = hasIndex ? index : position;
    if (typeof slot !== "number" || !Number.isInteger(slot) || slot < 0 || slot >= count) {
      throw new Error(`invalid embedding index ${String(slot)}`);
    }
    if (out[slot]) throw new Error(`duplicate embedding index ${slot}`);
    if (!Array.isArray(embedding) || embedding.length === 0 || !embedding.every((v) => typeof v === "number" && Number.isFinite(v))) {
      throw new Error(`embedding ${slot} is not a non-empty array of finite numbers`);
    }
    if (dims !== undefined && embedding.length !== dims) {
      throw new Error(`embedding ${slot} has ${embedding.length} dimensions, expected ${dims}`);
    }
    dims = embedding.length;
    out[slot] = embedding as number[];
  });
  return out as number[][];
}

const sleep = (ms: number) => new Promise<void>((resolve) => setTimeout(resolve, ms));

/**
 * Client for one or more OpenAI-compatible `/v1/embeddings` endpoints, with
 * sequential fallback and independent per-endpoint unhealthy TTLs (mirrors
 * localcrab's ResilientEmbeddingFunction).
 */
export class RemoteEmbedder {
  private readonly endpoints: RemoteEmbedEndpoint[];
  private readonly apiKey?: string;
  private readonly timeoutMs: number;
  private readonly healthTtlMs: number;
  private readonly maxChars: number;
  private readonly normalize: boolean;
  private readonly batchSize: number;
  private readonly delayMs: number;
  private readonly unhealthyUntil: number[];
  private requests = 0;

  constructor(endpoints: RemoteEmbedEndpoint[], opts: RemoteEmbedderOptions = {}) {
    this.endpoints = endpoints;
    this.apiKey = opts.apiKey;
    this.timeoutMs = opts.timeoutMs ?? DEFAULT_REMOTE_EMBED_TIMEOUT_MS;
    this.healthTtlMs = opts.healthTtlMs ?? DEFAULT_REMOTE_EMBED_HEALTH_TTL_MS;
    this.maxChars = opts.maxChars ?? DEFAULT_MAX_CHARS;
    this.normalize = opts.normalize ?? true;
    this.batchSize = opts.batchSize && opts.batchSize > 0 ? Math.floor(opts.batchSize) : Infinity;
    this.delayMs = opts.delayMs && opts.delayMs > 0 ? opts.delayMs : 0;
    this.unhealthyUntil = endpoints.map(() => 0);
  }

  private isHealthy(index: number): boolean {
    return Date.now() >= (this.unhealthyUntil[index] ?? 0);
  }

  private markUnhealthy(index: number): void {
    this.unhealthyUntil[index] = Date.now() + this.healthTtlMs;
  }

  private headers(): Record<string, string> {
    return {
      "Content-Type": "application/json",
      ...(this.apiKey ? { Authorization: `Bearer ${this.apiKey}` } : {}),
    };
  }

  /**
   * Embed a batch of texts. Inputs are sent in requests of at most `batchSize`
   * texts, `delayMs` apart. A text the server rejects (e.g. over its context
   * size) comes back as `null` instead of failing its whole request: a rejected
   * multi-text request is retried one text at a time. Throws
   * RemoteEmbedAllFailedError when no endpoint can be reached.
   */
  async embedBatch(texts: string[]): Promise<(number[] | null)[]> {
    if (texts.length === 0) return [];

    const inputs = texts.map((t) => (t.length > this.maxChars ? t.slice(0, this.maxChars) : t));
    const results: (number[] | null)[] = [];
    for (let start = 0; start < inputs.length; start += this.batchSize) {
      results.push(...await this.embedGroup(inputs.slice(start, start + this.batchSize)));
    }
    return results;
  }

  private async embedGroup(inputs: string[]): Promise<(number[] | null)[]> {
    try {
      return await this.request(inputs);
    } catch (error) {
      if (!(error instanceof RemoteEmbedInputError)) throw error;
      if (inputs.length === 1) {
        console.warn(`QMD Warning: remote embed rejected one input (${error.message}); it gets no vector.`);
        return [null];
      }
      const results: (number[] | null)[] = [];
      for (const input of inputs) results.push(...await this.embedGroup([input]));
      return results;
    }
  }

  /** One `/embeddings` request, trying each healthy endpoint in order. */
  private async request(inputs: string[]): Promise<number[][]> {
    if (this.delayMs > 0 && this.requests > 0) await sleep(this.delayMs);
    this.requests++;

    for (let i = 0; i < this.endpoints.length; i++) {
      if (!this.isHealthy(i)) continue;
      const endpoint = this.endpoints[i]!;
      try {
        const resp = await fetch(`${endpoint.apiBase}/embeddings`, {
          method: "POST",
          headers: this.headers(),
          body: JSON.stringify({ model: endpoint.modelId, input: inputs }),
          signal: AbortSignal.timeout(this.timeoutMs),
        });
        if (!resp.ok) {
          const detail = await resp.text().catch(() => "");
          if (isInputRejection(resp.status, detail)) {
            throw new RemoteEmbedInputError(`HTTP ${resp.status}: ${detail.slice(0, 200)}`);
          }
          throw new Error(`remote embed endpoint ${endpoint.apiBase} returned HTTP ${resp.status}`);
        }
        const vectors = parseEmbeddingsResponse(await resp.json(), inputs.length);
        return this.normalize ? vectors.map(l2Normalize) : vectors;
      } catch (error) {
        if (error instanceof RemoteEmbedInputError) throw error;
        console.warn(
          `QMD Warning: remote embed endpoint ${endpoint.apiBase} failed (${error instanceof Error ? error.message : String(error)}), marking unhealthy for ${this.healthTtlMs}ms.`
        );
        this.markUnhealthy(i);
      }
    }

    throw new RemoteEmbedAllFailedError("all remote embedding endpoints failed or unhealthy");
  }

  /** Convenience wrapper around embedBatch for a single text; [] when the server rejected it. */
  async embed(text: string): Promise<number[]> {
    const [result] = await this.embedBatch([text]);
    return result ?? [];
  }

  /** Health probe: GET `{apiBase}/models`, true only on HTTP 200, 5s timeout. */
  async ping(endpoint: RemoteEmbedEndpoint): Promise<boolean> {
    try {
      const resp = await fetch(`${endpoint.apiBase}/models`, {
        signal: AbortSignal.timeout(5000),
      });
      return resp.ok;
    } catch {
      return false;
    }
  }
}
