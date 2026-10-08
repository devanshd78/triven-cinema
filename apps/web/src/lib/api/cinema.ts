import type {
  AudioRetakeRequest,
  AsyncVideoGenerationResponse,
  BillingCatalogResponse,
  BillingMeResponse,
  CinemaElement,
  ElementListResponse,
  ElementType,
  ElementAssetRole,
  CombineScenesRequest,
  CombineScenesResponse,
  FactoryGenerationRequest,
  FactoryGenerationResponse,
  FullVideoGenerationRequest,
  FullVideoGenerationResponse,
  GenerationCapabilitiesResponse,
  GenerationJobResponse,
  MetricsSummaryResponse,
  ScenePlanRequest,
  ScenePlanResponse,
  VideoGenerationRequest,
  VideoGenerationResponse,
  YouTubePublishResponse,
  YouTubeStatusResponse,
} from "@/lib/types/generation";

// Production is same-origin through host Nginx. Empty is intentional.
// Browser requests always stay same-origin. Next.js rewrites /api and /media to FastAPI.
// This is required for the HttpOnly auth cookie to work reliably in local development
// (localhost and 127.0.0.1 are different cookie sites).
export const API_URL = "";

const WORKSPACE_HEADER = "X-Triven-Workspace";
const WORKSPACE_STORAGE_KEY = "triven_workspace_token";
let workspaceBootstrapPromise: Promise<void> | null = null;
let workspaceBootstrapController: AbortController | null = null;
let workspaceBootstrapped = false;
let accountEpoch = 0;
const chatWrites = new Map<string, Promise<unknown>>();

export class ApiError extends Error {
  constructor(message: string, public status: number) { super(message); }
}

export class GenerationJobError extends Error {
  constructor(message: string, public job: GenerationJobResponse) { super(message); }
}

export interface AuthUser {
  id: string;
  email: string;
  workspace_id: string;
}

export interface AuthMeResponse {
  authenticated: boolean;
  user: AuthUser | null;
}

export interface DemoOtpResponse {
  challenge_id: string;
  expires_in_seconds: number;
  demo_otp: string | null;
  demo_mode: boolean;
}

export interface ServerChatSession {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  workspace: Record<string, unknown>;
}

export interface ServerChatListResponse {
  chats: ServerChatSession[];
  count: number;
  max_saved: number;
}

function readWorkspaceToken(): string | null {
  if (typeof window === "undefined") return null;
  return window.sessionStorage.getItem(WORKSPACE_STORAGE_KEY);
}

function storeWorkspaceToken(response: Response) {
  if (typeof window === "undefined") return;
  const token = response.headers.get(WORKSPACE_HEADER);
  if (token) window.sessionStorage.setItem(WORKSPACE_STORAGE_KEY, token);
}

async function ensureWorkspaceBootstrap(): Promise<void> {
  if (typeof window === "undefined" || workspaceBootstrapped) return;
  if (!workspaceBootstrapPromise) {
    const epoch = accountEpoch;
    const controller = new AbortController();
    workspaceBootstrapController = controller;
    const bootstrap = (async () => {
      const response = await fetch(`${API_URL}/api/v1/identity/bootstrap`, {
        method: "POST",
        cache: "no-store",
        credentials: "include",
        signal: controller.signal,
      });
      if (epoch !== accountEpoch) throw new DOMException("Account changed", "AbortError");
      if (!response.ok) throw new Error("Unable to establish workspace session.");
      storeWorkspaceToken(response);
      workspaceBootstrapped = true;
    })().finally(() => {
      if (workspaceBootstrapPromise === bootstrap) {
        workspaceBootstrapPromise = null;
        workspaceBootstrapController = null;
      }
    });
    workspaceBootstrapPromise = bootstrap;
  }
  await workspaceBootstrapPromise;
}

async function readApiError(response: Response, fallback: string): Promise<string> {
  try {
    const data = await response.json();
    if (typeof data?.detail === "string") return data.detail;
    if (typeof data?.detail?.message === "string") return data.detail.message;
    if (Array.isArray(data?.detail)) {
      const messages = data.detail
        .map((item: { loc?: unknown[]; msg?: string }) => {
          const field = Array.isArray(item?.loc)
            ? item.loc.filter((part) => part !== "body").join(".")
            : "";
          const message = item?.msg || "Invalid value";
          return field ? `${field}: ${message}` : message;
        })
        .filter(Boolean);
      if (messages.length) return messages.join("\n");
    }
    return data?.message || fallback;
  } catch {
    return fallback;
  }
}

async function apiJson<T>(path: string, init?: RequestInit, fallback = "Request failed."): Promise<T> {
  const epoch = accountEpoch;
  await ensureWorkspaceBootstrap();
  if (epoch !== accountEpoch) throw new DOMException("Account changed", "AbortError");
  init?.signal?.throwIfAborted();
  const workspaceToken = readWorkspaceToken();
  const response = await fetch(`${API_URL}${path}`, {
    cache: "no-store",
    credentials: "include",
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
      ...(workspaceToken ? { [WORKSPACE_HEADER]: workspaceToken } : {}),
      ...(init?.headers || {}),
    },
  });
  if (epoch !== accountEpoch) throw new DOMException("Account changed", "AbortError");
  storeWorkspaceToken(response);
  if (!response.ok) throw new ApiError(await readApiError(response, fallback), response.status);
  const data = await response.json();
  if (epoch !== accountEpoch) throw new DOMException("Account changed", "AbortError");
  init?.signal?.throwIfAborted();
  return data;
}


async function publicJson<T>(path: string, init?: RequestInit, fallback = "Request failed."): Promise<T> {
  // Forward an existing signed anonymous-workspace token during login so the
  // first account can adopt Elements/history created before authentication.
  const workspaceToken = readWorkspaceToken();
  const response = await fetch(`${API_URL}${path}`, {
    cache: "no-store",
    credentials: "include",
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData) ? { "Content-Type": "application/json" } : {}),
      ...(workspaceToken ? { [WORKSPACE_HEADER]: workspaceToken } : {}),
      ...(init?.headers || {}),
    },
  });
  storeWorkspaceToken(response);
  if (!response.ok) throw new Error(await readApiError(response, fallback));
  return response.json();
}

export async function getAuthMe(): Promise<AuthMeResponse> {
  return publicJson("/api/v1/auth/me", undefined, "Unable to read login session.");
}

export async function requestLoginOtp(email: string): Promise<DemoOtpResponse> {
  return publicJson(
    "/api/v1/auth/otp/request",
    { method: "POST", body: JSON.stringify({ email }) },
    "Unable to create login code."
  );
}

export async function verifyLoginOtp(email: string, otp: string): Promise<AuthMeResponse> {
  const result = await publicJson<AuthMeResponse>(
    "/api/v1/auth/otp/verify",
    { method: "POST", body: JSON.stringify({ email, otp }) },
    "Unable to verify login code."
  );
  workspaceBootstrapped = false;
  return result;
}

export async function logoutCinema(): Promise<void> {
  accountEpoch += 1;
  workspaceBootstrapController?.abort();
  workspaceBootstrapController = null;
  workspaceBootstrapPromise = null;
  try {
    await publicJson<{ ok: boolean }>("/api/v1/auth/logout", { method: "POST" }, "Unable to sign out.");
  } finally {
    if (typeof window !== "undefined") window.sessionStorage.removeItem(WORKSPACE_STORAGE_KEY);
    workspaceBootstrapped = false;
  }
}

export async function listChatHistory(): Promise<ServerChatListResponse> {
  return apiJson("/api/v1/chats", undefined, "Unable to load previous chats.");
}

export async function saveChatHistoryItem(payload: ServerChatSession): Promise<ServerChatSession> {
  const epoch = accountEpoch;
  const previous = chatWrites.get(payload.id);
  const next = (async () => {
    await previous?.catch(() => undefined);
    if (epoch !== accountEpoch) throw new DOMException("Account changed", "AbortError");
    return apiJson<ServerChatSession>(
    `/api/v1/chats/${encodeURIComponent(payload.id)}`,
    {
      method: "PUT",
      body: JSON.stringify({
        title: payload.title,
        created_at: payload.created_at,
        updated_at: payload.updated_at,
        workspace: payload.workspace,
      }),
    },
    "Unable to save chat."
    );
  })();
  chatWrites.set(payload.id, next);
  try { return await next; }
  finally { if (chatWrites.get(payload.id) === next) chatWrites.delete(payload.id); }
}

export async function deleteChatHistoryItem(chatId: string): Promise<void> {
  const epoch = accountEpoch;
  await chatWrites.get(chatId)?.catch(() => undefined);
  if (epoch !== accountEpoch) throw new DOMException("Account changed", "AbortError");
  await apiJson(
    `/api/v1/chats/${encodeURIComponent(chatId)}`,
    { method: "DELETE" },
    "Unable to delete chat."
  );
}

export async function listElements(includeArchived = false): Promise<ElementListResponse> {
  return apiJson(`/api/v1/elements${includeArchived ? "?include_archived=true" : ""}`, undefined, "Unable to load Elements.");
}

async function uploadElement(path: string, form: FormData): Promise<CinemaElement> {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 120_000);
  try {
    return await apiJson(path, { method: "POST", body: form, signal: controller.signal }, "Unable to save reference images.");
  } catch (error) {
    if (controller.signal.aborted) throw new Error("Upload timed out. Retry with the same images to finish the save safely.");
    throw error;
  } finally {
    clearTimeout(timeout);
  }
}

export async function createElement(payload: {
  name: string;
  handle: string;
  type: ElementType;
  description?: string;
  files: File[];
  roles?: ElementAssetRole[];
  requestId?: string;
}): Promise<CinemaElement> {
  const form = new FormData();
  form.append("name", payload.name);
  form.append("handle", payload.handle);
  form.append("type", payload.type);
  form.append("description", payload.description || "");
  if (payload.requestId) form.append("request_id", payload.requestId);
  if (payload.roles?.length) form.append("roles", JSON.stringify(payload.roles));
  payload.files.forEach((file) => form.append("files", file));
  return uploadElement("/api/v1/elements", form);
}

export async function addElementAssets(elementId: string, files: File[], roles?: ElementAssetRole[], requestId?: string): Promise<CinemaElement> {
  const form = new FormData();
  if (requestId) form.append("request_id", requestId);
  if (roles?.length) form.append("roles", JSON.stringify(roles));
  files.forEach((file) => form.append("files", file));
  return uploadElement(`/api/v1/elements/${encodeURIComponent(elementId)}/assets`, form);
}

export async function updateElement(elementId: string, payload: Record<string, unknown>): Promise<CinemaElement> {
  return apiJson(`/api/v1/elements/${encodeURIComponent(elementId)}`, { method: "PATCH", body: JSON.stringify(payload) }, "Unable to update Element.");
}

export async function archiveElement(elementId: string): Promise<CinemaElement> {
  return apiJson(`/api/v1/elements/${encodeURIComponent(elementId)}`, { method: "DELETE" }, "Unable to archive Element.");
}

export async function getGenerationCapabilities(): Promise<GenerationCapabilitiesResponse> {
  return apiJson("/api/v1/generations/capabilities", undefined, "Unable to load generation capabilities.");
}

export async function generateScenePlan(payload: ScenePlanRequest): Promise<ScenePlanResponse> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), 25_000);
  try {
    return await apiJson(
      "/api/v1/generations/plan",
      { method: "POST", body: JSON.stringify(payload), signal: controller.signal },
      "Unable to generate scene plan."
    );
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("Storyboard planning timed out. Retry, or use Direct for a single-shot render.");
    }
    throw error;
  } finally {
    window.clearTimeout(timeout);
  }
}

export async function generateVideo(payload: VideoGenerationRequest): Promise<VideoGenerationResponse> {
  return apiJson(
    "/api/v1/generations/video",
    { method: "POST", body: JSON.stringify(payload) },
    "Unable to generate video."
  );
}

export async function createVideoGenerationJob(payload: VideoGenerationRequest): Promise<AsyncVideoGenerationResponse> {
  return apiJson(
    "/api/v1/generations/jobs/video",
    { method: "POST", body: JSON.stringify(payload) },
    "Unable to start generation job."
  );
}

export async function createFactoryGenerationJob(payload: FactoryGenerationRequest): Promise<AsyncVideoGenerationResponse> {
  return apiJson(
    "/api/v1/factory/jobs",
    { method: "POST", body: JSON.stringify(payload) },
    "Unable to start AI factory job."
  );
}

export async function getGenerationJob(jobId: string, signal?: AbortSignal): Promise<GenerationJobResponse> {
  const timeout = AbortSignal.timeout(25000);
  return apiJson(`/api/v1/generations/jobs/${jobId}`, { signal: signal ? AbortSignal.any([signal, timeout]) : timeout }, "Unable to read generation job.");
}

export async function listGenerationJobs(chatId?: string): Promise<GenerationJobResponse[]> {
  const response = await apiJson<GenerationJobResponse[] | { jobs: GenerationJobResponse[] }>(`/api/v1/generations/jobs${chatId ? `?chat_id=${encodeURIComponent(chatId)}` : ""}`);
  return Array.isArray(response) ? response : response.jobs;
}

export async function createCombineGenerationJob(payload: CombineScenesRequest): Promise<AsyncVideoGenerationResponse> {
  return apiJson("/api/v1/generations/jobs/combine", { method: "POST", body: JSON.stringify(payload) }, "Unable to start composition.");
}

export async function createAudioRetakeJob(payload: AudioRetakeRequest): Promise<AsyncVideoGenerationResponse> {
  return apiJson("/api/v1/generations/jobs/audio-retake", { method: "POST", body: JSON.stringify(payload) }, "Unable to start audio correction.");
}

export async function waitForJobResult<T>(
  jobId: string,
  onProgress?: (job: GenerationJobResponse) => void,
  pollMs = 1000,
  signal?: AbortSignal,
  onReconnect?: (message: string) => void,
): Promise<T> {
  let failures = 0;
  for (;;) {
    signal?.throwIfAborted();
    try {
      const job = await getGenerationJob(jobId, signal);
      signal?.throwIfAborted();
      failures = 0;
      onProgress?.(job);
      if (job.status === "completed" && job.result) return job.result as unknown as T;
      if (job.status === "failed") throw new GenerationJobError(job.error || "Generation job failed.", job);
    } catch (error) {
      if (error instanceof GenerationJobError || (error instanceof DOMException && error.name === "AbortError")) throw error;
      if (error instanceof ApiError && error.status >= 400 && error.status < 500 && ![408, 429].includes(error.status)) throw error;
      failures += 1;
      onReconnect?.("Connection interrupted. Your render continues; reconnecting…");
    }
    await new Promise<void>((resolve, reject) => {
      const cancel = () => { clearTimeout(timer); reject(new DOMException("Polling stopped", "AbortError")); };
      const timer = setTimeout(() => { signal?.removeEventListener("abort", cancel); resolve(); }, Math.min(15000, pollMs * 2 ** failures));
      signal?.addEventListener("abort", cancel, { once: true });
    });
  }
}

export function waitForVideoGenerationJob(
  jobId: string,
  onProgress?: (job: GenerationJobResponse) => void,
  pollMs = 1000
): Promise<VideoGenerationResponse> {
  return waitForJobResult<VideoGenerationResponse>(jobId, onProgress, pollMs);
}

export function waitForFactoryGenerationJob(
  jobId: string,
  onProgress?: (job: GenerationJobResponse) => void,
  pollMs = 1200
): Promise<FactoryGenerationResponse> {
  return waitForJobResult<FactoryGenerationResponse>(jobId, onProgress, pollMs);
}

export async function combineSceneVideos(payload: CombineScenesRequest): Promise<CombineScenesResponse> {
  return apiJson(
    "/api/v1/generations/combine",
    { method: "POST", body: JSON.stringify(payload) },
    "Unable to combine scene videos."
  );
}

export async function generateFullVideo(payload: FullVideoGenerationRequest): Promise<FullVideoGenerationResponse> {
  return apiJson(
    "/api/v1/generations/full-video",
    { method: "POST", body: JSON.stringify(payload) },
    "Unable to generate full video."
  );
}

export async function getMetricsSummary(): Promise<MetricsSummaryResponse> {
  return apiJson("/api/v1/generations/metrics/summary", undefined, "Unable to load render metrics.");
}

export async function getBillingCatalog(): Promise<BillingCatalogResponse> {
  return apiJson("/api/v1/billing/catalog", undefined, "Unable to load billing options.");
}

export async function getBillingMe(): Promise<BillingMeResponse> {
  return apiJson("/api/v1/billing/me", undefined, "Unable to load credit balance.");
}

export async function createCheckout(packId: string): Promise<{ checkout_url: string; session_id: string }> {
  return apiJson(
    "/api/v1/billing/checkout",
    { method: "POST", body: JSON.stringify({ pack_id: packId }) },
    "Unable to start Stripe Checkout."
  );
}

export async function verifyCheckout(sessionId: string): Promise<{ session_id: string; paid: boolean; balance_seconds: number }> {
  return apiJson(
    `/api/v1/billing/checkout/status?session_id=${encodeURIComponent(sessionId)}`,
    undefined,
    "Unable to verify Stripe Checkout."
  );
}

export async function createBillingPortal(): Promise<{ portal_url: string }> {
  return apiJson(
    "/api/v1/billing/portal",
    { method: "POST" },
    "Unable to open billing portal."
  );
}

export async function getYouTubeStatus(): Promise<YouTubeStatusResponse> {
  return apiJson("/api/v1/youtube/status", undefined, "Unable to load YouTube connection.");
}

export async function connectYouTube(): Promise<{ authorization_url: string }> {
  return apiJson(
    "/api/v1/youtube/connect",
    { method: "POST" },
    "Unable to start YouTube connection."
  );
}

export async function disconnectYouTube(): Promise<{ disconnected: boolean }> {
  return apiJson(
    "/api/v1/youtube/connection",
    { method: "DELETE" },
    "Unable to disconnect YouTube."
  );
}

export async function publishToYouTube(payload: {
  filename: string;
  title: string;
  description?: string;
  privacy?: "private" | "unlisted" | "public";
  tags?: string[];
  category_id?: string;
  publish_at?: string | null;
}): Promise<YouTubePublishResponse> {
  return apiJson(
    "/api/v1/youtube/publish",
    { method: "POST", body: JSON.stringify(payload) },
    "Unable to upload video to YouTube."
  );
}

export function absoluteApiUrl(path: string): string {
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return `${API_URL}${path}`;
}
