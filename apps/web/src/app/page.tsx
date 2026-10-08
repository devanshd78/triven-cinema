"use client";

import { useEffect, useEffectEvent, useMemo, useRef, useState } from "react";
import type { FormEvent, ReactNode } from "react";
import Image from "next/image";
import { ReferenceUploadPreview } from "@/components/ReferenceUploadPreview";
import { StudioVideo } from "@/components/StudioVideo";
import { studioPlayback } from "@/lib/media-playback";
import { referenceFileKey, referenceRoles, selectReferenceFiles } from "@/lib/element-uploads";
import { appliesToAllScenes, hasElementMention } from "@/lib/element-references";

import {
  absoluteApiUrl,
  addElementAssets,
  archiveElement,
  ApiError,
  GenerationJobError,
  createCombineGenerationJob,
  createAudioRetakeJob,
  listGenerationJobs,
  waitForJobResult,
  connectYouTube,
  createElement,
  createBillingPortal,
  createCheckout,
  createFactoryGenerationJob,
  createVideoGenerationJob,
  deleteChatHistoryItem,
  disconnectYouTube,
  generateScenePlan,
  getAuthMe,
  getBillingCatalog,
  getBillingMe,
  getGenerationCapabilities,
  getYouTubeStatus,
  listChatHistory,
  listElements,
  logoutCinema,
  requestLoginOtp,
  saveChatHistoryItem,
  updateElement,
  verifyCheckout,
  verifyLoginOtp,
} from "@/lib/api/cinema";
import type { AuthUser, ServerChatSession } from "@/lib/api/cinema";
import { reconcileChatSessions, invalidateSceneChain } from "@/lib/chat-history";
import { aggregateQCStatus, qcStatusLabel, resolveQCStatus } from "@/lib/qc-report";
import type { QCStatus, VideoQCReport } from "@/lib/qc-report";
import type {
  AspectRatio,
  AudioMode,
  AudioRetakeRequest,
  AudioRetakeResponse,
  BillingCatalogResponse,
  BillingMeResponse,
  CinemaElement,
  ContinuityMode,
  ElementAssetRole,
  ElementBinding,
  ElementReferenceMode,
  ElementType,
  ElementWardrobePolicy,
  DecoderName,
  FactoryGenerationResponse,
  GenerationCapabilitiesResponse,
  GenerationJobResponse,
  VideoGenerationRequest,
  VideoGenerationResponse,
  FactoryGenerationRequest,
  CombineScenesRequest,
  CombineScenesResponse,
  GenerationMode,
  RenderedSceneVideo,
  RenderQuality,
  RealismProfile,
  ScenePlanResponse,
  VideoModelName,
  VideoProviderName,
  YouTubePrivacy,
  YouTubeStatusResponse,
} from "@/lib/types/generation";

type ScenePromptMap = Record<number, string>;
type RenderedVideoMap = Record<number, RenderedSceneVideo>;
type ThemeMode = "light" | "dark";
type DirectorTab = "scene" | "camera" | "look" | "elements";
type CameraMove = "auto" | "static" | "dolly-in" | "dolly-out" | "pan-left" | "pan-right" | "tilt-up" | "tilt-down" | "orbit";
type LensPreset = "auto" | "18mm" | "24mm" | "35mm" | "50mm" | "85mm";
type ShotSize = "auto" | "wide" | "medium" | "close-up" | "extreme-close-up" | "over-the-shoulder";
type GenrePreset = "auto" | "general" | "drama" | "epic" | "action" | "comedy" | "horror";
type ColorPreset = "auto" | "neutral" | "warm" | "golden-hour" | "cool" | "moonlight" | "high-contrast";
type TempoPreset = "auto" | "slow" | "measured" | "dynamic";

type FinalVideo = {
  url: string;
  downloadUrl: string;
  filename: string;
  qualityNote: string;
  label: string;
  hasAudio: boolean;
  audioCodec: string | null;
  dimensions: string;
  estimatedCostUsd?: number | null;
  gpu?: string | null;
  youtubeUrl?: string | null;
  youtubePrivacy?: string | null;
  qcReport?: VideoQCReport;
  factoryResult?: FactoryGenerationResponse;
  aspectRatio?: AspectRatio;
  createdAt?: number;
} | null;

type PendingJob = {
  requestId: string;
  jobId?: string;
  kind: "video" | "factory" | "combine" | "audio_retake";
  sceneId?: number;
  renderSignature?: string;
  aspectRatio: AspectRatio;
  payload: VideoGenerationRequest | FactoryGenerationRequest | CombineScenesRequest | AudioRetakeRequest;
};

type StudioChatWorkspace = {
  prompt: string;
  mode: GenerationMode;
  aspectRatio: AspectRatio;
  sceneCount: number;
  durationSeconds: number;
  factoryTargetSeconds: number;
  factorySceneSeconds: number;
  quality: RenderQuality;
  audioMode: AudioMode;
  audioDirection: string;
  spokenScript: string;
  videoTakes: NonNullable<FinalVideo>[];
  activeJob: PendingJob | null;
  jobError: string;
  recoveredAssets: NonNullable<GenerationJobResponse["assets"]>;
  decoder: DecoderName;
  realismProfile: RealismProfile;
  seed: number;
  enhancePrompt: boolean;
  continuityMode: ContinuityMode;
  cameraMove: CameraMove;
  lensPreset: LensPreset;
  shotSize: ShotSize;
  genrePreset: GenrePreset;
  colorPreset: ColorPreset;
  tempoPreset: TempoPreset;
  selectedElementId: string | null;
  elementModes: Record<string, ElementReferenceMode>;
  elementWardrobePolicies: Record<string, ElementWardrobePolicy>;
  elementApplyAll: Record<string, boolean>;
  elementStrengths: Record<string, number>;
  result: ScenePlanResponse | null;
  scenePrompts: ScenePromptMap;
  renderedVideos: RenderedVideoMap;
  factoryResult: FactoryGenerationResponse | null;
  finalVideo: FinalVideo;
};

type StudioChatSession = {
  id: string;
  title: string;
  createdAt: number;
  updatedAt: number;
  dirty?: boolean;
  workspace: StudioChatWorkspace;
};

const CHAT_HISTORY_STORAGE_PREFIX = "triven-cinema-chat-history-v2";
const LEGACY_CHAT_HISTORY_STORAGE_KEY = "triven-cinema-chat-history-v1";
const DEFAULT_AUDIO_DIRECTION = "Natural synchronized ambience and Foley matching every visible action.";

function createEmptyChatWorkspace(): StudioChatWorkspace {
  return {
    prompt: "",
    mode: "factory",
    aspectRatio: "16:9",
    sceneCount: 2,
    durationSeconds: 15,
    factoryTargetSeconds: 30,
    factorySceneSeconds: 20,
    quality: "1080p",
    audioMode: "mastered",
    audioDirection: DEFAULT_AUDIO_DIRECTION,
    spokenScript: "",
    videoTakes: [],
    activeJob: null,
    jobError: "",
    recoveredAssets: [],
    decoder: "conv",
    realismProfile: "real_skin",
    seed: 42,
    enhancePrompt: false,
    continuityMode: "strict",
    cameraMove: "auto",
    lensPreset: "auto",
    shotSize: "auto",
    genrePreset: "auto",
    colorPreset: "auto",
    tempoPreset: "auto",
    selectedElementId: null,
    elementModes: {},
    elementWardrobePolicies: {},
    elementApplyAll: {},
    elementStrengths: {},
    result: null,
    scenePrompts: {},
    renderedVideos: {},
    factoryResult: null,
    finalVideo: null,
  };
}

function normalizeChatWorkspace(value: Partial<StudioChatWorkspace> | null | undefined): StudioChatWorkspace {
  const empty = createEmptyChatWorkspace();
  const savedVideo = value?.finalVideo;
  const savedFactoryResult = value?.factoryResult;
  return {
    ...empty,
    ...(value || {}),
    elementModes: value?.elementModes || {},
    elementWardrobePolicies: value?.elementWardrobePolicies || {},
    elementApplyAll: value?.elementApplyAll || {},
    elementStrengths: value?.elementStrengths || {},
    scenePrompts: value?.scenePrompts || {},
    renderedVideos: value?.renderedVideos || {},
    videoTakes: value?.videoTakes?.length ? value.videoTakes : savedVideo ? [savedVideo] : [],
    // Older saved Factory chats already contain verdicts, just not an attached
    // per-video report. Recover their QC status when history is reopened.
    finalVideo: savedVideo && !savedVideo.qcReport && savedFactoryResult && savedVideo.filename === savedFactoryResult.final_filename
      ? { ...savedVideo, qcReport: factoryQCReport(savedFactoryResult), factoryResult: savedFactoryResult }
      : savedVideo || null,
  };
}

function factoryQCReport(response: FactoryGenerationResponse): VideoQCReport {
  return {
    visual: resolveQCStatus(response.visual_qc_status, response.continuity_qc_passed, true, response.continuity_warnings),
    audio: resolveQCStatus(response.audio_qc_status, response.audio_qc_passed, response.audio_mode !== "mute", response.audio_warnings),
    visualWarnings: response.continuity_warnings || [],
    audioWarnings: response.audio_warnings || [],
    visualRetries: response.continuity_regenerations || 0,
    audioRetakes: response.audio_retake_count || 0,
  };
}

function chatTitleFromPrompt(value: string) {
  const normalized = value.replace(/\s+/g, " ").trim();
  if (!normalized) return "New chat";
  return normalized.length > 38 ? `${normalized.slice(0, 38).trimEnd()}…` : normalized;
}

function createChatId() {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return `chat-${crypto.randomUUID()}`;
  }
  return `chat-${Date.now()}-${Math.random().toString(36).slice(2, 9)}`;
}

function createChatSession(workspace = createEmptyChatWorkspace()): StudioChatSession {
  const now = Date.now();
  return {
    id: createChatId(),
    title: chatTitleFromPrompt(workspace.prompt),
    createdAt: now,
    updatedAt: now,
    dirty: true,
    workspace,
  };
}

function chatFromServer(entry: ServerChatSession): StudioChatSession {
  return {
    id: entry.id,
    title: entry.title || chatTitleFromPrompt(String(entry.workspace.prompt || "")),
    createdAt: entry.created_at,
    updatedAt: entry.updated_at,
    workspace: normalizeChatWorkspace(entry.workspace as Partial<StudioChatWorkspace>),
  };
}

function chatToServer(entry: StudioChatSession): ServerChatSession {
  return {
    id: entry.id,
    title: entry.title,
    created_at: entry.createdAt,
    updated_at: entry.updatedAt,
    workspace: entry.workspace as unknown as Record<string, unknown>,
  };
}

function parseStoredChatSessions(raw: string | null) {
  if (!raw) return [] as StudioChatSession[];
  try {
    const parsed: unknown = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [] as StudioChatSession[];
    return parsed
      .filter((entry): entry is StudioChatSession => Boolean(
        entry &&
        typeof entry === "object" &&
        "id" in entry && typeof entry.id === "string" &&
        "workspace" in entry && entry.workspace && typeof entry.workspace === "object"
      ))
      .map((entry) => ({
        ...entry,
        title: typeof entry.title === "string" && entry.title.trim() ? entry.title : chatTitleFromPrompt((entry.workspace as StudioChatWorkspace).prompt || ""),
        createdAt: Number.isFinite(entry.createdAt) ? entry.createdAt : Date.now(),
        updatedAt: Number.isFinite(entry.updatedAt) ? entry.updatedAt : Date.now(),
        workspace: normalizeChatWorkspace(entry.workspace),
      }))
      .sort((a, b) => b.updatedAt - a.updatedAt);
  } catch {
    return [] as StudioChatSession[];
  }
}

function chatStorageKey(userId: string | null | undefined) {
  return userId ? `${CHAT_HISTORY_STORAGE_PREFIX}:${userId}` : null;
}

function readStoredChatSessions(userId: string | null | undefined, includeLegacy = false) {
  if (typeof window === "undefined") return [] as StudioChatSession[];
  const key = chatStorageKey(userId);
  const scoped = key ? parseStoredChatSessions(window.localStorage.getItem(key)) : [];
  if (scoped.length || !includeLegacy) return scoped;
  return parseStoredChatSessions(window.localStorage.getItem(LEGACY_CHAT_HISTORY_STORAGE_KEY));
}

function persistChatSessions(sessions: StudioChatSession[], userId: string | null | undefined) {
  if (typeof window === "undefined") return;
  const key = chatStorageKey(userId);
  if (!key) return;
  try {
    window.localStorage.setItem(key, JSON.stringify(sessions));
  } catch {
    // A full localStorage should never break the actual Cinema generation flow.
  }
}

const DURATION_OPTIONS: Record<RenderQuality, number[]> = {
  preview: [5, 10, 15, 20],
  "1080p": [5, 10, 15, 20, 30],
  "4k": [5, 10, 15],
};

const FACTORY_SCENE_OPTIONS: Record<RenderQuality, number[]> = {
  preview: [15, 20],
  "1080p": [15, 20, 25, 30],
  "4k": [15],
};

// Final runtime is a user choice. The creator-grade preset must never overwrite it.
// Longer runtimes are assembled from continuity-locked scenes by Factory mode.
const FACTORY_TARGETS = [15, 20, 25, 30, 45, 60, 90, 120, 180, 300];

const CAMERA_MOVES: Array<{ value: CameraMove; label: string }> = [
  { value: "auto", label: "Auto" },
  { value: "static", label: "Static" },
  { value: "dolly-in", label: "Dolly In" },
  { value: "dolly-out", label: "Dolly Out" },
  { value: "pan-left", label: "Pan Left" },
  { value: "pan-right", label: "Pan Right" },
  { value: "tilt-up", label: "Tilt Up" },
  { value: "tilt-down", label: "Tilt Down" },
  { value: "orbit", label: "Orbit" },
];

const LENS_PRESETS: LensPreset[] = ["auto", "18mm", "24mm", "35mm", "50mm", "85mm"];
const SHOT_SIZES: Array<{ value: ShotSize; label: string }> = [
  { value: "auto", label: "Auto framing" },
  { value: "wide", label: "Wide" },
  { value: "medium", label: "Medium" },
  { value: "close-up", label: "Close-up" },
  { value: "extreme-close-up", label: "Extreme close-up" },
  { value: "over-the-shoulder", label: "Over the shoulder" },
];
const GENRE_PRESETS: GenrePreset[] = ["auto", "general", "drama", "epic", "action", "comedy", "horror"];
const COLOR_PRESETS: ColorPreset[] = ["auto", "neutral", "warm", "golden-hour", "cool", "moonlight", "high-contrast"];
const TEMPO_PRESETS: TempoPreset[] = ["auto", "slow", "measured", "dynamic"];

function Spinner() {
  return <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-current border-r-transparent" />;
}

function PlayIcon() {
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
      <path d="M8 6.5 18 12 8 17.5Z" fill="currentColor" />
    </svg>
  );
}

function DownloadIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4" aria-hidden="true">
      <path d="M12 4v11m0 0-4-4m4 4 4-4M5 20h14" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function SunIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" className="h-3.5 w-3.5" aria-hidden="true">
      <circle cx="12" cy="12" r="3.5" stroke="currentColor" strokeWidth="1.6" />
      <path d="M12 2.5v2M12 19.5v2M4.5 12h-2M21.5 12h-2M5.28 5.28l1.42 1.42M17.3 17.3l1.42 1.42M18.72 5.28 17.3 6.7M6.7 17.3l-1.42 1.42" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg viewBox="0 0 24 24" fill="none" className="h-3.5 w-3.5" aria-hidden="true">
      <path d="M20 15.1A8.2 8.2 0 0 1 8.9 4a8.2 8.2 0 1 0 11.1 11.1Z" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function NewChatIcon() {
  return <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4" aria-hidden="true"><path d="M13.5 5H6.8A2.8 2.8 0 0 0 4 7.8v9.4A2.8 2.8 0 0 0 6.8 20h9.4a2.8 2.8 0 0 0 2.8-2.8v-6.7" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/><path d="m11 13 1.1-3.25L18.35 3.5a1.52 1.52 0 0 1 2.15 2.15l-6.25 6.25L11 13Z" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/></svg>;
}

function ElementsIcon() {
  return <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4" aria-hidden="true"><circle cx="8" cy="8" r="3" stroke="currentColor" strokeWidth="1.6"/><rect x="13" y="5" width="6" height="6" rx="1.5" stroke="currentColor" strokeWidth="1.6"/><path d="M5 18h14M8 15v6M16 15v6" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round"/></svg>;
}

function FilmIcon() {
  return <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4" aria-hidden="true"><rect x="3.5" y="5" width="17" height="14" rx="2.2" stroke="currentColor" strokeWidth="1.6"/><path d="M8 5v14M16 5v14M3.5 9h4.5M3.5 15h4.5M16 9h4.5M16 15h4.5" stroke="currentColor" strokeWidth="1.35"/></svg>;
}

function PlusIcon() {
  return <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4" aria-hidden="true"><path d="M12 5v14M5 12h14" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round"/></svg>;
}

function ChatIcon() {
  return <svg viewBox="0 0 24 24" fill="none" className="h-4 w-4" aria-hidden="true"><path d="M20 11.5a7.2 7.2 0 0 1-7.5 7.2 8 8 0 0 1-3.2-.65L5 19.5l1.35-3.65A7 7 0 0 1 5 11.5a7.2 7.2 0 0 1 7.5-7.2A7.2 7.2 0 0 1 20 11.5Z" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round"/></svg>;
}

function TrashIcon() {
  return <svg viewBox="0 0 24 24" fill="none" className="h-3.5 w-3.5" aria-hidden="true"><path d="M4.5 7h15M9 7V4.8h6V7m-8.5 0 .7 12h9.6l.7-12M10 10.5v5M14 10.5v5" stroke="currentColor" strokeWidth="1.55" strokeLinecap="round" strokeLinejoin="round"/></svg>;
}

function aspectClass(aspectRatio: AspectRatio) {
  if (aspectRatio === "9:16") return "aspect-[9/16]";
  if (aspectRatio === "1:1") return "aspect-square";
  return "aspect-video";
}

function errorMessage(error: unknown, fallback: string) {
  return error instanceof Error ? error.message : fallback;
}

function formatCredits(seconds: number) {
  if (seconds >= 60) return `${(seconds / 60).toFixed(seconds % 60 === 0 ? 0 : 1)} min`;
  return `${seconds}s`;
}

function qualityLabel(quality: RenderQuality) {
  if (quality === "4k") return "4K master";
  if (quality === "1080p") return "1080p master";
  return "Source preview";
}

function escapeRegExp(value: string) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function primaryElementAsset(element: CinemaElement) {
  return element.assets.find((asset) => asset.id === element.primary_asset_id) || element.assets[0] || null;
}

function elementRoleLabel(role: ElementAssetRole) {
  return role.replaceAll("_", " ");
}

function elementTone(type: ElementType) {
  if (type === "character") return "element-character";
  if (type === "prop") return "element-prop";
  if (type === "location") return "element-location";
  return "element-style";
}

export default function Home() {
  const [theme, setTheme] = useState<ThemeMode>("light");
  const [authUser, setAuthUser] = useState<AuthUser | null>(null);
  const [authChecking, setAuthChecking] = useState(true);
  const [loginEmail, setLoginEmail] = useState("");
  const [loginOtp, setLoginOtp] = useState("");
  const [demoOtp, setDemoOtp] = useState<string | null>(null);
  const [loginStep, setLoginStep] = useState<"email" | "otp">("email");
  const [loginBusy, setLoginBusy] = useState(false);
  const [loginError, setLoginError] = useState("");
  const [prompt, setPrompt] = useState("");
  const [mode, setMode] = useState<GenerationMode>("factory");
  const [aspectRatio, setAspectRatio] = useState<AspectRatio>("16:9");
  const [sceneCount, setSceneCount] = useState(2);
  const [durationSeconds, setDurationSeconds] = useState(15);
  const [factoryTargetSeconds, setFactoryTargetSeconds] = useState(30);
  const [factorySceneSeconds, setFactorySceneSeconds] = useState(20);
  const [quality, setQuality] = useState<RenderQuality>("1080p");
  const [audioMode, setAudioMode] = useState<AudioMode>("mastered");
  const [audioDirection, setAudioDirection] = useState(DEFAULT_AUDIO_DIRECTION);
  const [spokenScript, setSpokenScript] = useState("");
  const [videoTakes, setVideoTakes] = useState<NonNullable<FinalVideo>[]>([]);
  const [activeJob, setActiveJob] = useState<PendingJob | null>(null);
  const [jobError, setJobError] = useState("");
  const [serverJobs, setServerJobs] = useState<GenerationJobResponse[]>([]);
  const [recoveredAssets, setRecoveredAssets] = useState<NonNullable<GenerationJobResponse["assets"]>>([]);
  const [historySaveState, setHistorySaveState] = useState<"saved" | "saving" | "offline">("saved");
  const [historyRetry, setHistoryRetry] = useState(0);
  const jobController = useRef<AbortController | null>(null);
  const sessionsRef = useRef<StudioChatSession[]>([]);
  const accountVersion = useRef(0);
  const provider: VideoProviderName = "modal";
  const model: VideoModelName = "ltx-2.5";
  const [decoder, setDecoder] = useState<DecoderName>("conv");
  const [realismProfile, setRealismProfile] = useState<RealismProfile>("real_skin");
  const [seed, setSeed] = useState(42);
  const [enhancePrompt, setEnhancePrompt] = useState(false);
  const [continuityMode, setContinuityMode] = useState<ContinuityMode>("strict");

  // Cinema Studio-style director controls. Defaults are intentionally "auto" so
  // prompt-only mode remains byte-for-byte user-authored until the creator opts in.
  const [directorTab, setDirectorTab] = useState<DirectorTab>("scene");
  const [cameraMove, setCameraMove] = useState<CameraMove>("auto");
  const [lensPreset, setLensPreset] = useState<LensPreset>("auto");
  const [shotSize, setShotSize] = useState<ShotSize>("auto");
  const [genrePreset, setGenrePreset] = useState<GenrePreset>("auto");
  const [colorPreset, setColorPreset] = useState<ColorPreset>("auto");
  const [tempoPreset, setTempoPreset] = useState<TempoPreset>("auto");
  const [showReferencePicker, setShowReferencePicker] = useState(false);
  const [showElementsLibrary, setShowElementsLibrary] = useState(false);
  const [chatSessions, setChatSessions] = useState<StudioChatSession[]>([]);
  const [activeChatId, setActiveChatId] = useState<string | null>(null);
  const [chatHistoryReady, setChatHistoryReady] = useState(false);

  const [elements, setElements] = useState<CinemaElement[]>([]);
  const [elementsLoading, setElementsLoading] = useState(false);
  const [elementBusy, setElementBusy] = useState(false);
  const [showElementCreator, setShowElementCreator] = useState(false);
  const [elementName, setElementName] = useState("");
  const [elementHandle, setElementHandle] = useState("");
  const [elementType, setElementType] = useState<ElementType>("character");
  const [elementFilter, setElementFilter] = useState<"all" | ElementType>("all");
  const [elementSearch, setElementSearch] = useState("");
  const [selectedElementId, setSelectedElementId] = useState<string | null>(null);
  const [elementDescription, setElementDescription] = useState("");
  const [elementFiles, setElementFiles] = useState<File[]>([]);
  const [elementUploadError, setElementUploadError] = useState("");
  const [elementFileRoles, setElementFileRoles] = useState<Record<string, ElementAssetRole>>({});
  const elementSaveRequest = useRef({ signature: "", id: "" });
  const elementAddRequests = useRef(new Map<string, string>());
  const creatorRef = useRef<HTMLDivElement>(null);
  const [elementModes, setElementModes] = useState<Record<string, ElementReferenceMode>>({});
  const [elementWardrobePolicies, setElementWardrobePolicies] = useState<Record<string, ElementWardrobePolicy>>({});
  const [elementApplyAll, setElementApplyAll] = useState<Record<string, boolean>>({});
  const [elementStrengths, setElementStrengths] = useState<Record<string, number>>({});
  const [mentionQuery, setMentionQuery] = useState<string | null>(null);

  const [publishToYouTube, setPublishToYouTube] = useState(false);
  const [youtubeTitle, setYoutubeTitle] = useState("");
  const [youtubeDescription, setYoutubeDescription] = useState("");
  const [youtubePrivacy, setYoutubePrivacy] = useState<YouTubePrivacy>("private");

  const [capabilities, setCapabilities] = useState<GenerationCapabilitiesResponse | null>(null);
  const [billingCatalog, setBillingCatalog] = useState<BillingCatalogResponse | null>(null);
  const [billingMe, setBillingMe] = useState<BillingMeResponse | null>(null);
  const [youtube, setYoutube] = useState<YouTubeStatusResponse | null>(null);

  const [result, setResult] = useState<ScenePlanResponse | null>(null);
  const [scenePrompts, setScenePrompts] = useState<ScenePromptMap>({});
  const [renderedVideos, setRenderedVideos] = useState<RenderedVideoMap>({});
  const [factoryResult, setFactoryResult] = useState<FactoryGenerationResponse | null>(null);
  const [finalVideo, setFinalVideo] = useState<FinalVideo>(null);

  const [editingScene, setEditingScene] = useState<number | null>(null);
  const [planning, setPlanning] = useState(false);
  const [directGenerating, setDirectGenerating] = useState(false);
  const [factoryGenerating, setFactoryGenerating] = useState(false);
  const [generatingScene, setGeneratingScene] = useState<number | null>(null);
  const [creatingFinal, setCreatingFinal] = useState(false);
  const [integrationBusy, setIntegrationBusy] = useState(false);
  const [progressMessage, setProgressMessage] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");

  const durationOptions = DURATION_OPTIONS[quality];
  const factoryDurationOptions = FACTORY_SCENE_OPTIONS[quality].filter((value) => value <= factoryTargetSeconds);
  const isBusy = Boolean(activeJob) || planning || directGenerating || factoryGenerating || generatingScene !== null || creatingFinal;
  const renderedSceneCount = Object.keys(renderedVideos).length;
  const allScenesRendered = !!result && result.scenes.length > 0 && result.scenes.every((scene) => Boolean(renderedVideos[scene.id]) && renderedVideos[scene.id].renderSignature === sceneRenderSignature(scene.id));
  const plannedDuration = useMemo(() => result ? result.scenes.length * durationSeconds : 0, [result, durationSeconds]);
  const filteredElements = useMemo(() => {
    const query = elementSearch.trim().toLowerCase();
    return elements.filter((element) => {
      if (elementFilter !== "all" && element.type !== elementFilter) return false;
      if (!query) return true;
      return element.name.toLowerCase().includes(query) || element.handle.toLowerCase().includes(query);
    });
  }, [elements, elementFilter, elementSearch]);
  const referencedElements = useMemo(
    () => elements.filter((element) => hasElementMention(prompt, element.handle) || Boolean(elementApplyAll[element.id])),
    [elements, prompt, elementApplyAll]
  );
  const activeElementLimit = capabilities?.elements?.max_active_per_scene ?? 6;
  const elementBindings = useMemo<ElementBinding[]>(
    () => referencedElements.slice(0, activeElementLimit).map((element) => ({
      element_id: element.id,
      version_id: element.current_version_id,
      handle: element.handle,
      reference_mode: elementModes[element.id] || "identity",
      wardrobe_policy: element.type === "character" ? (elementWardrobePolicies[element.id] || "reference") : "reference",
      strength: Math.max(0, Math.min(1, elementStrengths[element.id] ?? 1.0)),
      apply_to_all_scenes: appliesToAllScenes(element.type, elementApplyAll[element.id]),
    })),
    [referencedElements, activeElementLimit, elementModes, elementWardrobePolicies, elementStrengths, elementApplyAll]
  );
  const mentionSuggestions = useMemo(() => {
    if (mentionQuery == null) return [];
    const query = mentionQuery.toLowerCase();
    return elements.filter((element) =>
      element.handle.toLowerCase().startsWith(query) || element.name.toLowerCase().includes(query)
    ).slice(0, 8);
  }, [mentionQuery, elements]);
  const selectedElement = useMemo(
    () => elements.find((element) => element.id === selectedElementId) || referencedElements[0] || null,
    [elements, selectedElementId, referencedElements]
  );
  const studioPreviewElement = selectedElement || referencedElements[0] || null;
  const studioPreviewAsset = studioPreviewElement ? primaryElementAsset(studioPreviewElement) : null;
  const characterElements = useMemo(() => elements.filter((element) => element.type === "character"), [elements]);
  const quickCharacterElements = useMemo(
    () => characterElements.filter((element) => !referencedElements.some((active) => active.id === element.id)).slice(0, 4),
    [characterElements, referencedElements]
  );
  const latestRenderedVideo = useMemo(() => {
    const values = Object.values(renderedVideos);
    return values.length ? values[values.length - 1] : null;
  }, [renderedVideos]);
  const studioVideoUrl = finalVideo?.url || latestRenderedVideo?.url || null;
  const sortedChatSessions = useMemo(
    () => [...chatSessions].sort((a, b) => b.updatedAt - a.updatedAt),
    [chatSessions]
  );
  const activeChatTitle = useMemo(
    () => chatSessions.find((session) => session.id === activeChatId)?.title || chatTitleFromPrompt(prompt),
    [chatSessions, activeChatId, prompt]
  );

  const overlayOpen = showElementCreator || showReferencePicker || showElementsLibrary;
  useEffect(() => {
    studioPlayback.pauseAll();
    return () => studioPlayback.pauseAll();
  }, [activeChatId, authUser?.id, mode, studioVideoUrl, overlayOpen]);

  useEffect(() => {
    if (!overlayOpen) return;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => { document.body.style.overflow = previous; };
  }, [overlayOpen]);

  const closeCreatorOnEscape = useEffectEvent(() => { if (!elementBusy) setShowElementCreator(false); });
  useEffect(() => {
    if (!showElementCreator) return;
    const previous = document.activeElement as HTMLElement | null;
    const modal = creatorRef.current;
    modal?.querySelector<HTMLInputElement>("input")?.focus({ preventScroll: true });
    function trapFocus(event: KeyboardEvent) {
      if (event.key === "Escape") closeCreatorOnEscape();
      if (event.key !== "Tab" || !modal) return;
      const controls = Array.from(modal.querySelectorAll<HTMLElement>("button:not(:disabled), input:not(:disabled), textarea:not(:disabled), select:not(:disabled)"))
        .filter((control) => control.getClientRects().length > 0);
      const first = controls[0];
      const last = controls.at(-1);
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last?.focus(); }
      else if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first?.focus(); }
    }
    document.addEventListener("keydown", trapFocus);
    return () => {
      document.removeEventListener("keydown", trapFocus);
      previous?.focus({ preventScroll: true });
    };
  }, [showElementCreator]);

  useEffect(() => {
    let active = true;
    async function restoreLogin() {
      try {
        const session = await getAuthMe();
        if (!active) return;
        setAuthUser(session.authenticated ? session.user : null);
      } catch {
        if (active) setAuthUser(null);
      } finally {
        if (active) setAuthChecking(false);
      }
    }
    void restoreLogin();
    return () => { active = false; };
  }, []);

  useEffect(() => {
    const storedTheme = window.localStorage.getItem("triven-cinema-theme");
    const initialTheme: ThemeMode = storedTheme === "dark" ? "dark" : "light";
    document.documentElement.dataset.theme = initialTheme;
    if (initialTheme !== theme) {
      const timer = window.setTimeout(() => setTheme(initialTheme), 0);
      return () => window.clearTimeout(timer);
    }
    return undefined;
  }, [theme]);

  function selectTheme(nextTheme: ThemeMode) {
    setTheme(nextTheme);
    document.documentElement.dataset.theme = nextTheme;
    window.localStorage.setItem("triven-cinema-theme", nextTheme);
  }

  useEffect(() => {
    if (!authUser) return;
    let active = true;
    async function loadHistory() {
      await Promise.resolve();
      if (!active) return;
      setChatHistoryReady(false);
      setActiveChatId(null);
      const cached = readStoredChatSessions(authUser!.id);
      let sessions = cached;
      try {
        const response = await listChatHistory();
        if (!active) return;
        sessions = reconcileChatSessions(response.chats.map(chatFromServer), cached);
        // Reconcile locally completed renders after a failed or interrupted save.
        await Promise.allSettled(sessions.filter((session) => {
          const remote = response.chats.find((entry) => entry.id === session.id);
          return !remote || session.updatedAt > remote.updated_at;
        }).map((session) => saveChatHistoryItem(chatToServer(session))));
      } catch {
        if (!active) return;
        setHistorySaveState("offline");
      }
      if (!active) return;
      sessionsRef.current = sessions;
      setChatSessions(sessions);
      persistChatSessions(sessions, authUser!.id);
      const unfinished = sessions.find((session) => session.workspace.activeJob);
      if (unfinished) {
        setActiveChatId(unfinished.id);
        restoreWorkspace(unfinished.workspace);
      }
      setChatHistoryReady(true);
    }
    void loadHistory();
    return () => { active = false; };
  }, [authUser]);

  const restoreWorkspace = useEffectEvent((workspace: StudioChatWorkspace) => applyChatWorkspace(workspace));

  const syncWorkspace = useEffectEvent(() => {
    if (!chatHistoryReady) return;
    const workspace: StudioChatWorkspace = {
      prompt,
      mode,
      aspectRatio,
      sceneCount,
      durationSeconds,
      factoryTargetSeconds,
      factorySceneSeconds,
      quality,
      audioMode,
      audioDirection,
      spokenScript,
      videoTakes,
      activeJob,
      jobError,
      recoveredAssets,
      decoder,
      realismProfile,
      seed,
      enhancePrompt,
      continuityMode,
      cameraMove,
      lensPreset,
      shotSize,
      genrePreset,
      colorPreset,
      tempoPreset,
      selectedElementId,
      elementModes,
      elementWardrobePolicies,
      elementApplyAll,
      elementStrengths,
      result,
      scenePrompts,
      renderedVideos,
      factoryResult,
      finalVideo,
    };

    // Do not put a blank landing-state chat into history. The draft becomes a
    // real chat as soon as the creator starts writing a prompt.
    if (!activeChatId) {
      if (!prompt.trim()) return;
      const session = createChatSession(workspace);
      setActiveChatId(session.id);
      setChatSessions((current) => {
        const sorted = [session, ...current].sort((a, b) => b.updatedAt - a.updatedAt);
        persistChatSessions(sorted, authUser?.id);
        sessionsRef.current = sorted;
        return sorted;
      });
      return;
    }

    setChatSessions((current) => {
      const now = Date.now();
      let found = false;
      const next = current.map((session) => {
        if (session.id !== activeChatId) return session;
        found = true;
        return {
          ...session,
          title: prompt.trim() ? chatTitleFromPrompt(prompt) : session.title,
          updatedAt: now,
          dirty: true,
          workspace,
        };
      });
      if (!found) return current;
      const sorted = next.sort((a, b) => b.updatedAt - a.updatedAt);
      persistChatSessions(sorted, authUser?.id);
      sessionsRef.current = sorted;
      return sorted;
    });
  });

  useEffect(() => {
    const timer = window.setTimeout(() => syncWorkspace(), 0);
    return () => window.clearTimeout(timer);
  }, [
    authUser?.id,
    chatHistoryReady,
    activeChatId,
    prompt,
    mode,
    aspectRatio,
    sceneCount,
    durationSeconds,
    factoryTargetSeconds,
    factorySceneSeconds,
    quality,
    audioMode,
    audioDirection,
    spokenScript,
    videoTakes,
    activeJob,
    jobError,
    recoveredAssets,
    decoder,
    realismProfile,
    seed,
    enhancePrompt,
    continuityMode,
    cameraMove,
    lensPreset,
    shotSize,
    genrePreset,
    colorPreset,
    tempoPreset,
    selectedElementId,
    elementModes,
    elementWardrobePolicies,
    elementApplyAll,
    elementStrengths,
    result,
    scenePrompts,
    renderedVideos,
    factoryResult,
    finalVideo,
  ]);

  useEffect(() => {
    if (!authUser || !chatHistoryReady) return;
    const pending = chatSessions.filter((session) => session.dirty);
    if (!pending.length) return;
    const epoch = accountVersion.current;
    let active = true;
    const timer = window.setTimeout(async () => {
      setHistorySaveState("saving");
      try {
        await Promise.all(pending.map((session) => saveChatHistoryItem(chatToServer(session))));
        if (epoch === accountVersion.current) {
          const versions = new Map(pending.map((session) => [session.id, session.updatedAt]));
          setChatSessions((current) => {
            const next = current.map((session) => versions.get(session.id) === session.updatedAt ? { ...session, dirty: false } : session);
            sessionsRef.current = next;
            persistChatSessions(next, authUser.id);
            return next;
          });
          if (active) setHistorySaveState("saved");
        }
      } catch {
        if (active && epoch === accountVersion.current) setHistorySaveState("offline");
      }
    }, 700);
    return () => { active = false; window.clearTimeout(timer); };
  }, [authUser, chatHistoryReady, chatSessions, historyRetry]);

  useEffect(() => {
    if (historySaveState !== "offline") return;
    const timer = window.setTimeout(() => setHistoryRetry((current) => current + 1), 15000);
    const retry = () => setHistoryRetry((current) => current + 1);
    window.addEventListener("online", retry);
    return () => { window.clearTimeout(timer); window.removeEventListener("online", retry); };
  }, [historySaveState, historyRetry]);

  useEffect(() => () => { jobController.current?.abort(); }, []);

  async function recoverSavedJob(job: PendingJob) {
    if (jobController.current) return;
    try {
      if (job.kind === "audio_retake") {
        const response = await executeTrackedJob<AudioRetakeResponse>(job);
        rememberVideo(audioRetakeVideo(response), job.aspectRatio);
      } else if (job.kind === "factory") {
        const response = await executeTrackedJob<FactoryGenerationResponse>(job);
        setFactoryResult(response);
        rememberVideo(factoryVideoFromResponse(response), job.aspectRatio);
      } else if (job.kind === "combine") {
        const response = await executeTrackedJob<CombineScenesResponse>(job);
        rememberVideo({ url: absoluteApiUrl(response.final_video_url), downloadUrl: absoluteApiUrl(response.final_download_url), filename: response.final_filename, label: `${response.scene_count} scenes · recovered composition`, qualityNote: [response.quality_note, ...(response.warnings || [])].filter(Boolean).join(" "), hasAudio: response.media_info.has_audio, audioCodec: response.media_info.audio_codec, dimensions: `${response.media_info.width}×${response.media_info.height}`, qcReport: { visual: response.visual_qc_status || "unknown", audio: response.audio_qc_status || "unknown", visualWarnings: response.warnings || [], audioWarnings: [], visualRetries: 0, audioRetakes: 0 } }, job.aspectRatio);
      } else {
        const response = await executeTrackedJob<VideoGenerationResponse>(job);
        if (job.sceneId != null) {
          setRenderedVideos((current) => ({ ...current, [job.sceneId!]: { ...sceneVideoFromResponse(response), renderSignature: job.renderSignature } }));
          const take = { ...finalFromSceneResponse(response)!, aspectRatio: job.aspectRatio, label: `Scene ${job.sceneId} · recovered render`, createdAt: Date.now() };
          setVideoTakes((current) => [take, ...current.filter((item) => item.filename !== take.filename)]);
          setNotice("Scene recovered. Render remaining scenes or combine when ready.");
        } else rememberVideo(finalFromSceneResponse(response)!, job.aspectRatio);
      }
    } catch (err) {
      if (!(err instanceof DOMException && err.name === "AbortError")) setError(errorMessage(err, "Unable to recover this job."));
    } finally { setProgressMessage(""); }
  }

  const resumeSavedJob = useEffectEvent(recoverSavedJob);

  useEffect(() => {
    if (!authUser || !chatHistoryReady || !activeJob || jobController.current) return;
    void resumeSavedJob(activeJob);
  }, [authUser, chatHistoryReady, activeJob]);

  useEffect(() => {
    if (!authUser) return;
    let active = true;
    async function bootstrap() {
      setElementsLoading(true);
      const [caps, catalog, billing, yt, elementList] = await Promise.allSettled([
        getGenerationCapabilities(),
        getBillingCatalog(),
        getBillingMe(),
        getYouTubeStatus(),
        listElements(),
      ]);
      if (!active) return;
      if (caps.status === "fulfilled") setCapabilities(caps.value);
      if (catalog.status === "fulfilled") setBillingCatalog(catalog.value);
      if (billing.status === "fulfilled") setBillingMe(billing.value);
      if (yt.status === "fulfilled") setYoutube(yt.value);
      if (elementList.status === "fulfilled") setElements(elementList.value.elements);
      setElementsLoading(false);

      const params = new URLSearchParams(window.location.search);
      const sessionId = params.get("session_id");
      if (params.get("checkout") === "success" && sessionId) {
        try {
          const status = await verifyCheckout(sessionId);
          if (!active) return;
          setNotice(status.paid ? `Payment confirmed. ${formatCredits(status.balance_seconds)} generation credits available.` : "Payment is still processing.");
          setBillingMe(await getBillingMe());
        } catch (err) {
          if (active) setError(errorMessage(err, "Unable to verify payment."));
        }
      }
      if (params.get("youtube") === "connected") {
        if (!active) return;
        setNotice("YouTube channel connected. Factory jobs can now publish automatically.");
        try {
          const status = await getYouTubeStatus();
          if (active) setYoutube(status);
        } catch {
          // Connection succeeded; status refresh can recover on the next bootstrap.
        }
      }
      if (params.has("checkout") || params.has("youtube")) {
        window.history.replaceState({}, "", window.location.pathname);
      }
    }
    void bootstrap();
    return () => { active = false; };
  }, [authUser]);

  useEffect(() => {
    if (!authUser || !activeChatId) return;
    let active = true;
    void listGenerationJobs(activeChatId).then((jobs) => { if (active) setServerJobs(jobs); }).catch(() => { /* The saved chat still supplies local recovery. */ });
    return () => { active = false; };
  }, [authUser, activeChatId, activeJob]);

  function recoverServerJob(job: GenerationJobResponse) {
    if (isBusy || !activeChatId || (job.chat_id || job.payload.chat_id) !== activeChatId || !["video", "factory", "combine", "audio_retake"].includes(job.job_type)) return;
    const kind = job.job_type as PendingJob["kind"];
    const sceneIndex = typeof job.payload.scene_index === "number" ? job.payload.scene_index : undefined;
    void recoverSavedJob({ kind, requestId: String(job.payload.request_id || job.job_id), jobId: job.job_id, sceneId: sceneIndex == null ? undefined : result?.scenes[sceneIndex]?.id, aspectRatio: (job.payload.aspect_ratio as AspectRatio) || aspectRatio, payload: job.payload as unknown as PendingJob["payload"] });
  }

  function applyChatWorkspace(rawWorkspace: StudioChatWorkspace) {
    studioPlayback.pauseAll();
    const workspace = normalizeChatWorkspace(rawWorkspace);
    setServerJobs([]);
    setPrompt(workspace.prompt);
    setMode(workspace.mode);
    setAspectRatio(workspace.aspectRatio);
    setSceneCount(workspace.sceneCount);
    setDurationSeconds(workspace.durationSeconds);
    setFactoryTargetSeconds(workspace.factoryTargetSeconds);
    setFactorySceneSeconds(workspace.factorySceneSeconds);
    setQuality(workspace.quality);
    setAudioMode(workspace.audioMode);
    setAudioDirection(workspace.audioDirection);
    setSpokenScript(workspace.spokenScript);
    setVideoTakes(workspace.videoTakes);
    setActiveJob(workspace.activeJob);
    setJobError(workspace.jobError);
    setRecoveredAssets(workspace.recoveredAssets);
    setDecoder(workspace.decoder);
    setRealismProfile(workspace.realismProfile);
    setSeed(workspace.seed);
    setEnhancePrompt(workspace.enhancePrompt);
    setContinuityMode(workspace.continuityMode);
    setCameraMove(workspace.cameraMove);
    setLensPreset(workspace.lensPreset);
    setShotSize(workspace.shotSize);
    setGenrePreset(workspace.genrePreset);
    setColorPreset(workspace.colorPreset);
    setTempoPreset(workspace.tempoPreset);
    setSelectedElementId(workspace.selectedElementId);
    setElementModes(workspace.elementModes);
    setElementWardrobePolicies(workspace.elementWardrobePolicies);
    setElementApplyAll(workspace.elementApplyAll);
    setElementStrengths(workspace.elementStrengths);
    setResult(workspace.result);
    setScenePrompts(workspace.scenePrompts);
    setRenderedVideos(workspace.renderedVideos);
    setFactoryResult(workspace.factoryResult);
    setFinalVideo(workspace.finalVideo);
    setEditingScene(null);
    setProgressMessage("");
    setError("");
    setNotice("");
    setMentionQuery(null);
    setShowReferencePicker(false);
    setShowElementsLibrary(false);
    setDirectorTab("scene");
  }

  function handleNewChat() {
    if (isBusy) {
      setNotice("Wait for the current render to finish before starting a new chat.");
      return;
    }

    // The active chat is already kept in chatSessions by the workspace sync effect.
    // New Chat only switches the UI to a fresh unsaved draft; it will not appear in
    // Previous chats until the creator starts typing a prompt.
    persistChatSessions(chatSessions, authUser?.id);
    setActiveChatId(null);
    applyChatWorkspace(createEmptyChatWorkspace());
  }

  function handleOpenChat(session: StudioChatSession) {
    if (session.id === activeChatId) return;
    if (isBusy) {
      setNotice("Wait for the current render to finish before switching chats.");
      return;
    }
    setActiveChatId(session.id);
    applyChatWorkspace(session.workspace);
    persistChatSessions(chatSessions, authUser?.id);
  }

  async function handleDeleteChat(chatId: string) {
    // The current working chat is intentionally protected from deletion.
    if (chatId === activeChatId) return;
    const target = chatSessions.find((session) => session.id === chatId);
    if (!target) return;
    if (!window.confirm(`Delete “${target.title}” from chat history? Rendered media files will not be deleted.`)) return;

    const remaining = chatSessions.filter((session) => session.id !== chatId);
    sessionsRef.current = remaining;
    setChatSessions(remaining);
    persistChatSessions(remaining, authUser?.id);
    try {
      await deleteChatHistoryItem(chatId);
    } catch (err) {
      setError(errorMessage(err, "Unable to delete chat from your account."));
      setChatSessions((current) => [target, ...current].sort((a, b) => b.updatedAt - a.updatedAt));
    }
  }

  function handleQualityChange(nextQuality: RenderQuality) {
    setQuality(nextQuality);
    const options = DURATION_OPTIONS[nextQuality];
    const factoryOptions = FACTORY_SCENE_OPTIONS[nextQuality].filter((value) => value <= factoryTargetSeconds);
    setDurationSeconds((current) => options.includes(current) ? current : options[Math.min(2, options.length - 1)]);
    setFactorySceneSeconds((current) => {
      if (factoryOptions.includes(current)) return current;
      return factoryOptions[factoryOptions.length - 1] ?? Math.min(factoryTargetSeconds, 15);
    });
    invalidateRenderedMedia();
  }

  function handleFactoryTargetChange(nextTarget: number) {
    const maximum = capabilities?.max_factory_duration_seconds ?? 300;
    const next = Math.max(15, Math.min(maximum, Number(nextTarget) || 15));
    setFactoryTargetSeconds(next);
    setFactorySceneSeconds((current) => Math.min(current, next));
    invalidateRenderedMedia();
  }

  function handleFactorySceneDurationChange(nextSceneSeconds: number) {
    const next = Math.max(15, Math.min(factoryTargetSeconds, Number(nextSceneSeconds) || 15));
    setFactorySceneSeconds(next);
    invalidateRenderedMedia();
  }

  function resetOutput() {
    setResult(null);
    setScenePrompts({});
    setRenderedVideos({});
    setFactoryResult(null);
    // Previous takes remain available until a replacement has completed.

    setEditingScene(null);
    setProgressMessage("");
  }

  function invalidateRenderedMedia() {
    setRenderedVideos({});
    // Previous takes remain available until a replacement has completed.

  }

  async function refreshBilling() {
    try { setBillingMe(await getBillingMe()); } catch { /* optional integration */ }
  }

  function storeElement(element: CinemaElement) {
    setElements((current) => {
      if (element.status === "archived") return current.filter((item) => item.id !== element.id);
      return current.some((item) => item.id === element.id)
        ? current.map((item) => item.id === element.id ? element : item)
        : [element, ...current];
    });
  }

  function selectElementFiles(incoming: File[]) {
    if (elementBusy) return;
    const selected = selectReferenceFiles(elementFiles, incoming,
      capabilities?.elements?.max_assets_per_element ?? 8, capabilities?.elements?.max_upload_mb ?? 15);
    const roles = referenceRoles(elementType, selected.files, elementFileRoles);
    setElementFileRoles(Object.fromEntries(selected.files.map((file, index) => [referenceFileKey(file), roles[index]])));
    setElementFiles(selected.files);
    setElementUploadError(selected.error);
  }

  function updateMentionState(value: string) {
    setPrompt(value);
    const match = value.match(/(?:^|\s)@([A-Za-z0-9_-]*)$/);
    setMentionQuery(match ? match[1] : null);
  }

  function canActivateElement(element: CinemaElement) {
    const alreadyActive = hasElementMention(prompt, element.handle) || Boolean(elementApplyAll[element.id]);
    if (alreadyActive) return true;
    if (referencedElements.length >= activeElementLimit) {
      setError(`This LTX profile allows ${activeElementLimit} active Elements in one scene. Remove a reference before adding @${element.handle}.`);
      return false;
    }
    return true;
  }

  function insertElementMention(element: CinemaElement) {
    if (isBusy) return;
    if (!canActivateElement(element)) return;
    setSelectedElementId(element.id);
    const mention = `@${element.handle}`;
    if (!hasElementMention(prompt, element.handle)) {
      const spacer = prompt && !/\s$/.test(prompt) ? " " : "";
      setPrompt(`${prompt}${spacer}${mention} `);
    }
    setMentionQuery(null);
    setElementModes((current) => ({ ...current, [element.id]: current[element.id] || "identity" }));
    if (element.type === "character") setElementWardrobePolicies((current) => ({ ...current, [element.id]: current[element.id] || "reference" }));
    setElementStrengths((current) => ({ ...current, [element.id]: current[element.id] ?? 1.0 }));
    setElementApplyAll((current) => ({
      ...current,
      [element.id]: current[element.id] ?? (element.type === "character" || element.type === "style"),
    }));
  }

  function removeElementFromScene(element: CinemaElement) {
    if (isBusy) return;
    const mentionPattern = new RegExp(`(^|[^A-Za-z0-9_])@${escapeRegExp(element.handle)}(?![A-Za-z0-9_-])\\s*`, "gi");
    setPrompt((current) => current.replace(mentionPattern, "$1").replace(/ {2,}/g, " ").trimStart());
    setElementApplyAll((current) => ({ ...current, [element.id]: false }));
    if (selectedElementId === element.id) setSelectedElementId(null);
  }

  function promptWithDirectorControls(rawPrompt: string) {
    const directives: string[] = [];
    if (shotSize !== "auto") directives.push(`${shotSize.replaceAll("-", " ")} framing`);
    if (cameraMove !== "auto") directives.push(`${cameraMove.replaceAll("-", " ")} camera movement`);
    if (lensPreset !== "auto") directives.push(`${lensPreset} cinema lens`);
    if (genrePreset !== "auto") directives.push(`${genrePreset} genre language`);
    if (colorPreset !== "auto") directives.push(`${colorPreset.replaceAll("-", " ")} color palette`);
    if (tempoPreset !== "auto") directives.push(`${tempoPreset} performance tempo`);
    if (!directives.length) return rawPrompt.trim();
    return `${rawPrompt.trim()}\n\nDirector controls: ${directives.join("; ")}. Preserve all @Element identities and user-authored story details.`;
  }

  function chooseMention(element: CinemaElement) {
    if (isBusy) return;
    if (mentionQuery == null || !canActivateElement(element)) return;
    setSelectedElementId(element.id);
    const next = prompt.replace(/@([A-Za-z0-9_-]*)$/, `@${element.handle} `);
    setPrompt(next);
    setMentionQuery(null);
    setElementModes((current) => ({ ...current, [element.id]: current[element.id] || "identity" }));
    if (element.type === "character") setElementWardrobePolicies((current) => ({ ...current, [element.id]: current[element.id] || "reference" }));
    setElementStrengths((current) => ({ ...current, [element.id]: current[element.id] ?? 1.0 }));
    setElementApplyAll((current) => ({
      ...current,
      [element.id]: current[element.id] ?? (element.type === "character" || element.type === "style"),
    }));
  }

  function applyCreatorGradePreset() {
    if (isBusy) return;
    const activeCharacters = referencedElements.filter((element) => element.type === "character");
    // Creator-grade is a quality/consistency preset, not a duration preset.
    // Preserve the runtime and scene length the user selected.
    setQuality("1080p");
    setDecoder("diffusion");
    setContinuityMode("strict");
    setEnhancePrompt(false);
    invalidateRenderedMedia();
    if (activeCharacters.length > 1) {
      setRealismProfile("real_skin");
      setError(
        `Creator-grade solo presenter mode found ${activeCharacters.length} Character Elements (${activeCharacters.map((element) => `@${element.handle}`).join(", ")}). Keep only the one person who should appear in this shot; multiple identity sheets can blend faces.`
      );
      return;
    }
    if (activeCharacters.length) {
      setRealismProfile("identity_max");
      setElementModes((current) => {
        const next = { ...current };
        activeCharacters.forEach((element) => { next[element.id] = "identity"; });
        return next;
      });
      setElementWardrobePolicies((current) => {
        const next = { ...current };
        activeCharacters.forEach((element) => { next[element.id] = "prompt"; });
        return next;
      });
      setElementStrengths((current) => {
        const next = { ...current };
        activeCharacters.forEach((element) => { next[element.id] = 1.0; });
        return next;
      });
      setElementApplyAll((current) => {
        const next = { ...current };
        activeCharacters.forEach((element) => { next[element.id] = true; });
        return next;
      });
      setNotice(`Creator-grade preset applied for your selected ${factoryTargetSeconds}s runtime: 1080p Diffusion final, Identity Max, strict continuity, stage-2 Character lock and prompt-authoritative wardrobe.`);
    } else {
      setRealismProfile("real_skin");
      setNotice(`Creator-grade render settings applied for your selected ${factoryTargetSeconds}s runtime. Add a Character Element to enable Identity Max face locking.`);
    }
  }

  async function handleCreateElement() {
    if (isBusy || elementBusy) return;
    if (!elementName.trim() || !elementHandle.trim() || elementFiles.length === 0) {
      setElementUploadError("Enter a name, @handle and at least one reference image.");
      return;
    }
    if (!/^[A-Za-z][A-Za-z0-9_-]{0,31}$/.test(elementHandle)) {
      setElementUploadError("The handle must start with a letter and use up to 32 letters, numbers, underscores or hyphens.");
      return;
    }
    setElementBusy(true);
    setElementUploadError("");
    const roles = referenceRoles(elementType, elementFiles, elementFileRoles);
    const signature = JSON.stringify([elementName.trim(), elementHandle, elementType, elementDescription.trim(), roles, elementFiles.map(referenceFileKey)]);
    if (elementSaveRequest.current.signature !== signature) elementSaveRequest.current = { signature, id: crypto.randomUUID() };
    try {
      const created = await createElement({
        name: elementName.trim(),
        handle: elementHandle.trim().replace(/^@/, ""),
        type: elementType,
        description: elementDescription.trim(),
        files: elementFiles,
        roles,
        requestId: elementSaveRequest.current.id,
      });
      storeElement(created);
      setSelectedElementId(created.id);
      setShowElementCreator(false);
      setElementName("");
      setElementHandle("");
      setElementDescription("");
      setElementFiles([]);
      setElementFileRoles({});
      elementSaveRequest.current = { signature: "", id: "" };
      insertElementMention(created);
      setNotice(`Saved @${created.handle} as a reusable ${created.type} Element.`);
    } catch (err) {
      setElementUploadError(errorMessage(err, "Unable to save. Your selected images are still here; retry when the connection returns."));
    } finally {
      setElementBusy(false);
    }
  }

  async function handleAddElementReferences(element: CinemaElement, files: FileList | null) {
    if (isBusy || elementBusy) return;
    if (!files?.length) return;
    setElementBusy(true);
    setError("");
    try {
      const incoming = Array.from(files);
      const selected = selectReferenceFiles([], incoming,
        (capabilities?.elements?.max_assets_per_element ?? 8) - element.assets.length, capabilities?.elements?.max_upload_mb ?? 15);
      if (selected.error) throw new Error(selected.error);
      const signature = JSON.stringify([element.id, incoming.map(referenceFileKey)]);
      const requestId = elementAddRequests.current.get(signature) || crypto.randomUUID();
      elementAddRequests.current.set(signature, requestId);
      const updated = await addElementAssets(
        element.id,
        incoming,
        referenceRoles(element.type, incoming, {}, element.assets.map((asset) => asset.role)),
        requestId
      );
      storeElement(updated);
      setNotice(`Added reference images to @${element.handle}. A new immutable Element version was created.`);
    } catch (err) {
      setError(errorMessage(err, "Unable to add Element references."));
    } finally {
      setElementBusy(false);
    }
  }

  async function handleSetPrimaryElementAsset(element: CinemaElement, assetId: string) {
    if (isBusy) return;
    if (assetId === element.primary_asset_id) return;
    setElementBusy(true);
    setError("");
    try {
      storeElement(await updateElement(element.id, { primary_asset_id: assetId }));
      setNotice(`Updated the canonical reference for @${element.handle}. A new immutable Element version was created.`);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Unable to update the canonical Element reference.");
    } finally {
      setElementBusy(false);
    }
  }

  async function handleArchiveElement(element: CinemaElement) {
    if (isBusy) return;
    setElementBusy(true);
    setError("");
    try {
      storeElement(await archiveElement(element.id));
      setNotice(`Archived @${element.handle}. Existing renders remain tied to their saved version.`);
    } catch (err) {
      setError(errorMessage(err, "Unable to archive Element."));
    } finally {
      setElementBusy(false);
    }
  }

  function persistJobSnapshot(job: PendingJob | null, patch: Partial<StudioChatWorkspace> = {}) {
    if (!authUser || !activeChatId) return;
    const current = sessionsRef.current;
    const previous = current.find((session) => session.id === activeChatId);
    if (!previous) return;
    const session = { ...previous, updatedAt: Date.now(), dirty: true, workspace: { ...previous.workspace, ...patch, activeJob: job } };
    const next = current.map((item) => item.id === session.id ? session : item);
    sessionsRef.current = next;
    persistChatSessions(next, authUser.id);
    setChatSessions(next);
    void saveChatHistoryItem(chatToServer(session)).catch(() => setHistorySaveState("offline"));
  }

  function rememberVideo(video: NonNullable<FinalVideo>, ratio = aspectRatio) {
    const saved = { ...video, aspectRatio: video.aspectRatio || ratio, createdAt: video.createdAt || Date.now() };
    setFinalVideo(saved);
    setFactoryResult(saved.factoryResult || null);
    setVideoTakes((current) => [saved, ...current.filter((item) => item.filename !== saved.filename)]);
  }

  async function executeTrackedJob<T>(job: PendingJob): Promise<T> {
    const controller = new AbortController();
    jobController.current = controller;
    const account = accountVersion.current;
    setJobError("");
    setActiveJob(job);
    persistJobSnapshot(job);
    try {
      if (!job.jobId) {
        for (;;) {
          controller.signal.throwIfAborted();
          try {
            const started = job.kind === "factory"
              ? await createFactoryGenerationJob(job.payload as FactoryGenerationRequest)
              : job.kind === "audio_retake"
                ? await createAudioRetakeJob(job.payload as AudioRetakeRequest)
                : job.kind === "combine"
                ? await createCombineGenerationJob(job.payload as CombineScenesRequest)
                : await createVideoGenerationJob(job.payload as VideoGenerationRequest);
            job = { ...job, jobId: started.job_id };
            setActiveJob(job);
            persistJobSnapshot(job);
            break;
          } catch (err) {
            if (err instanceof ApiError && ![408, 429, 502, 503, 504].includes(err.status)) throw err;
            if (err instanceof DOMException && err.name === "AbortError") throw err;
            setProgressMessage("Reconnecting to your saved request. It will not create a duplicate render.");
            await new Promise((resolve) => setTimeout(resolve, 3000));
          }
        }
      }
      const response = await waitForJobResult<T>(job.jobId!, (status) => {
        setProgressMessage(`${status.message} ${status.progress}%`);
        if (status.assets?.length) setRecoveredAssets(status.assets);
      }, 1500, controller.signal, setProgressMessage);
      if (account !== accountVersion.current) throw new DOMException("Account changed", "AbortError");
      return response;
    } catch (err) {
      if (err instanceof DOMException && err.name === "AbortError") throw err;
      const message = errorMessage(err, "Generation failed.");
      setJobError(message);
      if (err instanceof GenerationJobError) setRecoveredAssets(err.job.assets || []);
      persistJobSnapshot(null, { jobError: message, recoveredAssets: err instanceof GenerationJobError ? err.job.assets || [] : recoveredAssets });
      throw err;
    } finally {
      if (account === accountVersion.current) {
        setActiveJob(null);
        jobController.current = null;
      }
    }
  }

  function newJob(kind: PendingJob["kind"], payload: PendingJob["payload"], sceneId?: number): PendingJob {
    const requestId = crypto.randomUUID();
    return { kind, requestId, sceneId, renderSignature: sceneId == null ? undefined : sceneRenderSignature(sceneId), aspectRatio, payload: { ...payload, request_id: requestId, chat_id: activeChatId || undefined } };
  }

  async function generateThroughJob(payload: VideoGenerationRequest, sceneId?: number) {
    return executeTrackedJob<VideoGenerationResponse>(newJob("video", payload, sceneId));
  }

  function finalFromSceneResponse(response: VideoGenerationResponse): FinalVideo {
    return {
      url: absoluteApiUrl(response.video_url),
      downloadUrl: absoluteApiUrl(response.download_url),
      filename: response.filename,
      qualityNote: [response.quality_note, ...(response.warnings || [])].filter(Boolean).join(" "),
      label: `${response.provider} · ${response.chunk_count} LTX chunk${response.chunk_count === 1 ? "" : "s"} · ${response.render_seconds.toFixed(1)}s render${response.detail_refined ? " · Real Skin refined" : ""}`,
      hasAudio: response.media_info.has_audio,
      audioCodec: response.media_info.audio_codec,
      dimensions: `${response.media_info.width ?? "?"}×${response.media_info.height ?? "?"}`,
      estimatedCostUsd: response.estimated_cost_usd,
      gpu: response.gpu,
      qcReport: {
        visual: resolveQCStatus(response.visual_qc_status, response.continuity_qc_passed, true, response.continuity_warnings),
        audio: resolveQCStatus(response.audio_qc_status, response.audio_qc_passed, false, response.audio_warnings || []),
        visualWarnings: [...(response.continuity_warnings || []), ...qcReasons(response.visual_qc)],
        audioWarnings: [...(response.audio_warnings || []), ...qcReasons(response.audio_qc)],
        visualRetries: response.continuity_regenerations || 0,
        audioRetakes: response.audio_retake_count || 0,
      },
    };
  }

  function factoryVideoFromResponse(response: FactoryGenerationResponse): NonNullable<FinalVideo> {
    return {
        url: absoluteApiUrl(response.final_video_url),
        downloadUrl: absoluteApiUrl(response.final_download_url),
        filename: response.final_filename,
        qualityNote: [response.quality_note, ...(response.warnings || [])].filter(Boolean).join(" "),
        label: `${response.scene_count} scenes · ${response.chunk_count} LTX chunks · ${response.provider} · ${response.total_render_seconds.toFixed(1)}s GPU render${response.detail_refined ? " · Real Skin refined" : ""}`,
        hasAudio: response.has_audio,
        audioCodec: response.has_audio ? "AAC / generated audio" : null,
        dimensions: `${response.width ?? "?"}×${response.height ?? "?"}`,
        estimatedCostUsd: response.estimated_cost_usd,
        gpu: response.gpu,
        youtubeUrl: response.youtube_url,
        youtubePrivacy: response.youtube_privacy,
        qcReport: factoryQCReport(response),
        factoryResult: response,
    };
  }

  function audioRetakeVideo(response: AudioRetakeResponse): NonNullable<FinalVideo> {
    return { filename: response.filename, url: absoluteApiUrl(response.video_url), downloadUrl: absoluteApiUrl(response.download_url), label: "Audio correction · original picture preserved", qualityNote: "The original take remains in Saved takes. Review the corrected dialogue before sharing.", hasAudio: true, audioCodec: response.media_info.audio_codec, dimensions: `${response.media_info.width}×${response.media_info.height}`, qcReport: { visual: response.visual_qc_status || "not_checked", audio: resolveQCStatus(response.audio_qc_status, response.audio_qc_passed, true, response.audio_warnings), visualWarnings: qcReasons(response.visual_qc), audioWarnings: response.audio_warnings || [], visualRetries: 0, audioRetakes: response.audio_retake_count || 1 } };
  }

  async function handleAudioRepair() {
    if (!finalVideo || !spokenScript.trim() || isBusy) return;
    setError("");
    try {
      const response = await executeTrackedJob<AudioRetakeResponse>(newJob("audio_retake", { filename: finalVideo.filename, spoken_script: spokenScript.trim(), audio_direction: audioDirection, seed }));
      rememberVideo(audioRetakeVideo(response), finalVideo.aspectRatio || aspectRatio);
      await refreshBilling();
    } catch (err) { setError(errorMessage(err, "Audio correction failed. Your original video is still available.")); }
    finally { setProgressMessage(""); }
  }

  async function handleFactory(cleanPrompt: string) {
    if (publishToYouTube && (!youtube?.enabled || !youtube.connected)) {
      throw new Error("Connect a YouTube channel before enabling automatic publishing.");
    }
    setFactoryGenerating(true);
    setProgressMessage(enhancePrompt ? "Starting AI-enhanced video factory..." : "Starting direct story sequencing (no Gemini rewrite)...");
    try {
      const response = await executeTrackedJob<FactoryGenerationResponse>(newJob("factory", {
        prompt: cleanPrompt,
        spoken_script: spokenScript,
        target_duration_seconds: factoryTargetSeconds,
        scene_duration_seconds: factorySceneSeconds,
        aspect_ratio: aspectRatio,
        quality,
        realism_profile: realismProfile,
        audio_mode: audioMode,
        audio_direction: audioDirection.trim() || null,
        provider,
        model,
        decoder,
        seed,
        continuity_mode: continuityMode,
        continuity_strength: referencedElements.some((element) => element.type === "character")
          ? (realismProfile === "identity_max" ? 1.0 : 0.95)
          : 0.9,
        continuity_qc_mode: quality !== "preview" ? "strict" : "auto",
        continuity_max_retries: realismProfile === "identity_max" ? 2 : 1,
        enhance_prompt: enhancePrompt,
        element_bindings: elementBindings,
        publish_to_youtube: publishToYouTube,
        youtube_title: youtubeTitle.trim() || null,
        youtube_description: youtubeDescription,
        youtube_privacy: youtubePrivacy,
        youtube_tags: [],
        youtube_category_id: "22",
        youtube_publish_at: null,
      }));
      setFactoryResult(response);
      rememberVideo(factoryVideoFromResponse(response));
      await refreshBilling();
    } finally {
      setFactoryGenerating(false);
      setProgressMessage("");
    }
  }

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    const cleanPrompt = prompt.trim();
    if (!cleanPrompt) {
      setError("Describe the video you want to create.");
      return;
    }
    const unknownHandles = [...cleanPrompt.matchAll(/(?<![A-Za-z0-9_])@([A-Za-z][A-Za-z0-9_-]{0,31})/g)]
      .map((match) => match[1]).filter((handle) => !elements.some((element) => element.handle.toLowerCase() === handle.toLowerCase()));
    if (unknownHandles.length) {
      setError(`Create or select saved Elements for ${[...new Set(unknownHandles)].map((handle) => `@${handle}`).join(", ")} before generating.`);
      return;
    }
    if (referencedElements.length > activeElementLimit) {
      setError(`This LTX profile allows ${activeElementLimit} active Elements in one scene. Remove ${referencedElements.length - activeElementLimit} reference${referencedElements.length - activeElementLimit === 1 ? "" : "s"} before generating.`);
      return;
    }
    if (realismProfile === "identity_max" && !referencedElements.some((element) => element.type === "character" && (elementModes[element.id] || "identity") === "identity")) {
      setError("Identity Max needs an active Character Element in Identity mode. Add @your-character or enable it across Factory scenes, then use a sharp real photo as the primary reference.");
      return;
    }
    setError("");
    setNotice("");
    resetOutput();
    const renderPrompt = promptWithDirectorControls(cleanPrompt);

    try {
      if (mode === "factory") {
        await handleFactory(renderPrompt);
        return;
      }
      if (mode === "direct") {
        setDirectGenerating(true);
        setProgressMessage("Rendering your prompt with LTX 2.5...");
        const response = await generateThroughJob({
          prompt: renderPrompt,
          spoken_script: spokenScript,
          aspect_ratio: aspectRatio,
          duration_seconds: durationSeconds,
          seed,
          decoder,
          enhance_prompt: enhancePrompt,
          quality,
          realism_profile: realismProfile,
          audio_mode: audioMode,
          audio_direction: audioDirection.trim() || null,
          provider,
          model,
          continuity_mode: "off",
          // Run visual QC without injecting continuity text into the native
          // LTX speech prompt. A failed QC must never discard the render.
          continuity_qc_mode: quality === "preview" ? "auto" : "strict",
          continuity_max_retries: 0, // Display the verdict without charging for a surprise QC-triggered GPU rerender.
          element_bindings: elementBindings,
        });
        rememberVideo(finalFromSceneResponse(response)!);
        await refreshBilling();
        return;
      }

      setPlanning(true);
      setProgressMessage("Creating continuity-locked storyboard shots with audio direction...");
      const response = await generateScenePlan({ prompt: renderPrompt, spoken_script: spokenScript, aspect_ratio: aspectRatio, scene_count: sceneCount });
      setResult(response);
      setScenePrompts(response.scenes.reduce((current, scene) => {
        current[scene.id] = scene.prompt;
        return current;
      }, {} as ScenePromptMap));
    } catch (err) {
      setError(errorMessage(err, "Generation failed."));
    } finally {
      setPlanning(false);
      setDirectGenerating(false);
      setProgressMessage("");
    }
  }

  function sceneRenderSignature(sceneId: number): string {
    const scene = result?.scenes.find((item) => item.id === sceneId);
    return JSON.stringify({ prompt: promptWithDirectorControls(scenePrompts[sceneId] || ""), spokenScript: scene?.spoken_script || "", aspectRatio: result?.aspect_ratio, durationSeconds, seed, decoder, quality, realismProfile, audioDirection, continuityMode, elementBindings });
  }

  async function renderScene(sceneId: number, referenceFrameFilename: string | null = null): Promise<RenderedSceneVideo> {
    if (!result) throw new Error("Storyboard is not available.");
    const scenePrompt = scenePrompts[sceneId]?.trim();
    if (!scenePrompt) throw new Error("This scene needs a prompt before rendering.");
    const sceneIndex = result.scenes.findIndex((scene) => scene.id === sceneId);
    const response = await generateThroughJob({
      prompt: promptWithDirectorControls(scenePrompt),
      spoken_script: result.scenes[sceneIndex]?.spoken_script || (result.scenes.length === 1 ? spokenScript : ""),
      aspect_ratio: result.aspect_ratio,
      duration_seconds: durationSeconds,
      seed,
      decoder,
      enhance_prompt: false,
      quality,
      realism_profile: realismProfile,
      audio_mode: "native",
      audio_direction: audioDirection.trim() || null,
      provider,
      model,
      continuity_mode: continuityMode,
      continuity_id: result.continuity_id,
      scene_index: Math.max(sceneIndex, 0),
      scene_count: result.scenes.length,
      character_bible: result.character_bible,
      style_bible: result.style_bible,
      entity_locks: result.entity_locks,
      visible_entity_counts: result.scenes[Math.max(sceneIndex, 0)]?.visible_entity_counts || {},
      continuity_qc_mode: quality === "preview" ? "auto" : "strict",
      continuity_max_retries: realismProfile === "identity_max" ? 2 : 1,
      reference_frame_filename: continuityMode === "strict" ? referenceFrameFilename : null,
      continuity_strength: 1.0,
      element_bindings: elementBindings,
    }, sceneId);
    return { ...sceneVideoFromResponse(response), renderSignature: sceneRenderSignature(sceneId) };
  }

  function sceneVideoFromResponse(response: VideoGenerationResponse): RenderedSceneVideo {
    return {
      url: absoluteApiUrl(response.video_url),
      downloadUrl: absoluteApiUrl(response.download_url),
      filename: response.filename,
      details: response.render_details,
      renderSeconds: response.render_seconds,
      wallSeconds: response.wall_seconds,
      provider: response.provider,
      gpu: response.gpu,
      mediaInfo: response.media_info,
      estimatedCostUsd: response.estimated_cost_usd,
      qualityNote: [response.quality_note, ...(response.warnings || [])].filter(Boolean).join(" "),
      realismProfile: response.realism_profile,
      detailRefined: response.detail_refined,
      audioMode: response.audio_mode,
      chunkCount: response.chunk_count,
      continuityMode: response.continuity_mode,
      continuityApplied: response.continuity_applied,
      referenceFrameFilename: response.reference_frame_filename,
      continuityFrameUrl: response.continuity_frame_url ? absoluteApiUrl(response.continuity_frame_url) : null,
      continuityFrameFilename: response.continuity_frame_filename,
      continuityQcPassed: response.continuity_qc_passed,
      visualQcStatus: response.visual_qc_status,
      audioQcStatus: response.audio_qc_status || "not_checked",
      audioWarnings: response.audio_warnings || [],
      continuityRegenerations: response.continuity_regenerations,
      continuityWarnings: response.continuity_warnings,
      elementsUsed: response.elements_used,
      elementReferenceMode: response.element_reference_mode,
    };
  }

  async function ensureScenesThrough(targetSceneId?: number, regenerate = false): Promise<RenderedVideoMap> {
    if (!result) throw new Error("Storyboard is not available.");
    const next: RenderedVideoMap = regenerate && targetSceneId != null ? invalidateSceneChain(renderedVideos, result.scenes.map((scene) => scene.id), targetSceneId) : { ...renderedVideos };
    if (regenerate) setRenderedVideos({ ...next });
    let previousFrame: string | null = null;
    let staleChain = false;
    for (const scene of result.scenes) {
      if (next[scene.id]?.renderSignature !== sceneRenderSignature(scene.id)) staleChain = true;
      if (staleChain) delete next[scene.id];
      if (next[scene.id]) {
        previousFrame = next[scene.id].continuityFrameFilename;
      } else {
        setGeneratingScene(scene.id);
        setProgressMessage(`Rendering scene ${scene.id}/${result.scenes.length}...`);
        const rendered = await renderScene(scene.id, previousFrame);
        next[scene.id] = rendered;
        const take: NonNullable<FinalVideo> = { url: rendered.url, downloadUrl: rendered.downloadUrl, filename: rendered.filename, qualityNote: rendered.qualityNote, label: `Scene ${scene.id} · ${rendered.details}`, hasAudio: rendered.mediaInfo.has_audio, audioCodec: rendered.mediaInfo.audio_codec, dimensions: `${rendered.mediaInfo.width}×${rendered.mediaInfo.height}`, aspectRatio: result.aspect_ratio, createdAt: Date.now(), qcReport: { visual: resolveQCStatus(rendered.visualQcStatus, rendered.continuityQcPassed, true, rendered.continuityWarnings), audio: rendered.audioQcStatus || "not_checked", visualWarnings: rendered.continuityWarnings || [], audioWarnings: rendered.audioWarnings || [], visualRetries: rendered.continuityRegenerations, audioRetakes: 0 } };
        setVideoTakes((current) => [take, ...current.filter((item) => item.filename !== take.filename)]);
        setRenderedVideos({ ...next });
        previousFrame = rendered.continuityFrameFilename;
      }
      if (targetSceneId && scene.id === targetSceneId) break;
    }
    return next;
  }

  async function handleRenderScene(sceneId: number) {
    setError("");
    try {
      await ensureScenesThrough(sceneId, Boolean(renderedVideos[sceneId]));
      await refreshBilling();
    } catch (err) {
      setError(errorMessage(err, "Scene render failed."));
    } finally {
      setGeneratingScene(null);
      setProgressMessage("");
    }
  }

  async function handleRenderMissingAndCombine() {
    if (!result) return;
    setError("");
    setCreatingFinal(true);
    try {
      const completed = await ensureScenesThrough();
      setGeneratingScene(null);
      setProgressMessage("Composing scenes, delivery quality and final audio master...");
      const combined = await executeTrackedJob<CombineScenesResponse>(newJob("combine", {
        scene_video_urls: result.scenes.map((scene) => completed[scene.id].url),
        aspect_ratio: result.aspect_ratio,
        quality,
        audio_mode: audioMode,
      }));
      rememberVideo({
        url: absoluteApiUrl(combined.final_video_url),
        downloadUrl: absoluteApiUrl(combined.final_download_url),
        filename: combined.final_filename,
        qualityNote: combined.quality_note,
        label: `${combined.scene_count} scenes · ${qualityLabel(combined.quality)} · ${combined.audio_mode} audio`,
        hasAudio: combined.media_info.has_audio,
        audioCodec: combined.media_info.audio_codec,
        dimensions: `${combined.media_info.width ?? "?"}×${combined.media_info.height ?? "?"}`,
        qcReport: {
          visual: combined.visual_qc_status || aggregateQCStatus(result.scenes.map((scene) => {
            const rendered = completed[scene.id];
            return resolveQCStatus(rendered.visualQcStatus, rendered.continuityQcPassed, rendered.continuityMode !== "off", rendered.continuityWarnings);
          })),
          audio: combined.audio_qc_status || aggregateQCStatus(Object.values(completed).map((video) => video.audioQcStatus || "not_checked")),
          visualWarnings: result.scenes.flatMap((scene) =>
            (completed[scene.id].continuityWarnings || []).map((warning) => `Scene ${scene.id}: ${warning}`)
          ),
          audioWarnings: result.scenes.flatMap((scene) => (completed[scene.id].audioWarnings || []).map((warning) => `Scene ${scene.id}: ${warning}`)),
          visualRetries: result.scenes.reduce((total, scene) => total + (completed[scene.id].continuityRegenerations || 0), 0),
          audioRetakes: 0,
        },
      });
      await refreshBilling();
    } catch (err) {
      setError(errorMessage(err, "Unable to create final video."));
    } finally {
      setGeneratingScene(null);
      setCreatingFinal(false);
      setProgressMessage("");
    }
  }

  function updateScenePrompt(sceneId: number, value: string) {
    setScenePrompts((current) => ({ ...current, [sceneId]: value }));
    setRenderedVideos((current) => invalidateSceneChain(current, result?.scenes.map((scene) => scene.id) || [], sceneId));
  }

  async function handleCheckout(packId: string) {
    setIntegrationBusy(true);
    setError("");
    try {
      const checkout = await createCheckout(packId);
      window.location.assign(checkout.checkout_url);
    } catch (err) {
      setError(errorMessage(err, "Unable to open checkout."));
      setIntegrationBusy(false);
    }
  }

  async function handleBillingPortal() {
    setIntegrationBusy(true);
    setError("");
    try {
      const portal = await createBillingPortal();
      window.location.assign(portal.portal_url);
    } catch (err) {
      setError(errorMessage(err, "Unable to open billing portal."));
      setIntegrationBusy(false);
    }
  }

  async function handleRequestLoginOtp(event: FormEvent) {
    event.preventDefault();
    setLoginBusy(true);
    setLoginError("");
    try {
      const response = await requestLoginOtp(loginEmail);
      setDemoOtp(response.demo_otp);
      setLoginOtp(response.demo_otp || "");
      setLoginStep("otp");
    } catch (err) {
      setLoginError(errorMessage(err, "Unable to create login code."));
    } finally {
      setLoginBusy(false);
    }
  }

  async function handleVerifyLoginOtp(event: FormEvent) {
    event.preventDefault();
    setLoginBusy(true);
    setLoginError("");
    try {
      const response = await verifyLoginOtp(loginEmail, loginOtp);
      if (!response.authenticated || !response.user) throw new Error("Login did not return an account session.");
      setAuthUser(response.user);
      setDemoOtp(null);
      setLoginOtp("");
      setLoginStep("email");
    } catch (err) {
      setLoginError(errorMessage(err, "Unable to sign in."));
    } finally {
      setLoginBusy(false);
    }
  }

  async function handleLogout() {
    if (isBusy || elementBusy || integrationBusy) {
      setNotice("Finish the current operation before signing out.");
      return;
    }
    try {
      await logoutCinema();
    } catch (err) {
      setError(errorMessage(err, "Unable to sign out. Please retry when the connection returns."));
      return;
    }
    accountVersion.current += 1;
    jobController.current?.abort();
    jobController.current = null;
    setAuthUser(null);
    setElements([]);
    setServerJobs([]);
    setBillingMe(null);
    setBillingCatalog(null);
    setYoutube(null);
    setCapabilities(null);
    setElementFiles([]);
    setElementFileRoles({});
    setElementUploadError("");
    elementSaveRequest.current = { signature: "", id: "" };
    elementAddRequests.current.clear();
    setElementName("");
    setElementHandle("");
    setElementDescription("");
    setShowElementCreator(false);
    setElementSearch("");
    setPublishToYouTube(false);
    setYoutubeTitle("");
    setYoutubeDescription("");
    setHistorySaveState("saved");
    sessionsRef.current = [];
    setChatSessions([]);
    setActiveChatId(null);
    setChatHistoryReady(false);
    applyChatWorkspace(createEmptyChatWorkspace());
  }

  async function handleYouTubeConnection() {
    setIntegrationBusy(true);
    setError("");
    try {
      if (youtube?.connected) {
        await disconnectYouTube();
        setYoutube(await getYouTubeStatus());
        setPublishToYouTube(false);
        setNotice("YouTube channel disconnected.");
      } else {
        const connect = await connectYouTube();
        window.location.assign(connect.authorization_url);
      }
    } catch (err) {
      setError(errorMessage(err, "Unable to update YouTube connection."));
    } finally {
      setIntegrationBusy(false);
    }
  }

  if (authChecking) {
    return (
      <main className="cinema-login-page bg-[var(--page-bg)] text-[var(--text)]">
        <div className="cinema-login-card cinema-login-card-loading"><div className="studio-logo-mark">T</div><Spinner /><span>Opening Cinema Studio…</span></div>
      </main>
    );
  }

  if (!authUser) {
    return (
      <main className="cinema-login-page bg-[var(--page-bg)] text-[var(--text)]">
        <section className="cinema-login-card">
          <div className="cinema-login-brand"><div className="studio-logo-mark">T</div><div><strong>Triven Cinema</strong><span>AI filmmaking studio</span></div></div>
          <div className="cinema-login-copy"><span className="cinema-login-kicker">Cinema Studio</span><h1>{loginStep === "email" ? "Sign in to your studio" : "Enter your login code"}</h1><p>{loginStep === "email" ? "Your chats, Elements and generation workspace will stay connected to this email." : `Code created for ${loginEmail}.`}</p></div>
          {loginStep === "email" ? (
            <form onSubmit={handleRequestLoginOtp} className="cinema-login-form">
              <label>Email address<input type="email" required autoComplete="email" value={loginEmail} onChange={(event) => setLoginEmail(event.target.value)} placeholder="you@example.com" /></label>
              <button type="submit" disabled={loginBusy || !loginEmail.trim()}>{loginBusy ? <Spinner /> : null}<span>Continue</span></button>
            </form>
          ) : (
            <form onSubmit={handleVerifyLoginOtp} className="cinema-login-form">
              {demoOtp ? <div className="cinema-demo-otp"><span>Demo OTP</span><strong>{demoOtp}</strong><small>This code is shown only because demo OTP mode is enabled.</small></div> : null}
              <label>One-time code<input inputMode="numeric" autoComplete="one-time-code" required value={loginOtp} onChange={(event) => setLoginOtp(event.target.value.replace(/\D/g, "").slice(0, 8))} placeholder="6-digit code" /></label>
              <button type="submit" disabled={loginBusy || loginOtp.length < 4}>{loginBusy ? <Spinner /> : null}<span>Enter Studio</span></button>
              <button type="button" className="cinema-login-back" onClick={() => { setLoginStep("email"); setLoginOtp(""); setDemoOtp(null); setLoginError(""); }}>Use another email</button>
            </form>
          )}
          {loginError ? <div className="cinema-login-error">{loginError}</div> : null}
          <div className="cinema-login-footnote">Your studio history stays connected to your account.</div>
        </section>
      </main>
    );
  }

  const finalActionLabel = creatingFinal
    ? "Creating final master"
    : allScenesRendered
      ? `Combine · ${qualityLabel(quality)}`
      : `Render all + ${qualityLabel(quality)}`;

  return (
    <main className="cinema-studio-page bg-[var(--page-bg)] text-[var(--text)]">
      <header className="studio-topbar">
        <div className="studio-topbar-left">
          <div className="studio-logo-mark">T</div>
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <span className="truncate text-sm font-semibold text-[var(--text-strong)]">{activeChatTitle}</span>
              <span className={`h-1.5 w-1.5 rounded-full ${historySaveState === "saved" ? "bg-emerald-500" : "bg-amber-500"}`} title={historySaveState === "saved" ? "Saved to your account" : historySaveState === "saving" ? "Saving…" : "Saved on this device; waiting to sync"} />
            </div>
            <div className="text-[10px] text-[var(--text-muted)]">Triven Cinema Studio</div>
          </div>
        </div>

        <div className="studio-topbar-center">
          <span className="studio-model-pill">Cinema Studio · LTX 2.5</span>
          <span className="studio-model-pill studio-model-pill-muted">Modal {capabilities?.gpu || "B200"}</span>
        </div>

        <div className="studio-topbar-right">
          <div className="theme-switch" role="group" aria-label="Color theme">
            <button type="button" className={`theme-switch-option ${theme === "light" ? "theme-switch-option-active" : ""}`} onClick={() => selectTheme("light")} aria-pressed={theme === "light"}>
              <SunIcon /><span>Light</span>
            </button>
            <button type="button" className={`theme-switch-option ${theme === "dark" ? "theme-switch-option-active" : ""}`} onClick={() => selectTheme("dark")} aria-pressed={theme === "dark"}>
              <MoonIcon /><span>Dark</span>
            </button>
          </div>
          {billingCatalog?.enabled && billingMe ? <span className="studio-status-pill">{formatCredits(billingMe.balance_seconds)} credits</span> : null}
          {youtube?.connected ? <span className="studio-status-pill">YouTube connected</span> : null}
          <div className="studio-account-pill"><span>{authUser.email}</span><button type="button" onClick={handleLogout} disabled={isBusy || elementBusy || integrationBusy}>Sign out</button></div>
        </div>
      </header>

      <div className="studio-workspace">
        <aside className="studio-sidebar" aria-label="Cinema Studio navigation">
          <div className="studio-sidebar-actions">
            <button type="button" className="studio-sidebar-action" onClick={handleNewChat} disabled={isBusy} title="Start a new chat">
              <NewChatIcon /><span>New Chat</span>
            </button>
            <button type="button" className={`studio-sidebar-action ${showElementsLibrary ? "studio-sidebar-action-active" : ""}`} onClick={() => setShowElementsLibrary(true)} title="My Elements">
              <ElementsIcon /><span>Elements</span>
            </button>
            <button type="button" className="studio-sidebar-action" onClick={() => document.getElementById("studio-output")?.scrollIntoView({ behavior: "smooth", block: "start" })} title="Generation tasks and outputs">
              <FilmIcon /><span>Tasks</span>
            </button>
          </div>

          <div className="studio-sidebar-divider" />

          <div className="studio-chat-history">
            <div className="studio-chat-history-title">Previous chats</div>
            <div className="studio-chat-history-list" role="list" aria-label="Previous Cinema chats">
              {chatHistoryReady && sortedChatSessions.length === 0 ? <div className="studio-chat-history-empty">No previous chats yet.</div> : null}
              {sortedChatSessions.map((session) => {
                const active = session.id === activeChatId;
                return (
                  <div key={session.id} className={`studio-chat-history-row ${active ? "studio-chat-history-row-active" : ""}`} role="listitem">
                    <button type="button" className="studio-chat-open" onClick={() => handleOpenChat(session)} disabled={isBusy && !active} aria-current={active ? "page" : undefined} title={session.title}>
                      <span className="studio-chat-history-icon"><ChatIcon /></span>
                      <span className="studio-chat-history-name">{session.title}</span>
                      {session.workspace.finalVideo?.qcReport && (
                        session.workspace.finalVideo.qcReport.visual === "failed" || session.workspace.finalVideo.qcReport.audio === "failed"
                      ) && <span title="This saved video failed quality control" className="shrink-0 text-[10px] font-semibold text-red-700 dark:text-red-300">QC failed</span>}
                    </button>
                    {!active ? (
                      <button type="button" className="studio-chat-delete" onClick={() => handleDeleteChat(session.id)} aria-label={`Delete ${session.title}`} title="Delete chat">
                        <TrashIcon />
                      </button>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </div>
        </aside>

        <section className="studio-main-column">
          <div className="studio-stage-shell">
            <div className="studio-stage-header">
              <div className="flex items-center gap-2">
                <div className="text-xs font-semibold text-[var(--text-strong)]">Scene</div>
                {finalVideo?.qcReport && <><span className="text-[10px] text-[var(--text-muted)]">Visual QC</span><QCStatusPill status={finalVideo.qcReport.visual} /></>}
                {finalVideo?.qcReport?.audio === "failed" && <><span className="text-[10px] text-[var(--text-muted)]">Audio QC</span><QCStatusPill status="failed" /></>}
                {referencedElements.length > 0 && <span className="studio-mini-pill">{referencedElements.length} Element{referencedElements.length === 1 ? "" : "s"}</span>}
              </div>
              <div className="flex items-center gap-2">
                <span className="studio-mini-pill">{aspectRatio}</span>
                <span className="studio-mini-pill">{quality === "preview" ? "Preview" : quality.toUpperCase()}</span>
                <span className="studio-mini-pill">{mode === "factory" ? `${factorySceneSeconds}s` : `${durationSeconds}s`}</span>
              </div>
            </div>

            <div className="studio-stage-area">
              <div className={`studio-canvas ${(finalVideo?.aspectRatio || aspectRatio) === "9:16" ? "studio-canvas-portrait" : (finalVideo?.aspectRatio || aspectRatio) === "1:1" ? "studio-canvas-square" : "studio-canvas-landscape"}`}>
                {studioVideoUrl ? (
                  <StudioVideo src={studioVideoUrl} controls playsInline preload="metadata" className="absolute inset-0 h-full w-full object-contain" />
                ) : studioPreviewAsset ? (
                  <Image unoptimized width={640} height={640} src={absoluteApiUrl(studioPreviewAsset.asset_url)} alt={studioPreviewElement?.name || "Element preview"} decoding="async" className="absolute inset-0 h-full w-full object-contain" />
                ) : (
                  <div className="studio-empty-stage">
                    <div className="studio-empty-orbit"><span>+</span></div>
                    <div className="mt-4 text-sm font-medium text-[var(--text-secondary)]">Build your cast, then describe the scene</div>
                    <div className="mt-1 max-w-sm text-center text-xs leading-5 text-[var(--text-muted)]">Upload a character or reference image, save it as an Element, then type <strong>@</strong> in the prompt to direct it.</div>
                    <button type="button" onClick={() => { setElementType("character"); setShowElementCreator(true); }} className="mt-4 studio-secondary-button">+ Create New Character</button>
                  </div>
                )}
              </div>
            </div>

          </div>

          <form onSubmit={handleSubmit} className="studio-bottom-deck"><fieldset disabled={isBusy} className="min-w-0 contents">
            <section className="studio-elements-strip">
              <div className="studio-section-title-row">
                <div>
                  <div className="studio-section-title">Elements</div>
                  <div className="studio-section-subtitle">Cast and references for this shot. Type @ in the prompt to reuse any saved Element.</div>
                </div>
                <div className="flex items-center gap-2">
                  <span className="studio-count-pill">{referencedElements.length}/{capabilities?.elements?.max_active_per_scene ?? 6}</span>
                  <button type="button" onClick={() => { setElementType("character"); setShowElementCreator(true); }} className="studio-small-action">+ Character</button>
                  <button type="button" onClick={() => setShowReferencePicker(true)} className="studio-small-action">+ Add</button>
                </div>
              </div>
              <div className="studio-element-strip-row">
                {referencedElements.map((element) => {
                  const asset = primaryElementAsset(element);
                  return (
                    <button key={element.id} type="button" onClick={() => { setSelectedElementId(element.id); setDirectorTab("elements"); }} className="studio-element-compact studio-element-compact-active">
                      <span className="studio-element-compact-thumb">{asset ? <Image unoptimized width={640} height={640} src={absoluteApiUrl(asset.asset_url)} alt={element.name} loading="lazy" decoding="async" /> : <span>{element.name.slice(0, 1)}</span>}</span>
                      <span className="min-w-0 text-left"><span className="block max-w-[92px] truncate text-[10px] font-semibold text-[var(--text)]">{element.name}</span><span className={`block text-[9px] ${elementTone(element.type)}`}>@{element.handle}</span></span>
                    </button>
                  );
                })}
                {quickCharacterElements.map((element) => {
                  const asset = primaryElementAsset(element);
                  return (
                    <button key={element.id} type="button" onClick={() => insertElementMention(element)} className="studio-element-compact">
                      <span className="studio-element-compact-thumb">{asset ? <Image unoptimized width={640} height={640} src={absoluteApiUrl(asset.asset_url)} alt={element.name} loading="lazy" decoding="async" /> : <span>{element.name.slice(0, 1)}</span>}</span>
                      <span className="min-w-0 text-left"><span className="block max-w-[92px] truncate text-[10px] font-semibold text-[var(--text-secondary)]">{element.name}</span><span className="block text-[9px] text-[var(--text-muted)]">saved character</span></span>
                    </button>
                  );
                })}
                {referencedElements.length === 0 && quickCharacterElements.length === 0 && (
                  <button type="button" onClick={() => setShowReferencePicker(true)} className="studio-element-empty-cta"><PlusIcon /><span>Add a character, prop or location</span></button>
                )}
              </div>
            </section>

            <section className="studio-prompt-panel">
              <div className="studio-prompt-heading">
                <div className="text-xs font-semibold text-[var(--text)]">Describe the shot</div>
                <span className="text-[10px] text-[var(--text-muted)]">@ mentions stay locked to saved Elements</span>
              </div>

              <div className="studio-prompt-editor-wrap">
                <textarea
                  value={prompt}
                  onChange={(event) => updateMentionState(event.target.value)}
                  onBlur={() => window.setTimeout(() => setMentionQuery(null), 120)}
                  rows={5}
                  placeholder="@Character walks through @Location holding @Prop. Describe motion, framing, dialogue and sound..."
                  className="studio-prompt-input"
                />
                {mentionQuery != null && mentionSuggestions.length > 0 && (
                  <div className="studio-mention-menu">
                    <div className="studio-mention-menu-title">Elements</div>
                    {mentionSuggestions.map((element) => {
                      const asset = primaryElementAsset(element);
                      return (
                        <button key={element.id} type="button" onMouseDown={(event) => { event.preventDefault(); chooseMention(element); }} className="studio-mention-option">
                          {asset ? <Image unoptimized width={640} height={640} src={absoluteApiUrl(asset.asset_url)} alt="" /> : <span className="studio-mention-empty">{element.name.slice(0, 1)}</span>}
                          <span className="min-w-0 flex-1">
                            <span className="block truncate text-xs font-semibold text-[var(--text)]">@{element.handle}</span>
                            <span className="block truncate text-[10px] text-[var(--text-muted)]">{element.name} · {element.type}</span>
                          </span>
                        </button>
                      );
                    })}
                  </div>
                )}
              </div>

              {referencedElements.length > 0 && (
                <div className="studio-prompt-references">
                  {referencedElements.map((element) => {
                    const asset = primaryElementAsset(element);
                    return (
                      <span key={element.id} className={`element-mention-chip ${elementTone(element.type)}`}>
                        {asset ? <Image unoptimized width={640} height={640} src={absoluteApiUrl(asset.asset_url)} alt="" className="h-5 w-5 rounded-full object-cover" /> : null}
                        @{element.handle}
                        <span className="element-hover-card">
                          {asset ? <Image unoptimized width={640} height={640} src={absoluteApiUrl(asset.asset_url)} alt={element.name} className="h-28 w-full rounded-lg object-cover" /> : null}
                          <span className="mt-2 block text-xs font-semibold text-[var(--text)]">{element.name}</span>
                          <span className="mt-0.5 block text-[10px] uppercase text-[var(--text-muted)]">{element.type} · v{element.current_version} · {element.assets.length} refs</span>
                        </span>
                      </span>
                    );
                  })}
                </div>
              )}

              <div className="studio-prompt-footer">
                <div className="studio-prompt-tools">
                  <select className="studio-toolbar-select" value={aspectRatio} onChange={(e) => { setAspectRatio(e.target.value as AspectRatio); invalidateRenderedMedia(); }}>
                    <option value="16:9">16:9</option><option value="9:16">9:16</option><option value="1:1">1:1</option>
                  </select>
                  <select className="studio-toolbar-select" value={quality} onChange={(e) => handleQualityChange(e.target.value as RenderQuality)}>
                    <option value="preview">Preview</option><option value="1080p">1080p</option><option value="4k">4K</option>
                  </select>
                  {mode === "factory" ? (
                    <select className="studio-toolbar-select" value={factorySceneSeconds} onChange={(e) => handleFactorySceneDurationChange(Number(e.target.value))}>
                      {factoryDurationOptions.map((value) => <option key={value} value={value}>{value}s scene</option>)}
                    </select>
                  ) : (
                    <select className="studio-toolbar-select" value={durationSeconds} onChange={(e) => { setDurationSeconds(Number(e.target.value)); invalidateRenderedMedia(); }}>
                      {durationOptions.map((value) => <option key={value} value={value}>{value}s</option>)}
                    </select>
                  )}
                  <label className="studio-ai-toggle"><input type="checkbox" checked={enhancePrompt} onChange={(e) => setEnhancePrompt(e.target.checked)} /><span>AI Director</span></label>
                </div>

                <button type="submit" disabled={isBusy || !chatHistoryReady} className="studio-generate-button">
                  {isBusy ? <Spinner /> : <PlayIcon />}
                  {factoryGenerating ? "Generating film" : directGenerating ? "Generating" : planning ? "Planning" : mode === "factory" ? "Generate" : mode === "direct" ? "Generate clip" : "Create storyboard"}
                </button>
              </div>
            </section></fieldset>
          </form>

          {progressMessage && <div className="studio-inline-notice studio-inline-info">{progressMessage}</div>}
          {error && <div className="studio-inline-notice studio-inline-error">{error}</div>}
          {notice && <div className="studio-inline-notice studio-inline-success">{notice}</div>}
        </section>

        <aside className="studio-inspector">
          <fieldset disabled={isBusy || elementBusy} className="studio-inspector-scroll min-w-0">
            <section className="studio-inspector-section">
              <div className="studio-inspector-title-row">
                <div>
                  <div className="studio-inspector-eyebrow">Cinema Studio</div>
                  <div className="studio-inspector-title">Generation mode</div>
                </div>
                <span className="studio-mini-pill">v11</span>
              </div>
              <div className="studio-mode-switch">
                {(["factory", "storyboard", "direct"] as GenerationMode[]).map((item) => (
                  <button key={item} type="button" disabled={isBusy} onClick={() => { setMode(item); resetOutput(); }} className={mode === item ? "studio-mode-active" : ""}>{item}</button>
                ))}
              </div>
            </section>

            <div className="studio-director-tabs" role="tablist" aria-label="Director panel">
              {([
                ["scene", "Scene"],
                ["camera", "Camera"],
                ["look", "Look"],
                ["elements", "Elements"],
              ] as Array<[DirectorTab, string]>).map(([tab, label]) => (
                <button key={tab} type="button" role="tab" aria-selected={directorTab === tab} onClick={() => setDirectorTab(tab)} className={directorTab === tab ? "studio-director-tab-active" : ""}>{label}</button>
              ))}
            </div>

            {directorTab === "elements" && (
              <section className="studio-inspector-section studio-inspector-section-flush">
                <div className="studio-inspector-title-row">
                  <div>
                    <div className="studio-inspector-eyebrow">My Elements</div>
                    <div className="studio-inspector-title">Cast & references</div>
                  </div>
                  <button type="button" onClick={() => { setElementType("character"); setShowElementCreator(true); }} className="studio-small-action">+ New</button>
                </div>

                <input value={elementSearch} onChange={(e) => setElementSearch(e.target.value)} className="studio-search-input" placeholder="Search Elements" />
                <div className="studio-filter-row">
                  {(["all", "character", "prop", "location", "style"] as const).map((value) => (
                    <button key={value} type="button" onClick={() => setElementFilter(value)} className={elementFilter === value ? "studio-filter-active" : ""}>{value === "all" ? "All" : `${value}s`}</button>
                  ))}
                </div>

                <div className="studio-element-library">
                  {elementsLoading && elements.length === 0 ? <div className="studio-empty-library">Loading Elements...</div> : null}
                  {!elementsLoading && filteredElements.length === 0 ? <div className="studio-empty-library">No matching Elements yet.</div> : null}
                  {filteredElements.map((element) => {
                    const asset = primaryElementAsset(element);
                    const active = hasElementMention(prompt, element.handle) || Boolean(elementApplyAll[element.id]);
                    return (
                      <button key={element.id} type="button" onClick={() => setSelectedElementId(element.id)} className={`studio-library-item ${selectedElement?.id === element.id ? "studio-library-item-selected" : ""}`}>
                        <span className="studio-library-thumb">{asset ? <Image unoptimized width={640} height={640} src={absoluteApiUrl(asset.asset_url)} alt={element.name} /> : <span>{element.name.slice(0, 1)}</span>}</span>
                        <span className="min-w-0 flex-1 text-left">
                          <span className="block truncate text-xs font-semibold text-[var(--text)]">{element.name}</span>
                          <span className="mt-0.5 block truncate text-[10px] text-[var(--text-muted)]">@{element.handle} · {element.type} · v{element.current_version}</span>
                        </span>
                        <span className={`studio-active-dot ${active ? "studio-active-dot-on" : ""}`} />
                      </button>
                    );
                  })}
                </div>

                {selectedElement && (
                  <div className="studio-selected-element">
                    <div className="flex items-center gap-3">
                      {primaryElementAsset(selectedElement) ? <Image unoptimized width={640} height={640} src={absoluteApiUrl(primaryElementAsset(selectedElement)!.asset_url)} alt={selectedElement.name} className="h-14 w-14 rounded-xl object-cover" /> : <div className="h-14 w-14 rounded-xl bg-[var(--empty-bg)]" />}
                      <div className="min-w-0 flex-1">
                        <div className="truncate text-sm font-semibold text-[var(--text)]">{selectedElement.name}</div>
                        <div className={`mt-0.5 text-[10px] uppercase ${elementTone(selectedElement.type)}`}>{selectedElement.type} · @{selectedElement.handle}</div>
                      </div>
                      <button type="button" onClick={() => insertElementMention(selectedElement)} className="studio-small-action">Use</button>
                    </div>

                    <div className="mt-3 grid gap-2">
                      <label className="studio-field-label">Reference mode
                        <select value={elementModes[selectedElement.id] || "identity"} onChange={(e) => setElementModes((current) => ({ ...current, [selectedElement.id]: e.target.value as ElementReferenceMode }))} className="studio-inspector-control">
                          <option value="identity">Identity / Element reference</option>
                          <option value="start_frame">Exact starting frame</option>
                        </select>
                      </label>
                      {selectedElement.type === "character" && (elementModes[selectedElement.id] || "identity") === "identity" && (
                        <label className="studio-field-label">Wardrobe source
                          <select value={elementWardrobePolicies[selectedElement.id] || "reference"} onChange={(e) => setElementWardrobePolicies((current) => ({ ...current, [selectedElement.id]: e.target.value as ElementWardrobePolicy }))} className="studio-inspector-control">
                            <option value="prompt">Follow scene prompt</option>
                            <option value="reference">Lock reference outfit · default</option>
                          </select>
                        </label>
                      )}
                      <label className="studio-field-label">Reference strength
                        <input type="range" min="0.55" max="1" step="0.05" value={Math.min(1, elementStrengths[selectedElement.id] ?? 1)} onChange={(e) => setElementStrengths((current) => ({ ...current, [selectedElement.id]: Number(e.target.value) }))} className="studio-range" />
                        <span className="studio-range-value">{Math.min(1, elementStrengths[selectedElement.id] ?? 1).toFixed(2)}</span>
                      </label>
                      <label className="studio-check-row"><input type="checkbox" checked={appliesToAllScenes(selectedElement.type, elementApplyAll[selectedElement.id])} onChange={(e) => setElementApplyAll((current) => ({ ...current, [selectedElement.id]: e.target.checked }))} /><span>Keep this Element active across all Factory scenes</span></label>
                    </div>

                    {selectedElement.type === "character" && <div className="mt-3 rounded-xl border border-[var(--border)] bg-[var(--panel-subtle)] px-3 py-2 text-[10px] leading-5 text-[var(--text-muted)]">Creator lock uses the Character reference in both IC-LoRA stages. For best identity, use a clean face close-up first, then full-body and profile views. Keep <strong>Follow scene prompt</strong> when the video needs a different outfit from the reference photo.</div>}

                    <div className="mt-3 flex gap-2 overflow-x-auto pb-1">
                      {selectedElement.assets.map((reference) => (
                        <button key={reference.id} type="button" onClick={() => void handleSetPrimaryElementAsset(selectedElement, reference.id)} className={`studio-asset-thumb relative ${reference.id === selectedElement.primary_asset_id ? "studio-asset-thumb-active" : ""}`} title={`${elementRoleLabel(reference.role)} · click to set canonical reference`}>
                          <Image unoptimized width={640} height={640} src={absoluteApiUrl(reference.asset_url)} alt="" />
                          <span className="absolute bottom-1 left-1 rounded bg-black/70 px-1 py-0.5 text-[8px] capitalize text-white">{elementRoleLabel(reference.role)}</span>
                        </button>
                      ))}
                      <label className="studio-asset-thumb studio-asset-add" title="Add references">+<input type="file" multiple accept="image/png,image/jpeg,image/webp" className="hidden" onChange={(e) => { void handleAddElementReferences(selectedElement, e.target.files); e.target.value = ""; }} /></label>
                    </div>
                    <div className="mt-3 flex items-center justify-between">
                      <span className="text-[10px] text-[var(--text-muted)]">{selectedElement.assets.length}/{capabilities?.elements?.max_assets_per_element ?? 8} references</span>
                      <div className="flex items-center gap-3">
                        {(hasElementMention(prompt, selectedElement.handle) || elementApplyAll[selectedElement.id]) && <button type="button" onClick={() => removeElementFromScene(selectedElement)} className="text-[10px] text-[var(--text-muted)]">Remove from scene</button>}
                        <button type="button" onClick={() => void handleArchiveElement(selectedElement)} className="text-[10px] text-[var(--danger-text)]">Archive</button>
                      </div>
                    </div>
                  </div>
                )}
              </section>
            )}

            {directorTab === "scene" && (
              <section className="studio-inspector-section studio-inspector-section-flush">
                <div className="studio-inspector-eyebrow">Scene</div>
                <div className="studio-inspector-title">Essentials</div>
                <p className="studio-inspector-copy">Format, quality and scene length live beside Generate. Keep this panel for only what changes the production.</p>
                <div className="mt-4 grid gap-3">
                  {mode === "factory" && (
                    <div className="studio-advanced-card">
                      <div className="flex items-start justify-between gap-3">
                        <div>
                          <div className="text-xs font-semibold text-[var(--text)]">Creator-grade talking head</div>
                          <div className="mt-1 text-[10px] leading-5 text-[var(--text-muted)]">1080p Diffusion final · user-selected runtime · strict visual QC · stage-2 Character lock · prompt-authoritative wardrobe. The preset never changes your duration.</div>
                        </div>
                        <button type="button" onClick={applyCreatorGradePreset} className="studio-small-action whitespace-nowrap">Apply preset</button>
                      </div>
                    </div>
                  )}
                  {mode === "factory" ? (
                    <>
                    <Control label="Final runtime"><select className="studio-inspector-control" value={factoryTargetSeconds} onChange={(e) => handleFactoryTargetChange(Number(e.target.value))}>{FACTORY_TARGETS.map((value) => <option key={value} value={value}>{value < 60 ? `${value} seconds` : `${value / 60} minute${value === 60 ? "" : "s"}`}</option>)}</select></Control>
                    <div className="-mt-1 text-[10px] leading-5 text-[var(--text-muted)]">Runtime is always your choice. Triven uses the longest safe identity-conditioned shot supported by the active worker and automatically continuity-chains the rest.</div>
                    </>
                  ) : mode === "storyboard" ? (
                    <Control label="Storyboard scenes"><select className="studio-inspector-control" value={sceneCount} onChange={(e) => setSceneCount(Number(e.target.value))}>{[2,3,4,6,8,10,12,16,20].map((value) => <option key={value} value={value}>{value} scenes</option>)}</select></Control>
                  ) : null}
                    <label className="studio-field-label">Spoken dialogue<textarea value={spokenScript} onChange={(e) => setSpokenScript(e.target.value)} rows={5} className="studio-inspector-textarea" placeholder="Exact words the character should say. Leave empty for no scripted dialogue." /></label>
                  <Control label="Audio"><select className="studio-inspector-control" value={audioMode} onChange={(e) => setAudioMode(e.target.value as AudioMode)}><option value="mastered">Generated + mastered</option><option value="native">Native LTX audio</option><option value="mute">Mute final video</option></select></Control>
                  <Control label="Continuity"><select className="studio-inspector-control" value={continuityMode} onChange={(e) => { setContinuityMode(e.target.value as ContinuityMode); invalidateRenderedMedia(); }}><option value="strict">Strict · identity + image</option><option value="balanced">Balanced · identity</option><option value="off">Off</option></select></Control>
                  <Control label="Realism"><select className="studio-inspector-control" value={realismProfile} onChange={(e) => { setRealismProfile(e.target.value as RealismProfile); invalidateRenderedMedia(); }}><option value="real_skin">Real Skin · recommended</option><option value="identity_max">Identity Max · use Character Element</option><option value="standard">Standard · faster</option></select></Control>
                  <div className="studio-advanced-card text-[10px] leading-5 text-[var(--text-muted)]">
                    {realismProfile === "standard"
                      ? "Standard keeps the normal production render without the extra detail pass."
                      : realismProfile === "identity_max"
                        ? "Identity Max keeps the Character/Ingredients reference active through BOTH LTX IC-LoRA stages, applies strict early/mid/late artifact QC, then uses the tiled Refine Details texture pass on final renders."
                        : "Real Skin adds the tiled LTX 2.5 Refine Details pass. Add a Character Element for persistent identity; use Follow scene prompt when the reference photo and requested wardrobe are different."}
                  </div>
                  {realismProfile === "identity_max" && referencedElements.filter((element) => element.type === "character").length > 1 && (
                    <div className="rounded-xl border border-amber-500/20 bg-amber-500/[0.05] px-3 py-2 text-[10px] leading-5 text-amber-700 dark:text-amber-300">Multiple Character identities are active. For a solo YouTube presenter, keep exactly one Character Element in the shot; extra @character references can compete and reduce face consistency.</div>
                  )}
                </div>

                <details className="studio-inspector-details studio-inspector-advanced">
                  <summary>Advanced</summary>
                  <div className="mt-3 grid gap-3">
                    <div className="grid grid-cols-2 gap-2"><Control label="Seed"><input className="studio-inspector-control" type="number" min={0} value={seed} onChange={(e) => { setSeed(Number(e.target.value) || 0); invalidateRenderedMedia(); }} /></Control><Control label="Preview decoder"><select className="studio-inspector-control" disabled={quality !== "preview"} value={quality === "preview" ? decoder : "diffusion"} onChange={(e) => setDecoder(e.target.value as DecoderName)}><option value="conv">Conv · fast</option><option value="diffusion">Diffusion · detailed</option></select></Control></div>
                    <label className="studio-field-label">Sound direction<textarea value={audioDirection} onChange={(e) => setAudioDirection(e.target.value)} rows={4} className="studio-inspector-textarea" placeholder="Ambience, Foley, music, voice style..." /></label>
                    {mode === "factory" && youtube?.enabled && (
                      <div className="studio-advanced-card">
                        <label className="studio-feature-toggle">
                          <span><strong>Publish to YouTube</strong><small>{youtube.connected ? youtube.channel_title || "Connected channel" : "Connect a channel first"}</small></span>
                          <input type="checkbox" disabled={!youtube.connected} checked={publishToYouTube} onChange={(e) => setPublishToYouTube(e.target.checked)} />
                        </label>
                        {publishToYouTube && <div className="mt-3 grid gap-2"><input className="studio-inspector-control" value={youtubeTitle} onChange={(e) => setYoutubeTitle(e.target.value)} placeholder="YouTube title" /><select className="studio-inspector-control" value={youtubePrivacy} onChange={(e) => setYoutubePrivacy(e.target.value as YouTubePrivacy)}><option value="private">Private</option><option value="unlisted" disabled={!youtube.public_uploads_allowed}>Unlisted</option><option value="public" disabled={!youtube.public_uploads_allowed}>Public</option></select><textarea className="studio-inspector-textarea" rows={3} value={youtubeDescription} onChange={(e) => setYoutubeDescription(e.target.value)} placeholder="Description" /></div>}
                      </div>
                    )}
                  </div>
                </details>
              </section>
            )}

            {directorTab === "camera" && (
              <section className="studio-inspector-section studio-inspector-section-flush">
                <div className="studio-inspector-eyebrow">Director</div>
                <div className="studio-inspector-title">Camera</div>
                <p className="studio-inspector-copy">Choose only what matters. Auto leaves the screenplay untouched.</p>
                <div className="mt-4 grid gap-3">
                  <Control label="Shot size"><select className="studio-inspector-control" value={shotSize} onChange={(e) => setShotSize(e.target.value as ShotSize)}>{SHOT_SIZES.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></Control>
                  <Control label="Movement"><select className="studio-inspector-control" value={cameraMove} onChange={(e) => setCameraMove(e.target.value as CameraMove)}>{CAMERA_MOVES.map((item) => <option key={item.value} value={item.value}>{item.label}</option>)}</select></Control>
                  <Control label="Lens"><select className="studio-inspector-control" value={lensPreset} onChange={(e) => setLensPreset(e.target.value as LensPreset)}>{LENS_PRESETS.map((value) => <option key={value} value={value}>{value === "auto" ? "Auto" : value}</option>)}</select></Control>
                </div>
              </section>
            )}

            {directorTab === "look" && (
              <section className="studio-inspector-section studio-inspector-section-flush">
                <div className="studio-inspector-eyebrow">Director</div>
                <div className="studio-inspector-title">Look</div>
                <p className="studio-inspector-copy">A small set of cinematic defaults instead of a wall of controls.</p>
                <div className="mt-4 grid gap-3">
                  <Control label="Genre"><select className="studio-inspector-control" value={genrePreset} onChange={(e) => setGenrePreset(e.target.value as GenrePreset)}>{GENRE_PRESETS.map((value) => <option key={value} value={value}>{value === "auto" ? "Auto" : value}</option>)}</select></Control>
                  <Control label="Colour"><select className="studio-inspector-control" value={colorPreset} onChange={(e) => setColorPreset(e.target.value as ColorPreset)}>{COLOR_PRESETS.map((value) => <option key={value} value={value}>{value === "auto" ? "Auto" : value.replaceAll("-", " ")}</option>)}</select></Control>
                  <Control label="Tempo"><select className="studio-inspector-control" value={tempoPreset} onChange={(e) => setTempoPreset(e.target.value as TempoPreset)}>{TEMPO_PRESETS.map((value) => <option key={value} value={value}>{value === "auto" ? "Auto" : value}</option>)}</select></Control>
                </div>
              </section>
            )}

            <details className="studio-inspector-details">
              <summary>Production & account</summary>
              <div className="mt-3 space-y-3">
                <IntegrationCard title="Production profile" subtitle="Current factory limits">
                  <InfoRow label="1080p scene" value={`${capabilities?.max_scene_duration_seconds_by_quality?.["1080p"] ?? 30}s`} />
                  <InfoRow label="4K scene" value={`${capabilities?.max_scene_duration_seconds_by_quality?.["4k"] ?? 15}s`} />
                  <InfoRow label="Factory runtime" value={`${capabilities?.max_factory_duration_seconds ?? 300}s`} />
                  <InfoRow label="Elements / scene" value={String(capabilities?.elements?.max_active_per_scene ?? 6)} />
                </IntegrationCard>
                <IntegrationCard title="Billing" subtitle={billingCatalog?.enabled ? "Stripe generation credits" : "Billing disabled"}>
                  {billingCatalog?.enabled && billingMe ? <><div className="mb-2 text-xs text-[var(--success-text)]">Balance {formatCredits(billingMe.balance_seconds)}</div>{billingCatalog.packs.filter((pack) => pack.available).map((pack) => <button key={pack.id} type="button" disabled={integrationBusy} onClick={() => void handleCheckout(pack.id)} className="studio-secondary-button mb-2 w-full">{pack.label} · {formatCredits(pack.credit_seconds)}</button>)}{billingMe.stripe_customer_id && <button type="button" onClick={handleBillingPortal} className="studio-secondary-button w-full">Manage billing</button>}</> : <p className="text-[10px] leading-5 text-[var(--text-muted)]">Configure Stripe to enable customer generation credits.</p>}
                </IntegrationCard>
                {youtube?.enabled && <button type="button" disabled={integrationBusy} onClick={handleYouTubeConnection} className="studio-secondary-button w-full">{youtube.connected ? "Disconnect YouTube" : "Connect YouTube"}</button>}
              </div>
            </details>
          </fieldset>
        </aside>
      </div>

      {showElementsLibrary && (
        <div className="studio-drawer-backdrop" role="dialog" aria-modal="true" aria-label="My Elements" onMouseDown={(event) => { if (event.target === event.currentTarget) setShowElementsLibrary(false); }}>
          <aside className="studio-elements-drawer">
            <div className="studio-drawer-header">
              <div>
                <div className="studio-inspector-eyebrow">Production library</div>
                <h2 className="mt-1 text-lg font-semibold text-[var(--text)]">My Elements</h2>
                <p className="mt-1 text-xs text-[var(--text-muted)]">Reusable characters, props, locations and styles. Click any Element to bind it to the current scene.</p>
              </div>
              <button type="button" onClick={() => setShowElementsLibrary(false)} className="studio-icon-button" aria-label="Close Elements">×</button>
            </div>
            <div className="studio-drawer-toolbar">
              <input value={elementSearch} onChange={(e) => setElementSearch(e.target.value)} className="studio-search-input" placeholder="Search characters, props, locations..." />
              <button type="button" onClick={() => { setShowElementsLibrary(false); setShowElementCreator(true); }} className="studio-primary-button"><PlusIcon /> New Element</button>
            </div>
            <div className="studio-filter-row studio-filter-row-wide">
              {(["all", "character", "prop", "location", "style"] as const).map((value) => (
                <button key={value} type="button" onClick={() => setElementFilter(value)} className={elementFilter === value ? "studio-filter-active" : ""}>{value === "all" ? "All" : `${value}s`}</button>
              ))}
            </div>
            <div className="studio-elements-grid">
              {filteredElements.map((element) => {
                const asset = primaryElementAsset(element);
                const active = hasElementMention(prompt, element.handle) || Boolean(elementApplyAll[element.id]);
                return (
                  <article key={element.id} className={`studio-element-grid-card ${active ? "studio-element-grid-card-active" : ""}`}>
                    <button type="button" className="studio-element-grid-preview" onClick={() => { setSelectedElementId(element.id); setDirectorTab("elements"); setShowElementsLibrary(false); }}>
                      {asset ? <Image unoptimized width={640} height={640} src={absoluteApiUrl(asset.asset_url)} alt={element.name} /> : <span>{element.name.slice(0, 1)}</span>}
                      <span className={`studio-reference-badge ${elementTone(element.type)}`}>{element.type}</span>
                    </button>
                    <div className="studio-element-grid-meta">
                      <div className="min-w-0"><div className="truncate text-xs font-semibold text-[var(--text)]">{element.name}</div><div className="truncate text-[10px] text-[var(--text-muted)]">@{element.handle} · {element.assets.length} refs · v{element.current_version}</div></div>
                      <button type="button" onClick={() => { insertElementMention(element); setShowElementsLibrary(false); }} className="studio-small-action">Use</button>
                    </div>
                  </article>
                );
              })}
              {!elementsLoading && filteredElements.length === 0 && <div className="studio-library-empty-large">No Elements match this filter.</div>}
            </div>
          </aside>
        </div>
      )}

      {showReferencePicker && (
        <div className="studio-modal-backdrop" role="dialog" aria-modal="true" aria-label="Add scene reference" onMouseDown={(event) => { if (event.target === event.currentTarget) setShowReferencePicker(false); }}>
          <div className="studio-reference-picker">
            <div className="studio-modal-header">
              <div><div className="studio-inspector-eyebrow">References</div><h2 className="mt-1 text-lg font-semibold text-[var(--text)]">Add to this scene</h2><p className="mt-1 text-xs text-[var(--text-muted)]">Select a saved Element. It will be inserted into the prompt as an @mention and resolved to its canonical reference during generation.</p></div>
              <button type="button" onClick={() => setShowReferencePicker(false)} className="studio-icon-button">×</button>
            </div>
            <div className="studio-reference-picker-toolbar">
              <div className="studio-reference-picker-tabs"><button type="button" className="studio-reference-picker-tab-active">Elements</button><button type="button" onClick={() => { setShowReferencePicker(false); setShowElementCreator(true); }}>Upload new</button></div>
              <span className="studio-count-pill">{referencedElements.length}/{activeElementLimit} active</span>
            </div>
            <div className="studio-reference-picker-grid">
              {elements.map((element) => {
                const asset = primaryElementAsset(element);
                const active = hasElementMention(prompt, element.handle) || Boolean(elementApplyAll[element.id]);
                return (
                  <button key={element.id} type="button" disabled={!active && referencedElements.length >= activeElementLimit} onClick={() => { insertElementMention(element); setShowReferencePicker(false); }} className={`studio-picker-card ${active ? "studio-picker-card-active" : ""}`}>
                    <span className="studio-picker-image">{asset ? <Image unoptimized width={640} height={640} src={absoluteApiUrl(asset.asset_url)} alt={element.name} /> : <span>{element.name.slice(0, 1)}</span>}</span>
                    <span className="min-w-0 text-left"><span className="block truncate text-xs font-semibold text-[var(--text)]">{element.name}</span><span className={`mt-0.5 block text-[10px] ${elementTone(element.type)}`}>@{element.handle} · {element.type}</span></span>
                    {active && <span className="studio-picker-check">✓</span>}
                  </button>
                );
              })}
            </div>
          </div>
        </div>
      )}

      {showElementCreator && (
        <div className="studio-modal-backdrop" role="dialog" aria-modal="true" aria-label="Create Element">
          <div className="studio-modal" ref={creatorRef}>
            <div className="studio-modal-header">
              <div><div className="studio-inspector-eyebrow">New Element</div><h2 className="mt-1 text-lg font-semibold text-[var(--text)]">Create a reusable visual identity</h2><p className="mt-1 text-xs text-[var(--text-muted)]">Characters, props and locations are saved once, then reused in prompts with @mentions.</p></div>
              <button type="button" disabled={elementBusy} aria-label="Close Element creator" onClick={() => setShowElementCreator(false)} className="studio-icon-button">×</button>
            </div>
            <div className="studio-modal-body"><fieldset disabled={elementBusy} className="studio-modal-fields">
              <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
                {(["character", "prop", "location", "style"] as ElementType[]).map((type) => <button key={type} type="button" onClick={() => { setElementType(type); setElementFileRoles({}); }} className={`studio-element-type-choice ${elementType === type ? "studio-element-type-choice-active" : ""}`}><span className={elementTone(type)}>{type}</span></button>)}
              </div>
              <div className="mt-4 grid gap-3 sm:grid-cols-2">
                <label className="studio-field-label">Name<input className="studio-inspector-control mt-1" maxLength={80} autoComplete="off" value={elementName} onChange={(e) => { const automaticHandle = elementName.replace(/[^A-Za-z0-9_-]/g, "").replace(/^[^A-Za-z]+/, "").slice(0, 32); if (!elementHandle || elementHandle === automaticHandle) setElementHandle(e.target.value.replace(/[^A-Za-z0-9_-]/g, "").replace(/^[^A-Za-z]+/, "").slice(0, 32)); setElementName(e.target.value); }} placeholder="Radha" /></label>
                <label className="studio-field-label">Prompt handle<div className="studio-handle-field mt-1"><span aria-hidden="true">@</span><input className="studio-inspector-control studio-handle-input" maxLength={32} autoComplete="off" spellCheck={false} value={elementHandle} onChange={(e) => setElementHandle(e.target.value.replace(/^@/, "").replace(/[^A-Za-z0-9_-]/g, ""))} placeholder="Radha" /></div></label>
              </div>
              <label className="mt-3 block studio-field-label">Identity / design description<textarea rows={3} maxLength={1600} value={elementDescription} onChange={(e) => setElementDescription(e.target.value)} className="studio-inspector-textarea mt-1" placeholder="Face, costume, materials, landmarks or other traits that must remain stable." /></label>
              <label className={`studio-upload-dropzone ${elementFiles.length ? "studio-upload-dropzone-compact" : ""}`} onDragOver={(event) => event.preventDefault()} onDrop={(event) => { event.preventDefault(); selectElementFiles(Array.from(event.dataTransfer.files)); }}>
                <span className="studio-upload-icon">+</span>
                <strong>{elementFiles.length ? "Add more references" : "Drop reference images or click to upload"}</strong>
                <small>{elementType === "character" ? "Use clear face, full-body, profile and costume views of the same person. Check the suggested labels below." : `PNG, JPEG or WEBP · up to ${capabilities?.elements?.max_assets_per_element ?? 8} references`}</small>
                <small>PNG, JPEG or WEBP · at least 128 × 128 · up to {capabilities?.elements?.max_upload_mb ?? 15} MB each · {capabilities?.elements?.max_assets_per_element ?? 8} images maximum</small>
                <input aria-label="Upload reference images" type="file" multiple accept="image/png,image/jpeg,image/webp,.jpg,.jpeg,.png,.webp" className="sr-only" onChange={(e) => { selectElementFiles(Array.from(e.target.files || [])); e.target.value = ""; }} />
              </label>
              {elementFiles.length > 0 && <div className="studio-upload-selection"><p className="studio-upload-selection-heading">{elementFiles.length} references selected <span>Check each image’s role</span></p><div className="studio-reference-uploads">{elementFiles.map((file, index) => {
                const key = referenceFileKey(file);
                const role = referenceRoles(elementType, elementFiles, elementFileRoles)[index];
                return <div key={key} className="studio-reference-upload">
                  <ReferenceUploadPreview file={file} />
                  <div className="min-w-0 flex-1"><p className="truncate text-xs" title={file.name}>{file.name}</p><label className="text-xs text-[var(--text-muted)]"><span className="sr-only">Reference role</span><select value={role} onChange={(event) => setElementFileRoles((current) => ({ ...current, [key]: event.target.value as ElementAssetRole }))} className="studio-inspector-control mt-1" aria-label={`Reference role for ${file.name}`}>
                    {(elementType === "character" ? ["face", "full_body", "profile", "costume", "support"] : [elementType === "prop" ? "object" : elementType, "support"]).map((value) => <option key={value} value={value}>{elementRoleLabel(value as ElementAssetRole)}</option>)}
                  </select></label></div>
                  <button type="button" className="studio-icon-button" aria-label={`Remove ${file.name}`} onClick={() => { setElementFiles((current) => current.filter((item) => item !== file)); setElementUploadError(""); }}>×</button>
                </div>;
              })}</div></div>}
            </fieldset></div>
            <div className="studio-modal-actions">
              {elementUploadError && <p role="alert" className="studio-upload-error">{elementUploadError}</p>}
              <div className="studio-modal-footer">
              <button type="button" disabled={elementBusy} onClick={() => setShowElementCreator(false)} className="studio-secondary-button">Cancel</button>
              <button type="button" onClick={() => void handleCreateElement()} disabled={elementBusy} className="studio-primary-button">{elementBusy ? "Saving..." : `Save @${elementHandle || "Element"}`}</button>
              </div>
            </div>
          </div>
        </div>
      )}

      <div id="studio-output" className="mx-auto w-full max-w-[1500px] px-4 pb-10 pt-5 sm:px-7 lg:px-10">
        {historySaveState === "offline" && <div role="status" className="studio-inline-notice studio-inline-info">Changes are saved on this device. Account sync will retry automatically. <button type="button" onClick={() => setHistoryRetry((current) => current + 1)} className="underline">Retry sync</button></div>}
        {(activeJob || jobError || serverJobs.length > 0) && <section className="mt-4 rounded-2xl border border-[var(--border)] bg-[var(--panel-bg)] p-4" aria-label="Generation jobs">
          <h2 className="text-sm font-semibold">Generation tasks</h2>
          {activeJob && <p role="status" className="mt-2 text-xs">{progressMessage || "Reconnecting to saved render…"} You can refresh and return to this job.</p>}
          {jobError && <p role="alert" className="mt-2 text-xs text-[var(--danger-text)]">{jobError}</p>}
          <div className="mt-2 space-y-2">{serverJobs.filter((job) => (job.chat_id || job.payload.chat_id) === activeChatId).slice(0, 20).map((job) => <div key={job.job_id} className="flex flex-wrap items-center justify-between gap-2 border-t border-[var(--border)] py-2 text-xs"><div><strong className="capitalize">{job.job_type} · {job.status}</strong><p className="text-[var(--text-muted)]">{job.error || job.message}</p></div>{job.status !== "failed" && <button type="button" disabled={isBusy} onClick={() => recoverServerJob(job)} className="studio-secondary-button">{job.status === "completed" ? "Open result" : "Resume tracking"}</button>}</div>)}</div>
        </section>}
        {recoveredAssets.length > 0 && <details className="mt-4 rounded-2xl border border-[var(--border)] p-4"><summary className="cursor-pointer text-sm font-semibold">Saved render assets ({recoveredAssets.length})</summary><p className="mt-2 text-xs text-[var(--text-muted)]">These outputs remain available even if a later step fails.</p><div className="mt-3 grid gap-3 sm:grid-cols-2">{recoveredAssets.filter((asset) => asset.filename.endsWith(".mp4")).map((asset) => <div key={asset.filename} className="min-w-0 rounded-xl border border-[var(--border)] p-3"><StudioVideo controls preload="metadata" src={asset.video_url || `/media/generated/${encodeURIComponent(asset.filename)}`} className="w-full" /><div className="mt-2 flex flex-wrap gap-2 text-xs"><span>Visual QC</span><QCStatusPill status={asset.visual_qc_status || asset.metadata?.visual_qc_status || "not_checked"} /><span>Audio QC</span><QCStatusPill status={asset.audio_qc_status || asset.metadata?.audio_qc_status || "not_checked"} /></div><a className="mt-2 inline-block text-xs underline" href={asset.download_url || `/api/v1/generations/download/${encodeURIComponent(asset.filename)}`}>Download retained clip</a></div>)}</div></details>}
        {videoTakes.length > 0 && <details className="mt-4 rounded-2xl border border-[var(--border)] bg-[var(--panel-bg)] p-4"><summary className="cursor-pointer text-sm font-semibold">Saved takes ({videoTakes.length})</summary><p className="mt-2 text-xs text-[var(--text-muted)]">Every successful take is kept when you change settings or retry.</p><div className="mt-3 space-y-2">{videoTakes.map((take, index) => <div key={take.filename} className="flex flex-wrap items-center justify-between gap-3 rounded-xl border border-[var(--border)] p-3"><div className="min-w-0"><p className="text-xs">Take {videoTakes.length - index} · {take.label}</p><div className="mt-1 flex flex-wrap gap-2"><span className="text-xs">Visual</span><QCStatusPill status={take.qcReport?.visual || "unknown"} /><span className="text-xs">Audio</span><QCStatusPill status={take.qcReport?.audio || "unknown"} /></div></div><div className="flex gap-2"><button type="button" disabled={isBusy} onClick={() => { studioPlayback.pauseAll(); setFinalVideo(take); setFactoryResult(take.factoryResult || null); }} className="studio-secondary-button">Watch & inspect</button><a href={take.downloadUrl} className="studio-secondary-button">Download</a></div></div>)}</div></details>}

        {finalVideo && (
          <section className="mt-5">
            <FinalVideoCard video={finalVideo} aspectRatio={finalVideo.aspectRatio || aspectRatio} />
            <details className="mt-3 rounded-2xl border border-[var(--border)] bg-[var(--panel-bg)] p-4"><summary className="cursor-pointer text-sm font-semibold">Correct audio using the existing video</summary><p className="mt-2 text-xs text-[var(--text-muted)]">The picture and original take are preserved. {billingCatalog?.enabled && billingCatalog.enforce_credits ? "This correction uses generation credits equal to the clip duration." : "A new audio take will be generated and checked."}</p><label className="mt-3 block text-xs">Exact spoken dialogue<textarea disabled={isBusy} value={spokenScript} onChange={(event) => setSpokenScript(event.target.value)} rows={4} className="studio-inspector-textarea mt-1" /></label><button type="button" onClick={() => void handleAudioRepair()} disabled={isBusy || !spokenScript.trim()} className="studio-primary-button mt-3">Generate corrected audio</button></details>
          </section>
        )}

        {!!factoryResult?.scene_results?.length && <details className="mt-5 rounded-2xl border border-[var(--border)] bg-[var(--panel-bg)] p-4"><summary className="cursor-pointer text-sm font-semibold">Inspect individual scenes ({factoryResult.scene_results.length})</summary><div className="mt-4 grid gap-4 lg:grid-cols-2">{factoryResult.scene_results.map((scene) => <article key={scene.filename} className="overflow-hidden rounded-xl border border-[var(--border)]"><StudioVideo src={scene.video_url} controls preload="metadata" className="w-full" /><div className="p-3"><h3 className="text-sm font-semibold">Scene {scene.scene_index + 1}</h3><div className="mt-2 flex flex-wrap items-center gap-2 text-xs"><span>Visual / identity QC</span><QCStatusPill status={scene.visual_qc_status || "not_checked"} /><span>Speech / audio QC</span><QCStatusPill status={scene.audio_qc_status || "not_checked"} /></div>{[scene.visual_qc, scene.audio_qc].flatMap(qcReasons).map((reason, index) => <p key={index} className="mt-2 text-xs">{reason}</p>)}<a href={scene.download_url} className="mt-3 inline-block text-xs underline">Download scene</a></div></article>)}</div></details>}

        {factoryResult && (
          <section className="mt-5 grid gap-3 sm:grid-cols-2 lg:grid-cols-7">
            <Stat label="Scenes" value={String(factoryResult.scene_count)} detail={`${factoryResult.chunk_count} native LTX chunks`} />
            <Stat label="Runtime" value={`${(factoryResult.actual_duration_seconds ?? factoryResult.target_duration_seconds).toFixed(1)}s`} detail={`${factoryResult.scene_duration_seconds}s target scene`} />
            <Stat label="Delivery" value={qualityLabel(factoryResult.quality)} detail={`${factoryResult.width ?? "?"}×${factoryResult.height ?? "?"} · ${factoryResult.audio_mode}`} />
            <Stat label="Planner" value={factoryResult.planner_source} detail={`${factoryResult.entity_locks.length} entity lock${factoryResult.entity_locks.length === 1 ? "" : "s"}`} />
            <Stat label="Elements" value={factoryResult.elements_used.length ? String(factoryResult.elements_used.length) : "None"} detail={factoryResult.elements_used.length ? `${factoryResult.elements_used.join(", ")} · ${factoryResult.element_reference_mode || "reference"}` : "Prompt-only generation"} />
            <Stat label="Visual QC" value={qcStatusLabel(factoryQCReport(factoryResult).visual)} detail={`${factoryResult.continuity_regenerations} auto-regeneration${factoryResult.continuity_regenerations === 1 ? "" : "s"}`} />
            <Stat label="Audio QC" value={qcStatusLabel(factoryQCReport(factoryResult).audio)} detail={`${factoryResult.audio_retake_count} LTX audio retake${factoryResult.audio_retake_count === 1 ? "" : "s"}`} />
          </section>
        )}

        {result && mode === "storyboard" && (
          <section className="mt-5 rounded-3xl border border-[var(--border)] bg-[var(--panel-bg)] p-5 sm:p-7">
            <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-center lg:justify-between">
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <h2 className="text-lg font-semibold">Storyboard</h2>
                  <span className="rounded-full bg-[var(--panel-subtle-strong)] px-2.5 py-1 text-[10px] text-[var(--text-muted)]">{result.planner_source}</span>
                  <span className="rounded-full bg-emerald-500/[0.06] px-2.5 py-1 text-[10px] text-[var(--accent-text)]">{continuityMode} continuity</span>
                </div>
                <p className="mt-1 text-xs text-[var(--text-muted)]">{result.scenes.length} scenes · {plannedDuration}s selected runtime · same seed · previous-frame chaining</p>
              </div>
              <button type="button" onClick={handleRenderMissingAndCombine} disabled={isBusy} className="flex h-11 items-center justify-center gap-2 rounded-xl bg-[var(--primary-bg)] px-5 text-sm font-medium text-[var(--primary-fg)] hover:bg-[var(--primary-hover)] disabled:opacity-40">
                {creatingFinal ? <Spinner /> : <PlayIcon />}{creatingFinal ? progressMessage || "Creating final video" : finalActionLabel}
              </button>
            </div>

            <div className="grid gap-5 lg:grid-cols-2">
              {result.scenes.map((scene, index) => {
                const video = renderedVideos[scene.id];
                const isGenerating = generatingScene === scene.id;
                const isEditing = editingScene === scene.id;
                return (
                  <article key={scene.id} className="overflow-hidden rounded-2xl border border-[var(--border)] bg-[var(--panel-bg)]">
                    {video ? (
                      <div className="bg-black"><StudioVideo src={video.url} controls playsInline className={`w-full object-contain ${aspectClass(result.aspect_ratio)}`} /></div>
                    ) : (
                      <div className={`relative flex items-center justify-center bg-[var(--empty-bg)] ${aspectClass(result.aspect_ratio)}`}><div className="text-center text-[var(--text-muted)]">{isGenerating ? <Spinner /> : <PlayIcon />}<div className="mt-3 text-xs">{isGenerating ? "Rendering with LTX..." : "Not rendered"}</div></div></div>
                    )}
                    <div className="p-5 sm:p-6">
                      <div className="flex items-start justify-between gap-4">
                        <div className="min-w-0"><div className="mb-2 text-[10px] uppercase tracking-[0.2em] text-[var(--text-muted)]">Scene {String(index + 1).padStart(2, "0")}</div><h3 className="truncate text-base font-medium text-[var(--text)]">{scene.title}</h3></div>
                        <div className="shrink-0 rounded-lg border border-[var(--border)] px-2.5 py-1 text-[10px] text-[var(--text-muted)]">{durationSeconds}s render</div>
                      </div>
                      {isEditing ? (
                        <textarea disabled={isBusy} value={scenePrompts[scene.id] || ""} onChange={(e) => updateScenePrompt(scene.id, e.target.value)} rows={8} className="mt-5 w-full resize-none rounded-xl border border-[var(--border)] bg-[var(--input-bg)] p-4 text-sm leading-6 text-[var(--text)] outline-none" />
                      ) : <p className="mt-5 line-clamp-6 text-sm leading-6 text-[var(--text-muted)]">{scenePrompts[scene.id]}</p>}
                      <details className="mt-3 text-xs text-[var(--text-muted)]"><summary className="cursor-pointer">Scene dialogue</summary><textarea aria-label={`Spoken dialogue for scene ${scene.id}`} disabled={isBusy} value={scene.spoken_script || ""} onChange={(event) => { const value = event.target.value; setResult((current) => current ? { ...current, scenes: current.scenes.map((item) => item.id === scene.id ? { ...item, spoken_script: value } : item) } : current); setRenderedVideos((current) => invalidateSceneChain(current, result.scenes.map((item) => item.id), scene.id)); }} rows={3} className="studio-inspector-textarea mt-2" placeholder="Exact words for this scene" /></details>
                      {video && (
                        <div className="mt-4 space-y-2">
                          {video.renderSignature !== sceneRenderSignature(scene.id) && <p className="text-xs text-amber-700">Settings changed. This saved take remains available; the next render will update this scene and its downstream references.</p>}
                          <div className="flex flex-wrap items-center gap-2 text-xs text-[var(--text)]">
                            <span className="font-medium">Scene visual QC:</span>
                            <QCStatusPill status={resolveQCStatus(video.visualQcStatus, video.continuityQcPassed, video.continuityMode !== "off", video.continuityWarnings)} />
                            {video.continuityRegenerations > 0 && <span className="text-[var(--text-muted)]">{video.continuityRegenerations} retry attempt{video.continuityRegenerations === 1 ? "" : "s"}</span>}
                          </div>
                          <div className="flex items-center gap-2 text-xs"><span>Speech / audio QC:</span><QCStatusPill status={video.audioQcStatus || "not_checked"} /></div>
                          {video.continuityWarnings?.length > 0 && (
                            <details className="text-xs text-[var(--text-muted)]">
                              <summary className="cursor-pointer">QC details ({video.continuityWarnings.length})</summary>
                              {video.continuityWarnings.map((warning, warningIndex) => <p key={warningIndex} className="mt-1 leading-5">{warning}</p>)}
                            </details>
                          )}
                          <div className="rounded-xl bg-[var(--panel-subtle)] px-3 py-2 text-[11px] leading-5 text-[var(--text-muted)]">{video.details} · {video.chunkCount} chunk{video.chunkCount === 1 ? "" : "s"} · {video.renderSeconds.toFixed(1)}s render · {video.mediaInfo.has_audio ? `audio ${video.mediaInfo.audio_codec || "present"}` : "no audio"}{video.continuityMode === "strict" ? video.continuityApplied ? " · conditioned" : " · anchor" : ""}</div>
                        </div>
                      )}
                      <div className="mt-5 flex items-center justify-between gap-3 border-t border-[var(--border)] pt-4">
                        <button type="button" disabled={isBusy && !isEditing} onClick={() => setEditingScene(isEditing ? null : scene.id)} className="h-9 rounded-lg px-3 text-xs text-[var(--text-muted)] hover:text-[var(--text)] disabled:opacity-40">{isEditing ? "Done editing" : "Edit prompt"}</button>
                        <div className="flex items-center gap-2">
                          {video && <a href={video.downloadUrl} className="flex h-9 items-center gap-2 rounded-lg border border-[var(--border)] px-3 text-xs text-[var(--text-muted)] hover:text-[var(--text)]"><DownloadIcon /> Clip</a>}
                          <button type="button" disabled={isBusy} onClick={() => handleRenderScene(scene.id)} className="flex h-9 items-center gap-2 rounded-lg bg-[var(--primary-bg)] px-4 text-xs font-medium text-[var(--primary-fg)] hover:bg-[var(--primary-hover)] disabled:opacity-40">{isGenerating ? <Spinner /> : <PlayIcon />}{video ? "Regenerate chain" : "Render through scene"}</button>
                        </div>
                      </div>
                    </div>
                  </article>
                );
              })}
            </div>
            <div className="mt-5 text-xs text-[var(--text-muted)]">{renderedSceneCount}/{result.scenes.length} rendered · {allScenesRendered ? "Ready to combine" : "Scenes render sequentially so strict continuity has the previous frame"}</div>
          </section>
        )}
      </div>
    </main>
  );
}

function Control({ label, children }: { label: string; children: ReactNode }) {
  return <label className="block"><span className="mb-1.5 block text-[10px] font-medium uppercase tracking-[0.16em] text-[var(--text-muted)]">{label}</span>{children}</label>;
}

function IntegrationCard({ title, subtitle, children }: { title: string; subtitle: string; children: ReactNode }) {
  return <div className="rounded-2xl border border-[var(--border)] bg-[var(--panel-bg)] p-4"><div className="text-xs font-semibold text-[var(--text)]">{title}</div><div className="mt-1 mb-4 text-[11px] leading-5 text-[var(--text-muted)]">{subtitle}</div>{children}</div>;
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return <div className="flex items-center justify-between border-t border-[var(--border)] py-2 text-[11px]"><span className="text-[var(--text-muted)]">{label}</span><span className="text-[var(--text-secondary)]">{value}</span></div>;
}

function Stat({ label, value, detail }: { label: string; value: string; detail: string }) {
  return <div className="rounded-2xl border border-[var(--border)] bg-[var(--panel-bg)] p-4"><div className="text-[10px] uppercase tracking-[0.16em] text-[var(--text-muted)]">{label}</div><div className="mt-2 text-lg font-semibold text-[var(--text)]">{value}</div><div className="mt-1 text-[11px] text-[var(--text-muted)]">{detail}</div></div>;
}

function qcReasons(report?: Record<string, unknown>): string[] {
  if (!report) return [];
  const reasons: string[] = [];
  for (const key of ["warnings", "issues", "violations", "reason", "note", "error"]) {
    const value = report[key];
    if (typeof value === "string" && value.trim()) reasons.push(value);
    if (Array.isArray(value)) for (const item of value) {
      if (typeof item === "string") reasons.push(item);
      else if (item && typeof item === "object" && typeof item.reason === "string") reasons.push(item.reason);
    }
  }
  return [...new Set(reasons)];
}

function QCStatusPill({ status }: { status: QCStatus }) {
  const appearance = status === "failed"
    ? "border-red-500/30 bg-red-500/10 text-red-700 dark:text-red-300"
    : status === "passed"
      ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
      : status === "unavailable"
        ? "border-amber-500/30 bg-amber-500/10 text-amber-800 dark:text-amber-300"
        : "border-[var(--border)] bg-[var(--panel-subtle)] text-[var(--text-muted)]";
  return <span className={`inline-flex rounded-full border px-2.5 py-1 text-[11px] font-semibold ${appearance}`}>{qcStatusLabel(status)}</span>;
}

function VideoQCPanel({ report }: { report?: VideoQCReport }) {
  // Old videos did not store a QC verdict. Never silently label them "Passed".
  const visual = report?.visual || "unknown";
  const audio = report?.audio || "unknown";
  const hasFailure = visual === "failed" || audio === "failed";
  const reviewNote = hasFailure
    ? "QC failed: the MP4 is saved, but the result is not approved. Check the issues below before sharing."
    : visual === "unavailable" || audio === "unavailable"
      ? "QC could not complete. The video is available, but its quality is unverified."
      : "QC status is saved with this video and will remain visible in chat history.";
  return (
    <section aria-label="Video quality control results" className="border-t border-[var(--border)] px-5 py-4 sm:px-6">
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-[var(--text)]">Quality control results</h3>
        <p className="mt-1 text-xs leading-5 text-[var(--text-muted)]">{reviewNote}</p>
      </div>
      <div className="grid gap-3 sm:grid-cols-2">
        {([
          { label: "Visual / identity QC", status: visual, warnings: report?.visualWarnings || [], attempts: report?.visualRetries || 0, retryLabel: "visual retries" },
          { label: "Speech / audio QC", status: audio, warnings: report?.audioWarnings || [], attempts: report?.audioRetakes || 0, retryLabel: "audio retakes" },
        ] as const).map((item) => (
          <div key={item.label} className="rounded-xl border border-[var(--border)] bg-[var(--panel-subtle)] p-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <span className="text-xs font-semibold text-[var(--text)]">{item.label}</span>
              <QCStatusPill status={item.status} />
            </div>
            {item.attempts > 0 && <p className="mt-2 text-[11px] text-[var(--text-muted)]">{item.attempts} {item.retryLabel}</p>}
            {item.warnings.length > 0 ? (
              <div className="mt-2 space-y-1">
                {item.warnings.slice(0, 3).map((warning, index) => <p key={index} className="break-words text-xs leading-5 text-[var(--text-muted)]">{warning}</p>)}
                {item.warnings.length > 3 && (
                  <details className="text-xs text-[var(--text-muted)]">
                    <summary className="cursor-pointer">Show {item.warnings.length - 3} more QC detail{item.warnings.length === 4 ? "" : "s"}</summary>
                    {item.warnings.slice(3).map((warning, index) => <p key={index} className="mt-1 break-words leading-5">{warning}</p>)}
                  </details>
                )}
              </div>
            ) : (
              <p className="mt-2 text-[11px] leading-5 text-[var(--text-muted)]">
                {item.status === "passed" ? "Inspection completed without a detected violation." :
                  item.status === "not_checked" ? "No separate inspection was performed." :
                  item.status === "unknown" ? "This output does not contain a saved QC report." :
                  item.status === "unavailable" ? "The quality-check provider could not complete this inspection." :
                  "Review the rendered video carefully."}
              </p>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

function FinalVideoCard({ video, aspectRatio }: { video: NonNullable<FinalVideo>; aspectRatio: AspectRatio }) {
  const qcFailed = video.qcReport?.visual === "failed" || video.qcReport?.audio === "failed";
  const qcUnavailable = video.qcReport?.visual === "unavailable" || video.qcReport?.audio === "unavailable";
  return (
    <div className="overflow-hidden rounded-3xl border border-[var(--border)] bg-[var(--panel-bg)]">
      <div className="flex flex-col gap-3 border-b border-[var(--border)] px-5 py-4 sm:flex-row sm:items-center sm:justify-between sm:px-6">
        <div>
          <div className="flex flex-wrap items-center gap-2 text-sm font-medium text-[var(--text-strong)]">Final video
            <span className="rounded-full bg-emerald-500/10 px-2.5 py-1 text-[10px] text-[var(--accent-text)]">Rendered</span>
            {qcFailed ? <QCStatusPill status="failed" /> : qcUnavailable ? <QCStatusPill status="unavailable" /> : null}
          </div>
          <div className="mt-1 text-xs text-[var(--text-muted)]">{video.label}</div>
          <div className="mt-1 text-[11px] text-[var(--text-faint)]">{video.dimensions} · {video.hasAudio ? `audio ${video.audioCodec || "present"}` : "no audio stream"}{video.gpu ? ` · ${video.gpu}` : ""}{video.estimatedCostUsd != null ? ` · est. $${video.estimatedCostUsd.toFixed(4)}` : ""}</div>
        </div>
        <div className="flex flex-wrap gap-2">
          {video.youtubeUrl && <a href={video.youtubeUrl} target="_blank" rel="noreferrer" className="flex h-10 items-center justify-center rounded-xl border border-red-500/20 bg-red-500/[0.06] px-4 text-xs font-medium text-[var(--danger-text)]">Open YouTube · {video.youtubePrivacy}</a>}
          <a href={video.downloadUrl} className="flex h-10 items-center justify-center gap-2 rounded-xl bg-[var(--primary-bg)] px-4 text-xs font-medium text-[var(--primary-fg)] hover:bg-[var(--primary-hover)]"><DownloadIcon /> Download MP4</a>
        </div>
      </div>
      <div className={`mx-auto bg-black ${aspectRatio === "9:16" ? "max-w-[430px]" : aspectRatio === "1:1" ? "max-w-[760px]" : "w-full"}`}><StudioVideo src={video.url} controls playsInline className={`w-full object-contain ${aspectClass(aspectRatio)}`} /></div>
      <VideoQCPanel report={video.qcReport} />
      <div className="border-t border-[var(--border)] px-5 py-4 text-[11px] leading-5 text-[var(--text-muted)] sm:px-6">{video.qualityNote}</div>
    </div>
  );
}
