"use client";

import { useCallback, useEffect, useState } from "react";

import {
    createGeminiTtsSamplePackApiV1SuperuserGeminiTtsSamplePacksPost,
    generateGeminiTtsSampleVoiceApiV1SuperuserGeminiTtsSamplePacksPackIdVoicesVoiceIdGeneratePost,
    getGeminiTtsSamplePlaybackUrlApiV1SuperuserGeminiTtsSamplePacksPackIdAssetsAssetIdPlaybackUrlGet,
    listGeminiTtsSamplePacksApiV1SuperuserGeminiTtsSamplePacksGet,
} from "@/client/sdk.gen";
import type {
    GeminiTtsSampleAssetResponse,
    GeminiTtsSamplePackResponse,
} from "@/client/types.gen";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { useAuth } from "@/lib/auth";

import { describeSampleError } from "./sampleErrors";

const FORM_FIELDS = ["model_id", "voice_id", "location", "language", "style_text", "sample_text"] as const;
type FormKey = (typeof FORM_FIELDS)[number];

const PROVIDER = "google_vertex";
const PROVIDER_LABEL = "Google Vertex AI";
const POLL_MS = 4000;

type VoiceState =
    | { kind: "generating" }
    | { kind: "ready"; asset: GeminiTtsSampleAssetResponse }
    | { kind: "failed"; message: string }
    | { kind: "none" };

/**
 * One answer per voice: is its sample generated? Internal versions and attempts
 * are not shown. A playable sample always wins (a replacement that is running
 * or failed must not hide audio that still plays); otherwise the latest attempt
 * decides.
 */
function voiceState(versions: GeminiTtsSampleAssetResponse[]): VoiceState {
    const current = versions.find((asset) => asset.is_current && asset.status === "completed");
    if (current) return { kind: "ready", asset: current };
    const latest = versions[versions.length - 1];
    if (latest?.status === "queued" || latest?.status === "running") return { kind: "generating" };
    if (latest?.status === "failed") return { kind: "failed", message: latest.error_message || "Generation failed" };
    return { kind: "none" };
}

export default function GeminiTtsSamplesPage() {
    const auth = useAuth();
    const [packs, setPacks] = useState<GeminiTtsSamplePackResponse[]>([]);
    const [form, setForm] = useState<Record<FormKey, string>>({
        model_id: "gemini-3.1-flash-tts-preview",
        voice_id: "",
        location: "global",
        language: "en-US",
        style_text: "warm and conversational",
        sample_text: "Hello, this is a voice sample.",
    });
    const [message, setMessage] = useState<{ text: string; isError: boolean } | null>(null);
    // Voices with a request in flight in this tab: a second click is ignored
    // here, and the server refuses a duplicate from another tab.
    const [busy, setBusy] = useState<Set<string>>(new Set());
    // Signed URLs expire after an hour: a fresh one replaces a stale one on error.
    const [freshUrls, setFreshUrls] = useState<Record<number, string>>({});

    const load = useCallback(async () => {
        const response = await listGeminiTtsSamplePacksApiV1SuperuserGeminiTtsSamplePacksGet();
        if (response.error) throw new Error(describeSampleError(response.error, "Failed to load sample packs"));
        setPacks(response.data ?? []);
    }, []);

    const fail = (error: unknown) =>
        setMessage({ text: error instanceof Error ? error.message : "Request failed", isError: true });

    useEffect(() => {
        if (!auth.isAuthenticated) return;
        load().catch(fail);
    }, [auth.isAuthenticated, load]);

    const anyGenerating = packs.some((pack) =>
        Array.from(new Set(pack.assets.map((asset) => asset.voice_id))).some(
            (voiceId) => voiceState(pack.assets.filter((asset) => asset.voice_id === voiceId)).kind === "generating",
        ),
    );
    useEffect(() => {
        if (!anyGenerating) return;
        const timer = setInterval(() => load().catch(() => undefined), POLL_MS);
        return () => clearInterval(timer);
    }, [anyGenerating, load]);

    const createPack = async (event: React.FormEvent) => {
        event.preventDefault();
        const voice = form.voice_id.trim();
        try {
            const response = await createGeminiTtsSamplePackApiV1SuperuserGeminiTtsSamplePacksPost({
                body: { ...form, voice_id: voice || undefined },
            });
            if (response.error) throw new Error(describeSampleError(response.error));
            setMessage({ text: voice ? `${voice} is generating.` : "Batch generation started.", isError: false });
            await load();
        } catch (error) {
            fail(error);
        }
    };

    // Generate a voice that has no sample, or retry one whose generation failed.
    // Only that voice is requested: other voices, and their playable samples,
    // are never touched. (The API takes regenerate=true for a failed voice.)
    const generate = async (pack: GeminiTtsSamplePackResponse, voiceId: string, isRetry: boolean) => {
        const key = `${pack.id}:${voiceId}`;
        if (busy.has(key)) return;
        setBusy((current) => new Set(current).add(key));
        try {
            const response =
                await generateGeminiTtsSampleVoiceApiV1SuperuserGeminiTtsSamplePacksPackIdVoicesVoiceIdGeneratePost({
                    path: { pack_id: pack.id, voice_id: voiceId },
                    body: { regenerate: isRetry },
                });
            if (response.error) {
                if (response.response?.status === 409) {
                    // Another tab (or an earlier click) already started it.
                    setMessage({ text: `${voiceId} is already generating.`, isError: false });
                    await load();
                    return;
                }
                throw new Error(describeSampleError(response.error));
            }
            setMessage({ text: `${voiceId} is generating.`, isError: false });
            await load();
        } catch (error) {
            fail(error);
        } finally {
            setBusy((current) => {
                const next = new Set(current);
                next.delete(key);
                return next;
            });
        }
    };

    const refreshPlayback = async (pack: GeminiTtsSamplePackResponse, asset: GeminiTtsSampleAssetResponse) => {
        try {
            const response =
                await getGeminiTtsSamplePlaybackUrlApiV1SuperuserGeminiTtsSamplePacksPackIdAssetsAssetIdPlaybackUrlGet({
                    path: { pack_id: pack.id, asset_id: asset.id },
                });
            if (response.error) throw new Error(describeSampleError(response.error, "Playback unavailable"));
            const url = (response.data as { url?: string } | undefined)?.url;
            if (!url) throw new Error("Playback unavailable");
            setFreshUrls((urls) => ({ ...urls, [asset.id]: url }));
        } catch (error) {
            fail(error);
        }
    };

    // Only samples of the selected provider, model and (when given) voice are
    // shown, so a previous selection's audio is never presented as this one's.
    const selectedModel = form.model_id.trim();
    const selectedVoice = form.voice_id.trim();
    const visiblePacks = packs.filter((pack) => pack.provider === PROVIDER && pack.model_id === selectedModel);

    return (
        <main className="container mx-auto max-w-6xl space-y-6 p-6">
            <div>
                <h1 className="text-2xl font-semibold">Gemini TTS samples</h1>
                <p className="text-sm text-muted-foreground">
                    Generate and audition voice samples. Each voice is either generating, ready to play, or failed
                    with a retry. Generation is billable.
                </p>
            </div>
            {message && (
                <p role={message.isError ? "alert" : "status"} className="rounded-md border p-3 text-sm whitespace-pre-line">
                    {message.text}
                </p>
            )}
            <Card>
                <CardHeader>
                    <CardTitle>Generate all voices</CardTitle>
                </CardHeader>
                <CardContent>
                    <form onSubmit={createPack} className="grid gap-4 md:grid-cols-2">
                        <div className="space-y-1">
                            <Label>Provider</Label>
                            <p className="rounded border bg-muted p-2 text-sm">{PROVIDER_LABEL}</p>
                        </div>
                        {FORM_FIELDS.map((key) => (
                            <div key={key} className="space-y-1">
                                <Label htmlFor={key}>
                                    {key}
                                    {key === "voice_id" ? " (optional; omit for all 30)" : ""}
                                </Label>
                                <Input
                                    id={key}
                                    value={form[key]}
                                    onChange={(event) => setForm({ ...form, [key]: event.target.value })}
                                    required={key !== "voice_id"}
                                />
                            </div>
                        ))}
                        <Button type="submit" className="md:col-span-2">
                            {selectedVoice ? "Generate selected voice" : "Generate all voices"}
                        </Button>
                    </form>
                </CardContent>
            </Card>
            {visiblePacks.length === 0 && (
                <p className="text-sm text-muted-foreground">
                    No samples yet for {PROVIDER_LABEL} · {selectedModel || "this model"}.
                </p>
            )}
            {visiblePacks.map((pack) => {
                const voiceIds = Array.from(new Set(pack.assets.map((asset) => asset.voice_id))).filter(
                    (voiceId) => !selectedVoice || voiceId === selectedVoice,
                );
                return (
                    <Card key={pack.id}>
                        <CardHeader>
                            <CardTitle>
                                {PROVIDER_LABEL} · {pack.model_id}
                            </CardTitle>
                            <p className="text-sm text-muted-foreground">
                                {pack.language} · {pack.location} · {pack.style_text}
                            </p>
                        </CardHeader>
                        <CardContent className="space-y-2">
                            {voiceIds.map((voiceId) => {
                                const state = voiceState(pack.assets.filter((asset) => asset.voice_id === voiceId));
                                const inFlight = busy.has(`${pack.id}:${voiceId}`);
                                const src =
                                    state.kind === "ready"
                                        ? (freshUrls[state.asset.id] ?? state.asset.sample_url ?? undefined)
                                        : undefined;
                                return (
                                    <div key={voiceId} className="flex flex-wrap items-center gap-3 rounded border p-3">
                                        <span className="w-24 font-medium">{voiceId}</span>
                                        {state.kind === "ready" && (
                                            <>
                                                <span className="text-sm text-green-700">Ready</span>
                                                {src && (
                                                    <audio
                                                        controls
                                                        src={src}
                                                        className="h-8"
                                                        aria-label={`${voiceId} sample`}
                                                        onError={() => {
                                                            if (!freshUrls[state.asset.id]) void refreshPlayback(pack, state.asset);
                                                        }}
                                                    />
                                                )}
                                            </>
                                        )}
                                        {state.kind === "generating" && (
                                            <span className="text-sm text-muted-foreground">Generating…</span>
                                        )}
                                        {state.kind === "none" && (
                                            <>
                                                <span className="text-sm text-muted-foreground">Not generated</span>
                                                <Button size="sm" variant="outline" disabled={inFlight} onClick={() => generate(pack, voiceId, false)}>
                                                    Generate
                                                </Button>
                                            </>
                                        )}
                                        {state.kind === "failed" && (
                                            <>
                                                <span className="text-sm text-destructive">Failed: {state.message}</span>
                                                <Button size="sm" variant="outline" disabled={inFlight} onClick={() => generate(pack, voiceId, true)}>
                                                    Retry
                                                </Button>
                                            </>
                                        )}
                                    </div>
                                );
                            })}
                        </CardContent>
                    </Card>
                );
            })}
        </main>
    );
}
