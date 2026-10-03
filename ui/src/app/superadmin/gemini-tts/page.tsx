"use client";

import { useCallback, useEffect, useState } from "react";

import {
    createGeminiTtsSamplePackApiV1SuperuserGeminiTtsSamplePacksPost,
    generateGeminiTtsSampleVoiceApiV1SuperuserGeminiTtsSamplePacksPackIdVoicesVoiceIdGeneratePost,
    getGeminiTtsSamplePlaybackUrlApiV1SuperuserGeminiTtsSamplePacksPackIdAssetsAssetIdPlaybackUrlGet,
    listGeminiTtsSamplePacksApiV1SuperuserGeminiTtsSamplePacksGet,
    retryGeminiTtsSamplePackApiV1SuperuserGeminiTtsSamplePacksPackIdRetryPost,
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

const isActive = (asset?: GeminiTtsSampleAssetResponse) =>
    asset?.status === "queued" || asset?.status === "running";

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

    const createPack = async (event: React.FormEvent) => {
        event.preventDefault();
        const voice = form.voice_id.trim();
        try {
            const response = await createGeminiTtsSamplePackApiV1SuperuserGeminiTtsSamplePacksPost({
                body: { ...form, voice_id: voice || undefined },
            });
            if (response.error) throw new Error(describeSampleError(response.error));
            setMessage({ text: voice ? `${voice} queued.` : "Batch generation queued.", isError: false });
            await load();
        } catch (error) {
            fail(error);
        }
    };

    const generate = async (pack: GeminiTtsSamplePackResponse, voiceId: string, regenerate: boolean) => {
        if (
            regenerate &&
            !window.confirm("Regenerate this one voice? This sends a new Google TTS request and may incur usage.")
        ) {
            return;
        }
        try {
            const response =
                await generateGeminiTtsSampleVoiceApiV1SuperuserGeminiTtsSamplePacksPackIdVoicesVoiceIdGeneratePost({
                    path: { pack_id: pack.id, voice_id: voiceId },
                    body: { regenerate },
                });
            if (response.error) throw new Error(describeSampleError(response.error));
            setMessage({ text: `${voiceId} queued.`, isError: false });
            await load();
        } catch (error) {
            fail(error);
        }
    };

    // Failed voices are recovered by the pack-level retry (the API refuses a
    // plain per-voice generate for them), which only touches failed voices and
    // never replaces a playable sample.
    const retryFailed = async (pack: GeminiTtsSamplePackResponse) => {
        const count = pack.eligible_voices ?? 0;
        if (
            !window.confirm(
                `Retry ${count} failed voice${count === 1 ? "" : "s"}? Each sends a new Google TTS request and may incur usage.`,
            )
        ) {
            return;
        }
        try {
            const response = await retryGeminiTtsSamplePackApiV1SuperuserGeminiTtsSamplePacksPackIdRetryPost({
                path: { pack_id: pack.id },
            });
            if (response.error) throw new Error(describeSampleError(response.error));
            const queued = response.data?.retry_summary?.enqueued_jobs ?? count;
            setMessage({ text: `${queued} failed voice${queued === 1 ? "" : "s"} queued for retry.`, isError: false });
            await load();
        } catch (error) {
            fail(error);
            await load().catch(() => undefined);
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

    return (
        <main className="container mx-auto max-w-6xl space-y-6 p-6">
            <div>
                <h1 className="text-2xl font-semibold">Gemini TTS samples</h1>
                <p className="text-sm text-muted-foreground">
                    Batch-generate packs or audition one canonical voice at a time. Current versions are played by
                    default.
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
                            {form.voice_id.trim() ? "Generate selected voice" : "Generate all voices"}
                        </Button>
                    </form>
                </CardContent>
            </Card>
            {packs.map((pack) => {
                const voiceIds = Array.from(new Set(pack.assets.map((asset) => asset.voice_id)));
                const eligible = pack.eligible_voices ?? 0;
                return (
                    <Card key={pack.id}>
                        <CardHeader>
                            <CardTitle>
                                Pack #{pack.id} · {pack.model_id} · {pack.status}
                            </CardTitle>
                            <p className="text-sm text-muted-foreground">
                                {pack.language} · {pack.location} · catalog {pack.catalog_revision}
                            </p>
                        </CardHeader>
                        <CardContent className="space-y-2">
                            {voiceIds.map((voiceId) => {
                                const versions = pack.assets.filter((asset) => asset.voice_id === voiceId);
                                const current = versions.find((asset) => asset.is_current && asset.status === "completed");
                                const latest = versions[versions.length - 1];
                                const failedOnly = !current && latest?.status === "failed";
                                const src = current ? (freshUrls[current.id] ?? current.sample_url ?? undefined) : undefined;
                                return (
                                    <div key={voiceId} className="flex flex-wrap items-center gap-3 rounded border p-3">
                                        <span className="w-24 font-medium">{voiceId}</span>
                                        {current && src ? (
                                            <audio
                                                controls
                                                src={src}
                                                className="h-8"
                                                aria-label={`${voiceId} sample`}
                                                onError={() => {
                                                    if (!freshUrls[current.id]) void refreshPlayback(pack, current);
                                                }}
                                            />
                                        ) : (
                                            <span className="text-sm text-muted-foreground">
                                                {latest?.status || "not generated"}
                                            </span>
                                        )}
                                        {failedOnly ? (
                                            <span className="text-sm text-destructive">
                                                {latest?.error_message || "Generation failed"} — use “Retry failed voices”
                                            </span>
                                        ) : (
                                            <Button
                                                size="sm"
                                                variant="outline"
                                                disabled={isActive(latest)}
                                                onClick={() => generate(pack, voiceId, Boolean(current))}
                                            >
                                                {current ? "Regenerate" : "Generate"}
                                            </Button>
                                        )}
                                        <details className="text-sm">
                                            <summary>History ({versions.length})</summary>
                                            {versions.map((version) => (
                                                <div key={version.id}>
                                                    v{version.version}: {version.status}
                                                    {version.is_current ? " · current" : ""}
                                                </div>
                                            ))}
                                        </details>
                                    </div>
                                );
                            })}
                            <Button variant="secondary" disabled={eligible === 0} onClick={() => retryFailed(pack)}>
                                Retry failed voices ({eligible})
                            </Button>
                        </CardContent>
                    </Card>
                );
            })}
        </main>
    );
}
