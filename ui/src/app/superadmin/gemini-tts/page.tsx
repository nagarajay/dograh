"use client";

import { useCallback, useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { resolveBrowserBackendUrl } from "@/lib/apiClient";
import { useAuth } from "@/lib/auth";

type Asset = {
    id: number; voice_id: string; gender: string; status: string; version: number;
    is_current: boolean; sample_url?: string | null; error_message?: string | null;
};
type Pack = { id: number; model_id: string; catalog_revision: string; location: string;
    language: string; style_text: string; sample_text: string; status: string; assets: Asset[] };

const API_PATH = "/api/v1/superuser/gemini-tts/sample-packs";

export default function GeminiTtsSamplesPage() {
    const { getAccessToken } = useAuth();
    const [packs, setPacks] = useState<Pack[]>([]);
    const [form, setForm] = useState({ model_id: "gemini-3.1-flash-tts-preview", voice_id: "", location: "global", language: "en-US", style_text: "warm and conversational", sample_text: "Hello, this is a voice sample." });
    const [message, setMessage] = useState<string | null>(null);

    const request = useCallback(async (path: string, init?: RequestInit) => {
        const token = await getAccessToken();
        const response = await fetch(`${resolveBrowserBackendUrl()}${API_PATH}${path}`, {
            ...init,
            headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}`, ...(init?.headers || {}) },
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || "Request failed");
        return payload;
    }, [getAccessToken]);

    const load = useCallback(async () => setPacks(await request("")), [request]);
    useEffect(() => { load().catch((error) => setMessage(error.message)); }, [load]);

    const createPack = async (event: React.FormEvent) => {
        event.preventDefault();
        try { const body = { ...form, voice_id: form.voice_id.trim() || undefined }; await request("", { method: "POST", body: JSON.stringify(body) }); setMessage(form.voice_id.trim() ? `${form.voice_id} queued.` : "Batch generation queued."); await load(); }
        catch (error) { setMessage(error instanceof Error ? error.message : "Request failed"); }
    };

    const generate = async (pack: Pack, voice: Asset, regenerate: boolean) => {
        if (regenerate && !window.confirm("Regenerate this one voice? This sends a new Google TTS request and may incur usage.")) return;
        try { await request(`/${pack.id}/voices/${encodeURIComponent(voice.voice_id)}/generate`, { method: "POST", body: JSON.stringify({ regenerate }) }); setMessage(`${voice.voice_id} queued.`); await load(); }
        catch (error) { setMessage(error instanceof Error ? error.message : "Request failed"); }
    };

    return <main className="container mx-auto max-w-6xl space-y-6 p-6">
        <div><h1 className="text-2xl font-semibold">Gemini TTS samples</h1><p className="text-sm text-muted-foreground">Batch-generate packs or audition one canonical voice at a time. Current versions are played by default.</p></div>
        {message && <p className="rounded-md border p-3 text-sm">{message}</p>}
        <Card><CardHeader><CardTitle>Generate all voices</CardTitle></CardHeader><CardContent><form onSubmit={createPack} className="grid gap-4 md:grid-cols-2">
            {(["model_id", "voice_id", "location", "language", "style_text", "sample_text"] as const).map((key) => <div key={key} className="space-y-1"><Label htmlFor={key}>{key}{key === "voice_id" ? " (optional; omit for all 30)" : ""}</Label><Input id={key} value={form[key]} onChange={(event) => setForm({ ...form, [key]: event.target.value })} required={key !== "voice_id"} /></div>)}
            <Button type="submit" className="md:col-span-2">{form.voice_id.trim() ? "Generate selected voice" : "Generate all voices"}</Button>
        </form></CardContent></Card>
        {packs.map((pack) => <Card key={pack.id}><CardHeader><CardTitle>Pack #{pack.id} · {pack.model_id} · {pack.status}</CardTitle><p className="text-sm text-muted-foreground">{pack.language} · {pack.location} · catalog {pack.catalog_revision}</p></CardHeader><CardContent className="space-y-2">
            {Array.from(new Set(pack.assets.map((asset) => asset.voice_id))).map((voiceId) => { const versions = pack.assets.filter((asset) => asset.voice_id === voiceId); const current = versions.find((asset) => asset.is_current); const latest = versions[versions.length - 1]; return <div key={voiceId} className="flex flex-wrap items-center gap-3 rounded border p-3"><span className="w-24 font-medium">{voiceId}</span>{current?.sample_url ? <audio controls src={current.sample_url} className="h-8" /> : <span className="text-sm text-muted-foreground">{latest?.status || "not generated"}</span>}<Button size="sm" variant="outline" disabled={latest?.status === "queued" || latest?.status === "running"} onClick={() => generate(pack, latest || ({ voice_id: voiceId } as Asset), Boolean(current))}>{current ? "Regenerate" : "Generate"}</Button><details className="text-sm"><summary>History ({versions.length})</summary>{versions.map((version) => <div key={version.id}>v{version.version}: {version.status}{version.is_current ? " · current" : ""}</div>)}</details></div>; })}
            <Button variant="secondary" onClick={async () => { try { await request(`/${pack.id}/retry`, { method: "POST" }); setMessage("Failed assets queued for retry."); await load(); } catch (error) { setMessage(error instanceof Error ? error.message : "Request failed"); } }}>Retry failed assets only</Button>
        </CardContent></Card>)}
    </main>;
}
