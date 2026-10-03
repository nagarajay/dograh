import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import GeminiTtsSamplesPage from './page';

const m = vi.hoisted(() => ({
    list: vi.fn(),
    create: vi.fn(),
    generate: vi.fn(),
    playback: vi.fn(),
    useAuth: vi.fn(),
}));

vi.mock('@/client/sdk.gen', () => ({
    listGeminiTtsSamplePacksApiV1SuperuserGeminiTtsSamplePacksGet: m.list,
    createGeminiTtsSamplePackApiV1SuperuserGeminiTtsSamplePacksPost: m.create,
    generateGeminiTtsSampleVoiceApiV1SuperuserGeminiTtsSamplePacksPackIdVoicesVoiceIdGeneratePost: m.generate,
    getGeminiTtsSamplePlaybackUrlApiV1SuperuserGeminiTtsSamplePacksPackIdAssetsAssetIdPlaybackUrlGet: m.playback,
}));
vi.mock('@/lib/auth', () => ({ useAuth: m.useAuth }));

type Asset = Record<string, unknown>;
const asset = (over: Asset): Asset => ({
    id: 1, voice_id: 'Achernar', gender: 'Female', version: 1, is_current: false,
    status: 'completed', attempts: 1, ...over,
});
const pack = (assets: Asset[], over: Record<string, unknown> = {}) => ({
    id: 5, provider: 'google_vertex', model_id: 'gemini-3.1-flash-tts-preview',
    catalog_revision: 'r1', location: 'global', language: 'en-US', style_text: 'warm',
    sample_text: 'hi', request_fingerprint: 'f', status: 'partial', total_assets: assets.length,
    completed_assets: 0, failed_assets: 0, eligible_voices: 0, assets, ...over,
});
const ready = (voice: string, id: number, url = `https://s/${voice}.wav`): Asset =>
    asset({ id, voice_id: voice, is_current: true, sample_url: url });
const failed = (voice: string, id: number, over: Asset = {}): Asset =>
    asset({ id, voice_id: voice, status: 'failed', error_message: 'quota: rate limited', ...over });

const show = async (packs: unknown[]) => {
    m.list.mockResolvedValue({ data: packs });
    render(<GeminiTtsSamplesPage />);
    await waitFor(() => expect(m.list).toHaveBeenCalled());
};

describe('GeminiTtsSamplesPage', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        m.useAuth.mockReturnValue({ isAuthenticated: true });
    });
    afterEach(() => {
        cleanup();
        vi.useRealTimers();
        vi.restoreAllMocks();
    });

    it('waits for authentication before loading', () => {
        m.useAuth.mockReturnValue({ isAuthenticated: false });
        render(<GeminiTtsSamplesPage />);
        expect(m.list).not.toHaveBeenCalled();
    });

    describe('one simple state per voice', () => {
        it('shows ready / generating / failed / not generated, and nothing about versions or attempts', async () => {
            await show([
                pack([
                    ready('Achernar', 1),
                    asset({ id: 2, voice_id: 'Achird', status: 'running' }),
                    failed('Algenib', 3),
                    // Two earlier attempts and a replacement for a ready voice: all hidden.
                    failed('Algieba', 4, { version: 1 }),
                    ready('Algieba', 5),
                    failed('Algieba', 6, { version: 3 }),
                ]),
            ]);
            const text = document.body.textContent ?? '';
            expect(text).toContain('Ready');
            expect(text).toContain('Generating…');
            expect(text).toContain('Failed: quota: rate limited');
            expect(screen.getAllByLabelText(/sample$/)).toHaveLength(2); // Achernar + Algieba
            expect(text).not.toMatch(/History|version|v\d|attempt|Regenerate|Retry failed voices/i);
            expect(screen.queryByRole('button', { name: /regenerate/i })).toBeNull();
            expect(screen.queryByRole('button', { name: /retry failed voices/i })).toBeNull();
        });

        it('keeps a playable sample visible while a replacement runs or after it failed', async () => {
            await show([pack([ready('Achernar', 1), asset({ id: 2, version: 2, status: 'queued' })])]);
            expect(screen.getByLabelText('Achernar sample').getAttribute('src')).toBe('https://s/Achernar.wav');
            expect(screen.queryByText('Generating…')).toBeNull();
        });

        it('offers Generate only for a voice with no sample', async () => {
            await show([pack([asset({ id: 9, voice_id: 'Puck', status: 'completed', is_current: false })])]);
            m.generate.mockResolvedValue({ data: pack([]) });
            fireEvent.click(screen.getByRole('button', { name: 'Generate' }));
            await waitFor(() =>
                expect(m.generate).toHaveBeenCalledWith({ path: { pack_id: 5, voice_id: 'Puck' }, body: { regenerate: false } }),
            );
        });
    });

    describe('individual retry of a failed voice', () => {
        it('requests only that voice; successful samples are not requested again', async () => {
            await show([pack([ready('Achernar', 1), failed('Achird', 2), failed('Algenib', 3)])]);
            m.generate.mockResolvedValue({ data: pack([]) });
            const retries = screen.getAllByRole('button', { name: 'Retry' });
            expect(retries).toHaveLength(2); // one per failed voice, none for the ready one
            fireEvent.click(retries[0]);
            await waitFor(() => expect(m.generate).toHaveBeenCalledTimes(1));
            expect(m.generate).toHaveBeenCalledWith({ path: { pack_id: 5, voice_id: 'Achird' }, body: { regenerate: true } });
            expect(screen.getByLabelText('Achernar sample').getAttribute('src')).toBe('https://s/Achernar.wav');
            expect((await screen.findByRole('status')).textContent).toContain('Achird is generating.');
        });

        it('reloads so the voice shows Generating… after a retry', async () => {
            await show([pack([failed('Achird', 2)])]);
            m.generate.mockResolvedValue({ data: pack([]) });
            m.list.mockResolvedValue({ data: [pack([failed('Achird', 2), asset({ id: 3, voice_id: 'Achird', version: 2, status: 'queued' })])] });
            fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
            expect(await screen.findByText('Generating…')).toBeTruthy();
            expect(screen.queryByRole('button', { name: 'Retry' })).toBeNull();
        });
    });

    describe('duplicate generation', () => {
        it('ignores repeated clicks while the request is in flight', async () => {
            await show([pack([failed('Achird', 2)])]);
            let finish: (v: unknown) => void = () => undefined;
            m.generate.mockReturnValue(new Promise((resolve) => (finish = resolve)));
            const button = screen.getByRole('button', { name: 'Retry' });
            fireEvent.click(button);
            fireEvent.click(button);
            fireEvent.click(button);
            expect(m.generate).toHaveBeenCalledTimes(1);
            expect((screen.getByRole('button', { name: 'Retry' }) as HTMLButtonElement).disabled).toBe(true);
            await act(async () => finish({ data: pack([]) }));
        });

        it('treats a 409 from another tab as already generating, not as an error', async () => {
            await show([pack([failed('Achird', 2)])]);
            m.generate.mockResolvedValue({ error: { detail: "voice 'Achird' is already queued" }, response: { status: 409 } });
            m.list.mockResolvedValue({ data: [pack([failed('Achird', 2), asset({ id: 3, voice_id: 'Achird', version: 2, status: 'running' })])] });
            fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
            expect((await screen.findByRole('status')).textContent).toContain('already generating');
            expect(screen.queryByRole('alert')).toBeNull();
            expect(await screen.findByText('Generating…')).toBeTruthy();
        });

        it('polls while something is generating and stops when done', async () => {
            vi.useFakeTimers();
            m.list.mockResolvedValue({ data: [pack([asset({ id: 2, voice_id: 'Achird', status: 'running' })])] });
            render(<GeminiTtsSamplesPage />);
            await act(async () => undefined);
            expect(m.list).toHaveBeenCalledTimes(1);
            m.list.mockResolvedValue({ data: [pack([ready('Achird', 2)])] });
            await act(async () => vi.advanceTimersByTimeAsync(4100));
            expect(m.list).toHaveBeenCalledTimes(2);
            expect(screen.getByLabelText('Achird sample')).toBeTruthy();
            await act(async () => vi.advanceTimersByTimeAsync(20000));
            expect(m.list).toHaveBeenCalledTimes(2); // nothing generating: no more polling
        });
    });

    describe('selected provider, model and voice', () => {
        const other = pack([ready('Achernar', 1, 'https://s/old-model.wav')], { id: 8, model_id: 'gemini-2.5-flash-tts' });
        const wrongProvider = pack([ready('Achernar', 2, 'https://s/other-provider.wav')], { id: 9, provider: 'elevenlabs' });
        const selected = pack([ready('Achernar', 3, 'https://s/selected.wav'), ready('Puck', 4, 'https://s/puck.wav')]);

        it("never shows another model's or provider's sample as the current selection", async () => {
            await show([other, wrongProvider, selected]);
            const sources = screen.getAllByLabelText(/sample$/).map((el) => el.getAttribute('src'));
            expect(sources).toEqual(['https://s/selected.wav', 'https://s/puck.wav']);
        });

        it('follows the model the operator selects', async () => {
            await show([other, selected]);
            fireEvent.change(screen.getByLabelText(/model_id/), { target: { value: 'gemini-2.5-flash-tts' } });
            expect(screen.getAllByLabelText(/sample$/).map((el) => el.getAttribute('src'))).toEqual(['https://s/old-model.wav']);
            fireEvent.change(screen.getByLabelText(/model_id/), { target: { value: 'gemini-9-unknown' } });
            expect(screen.queryAllByLabelText(/sample$/)).toHaveLength(0);
            expect(screen.getByText(/No samples yet for Google Vertex AI · gemini-9-unknown/)).toBeTruthy();
        });

        it('shows only the selected voice', async () => {
            await show([selected]);
            fireEvent.change(screen.getByLabelText(/voice_id/), { target: { value: 'Puck' } });
            expect(screen.getAllByLabelText(/sample$/).map((el) => el.getAttribute('src'))).toEqual(['https://s/puck.wav']);
            expect(screen.queryByText('Achernar')).toBeNull();
        });
    });

    describe('error display', () => {
        it.each([
            ['a string detail', { detail: 'sample pack not found' }, 'sample pack not found'],
            ['an object detail', { detail: { message: 'Some sample jobs could not be queued.' } }, 'Some sample jobs could not be queued.'],
            ['a validation array', { detail: [{ loc: ['body', 'style_text'], msg: 'Field required' }] }, 'Field required'],
        ])('shows %s as text, never [object Object]', async (_name, error, expected) => {
            m.list.mockResolvedValue({ error });
            render(<GeminiTtsSamplesPage />);
            const alert = await screen.findByRole('alert');
            expect(alert.textContent).toContain(expected);
            expect(alert.textContent).not.toContain('[object Object]');
        });

        it('shows a non-conflict generate failure', async () => {
            await show([pack([failed('Achird', 2)])]);
            m.generate.mockResolvedValue({ error: { detail: 'sample storage unavailable' }, response: { status: 503 } });
            fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
            expect((await screen.findByRole('alert')).textContent).toContain('sample storage unavailable');
            expect((screen.getByRole('button', { name: 'Retry' }) as HTMLButtonElement).disabled).toBe(false);
        });

        it('shows a create failure with its validation messages', async () => {
            await show([]);
            m.create.mockResolvedValue({ error: { detail: [{ loc: ['body', 'sample_text'], msg: 'String should have at least 1 character' }] } });
            fireEvent.click(screen.getByRole('button', { name: 'Generate all voices' }));
            expect((await screen.findByRole('alert')).textContent).toContain('at least 1 character');
        });

        it('sends the selected voice, or none for a whole batch', async () => {
            await show([]);
            m.create.mockResolvedValue({ data: pack([]) });
            fireEvent.click(screen.getByRole('button', { name: 'Generate all voices' }));
            await waitFor(() => expect(m.create).toHaveBeenCalledTimes(1));
            expect(m.create.mock.calls[0][0].body.voice_id).toBeUndefined();
            fireEvent.change(screen.getByLabelText(/voice_id/), { target: { value: ' Puck ' } });
            fireEvent.click(screen.getByRole('button', { name: 'Generate selected voice' }));
            await waitFor(() => expect(m.create).toHaveBeenCalledTimes(2));
            expect(m.create.mock.calls[1][0].body.voice_id).toBe('Puck');
        });
    });

    describe('playback', () => {
        it('fetches a fresh signed URL when the stored one has expired', async () => {
            await show([pack([ready('Achernar', 7, 'https://s/old.wav')])]);
            m.playback.mockResolvedValue({ data: { asset_id: 7, mime_type: 'audio/wav', url: 'https://s/new.wav' } });
            fireEvent.error(screen.getByLabelText('Achernar sample'));
            await waitFor(() => expect(m.playback).toHaveBeenCalledWith({ path: { pack_id: 5, asset_id: 7 } }));
            await waitFor(() => expect(screen.getByLabelText('Achernar sample').getAttribute('src')).toBe('https://s/new.wav'));
        });

        it('shows why playback is unavailable', async () => {
            await show([pack([ready('Achernar', 7)])]);
            m.playback.mockResolvedValue({ error: { detail: 'sample storage unavailable' } });
            fireEvent.error(screen.getByLabelText('Achernar sample'));
            expect((await screen.findByRole('alert')).textContent).toContain('sample storage unavailable');
        });
    });
});
