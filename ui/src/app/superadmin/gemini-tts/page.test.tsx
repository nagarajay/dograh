import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import GeminiTtsSamplesPage from './page';

const m = vi.hoisted(() => ({
    list: vi.fn(),
    create: vi.fn(),
    generate: vi.fn(),
    retry: vi.fn(),
    playback: vi.fn(),
    useAuth: vi.fn(),
}));

vi.mock('@/client/sdk.gen', () => ({
    listGeminiTtsSamplePacksApiV1SuperuserGeminiTtsSamplePacksGet: m.list,
    createGeminiTtsSamplePackApiV1SuperuserGeminiTtsSamplePacksPost: m.create,
    generateGeminiTtsSampleVoiceApiV1SuperuserGeminiTtsSamplePacksPackIdVoicesVoiceIdGeneratePost: m.generate,
    retryGeminiTtsSamplePackApiV1SuperuserGeminiTtsSamplePacksPackIdRetryPost: m.retry,
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

const show = async (packs: unknown[]) => {
    m.list.mockResolvedValue({ data: packs });
    render(<GeminiTtsSamplesPage />);
    await waitFor(() => expect(m.list).toHaveBeenCalled());
};

describe('GeminiTtsSamplesPage', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        m.useAuth.mockReturnValue({ isAuthenticated: true });
        vi.spyOn(window, 'confirm').mockReturnValue(true);
    });
    afterEach(() => {
        cleanup();
        vi.restoreAllMocks();
    });

    it('waits for authentication before loading', () => {
        m.useAuth.mockReturnValue({ isAuthenticated: false });
        render(<GeminiTtsSamplesPage />);
        expect(m.list).not.toHaveBeenCalled();
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

    describe('failed voices and retry', () => {
        const failedPack = () =>
            pack([asset({ id: 1, status: 'failed', error_message: 'quota: rate limited' })], { eligible_voices: 1, failed_assets: 1 });

        it('offers the recovery operation, not a per-voice Generate that the API would refuse', async () => {
            await show([failedPack()]);
            expect(screen.queryByRole('button', { name: 'Generate' })).toBeNull();
            expect(screen.getByText(/quota: rate limited/)).toBeTruthy();
            m.retry.mockResolvedValue({ data: { ...failedPack(), retry_summary: { enqueued_jobs: 1 } } });
            fireEvent.click(screen.getByRole('button', { name: 'Retry failed voices (1)' }));
            await waitFor(() => expect(m.retry).toHaveBeenCalledWith({ path: { pack_id: 5 } }));
            expect(m.generate).not.toHaveBeenCalled();
            expect((await screen.findByRole('status')).textContent).toContain('1 failed voice queued for retry.');
        });

        it('does nothing when the operator declines the cost confirmation', async () => {
            vi.spyOn(window, 'confirm').mockReturnValue(false);
            await show([failedPack()]);
            fireEvent.click(screen.getByRole('button', { name: 'Retry failed voices (1)' }));
            expect(m.retry).not.toHaveBeenCalled();
        });

        it('disables retry when no voice is eligible', async () => {
            await show([pack([asset({ is_current: true, sample_url: 'https://s/a.wav' })])]);
            expect((screen.getByRole('button', { name: /Retry failed voices/ }) as HTMLButtonElement).disabled).toBe(true);
        });

        it('reports a partially queued retry with its counts', async () => {
            await show([failedPack()]);
            m.retry.mockResolvedValue({
                error: { detail: { message: 'Some sample jobs could not be queued; poll the pack.', retry_summary: { selected_voices: 3, enqueued_jobs: 2, enqueue_failures: 1, enqueue_conflicts: 0 } } },
            });
            fireEvent.click(screen.getByRole('button', { name: 'Retry failed voices (1)' }));
            const alert = await screen.findByRole('alert');
            expect(alert.textContent).toContain('Some sample jobs could not be queued');
            expect(alert.textContent).toContain('2 of 3 queued, 1 failed, 0 conflicting');
            expect(alert.textContent).not.toContain('[object Object]');
        });
    });

    describe('generation and conflicts', () => {
        it('generates a voice that has no sample yet without regenerate', async () => {
            await show([pack([asset({ id: 9, voice_id: 'Puck', status: 'completed', is_current: false, version: 1 })])]);
            // A completed-but-not-current version has no playable sample: plain Generate.
            m.generate.mockResolvedValue({ data: pack([]) });
            fireEvent.click(screen.getByRole('button', { name: 'Generate' }));
            await waitFor(() => expect(m.generate).toHaveBeenCalledWith({ path: { pack_id: 5, voice_id: 'Puck' }, body: { regenerate: false } }));
            expect(window.confirm).not.toHaveBeenCalled();
        });

        it('asks before regenerating a playable voice, then sends regenerate=true', async () => {
            await show([pack([asset({ is_current: true, sample_url: 'https://s/a.wav' })])]);
            m.generate.mockResolvedValue({ data: pack([]) });
            fireEvent.click(screen.getByRole('button', { name: 'Regenerate' }));
            expect(window.confirm).toHaveBeenCalledWith(expect.stringContaining('may incur usage'));
            await waitFor(() => expect(m.generate).toHaveBeenCalledWith({ path: { pack_id: 5, voice_id: 'Achernar' }, body: { regenerate: true } }));
        });

        it('does not regenerate when the confirmation is declined', async () => {
            vi.spyOn(window, 'confirm').mockReturnValue(false);
            await show([pack([asset({ is_current: true, sample_url: 'https://s/a.wav' })])]);
            fireEvent.click(screen.getByRole('button', { name: 'Regenerate' }));
            expect(m.generate).not.toHaveBeenCalled();
        });

        it('shows a 409 conflict message', async () => {
            await show([pack([asset({ is_current: true, sample_url: 'https://s/a.wav' })])]);
            m.generate.mockResolvedValue({ error: { detail: "voice 'Achernar' is already running" } });
            fireEvent.click(screen.getByRole('button', { name: 'Regenerate' }));
            expect((await screen.findByRole('alert')).textContent).toContain("already running");
        });

        it('disables generation while a version is queued or running', async () => {
            await show([
                pack([
                    asset({ id: 1, is_current: true, sample_url: 'https://s/a.wav' }),
                    asset({ id: 2, version: 2, status: 'running' }),
                ]),
            ]);
            expect((screen.getByRole('button', { name: 'Regenerate' }) as HTMLButtonElement).disabled).toBe(true);
        });
    });

    describe('playback', () => {
        const playable = () => pack([asset({ id: 7, is_current: true, sample_url: 'https://s/old.wav' })]);

        it('plays the current version from its signed URL', async () => {
            await show([playable()]);
            expect(screen.getByLabelText('Achernar sample').getAttribute('src')).toBe('https://s/old.wav');
        });

        it('fetches a fresh signed URL when the stored one has expired', async () => {
            await show([playable()]);
            m.playback.mockResolvedValue({ data: { asset_id: 7, mime_type: 'audio/wav', url: 'https://s/new.wav' } });
            fireEvent.error(screen.getByLabelText('Achernar sample'));
            await waitFor(() => expect(m.playback).toHaveBeenCalledWith({ path: { pack_id: 5, asset_id: 7 } }));
            await waitFor(() => expect(screen.getByLabelText('Achernar sample').getAttribute('src')).toBe('https://s/new.wav'));
        });

        it('shows why playback is unavailable', async () => {
            await show([playable()]);
            m.playback.mockResolvedValue({ error: { detail: 'sample storage unavailable' } });
            fireEvent.error(screen.getByLabelText('Achernar sample'));
            expect((await screen.findByRole('alert')).textContent).toContain('sample storage unavailable');
        });

        it('plays only the current version, never a failed or older one', async () => {
            await show([
                pack([
                    asset({ id: 1, status: 'failed' }),
                    asset({ id: 2, version: 2, is_current: true, sample_url: 'https://s/current.wav' }),
                    asset({ id: 3, version: 3, status: 'failed' }),
                ]),
            ]);
            expect(screen.getAllByLabelText('Achernar sample')).toHaveLength(1);
            expect(screen.getByLabelText('Achernar sample').getAttribute('src')).toBe('https://s/current.wav');
        });
    });
});
