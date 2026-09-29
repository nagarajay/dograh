import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const client = vi.hoisted(() => ({
    createRecordingsApiV1WorkflowRecordingsPost: vi.fn(),
    deleteRecordingApiV1WorkflowRecordingsRecordingIdDelete: vi.fn(),
    getUploadUrlsApiV1WorkflowRecordingsUploadUrlPost: vi.fn(),
    listRecordingsApiV1WorkflowRecordingsGet: vi.fn(),
    transcribeAudioApiV1WorkflowRecordingsTranscribePost: vi.fn(),
}));

vi.mock("@/client", () => client);
vi.mock("posthog-js", () => ({ default: { capture: vi.fn() } }));
vi.mock("@/context/UserConfigContext", () => ({
    useUserConfig: () => ({
        userConfig: { tts: { provider: "test", model: "m", voice: "v" } },
    }),
}));
vi.mock("@/hooks/useAudioPlayback", () => ({
    useAudioPlayback: () => ({
        playingId: null,
        play: vi.fn(),
        stop: vi.fn(),
        togglePlayback: vi.fn(),
    }),
}));

import { RecordingsDialog } from "./RecordingsDialog";

describe("RecordingsDialog upload", () => {
    beforeEach(() => {
        vi.clearAllMocks();
        client.listRecordingsApiV1WorkflowRecordingsGet.mockResolvedValue({
            data: { recordings: [] },
        });
        client.transcribeAudioApiV1WorkflowRecordingsTranscribePost.mockResolvedValue({
            data: { transcript: "hello there" },
        });
        client.getUploadUrlsApiV1WorkflowRecordingsUploadUrlPost.mockResolvedValue({
            data: {
                items: [
                    {
                        upload_url: "http://storage.test/put",
                        recording_id: "rec12345",
                        storage_key: "recordings/1/rec12345/a.wav",
                        upload_token: "signed.token",
                    },
                ],
            },
        });
        client.createRecordingsApiV1WorkflowRecordingsPost.mockResolvedValue({
            data: { recordings: [] },
        });
        vi.stubGlobal("fetch", vi.fn().mockResolvedValue({ ok: true }));
    });

    it("sends the server-issued upload token when creating the recording", async () => {
        const { container } = render(
            <RecordingsDialog open onOpenChange={vi.fn()} workflowId={1} />,
        );

        const input = container.ownerDocument.querySelector(
            'input[type="file"]',
        ) as HTMLInputElement;
        const file = new File(["RIFFdata"], "a.wav", { type: "audio/wav" });
        fireEvent.change(input, { target: { files: [file] } });

        // Auto-transcription is best-effort; the transcript a person types is
        // what gates the upload.
        const transcript = (await screen.findByPlaceholderText(
            /What does this recording say\?|Transcribing/,
        )) as HTMLTextAreaElement;
        await waitFor(() => expect(transcript.disabled).toBe(false));
        fireEvent.change(transcript, { target: { value: "hello there" } });

        const upload = (await screen.findByRole("button", {
            name: /Upload 1 Recording/,
        })) as HTMLButtonElement;
        await waitFor(() => expect(upload.disabled).toBe(false));
        fireEvent.click(upload);

        await waitFor(() =>
            expect(
                client.createRecordingsApiV1WorkflowRecordingsPost,
            ).toHaveBeenCalledTimes(1),
        );
        const body =
            client.createRecordingsApiV1WorkflowRecordingsPost.mock.calls[0][0].body;
        expect(body.recordings[0]).toMatchObject({
            recording_id: "rec12345",
            storage_key: "recordings/1/rec12345/a.wav",
            upload_token: "signed.token",
        });
    });
});
