"use client";

import { Loader2, PlayCircle } from "lucide-react";
import { useState } from "react";

import { AudioControls, ConnectionStatus } from "@/app/workflow/[workflowId]/run/[runId]/components";
import { useWebSocketRTC } from "@/app/workflow/[workflowId]/run/[runId]/hooks";
import { createSuperadminTestRunApiV1SuperuserOrganizationsOrganizationIdWorkflowsWorkflowIdTestRunPost } from "@/client/sdk.gen";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { RealtimeFeedback } from "@/components/workflow/conversation";
import { detailFromError } from "@/lib/apiError";
import { useAuth } from "@/lib/auth";

interface Props {
    organizationId: number;
    workflowId: number;
    agentName: string;
}

/**
 * Browser test of another organization's agent.
 *
 * The run is created through the super-admin endpoint, which binds it to the
 * agent's draft definition and stamps it as a super-admin test — nothing is
 * published and no configuration is written. The call then executes inside the
 * customer's organization, against their model and telephony configuration,
 * which is what makes it evidence the agent works.
 */
export function SuperadminAgentTest({ organizationId, workflowId, agentName }: Props) {
    const { getAccessToken } = useAuth();
    const [runId, setRunId] = useState<number | null>(null);
    const [accessToken, setAccessToken] = useState<string | null>(null);
    const [isCreating, setIsCreating] = useState(false);
    const [error, setError] = useState("");

    const startTestRun = async () => {
        setIsCreating(true);
        setError("");
        try {
            const token = await getAccessToken();
            if (!token) throw new Error("Missing admin access token");

            const response =
                await createSuperadminTestRunApiV1SuperuserOrganizationsOrganizationIdWorkflowsWorkflowIdTestRunPost({
                    path: { organization_id: organizationId, workflow_id: workflowId },
                    body: { name: `superadmin-test-${agentName}` },
                });

            if (response.error) {
                throw new Error(detailFromError(response.error, "Failed to start test run"));
            }
            if (!response.data) {
                throw new Error("Failed to start test run");
            }

            setAccessToken(token);
            setRunId(response.data.id);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to start test run");
        } finally {
            setIsCreating(false);
        }
    };

    return (
        <Card>
            <CardHeader>
                <CardTitle>Test agent</CardTitle>
                <CardDescription>
                    Runs the draft definition in this organization. Nothing is published and no
                    configuration is changed. The call consumes the customer&apos;s provider
                    credits and a concurrency slot, and is excluded from their usage reporting.
                </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
                {error && <p className="text-sm text-destructive">{error}</p>}

                {runId === null ? (
                    <Button onClick={startTestRun} disabled={isCreating}>
                        {isCreating ? (
                            <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                        ) : (
                            <PlayCircle className="mr-2 h-4 w-4" />
                        )}
                        Start test run
                    </Button>
                ) : (
                    <TestRunSession
                        workflowId={workflowId}
                        runId={runId}
                        accessToken={accessToken}
                    />
                )}
            </CardContent>
        </Card>
    );
}

function TestRunSession({
    workflowId,
    runId,
    accessToken,
}: {
    workflowId: number;
    runId: number;
    accessToken: string | null;
}) {
    const {
        audioRef,
        audioInputs,
        selectedAudioInput,
        setSelectedAudioInput,
        connectionActive,
        permissionError,
        isCompleted,
        connectionStatus,
        start,
        stop,
        isStarting,
        getAudioInputDevices,
        feedbackMessages,
    } = useWebSocketRTC({
        workflowId,
        workflowRunId: runId,
        accessToken,
        // The preflight checks validate the caller's own organization, which is
        // not the one under test. The server still authorizes the run.
        skipPreflightValidation: true,
    });

    return (
        <div className="space-y-4">
            <p className="text-xs text-muted-foreground">Run #{runId}</p>
            <ConnectionStatus connectionStatus={connectionStatus} />
            <AudioControls
                audioInputs={audioInputs}
                selectedAudioInput={selectedAudioInput}
                setSelectedAudioInput={setSelectedAudioInput}
                isCompleted={isCompleted}
                connectionActive={connectionActive}
                permissionError={permissionError}
                start={start}
                stop={stop}
                isStarting={isStarting}
                getAudioInputDevices={getAudioInputDevices}
            />
            <audio ref={audioRef} autoPlay />
            <RealtimeFeedback
                mode="live"
                messages={feedbackMessages}
                isCallActive={connectionActive}
                isCallCompleted={isCompleted}
            />
        </div>
    );
}
