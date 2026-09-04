"use client";

import { ArrowLeft, Loader2 } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { inspectOrganizationWorkflowApiV1SuperuserOrganizationsOrganizationIdWorkflowsWorkflowIdInspectionGet } from "@/client/sdk.gen";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from "@/components/ui/table";
import { detailFromError } from "@/lib/apiError";
import { useAuth } from "@/lib/auth";
import { formatDateTime } from "@/lib/dateTime";

import { SuperadminAgentTest } from "./SuperadminAgentTest";

interface InspectedNode {
    id: string;
    type?: string | null;
    name?: string | null;
    narrative: Record<string, unknown>;
    tool_uuids: string[];
    document_uuids: string[];
    mcp_tool_filters?: Record<string, unknown> | null;
}

interface Inspection {
    workflow: {
        id: number;
        workflow_uuid: string;
        name: string;
        status: string;
        is_published: boolean;
        total_runs: number;
        created_at?: string | null;
    };
    organization_id: number;
    inspected_source: string;
    inspected_definition_id?: number | null;
    inspected_version_number?: number | null;
    published_definition_id?: number | null;
    published_version_number?: number | null;
    published_at?: string | null;
    has_unpublished_draft: boolean;
    global_prompt?: string | null;
    nodes: InspectedNode[];
    edges: Array<{
        source?: string | null;
        target?: string | null;
        label?: string | null;
        condition?: string | null;
    }>;
    tools: Array<{
        tool_uuid: string;
        name?: string | null;
        description?: string | null;
        category?: string | null;
        status?: string | null;
        resolved: boolean;
    }>;
    documents: Array<{
        document_uuid: string;
        filename?: string | null;
        retrieval_mode?: string | null;
        processing_status?: string | null;
        total_chunks?: number | null;
        resolved: boolean;
    }>;
    model_configuration: Record<string, unknown>;
    workflow_configurations: Record<string, unknown>;
    template_context_variables: Record<string, unknown>;
    attached_phone_numbers: string[];
}

const NARRATIVE_LABELS: Record<string, string> = {
    prompt: "Instructions",
    greeting: "Greeting",
    greeting_type: "Greeting type",
    extraction_enabled: "Extraction enabled",
    extraction_prompt: "Extraction prompt",
    extraction_variables: "Extraction variables",
    allow_interrupt: "Interruptible",
    add_global_prompt: "Includes global prompt",
    delayed_start: "Delayed start",
    delayed_start_duration: "Delayed start (s)",
    pre_call_fetch_mode: "Pre-call fetch",
    pre_call_fetch_url: "Pre-call fetch URL",
};

function renderValue(value: unknown) {
    if (value === null || value === undefined) return "—";
    if (typeof value === "boolean") return value ? "Yes" : "No";
    if (typeof value === "object") {
        return (
            <pre className="overflow-x-auto whitespace-pre-wrap break-words rounded bg-muted p-2 text-xs">
                {JSON.stringify(value, null, 2)}
            </pre>
        );
    }
    const text = String(value);
    if (text.includes("\n") || text.length > 120) {
        return (
            <pre className="overflow-x-auto whitespace-pre-wrap break-words rounded bg-muted p-2 text-xs">
                {text}
            </pre>
        );
    }
    return text;
}

function ServiceRow({ label, service }: { label: string; service: unknown }) {
    const record = (service ?? {}) as Record<string, unknown>;
    if (Object.keys(record).length === 0) {
        return (
            <TableRow>
                <TableCell className="font-medium">{label}</TableCell>
                <TableCell colSpan={3} className="text-muted-foreground">
                    Not configured
                </TableCell>
            </TableRow>
        );
    }
    return (
        <TableRow>
            <TableCell className="font-medium">{label}</TableCell>
            <TableCell>{String(record.provider ?? "—")}</TableCell>
            <TableCell>{String(record.model ?? record.voice_id ?? "—")}</TableCell>
            <TableCell className="font-mono text-xs">
                {String(record.api_key ?? "—")}
            </TableCell>
        </TableRow>
    );
}

export default function SuperadminAgentInspectionPage() {
    const params = useParams<{ organizationId: string; workflowId: string }>();
    const organizationId = Number(params.organizationId);
    const workflowId = Number(params.workflowId);
    const auth = useAuth();

    const [inspection, setInspection] = useState<Inspection | null>(null);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState("");

    const fetchInspection = useCallback(async () => {
        if (!auth.isAuthenticated || Number.isNaN(organizationId) || Number.isNaN(workflowId)) {
            return;
        }
        setIsLoading(true);
        setError("");
        try {
            const response =
                await inspectOrganizationWorkflowApiV1SuperuserOrganizationsOrganizationIdWorkflowsWorkflowIdInspectionGet({
                    path: { organization_id: organizationId, workflow_id: workflowId },
                });
            if (response.error) {
                throw new Error(detailFromError(response.error, "Failed to load agent"));
            }
            setInspection(response.data as unknown as Inspection);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load agent");
        } finally {
            setIsLoading(false);
        }
    }, [auth.isAuthenticated, organizationId, workflowId]);

    useEffect(() => {
        fetchInspection();
    }, [fetchInspection]);

    if (isLoading) {
        return (
            <div className="flex h-64 items-center justify-center">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
        );
    }

    if (error || !inspection) {
        return (
            <div className="container mx-auto px-4 py-6">
                <p className="text-sm text-destructive">{error || "Agent not found."}</p>
            </div>
        );
    }

    const { workflow, model_configuration: modelConfiguration } = inspection;

    return (
        <div className="container mx-auto space-y-4 px-4 py-6">
            <Button variant="ghost" size="sm" asChild>
                <Link href={`/superadmin/organizations/${inspection.organization_id}`}>
                    <ArrowLeft className="mr-2 h-4 w-4" />
                    Organization {inspection.organization_id}
                </Link>
            </Button>

            <Card>
                <CardHeader>
                    <CardTitle className="flex flex-wrap items-center gap-2">
                        {workflow.name}
                        <Badge variant={workflow.is_published ? "default" : "outline"}>
                            {workflow.is_published ? "Published" : "Draft only"}
                        </Badge>
                        {inspection.has_unpublished_draft && workflow.is_published && (
                            <Badge variant="destructive">Unpublished draft</Badge>
                        )}
                    </CardTitle>
                    <CardDescription className="font-mono text-xs">
                        {workflow.workflow_uuid}
                    </CardDescription>
                </CardHeader>
                <CardContent className="grid gap-4 md:grid-cols-4">
                    <div>
                        <p className="text-xs text-muted-foreground">Inspecting</p>
                        <p className="text-sm">
                            {inspection.inspected_source}
                            {inspection.inspected_version_number != null &&
                                ` v${inspection.inspected_version_number}`}
                        </p>
                    </div>
                    <div>
                        <p className="text-xs text-muted-foreground">Published version</p>
                        <p className="text-sm">
                            {inspection.published_version_number != null
                                ? `v${inspection.published_version_number}`
                                : "None"}
                        </p>
                    </div>
                    <div>
                        <p className="text-xs text-muted-foreground">Published at</p>
                        <p className="text-sm">
                            {inspection.published_at
                                ? formatDateTime(inspection.published_at)
                                : "—"}
                        </p>
                    </div>
                    <div>
                        <p className="text-xs text-muted-foreground">Runs</p>
                        <p className="text-sm">
                            <Link
                                href={`/superadmin/runs?organization_id=${inspection.organization_id}`}
                                className="underline-offset-2 hover:underline"
                            >
                                {workflow.total_runs}
                            </Link>
                        </p>
                    </div>
                </CardContent>
            </Card>

            <SuperadminAgentTest
                organizationId={inspection.organization_id}
                workflowId={workflow.id}
                agentName={workflow.name}
            />

            <Card>
                <CardHeader>
                    <CardTitle>Models</CardTitle>
                    <CardDescription>
                        Effective configuration for this agent. Keys are masked.
                    </CardDescription>
                </CardHeader>
                <CardContent className="overflow-x-auto">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Service</TableHead>
                                <TableHead>Provider</TableHead>
                                <TableHead>Model / voice</TableHead>
                                <TableHead>Key</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            <ServiceRow label="LLM" service={modelConfiguration.llm} />
                            <ServiceRow label="STT" service={modelConfiguration.stt} />
                            <ServiceRow label="TTS" service={modelConfiguration.tts} />
                            <ServiceRow label="Embeddings" service={modelConfiguration.embeddings} />
                            {Boolean(modelConfiguration.is_realtime) && (
                                <ServiceRow label="Realtime" service={modelConfiguration.realtime} />
                            )}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Channels</CardTitle>
                </CardHeader>
                <CardContent>
                    {inspection.attached_phone_numbers.length === 0 ? (
                        <p className="text-sm text-muted-foreground">
                            No inbound number routes to this agent.
                        </p>
                    ) : (
                        <ul className="space-y-1 font-mono text-sm">
                            {inspection.attached_phone_numbers.map((address) => (
                                <li key={address}>{address}</li>
                            ))}
                        </ul>
                    )}
                </CardContent>
            </Card>

            {inspection.global_prompt && (
                <Card>
                    <CardHeader>
                        <CardTitle>Global instructions</CardTitle>
                    </CardHeader>
                    <CardContent>{renderValue(inspection.global_prompt)}</CardContent>
                </Card>
            )}

            <Card>
                <CardHeader>
                    <CardTitle>Conversation nodes</CardTitle>
                    <CardDescription>
                        The instructions this agent actually runs, node by node.
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    {inspection.nodes.map((node) => (
                        <div key={node.id} className="rounded-md border p-3">
                            <div className="mb-2 flex flex-wrap items-center gap-2">
                                <span className="font-medium">{node.name || node.id}</span>
                                <Badge variant="outline">{node.type || "unknown"}</Badge>
                                <span className="font-mono text-xs text-muted-foreground">
                                    {node.id}
                                </span>
                            </div>
                            <dl className="space-y-2">
                                {Object.entries(node.narrative).map(([key, value]) => (
                                    <div key={key}>
                                        <dt className="text-xs text-muted-foreground">
                                            {NARRATIVE_LABELS[key] || key}
                                        </dt>
                                        <dd className="text-sm">{renderValue(value)}</dd>
                                    </div>
                                ))}
                            </dl>
                            {(node.tool_uuids.length > 0 || node.document_uuids.length > 0) && (
                                <p className="mt-2 text-xs text-muted-foreground">
                                    {node.tool_uuids.length} tool(s), {node.document_uuids.length}{" "}
                                    document(s)
                                </p>
                            )}
                        </div>
                    ))}
                    {inspection.nodes.length === 0 && (
                        <p className="text-sm text-muted-foreground">
                            This agent has no nodes defined.
                        </p>
                    )}
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Transitions</CardTitle>
                </CardHeader>
                <CardContent className="overflow-x-auto">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>From</TableHead>
                                <TableHead>To</TableHead>
                                <TableHead>Label</TableHead>
                                <TableHead>Condition</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {inspection.edges.map((edge, index) => (
                                <TableRow key={`${edge.source}-${edge.target}-${index}`}>
                                    <TableCell className="font-mono text-xs">{edge.source}</TableCell>
                                    <TableCell className="font-mono text-xs">{edge.target}</TableCell>
                                    <TableCell>{edge.label || "—"}</TableCell>
                                    <TableCell>{edge.condition || "—"}</TableCell>
                                </TableRow>
                            ))}
                            {inspection.edges.length === 0 && (
                                <TableRow>
                                    <TableCell colSpan={4} className="text-center text-muted-foreground">
                                        No transitions.
                                    </TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Tools</CardTitle>
                    <CardDescription>
                        Catalog entries only — tool definitions can carry credentials and are not
                        read here.
                    </CardDescription>
                </CardHeader>
                <CardContent className="overflow-x-auto">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Name</TableHead>
                                <TableHead>Category</TableHead>
                                <TableHead>Status</TableHead>
                                <TableHead>UUID</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {inspection.tools.map((tool) => (
                                <TableRow key={tool.tool_uuid}>
                                    <TableCell>
                                        {tool.resolved ? tool.name : "Unresolved reference"}
                                    </TableCell>
                                    <TableCell>{tool.category || "—"}</TableCell>
                                    <TableCell>{tool.status || "—"}</TableCell>
                                    <TableCell className="font-mono text-xs">
                                        {tool.tool_uuid}
                                    </TableCell>
                                </TableRow>
                            ))}
                            {inspection.tools.length === 0 && (
                                <TableRow>
                                    <TableCell colSpan={4} className="text-center text-muted-foreground">
                                        No tools attached.
                                    </TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Knowledge</CardTitle>
                </CardHeader>
                <CardContent className="overflow-x-auto">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Document</TableHead>
                                <TableHead>Retrieval</TableHead>
                                <TableHead>Processing</TableHead>
                                <TableHead className="text-right">Chunks</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {inspection.documents.map((document) => (
                                <TableRow key={document.document_uuid}>
                                    <TableCell>
                                        {document.resolved
                                            ? document.filename
                                            : "Unresolved reference"}
                                    </TableCell>
                                    <TableCell>{document.retrieval_mode || "—"}</TableCell>
                                    <TableCell>{document.processing_status || "—"}</TableCell>
                                    <TableCell className="text-right">
                                        {document.total_chunks ?? "—"}
                                    </TableCell>
                                </TableRow>
                            ))}
                            {inspection.documents.length === 0 && (
                                <TableRow>
                                    <TableCell colSpan={4} className="text-center text-muted-foreground">
                                        No knowledge documents attached.
                                    </TableCell>
                                </TableRow>
                            )}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Context variables</CardTitle>
                    <CardDescription>
                        Template defaults a run starts from.
                    </CardDescription>
                </CardHeader>
                <CardContent>
                    {renderValue(inspection.template_context_variables)}
                </CardContent>
            </Card>
        </div>
    );
}
