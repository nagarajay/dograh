"use client";

import { AlertTriangle, ArrowLeft, CheckCircle2, Loader2, XCircle } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import {
    getOrganizationApiV1SuperuserOrganizationsOrganizationIdGet,
    listOrganizationWorkflowsApiV1SuperuserOrganizationsOrganizationIdWorkflowsGet,
} from "@/client/sdk.gen";
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

interface TelephonyConfigurationState {
    id: number;
    name: string;
    provider: string;
    is_default_outbound: boolean;
    inactive: boolean;
    inactive_since?: string | null;
    inactive_reason?: string | null;
    phone_number_count: number;
    created_at?: string | null;
}

interface OrganizationDetail {
    organization: {
        id: number;
        provider_id: string;
        display_name?: string | null;
        external_reference?: string | null;
        created_at?: string | null;
        workflow_count: number;
        run_count: number;
        user_count: number;
        last_run_at?: string | null;
    };
    users: Array<{
        id: number;
        email?: string | null;
        provider_id?: string | null;
        is_superuser: boolean;
    }>;
    operational_state: {
        bootstrap_state: string;
        bootstrap_updated_at?: string | null;
        model_configuration_present: boolean;
        model_configuration_updated_at?: string | null;
        model_configuration_last_validated_at?: string | null;
        langfuse_configured: boolean;
        active_api_key_count: number;
        telephony_configurations: TelephonyConfigurationState[];
    };
}

interface WorkflowSummary {
    id: number;
    workflow_uuid: string;
    name: string;
    status: string;
    folder_id?: number | null;
    is_published: boolean;
    total_runs: number;
    created_at?: string | null;
}

const BOOTSTRAP_LABELS: Record<string, { label: string; tone: "ok" | "warn" | "bad" }> = {
    completed: { label: "Provisioned", tone: "ok" },
    in_progress: { label: "Provisioning", tone: "warn" },
    stalled: { label: "Stalled", tone: "bad" },
    never_started: { label: "Never started", tone: "bad" },
};

function StateIcon({ ok }: { ok: boolean }) {
    return ok ? (
        <CheckCircle2 className="h-4 w-4 text-green-600" />
    ) : (
        <XCircle className="h-4 w-4 text-destructive" />
    );
}

export default function SuperadminOrganizationDetailPage() {
    const params = useParams<{ organizationId: string }>();
    const organizationId = Number(params.organizationId);
    const auth = useAuth();

    const [detail, setDetail] = useState<OrganizationDetail | null>(null);
    const [workflows, setWorkflows] = useState<WorkflowSummary[]>([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState("");

    const fetchAll = useCallback(async () => {
        if (!auth.isAuthenticated || Number.isNaN(organizationId)) return;
        setIsLoading(true);
        setError("");

        try {
            const [detailResponse, workflowsResponse] = await Promise.all([
                getOrganizationApiV1SuperuserOrganizationsOrganizationIdGet({
                    path: { organization_id: organizationId },
                }),
                listOrganizationWorkflowsApiV1SuperuserOrganizationsOrganizationIdWorkflowsGet({
                    path: { organization_id: organizationId },
                }),
            ]);

            if (detailResponse.error) {
                throw new Error(detailFromError(detailResponse.error, "Failed to load organization"));
            }
            if (workflowsResponse.error) {
                throw new Error(detailFromError(workflowsResponse.error, "Failed to load agents"));
            }

            setDetail(detailResponse.data as unknown as OrganizationDetail);
            setWorkflows((workflowsResponse.data?.workflows ?? []) as WorkflowSummary[]);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load organization");
        } finally {
            setIsLoading(false);
        }
    }, [auth.isAuthenticated, organizationId]);

    useEffect(() => {
        fetchAll();
    }, [fetchAll]);

    if (isLoading) {
        return (
            <div className="flex h-64 items-center justify-center">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
        );
    }

    if (error || !detail) {
        return (
            <div className="container mx-auto px-4 py-6">
                <p className="text-sm text-destructive">{error || "Organization not found."}</p>
            </div>
        );
    }

    const { organization, users, operational_state: state } = detail;
    const bootstrap = BOOTSTRAP_LABELS[state.bootstrap_state] ?? {
        label: state.bootstrap_state,
        tone: "warn" as const,
    };

    return (
        <div className="container mx-auto space-y-4 px-4 py-6">
            <div className="flex items-center gap-2">
                <Button variant="ghost" size="sm" asChild>
                    <Link href="/superadmin/organizations">
                        <ArrowLeft className="mr-2 h-4 w-4" />
                        Organizations
                    </Link>
                </Button>
            </div>

            <Card>
                <CardHeader>
                    {/* See the organizations list: an organization with no display
                        name is unnamed, not a client named "Unnamed client". */}
                    <CardTitle>
                        {organization.display_name || (
                            <span className="font-mono text-base text-muted-foreground">
                                {organization.provider_id}
                            </span>
                        )}
                    </CardTitle>
                    <CardDescription className="font-mono text-xs">
                        Dograh org {organization.id} · {organization.provider_id}
                        {organization.external_reference
                            ? ` · ref ${organization.external_reference}`
                            : ""}
                    </CardDescription>
                </CardHeader>
                <CardContent className="grid gap-4 md:grid-cols-4">
                    <div>
                        <p className="text-xs text-muted-foreground">Agents</p>
                        <p className="text-lg font-medium">{organization.workflow_count}</p>
                    </div>
                    <div>
                        <p className="text-xs text-muted-foreground">Runs</p>
                        <p className="text-lg font-medium">{organization.run_count}</p>
                    </div>
                    <div>
                        <p className="text-xs text-muted-foreground">Last run</p>
                        <p className="text-sm">
                            {organization.last_run_at ? formatDateTime(organization.last_run_at) : "Never"}
                        </p>
                    </div>
                    <div>
                        <p className="text-xs text-muted-foreground">Created</p>
                        <p className="text-sm">
                            {organization.created_at ? formatDateTime(organization.created_at) : "—"}
                        </p>
                    </div>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Provisioning state</CardTitle>
                    <CardDescription>
                        Derived from Dograh configuration. No credentials or secret values are read
                        into this view.
                    </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                    <div className="grid gap-4 md:grid-cols-4">
                        <div>
                            <p className="text-xs text-muted-foreground">Bootstrap</p>
                            <Badge
                                variant={bootstrap.tone === "ok" ? "default" : "destructive"}
                                className="mt-1"
                            >
                                {bootstrap.label}
                            </Badge>
                            {state.bootstrap_updated_at && (
                                <p className="mt-1 text-xs text-muted-foreground">
                                    {formatDateTime(state.bootstrap_updated_at)}
                                </p>
                            )}
                        </div>
                        <div>
                            <p className="text-xs text-muted-foreground">Model configuration</p>
                            <div className="mt-1 flex items-center gap-2 text-sm">
                                <StateIcon ok={state.model_configuration_present} />
                                {state.model_configuration_present ? "Configured" : "Missing"}
                            </div>
                            {state.model_configuration_last_validated_at && (
                                <p className="mt-1 text-xs text-muted-foreground">
                                    Validated {formatDateTime(state.model_configuration_last_validated_at)}
                                </p>
                            )}
                        </div>
                        <div>
                            <p className="text-xs text-muted-foreground">Langfuse tracing</p>
                            <div className="mt-1 flex items-center gap-2 text-sm">
                                <StateIcon ok={state.langfuse_configured} />
                                {state.langfuse_configured ? "Configured" : "Not configured"}
                            </div>
                        </div>
                        <div>
                            <p className="text-xs text-muted-foreground">Active API keys</p>
                            <p className="text-lg font-medium">{state.active_api_key_count}</p>
                        </div>
                    </div>

                    <div>
                        <p className="mb-2 text-sm font-medium">Telephony</p>
                        {state.telephony_configurations.length === 0 ? (
                            <p className="text-sm text-muted-foreground">
                                No telephony configuration.
                            </p>
                        ) : (
                            <div className="overflow-x-auto">
                                <Table>
                                    <TableHeader>
                                        <TableRow>
                                            <TableHead>Name</TableHead>
                                            <TableHead>Provider</TableHead>
                                            <TableHead>State</TableHead>
                                            <TableHead className="text-right">Numbers</TableHead>
                                            <TableHead>Default outbound</TableHead>
                                        </TableRow>
                                    </TableHeader>
                                    <TableBody>
                                        {state.telephony_configurations.map((configuration) => (
                                            <TableRow key={configuration.id}>
                                                <TableCell>{configuration.name}</TableCell>
                                                <TableCell>{configuration.provider}</TableCell>
                                                <TableCell>
                                                    {configuration.inactive ? (
                                                        <span className="flex items-center gap-2 text-sm text-destructive">
                                                            <AlertTriangle className="h-4 w-4" />
                                                            {configuration.inactive_reason || "Inactive"}
                                                        </span>
                                                    ) : (
                                                        <span className="text-sm">Active</span>
                                                    )}
                                                </TableCell>
                                                <TableCell className="text-right">
                                                    {configuration.phone_number_count}
                                                </TableCell>
                                                <TableCell>
                                                    {configuration.is_default_outbound ? "Yes" : "No"}
                                                </TableCell>
                                            </TableRow>
                                        ))}
                                    </TableBody>
                                </Table>
                            </div>
                        )}
                    </div>
                </CardContent>
            </Card>

            <Card>
                <CardHeader className="flex flex-row items-center justify-between">
                    <div>
                        <CardTitle>Agents</CardTitle>
                        <CardDescription>Workflows owned by this organization.</CardDescription>
                    </div>
                    <Button variant="outline" size="sm" asChild>
                        <Link href={`/superadmin/runs?organization_id=${organization.id}`}>
                            View runs
                        </Link>
                    </Button>
                </CardHeader>
                <CardContent className="overflow-x-auto">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>ID</TableHead>
                                <TableHead>Name</TableHead>
                                <TableHead>Status</TableHead>
                                <TableHead>Published</TableHead>
                                <TableHead className="text-right">Runs</TableHead>
                                <TableHead>Created</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {workflows.length === 0 && (
                                <TableRow>
                                    <TableCell colSpan={6} className="text-center text-muted-foreground">
                                        This organization has no agents.
                                    </TableCell>
                                </TableRow>
                            )}
                            {workflows.map((workflow) => (
                                <TableRow key={workflow.id}>
                                    <TableCell>
                                        <Link
                                            href={`/superadmin/organizations/${organization.id}/workflows/${workflow.id}`}
                                            className="font-medium underline-offset-2 hover:underline"
                                        >
                                            {workflow.id}
                                        </Link>
                                    </TableCell>
                                    <TableCell>{workflow.name}</TableCell>
                                    <TableCell>{workflow.status}</TableCell>
                                    <TableCell>{workflow.is_published ? "Yes" : "Draft only"}</TableCell>
                                    <TableCell className="text-right">{workflow.total_runs}</TableCell>
                                    <TableCell>
                                        {workflow.created_at ? formatDateTime(workflow.created_at) : "—"}
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>

            <Card>
                <CardHeader>
                    <CardTitle>Members</CardTitle>
                </CardHeader>
                <CardContent className="overflow-x-auto">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>User ID</TableHead>
                                <TableHead>Email</TableHead>
                                <TableHead>Provider ID</TableHead>
                                <TableHead>Superuser</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {users.map((member) => (
                                <TableRow key={member.id}>
                                    <TableCell>{member.id}</TableCell>
                                    <TableCell>{member.email || "—"}</TableCell>
                                    <TableCell className="font-mono text-xs">
                                        {member.provider_id || "—"}
                                    </TableCell>
                                    <TableCell>{member.is_superuser ? "Yes" : "No"}</TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                </CardContent>
            </Card>
        </div>
    );
}
