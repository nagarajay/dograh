"use client";

import { AlertTriangle, CheckCircle, Loader2 } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { getWorkflowRunsApiV1SuperuserWorkflowRunsGet } from "@/client/sdk.gen";
import type { SuperuserWorkflowRunResponse } from "@/client/types.gen";
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

import { superadminRunsFilterParam, superadminRunsHref } from "../lib/runLinks";

interface Props {
    organizationId: number;
    /** Present on an agent page: the panel then lists that agent's runs only. */
    workflowId?: number;
    /** Rows to show inline. The full list lives on the runs page. */
    limit?: number;
    title?: string;
    description?: string;
    /** Agent column links back into the organization's agent view. */
    showAgentColumn?: boolean;
}

function duration(run: SuperuserWorkflowRunResponse): string {
    const seconds = run.usage_info?.call_duration_seconds;
    if (run.is_completed && typeof seconds === "number") {
        return `${seconds.toFixed(2)}s`;
    }
    return "—";
}

/**
 * Recent call logs for one organization, optionally narrowed to one agent.
 *
 * Reads the same super-admin runs endpoint the runs page uses, with the same
 * `organization_id` scope, so what is counted on the organization overview and
 * what is listed here cannot disagree. This is a summary view on purpose:
 * filtering, sorting, transcripts and recordings stay on the runs page.
 */
export function SuperadminRunsPanel({
    organizationId,
    workflowId,
    limit = 10,
    title = "Runs / call logs",
    description,
    showAgentColumn = true,
}: Props) {
    const auth = useAuth();
    const [runs, setRuns] = useState<SuperuserWorkflowRunResponse[]>([]);
    const [totalCount, setTotalCount] = useState(0);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState("");

    const fetchRuns = useCallback(async () => {
        if (!auth.isAuthenticated || Number.isNaN(organizationId)) return;
        setIsLoading(true);
        setError("");
        try {
            const filters = superadminRunsFilterParam(workflowId);
            const response = await getWorkflowRunsApiV1SuperuserWorkflowRunsGet({
                query: {
                    page: 1,
                    limit,
                    organization_id: organizationId,
                    ...(filters && { filters }),
                },
            });
            if (response.error) {
                throw new Error(detailFromError(response.error, "Failed to load runs"));
            }
            setRuns(response.data?.workflow_runs ?? []);
            setTotalCount(response.data?.total_count ?? 0);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load runs");
        } finally {
            setIsLoading(false);
        }
    }, [auth.isAuthenticated, limit, organizationId, workflowId]);

    useEffect(() => {
        fetchRuns();
    }, [fetchRuns]);

    return (
        <Card>
            <CardHeader className="flex flex-row items-center justify-between">
                <div>
                    <CardTitle>{title}</CardTitle>
                    <CardDescription>
                        {description ??
                            (isLoading
                                ? "Loading…"
                                : `Showing ${runs.length} of ${totalCount} run(s).`)}
                    </CardDescription>
                </div>
                <Button variant="outline" size="sm" asChild>
                    <Link href={superadminRunsHref({ organizationId, workflowId })}>
                        View all runs
                    </Link>
                </Button>
            </CardHeader>
            <CardContent className="overflow-x-auto">
                {error && <p className="text-sm text-destructive">{error}</p>}
                {isLoading && !error && (
                    <div className="flex h-24 items-center justify-center">
                        <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
                    </div>
                )}
                {!isLoading && !error && (
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Run</TableHead>
                                {showAgentColumn && <TableHead>Agent</TableHead>}
                                <TableHead>Status</TableHead>
                                <TableHead>Disposition</TableHead>
                                <TableHead>Channel</TableHead>
                                <TableHead className="text-right">Duration</TableHead>
                                <TableHead>Started</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            {runs.length === 0 && (
                                <TableRow>
                                    <TableCell
                                        colSpan={showAgentColumn ? 7 : 6}
                                        className="text-center text-muted-foreground"
                                    >
                                        No runs yet.
                                    </TableCell>
                                </TableRow>
                            )}
                            {runs.map((run) => (
                                <TableRow key={run.id}>
                                    <TableCell className="font-mono text-sm">
                                        <div className="flex flex-col gap-1">
                                            <span>#{run.id}</span>
                                            {run.is_superadmin_test && (
                                                <Badge
                                                    variant="outline"
                                                    className="w-fit border-amber-500 font-sans text-amber-700"
                                                >
                                                    Super-admin test
                                                </Badge>
                                            )}
                                        </div>
                                    </TableCell>
                                    {showAgentColumn && (
                                        <TableCell>
                                            <Link
                                                href={`/superadmin/organizations/${organizationId}/workflows/${run.workflow_id}`}
                                                className="underline-offset-2 hover:underline"
                                            >
                                                {run.workflow_name || `Agent ${run.workflow_id}`}
                                            </Link>
                                        </TableCell>
                                    )}
                                    <TableCell>
                                        {run.is_completed ? (
                                            <CheckCircle className="h-4 w-4 text-green-600" />
                                        ) : (
                                            <AlertTriangle className="h-4 w-4 text-yellow-500" />
                                        )}
                                    </TableCell>
                                    <TableCell>
                                        {run.gathered_context?.mapped_call_disposition ? (
                                            <Badge variant="default">
                                                {String(run.gathered_context.mapped_call_disposition)}
                                            </Badge>
                                        ) : (
                                            <span className="text-sm text-muted-foreground">—</span>
                                        )}
                                    </TableCell>
                                    <TableCell className="text-sm">{run.mode}</TableCell>
                                    <TableCell className="text-right text-sm">
                                        {duration(run)}
                                    </TableCell>
                                    <TableCell className="text-sm">
                                        {formatDateTime(run.created_at)}
                                    </TableCell>
                                </TableRow>
                            ))}
                        </TableBody>
                    </Table>
                )}
            </CardContent>
        </Card>
    );
}
