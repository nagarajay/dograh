"use client";

import { ChevronLeft, ChevronRight, Loader2, RefreshCw, Search } from "lucide-react";
import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { listOrganizationsApiV1SuperuserOrganizationsGet } from "@/client/sdk.gen";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
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

interface OrganizationSummary {
    id: number;
    provider_id: string;
    display_name?: string | null;
    external_reference?: string | null;
    created_at?: string | null;
    workflow_count: number;
    run_count: number;
    user_count: number;
    last_run_at?: string | null;
}

const LIMIT = 50;

export default function SuperadminOrganizationsPage() {
    const auth = useAuth();
    const [organizations, setOrganizations] = useState<OrganizationSummary[]>([]);
    const [page, setPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [totalCount, setTotalCount] = useState(0);
    const [searchInput, setSearchInput] = useState("");
    const [search, setSearch] = useState("");
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState("");

    const fetchOrganizations = useCallback(async (nextPage: number, nextSearch: string) => {
        if (!auth.isAuthenticated) return;
        setIsLoading(true);
        setError("");

        try {
            const response = await listOrganizationsApiV1SuperuserOrganizationsGet({
                query: {
                    page: nextPage,
                    limit: LIMIT,
                    ...(nextSearch ? { search: nextSearch } : {}),
                },
            });

            if (response.error) {
                throw new Error(detailFromError(response.error, "Failed to load organizations"));
            }

            const data = response.data;
            if (data) {
                setOrganizations(data.organizations as OrganizationSummary[]);
                setPage(data.page);
                setTotalPages(data.total_pages);
                setTotalCount(data.total_count);
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to load organizations");
        } finally {
            setIsLoading(false);
        }
    }, [auth.isAuthenticated]);

    useEffect(() => {
        fetchOrganizations(page, search);
        // Refetching is driven by explicit page/search changes, not by the
        // callback identity.
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [page, search, auth.isAuthenticated]);

    return (
        <div className="container mx-auto space-y-4 px-4 py-6">
            <Card>
                <CardHeader className="flex flex-row items-start justify-between gap-4">
                    <div>
                        <CardTitle>Organizations</CardTitle>
                        <CardDescription>
                            Every organization known to this Dograh deployment, including ones
                            that have never run an agent.
                        </CardDescription>
                    </div>
                    <Button
                        variant="outline"
                        size="sm"
                        onClick={() => fetchOrganizations(page, search)}
                        disabled={isLoading}
                    >
                        <RefreshCw className="mr-2 h-4 w-4" />
                        Refresh
                    </Button>
                </CardHeader>
                <CardContent className="space-y-4">
                    <form
                        className="flex gap-2"
                        onSubmit={(event) => {
                            event.preventDefault();
                            setPage(1);
                            setSearch(searchInput.trim());
                        }}
                    >
                        <Input
                            value={searchInput}
                            onChange={(event) => setSearchInput(event.target.value)}
                            placeholder="Search by client name, AVSIQ reference, or provider id"
                            className="max-w-md"
                        />
                        <Button type="submit" variant="secondary" disabled={isLoading}>
                            <Search className="mr-2 h-4 w-4" />
                            Search
                        </Button>
                    </form>

                    {error && <p className="text-sm text-destructive">{error}</p>}

                    {isLoading ? (
                        <div className="flex h-40 items-center justify-center">
                            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
                        </div>
                    ) : (
                        <div className="overflow-x-auto">
                            <Table>
                                <TableHeader>
                                    <TableRow>
                                        <TableHead>Client</TableHead>
                                        <TableHead>Dograh org</TableHead>
                                        <TableHead>External reference</TableHead>
                                        <TableHead className="text-right">Agents</TableHead>
                                        <TableHead className="text-right">Runs</TableHead>
                                        <TableHead className="text-right">Users</TableHead>
                                        <TableHead>Last run</TableHead>
                                        <TableHead>Created</TableHead>
                                    </TableRow>
                                </TableHeader>
                                <TableBody>
                                    {organizations.length === 0 && (
                                        <TableRow>
                                            <TableCell colSpan={8} className="text-center text-muted-foreground">
                                                {search
                                                    ? "No organizations match this search."
                                                    : "No client organizations yet. One is created when the first AVSIQ client is provisioned."}
                                            </TableCell>
                                        </TableRow>
                                    )}
                                    {organizations.map((organization) => (
                                        <TableRow key={organization.id}>
                                            <TableCell>
                                                <Link
                                                    href={`/superadmin/organizations/${organization.id}`}
                                                    className="font-medium underline-offset-2 hover:underline"
                                                >
                                                    {/* No "Unnamed client" fallback: a row without a
                                                        display name is an organization whose identity was
                                                        never recorded, not a client called nothing. Show
                                                        the identity that does exist — the provider id —
                                                        so the gap is visible rather than papered over. */}
                                                    {organization.display_name || (
                                                        <span className="font-mono text-xs text-muted-foreground">
                                                            {organization.provider_id}
                                                        </span>
                                                    )}
                                                </Link>
                                            </TableCell>
                                            <TableCell className="font-mono text-xs">
                                                {organization.id} · {organization.provider_id}
                                            </TableCell>
                                            <TableCell className="font-mono text-xs">
                                                {organization.external_reference || "—"}
                                            </TableCell>
                                            <TableCell className="text-right">{organization.workflow_count}</TableCell>
                                            <TableCell className="text-right">
                                                {organization.run_count === 0 ? (
                                                    <Badge variant="outline">0</Badge>
                                                ) : (
                                                    organization.run_count
                                                )}
                                            </TableCell>
                                            <TableCell className="text-right">{organization.user_count}</TableCell>
                                            <TableCell>
                                                {organization.last_run_at
                                                    ? formatDateTime(organization.last_run_at)
                                                    : "—"}
                                            </TableCell>
                                            <TableCell>
                                                {organization.created_at
                                                    ? formatDateTime(organization.created_at)
                                                    : "—"}
                                            </TableCell>
                                        </TableRow>
                                    ))}
                                </TableBody>
                            </Table>
                        </div>
                    )}

                    <div className="flex items-center justify-between">
                        <p className="text-sm text-muted-foreground">
                            {totalCount} organization{totalCount === 1 ? "" : "s"}
                        </p>
                        <div className="flex items-center gap-2">
                            <Button
                                variant="outline"
                                size="sm"
                                disabled={page <= 1 || isLoading}
                                onClick={() => setPage((current) => Math.max(1, current - 1))}
                            >
                                <ChevronLeft className="h-4 w-4" />
                            </Button>
                            <span className="text-sm">
                                Page {page} of {totalPages}
                            </span>
                            <Button
                                variant="outline"
                                size="sm"
                                disabled={page >= totalPages || isLoading}
                                onClick={() => setPage((current) => current + 1)}
                            >
                                <ChevronRight className="h-4 w-4" />
                            </Button>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </div>
    );
}
