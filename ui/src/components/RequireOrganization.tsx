"use client";

import type { ReactNode } from "react";

import SpinLoader from "@/components/SpinLoader";
import { useOrgConfig } from "@/context/OrgConfigContext";

/**
 * Renders organisation-scoped content only for a caller with an authorised
 * organisation context.
 *
 * A platform super-admin belongs to no organisation on purpose, so a page that
 * edits an organisation's own configuration has nothing to show them. Rather
 * than picking a tenant for them, this says so and sends nothing organisation
 * scoped; acting inside a specific organisation is the super-admin console's
 * job, through its explicit, audited paths.
 */
export function RequireOrganization({
    children,
    what = "This page",
}: {
    children: ReactNode;
    what?: string;
}) {
    const { hasOrganization, loading } = useOrgConfig();

    if (loading) {
        return <SpinLoader />;
    }

    if (!hasOrganization) {
        return (
            <div className="rounded-lg border p-6" role="status">
                <h2 className="text-lg font-semibold">Organization required</h2>
                <p className="mt-1 text-sm text-muted-foreground">
                    {what} configures an organization&apos;s own settings, and this account
                    is not a member of one. Platform administrators manage
                    organizations from the Super Admin console.
                </p>
            </div>
        );
    }

    return <>{children}</>;
}
