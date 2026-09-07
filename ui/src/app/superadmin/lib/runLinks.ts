/**
 * Links and filter payloads that keep the super-admin console's run views
 * organization-scoped.
 *
 * Two encodings exist for the same filter and they are not interchangeable:
 * the runs page reads filters from the URL as `{id, value}` (see
 * `decodeFiltersFromURL`), while the API expects `{attribute, type, value}`.
 * Both are produced here so a caller cannot pick the wrong one.
 */

export interface RunScope {
    organizationId: number;
    /** Restrict to a single agent. Omitted means every agent in the organization. */
    workflowId?: number;
}

/** URL of the full runs page, pre-scoped to this organization (and agent). */
export function superadminRunsHref({ organizationId, workflowId }: RunScope): string {
    const params = new URLSearchParams();
    params.set("organization_id", String(organizationId));
    if (workflowId !== undefined) {
        params.set(
            "filters",
            JSON.stringify([{ id: "workflowId", value: { value: workflowId } }]),
        );
    }
    return `/superadmin/runs?${params.toString()}`;
}

/**
 * JSON-encoded `filters` query value for the super-admin runs API, or
 * undefined when no agent narrowing is requested. The organization scope is
 * not encoded here — it travels as the endpoint's own `organization_id`
 * parameter, which is what the backend enforces.
 */
export function superadminRunsFilterParam(workflowId?: number): string | undefined {
    if (workflowId === undefined) return undefined;
    return JSON.stringify([
        { attribute: "workflowId", type: "number", value: { value: workflowId } },
    ]);
}
