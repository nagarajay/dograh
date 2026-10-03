import { detailFromError } from "@/lib/apiError";

interface RetrySummary {
    enqueued_jobs?: number;
    enqueue_conflicts?: number;
    enqueue_failures?: number;
    selected_voices?: number;
}

/**
 * Message for a failed sample-library call. Handles the three shapes the API
 * returns (string, `{ message, retry_summary }` object, validation array) and
 * adds the queue counts of a partial bulk retry so the operator knows what ran.
 */
export function describeSampleError(error: unknown, fallback = "Request failed"): string {
    const base = detailFromError(error, fallback);
    const detail = (error as { detail?: { retry_summary?: RetrySummary } } | undefined)?.detail;
    const summary = detail && typeof detail === "object" ? detail.retry_summary : undefined;
    if (!summary) return base;
    return (
        `${base} (${summary.enqueued_jobs ?? 0} of ${summary.selected_voices ?? 0} queued, ` +
        `${summary.enqueue_failures ?? 0} failed, ${summary.enqueue_conflicts ?? 0} conflicting)`
    );
}
