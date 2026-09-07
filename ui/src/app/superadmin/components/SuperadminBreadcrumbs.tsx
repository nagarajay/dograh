"use client";

import { ChevronRight } from "lucide-react";
import Link from "next/link";

export interface Crumb {
    label: string;
    /** Omitted on the current page, which is rendered as plain text. */
    href?: string;
}

/**
 * Organizations → Organization → Agent → Runs.
 *
 * Navigation only: every page behind these links enforces its own superuser
 * check and its own organization scope, so a crumb never widens access.
 */
export function SuperadminBreadcrumbs({ items }: { items: Crumb[] }) {
    return (
        <nav aria-label="Breadcrumb">
            <ol className="flex flex-wrap items-center gap-1 text-sm text-muted-foreground">
                {items.map((item, index) => {
                    const isLast = index === items.length - 1;
                    return (
                        <li key={`${item.label}-${index}`} className="flex items-center gap-1">
                            {item.href && !isLast ? (
                                <Link
                                    href={item.href}
                                    className="rounded px-1 py-0.5 hover:bg-muted hover:text-foreground"
                                >
                                    {item.label}
                                </Link>
                            ) : (
                                <span className="px-1 py-0.5 font-medium text-foreground">
                                    {item.label}
                                </span>
                            )}
                            {!isLast && <ChevronRight className="h-3.5 w-3.5" />}
                        </li>
                    );
                })}
            </ol>
        </nav>
    );
}
