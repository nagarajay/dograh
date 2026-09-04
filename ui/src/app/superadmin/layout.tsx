"use client";

import { Building2, Loader2, ShieldAlert, UserCog } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { useIsSuperuser } from "@/hooks/useIsSuperuser";
import { cn } from "@/lib/utils";

const SUPERADMIN_NAV = [
    { title: "Organizations", href: "/superadmin/organizations", icon: Building2 },
    { title: "Agent Runs", href: "/superadmin/runs", icon: ShieldAlert },
    { title: "Impersonate", href: "/superadmin", icon: UserCog },
];

/**
 * Persistent shell for the super-admin console.
 *
 * The nav is rendered for superusers only, but it is not the access control:
 * every endpoint behind these pages requires a superuser session of its own,
 * so hiding the links is a usability choice, not a boundary.
 */
export default function SuperadminLayout({ children }: { children: React.ReactNode }) {
    const pathname = usePathname();
    const { isSuperuser, isLoading } = useIsSuperuser();

    if (isLoading) {
        return (
            <div className="flex h-64 items-center justify-center">
                <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            </div>
        );
    }

    if (!isSuperuser) {
        return (
            <div className="container mx-auto py-10">
                <h1 className="text-xl font-semibold">Not available</h1>
                <p className="mt-2 text-sm text-muted-foreground">
                    This area requires Dograh super-admin access.
                </p>
            </div>
        );
    }

    return (
        <div className="flex flex-col">
            <nav className="sticky top-0 z-10 border-b bg-background">
                <div className="container mx-auto flex gap-1 px-4 py-2">
                    {SUPERADMIN_NAV.map((item) => {
                        const isActive =
                            item.href === "/superadmin"
                                ? pathname === "/superadmin"
                                : pathname.startsWith(item.href);
                        return (
                            <Link
                                key={item.href}
                                href={item.href}
                                className={cn(
                                    "flex items-center gap-2 rounded-md px-3 py-2 text-sm transition-colors",
                                    isActive
                                        ? "bg-muted font-medium text-foreground"
                                        : "text-muted-foreground hover:bg-muted/60"
                                )}
                            >
                                <item.icon className="h-4 w-4" />
                                {item.title}
                            </Link>
                        );
                    })}
                </div>
            </nav>
            {children}
        </div>
    );
}
