import { useEffect, useState } from "react";

import { getAuthUserApiV1UserAuthUserGet } from "@/client/sdk.gen";
import { useAuth } from "@/lib/auth";

/**
 * Whether the signed-in user is a Dograh superuser.
 *
 * Read from the backend rather than from the auth provider's profile: the
 * superuser flag lives on Dograh's own user row, and the console must not
 * infer privilege from anything the client can set. This only decides what
 * the UI offers — every superuser endpoint enforces the flag itself.
 */
export function useIsSuperuser(): { isSuperuser: boolean; isLoading: boolean } {
    const { user, loading: authLoading } = useAuth();
    const [isSuperuser, setIsSuperuser] = useState(false);
    const [isLoading, setIsLoading] = useState(true);

    useEffect(() => {
        if (authLoading) return;
        if (!user) {
            setIsSuperuser(false);
            setIsLoading(false);
            return;
        }

        let active = true;
        setIsLoading(true);

        const loadAuthUser = async () => {
            try {
                const response = await getAuthUserApiV1UserAuthUserGet();
                if (active) {
                    setIsSuperuser(Boolean(response.data?.is_superuser));
                }
            } catch {
                if (active) {
                    setIsSuperuser(false);
                }
            } finally {
                if (active) {
                    setIsLoading(false);
                }
            }
        };

        loadAuthUser();

        return () => {
            active = false;
        };
    }, [user, authLoading]);

    return { isSuperuser, isLoading };
}
