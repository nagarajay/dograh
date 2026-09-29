import { renderHook, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { useDispositionCodes } from './useDispositionCodes';

const { getCodesMock, useAuthMock, useOrgConfigMock } = vi.hoisted(() => ({
    getCodesMock: vi.fn(),
    useAuthMock: vi.fn(),
    useOrgConfigMock: vi.fn(),
}));

vi.mock('@/client/sdk.gen', () => ({
    getDispositionCodesApiV1OrganizationsDispositionCodesGet: getCodesMock,
}));
vi.mock('@/lib/auth', () => ({ useAuth: useAuthMock }));
vi.mock('@/context/OrgConfigContext', () => ({ useOrgConfig: useOrgConfigMock }));

describe('useDispositionCodes', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        useAuthMock.mockReturnValue({ user: { id: 'u' }, loading: false });
        getCodesMock.mockResolvedValue({
            data: { codes: ['a'], end_task_reason_codes: ['b'], system_codes: ['c'] },
            error: undefined,
        });
    });

    it('does not request an organisation catalog without an organisation', async () => {
        useOrgConfigMock.mockReturnValue({ hasOrganization: false, loading: false });
        const { result } = renderHook(() => useDispositionCodes());

        await waitFor(() => expect(result.current.isLoading).toBe(false));
        expect(getCodesMock).not.toHaveBeenCalled();
        expect(result.current.codes).toEqual([]);
    });

    it('loads the catalog for an authorised organisation', async () => {
        useOrgConfigMock.mockReturnValue({ hasOrganization: true, loading: false });
        const { result } = renderHook(() => useDispositionCodes());

        await waitFor(() => expect(result.current.codes).toEqual(['a']));
        expect(getCodesMock).toHaveBeenCalledTimes(1);
    });
});
