import { render, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { TelephonyConfigWarningsProvider } from './TelephonyConfigWarningsContext';

const { getWarningsMock, useAuthMock, useOrgConfigMock } = vi.hoisted(() => ({
    getWarningsMock: vi.fn(),
    useAuthMock: vi.fn(),
    useOrgConfigMock: vi.fn(),
}));

vi.mock('@/client/sdk.gen', () => ({
    getTelephonyConfigWarningsApiV1OrganizationsTelephonyConfigWarningsGet: getWarningsMock,
}));
vi.mock('@/lib/auth', () => ({ useAuth: useAuthMock }));
vi.mock('@/context/OrgConfigContext', () => ({ useOrgConfig: useOrgConfigMock }));

describe('TelephonyConfigWarningsProvider', () => {
    beforeEach(() => {
        vi.clearAllMocks();
        useAuthMock.mockReturnValue({ loading: false, isAuthenticated: true });
        getWarningsMock.mockResolvedValue({ data: {} });
    });

    it('sends no request for an account with no organisation', async () => {
        useOrgConfigMock.mockReturnValue({ hasOrganization: false });
        render(<TelephonyConfigWarningsProvider>x</TelephonyConfigWarningsProvider>);

        await new Promise((r) => setTimeout(r, 30));
        expect(getWarningsMock).not.toHaveBeenCalled();
    });

    it('fetches once an organisation is authorised', async () => {
        useOrgConfigMock.mockReturnValue({ hasOrganization: true });
        render(<TelephonyConfigWarningsProvider>x</TelephonyConfigWarningsProvider>);

        await waitFor(() => expect(getWarningsMock).toHaveBeenCalledTimes(1));
    });
});
