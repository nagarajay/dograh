import { render, screen } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import { RequireOrganization } from './RequireOrganization';

const { useOrgConfigMock } = vi.hoisted(() => ({ useOrgConfigMock: vi.fn() }));

vi.mock('@/context/OrgConfigContext', () => ({ useOrgConfig: useOrgConfigMock }));
vi.mock('@/components/SpinLoader', () => ({ default: () => <div>loading</div> }));

describe('RequireOrganization', () => {
    beforeEach(() => vi.clearAllMocks());

    it('explains itself and renders nothing organisation-scoped without an organisation', () => {
        useOrgConfigMock.mockReturnValue({ hasOrganization: false, loading: false });
        render(
            <RequireOrganization what="Model configuration">
                <div>org-scoped content</div>
            </RequireOrganization>,
        );

        expect(screen.queryByText('org-scoped content')).toBeNull();
        expect(screen.getByRole('status').textContent).toContain('Organization required');
    });

    it('renders the content for an authorised organisation', () => {
        useOrgConfigMock.mockReturnValue({ hasOrganization: true, loading: false });
        render(
            <RequireOrganization>
                <div>org-scoped content</div>
            </RequireOrganization>,
        );

        expect(screen.getByText('org-scoped content')).toBeTruthy();
    });

    it('does not decide while the organisation context is still loading', () => {
        useOrgConfigMock.mockReturnValue({ hasOrganization: false, loading: true });
        render(
            <RequireOrganization>
                <div>org-scoped content</div>
            </RequireOrganization>,
        );

        expect(screen.queryByText('org-scoped content')).toBeNull();
        expect(screen.queryByRole('status')).toBeNull();
    });
});
