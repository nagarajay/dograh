import { describe, expect, it } from 'vitest';

import { detailFromError } from './apiError';

describe('detailFromError', () => {
    it('reads a string detail', () => {
        expect(detailFromError({ detail: 'sample pack not found' })).toBe('sample pack not found');
    });

    it('reads the message of an object detail instead of rendering [object Object]', () => {
        const text = detailFromError({
            detail: { message: 'Some sample jobs could not be queued.', retry_summary: { enqueue_failures: 1 } },
        });
        expect(text).toBe('Some sample jobs could not be queued.');
        expect(text).not.toContain('[object Object]');
    });

    it('falls back for an object detail with no message', () => {
        expect(detailFromError({ detail: { code: 'x' } }, 'Request failed')).toBe('Request failed');
    });

    it('joins validation arrays', () => {
        expect(
            detailFromError({ detail: [{ loc: ['body', 'style_text'], msg: 'Field required' }, { message: 'bad', model: 'm' }] }),
        ).toBe('Field required\nm: bad');
    });

    it('accepts a plain string and unknown shapes', () => {
        expect(detailFromError('boom')).toBe('boom');
        expect(detailFromError(undefined, 'fallback')).toBe('fallback');
        expect(detailFromError({}, 'fallback')).toBe('fallback');
    });
});
