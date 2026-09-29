from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from api.routes.workflow_recording import _make_upload_token, _verify_upload_token


def _token(org=7, recording_id="abcd1234", key=None):
    key = key or f"recordings/{org}/{recording_id}/hello.wav"
    return _make_upload_token(
        organization_id=org,
        recording_id=recording_id,
        storage_key=key,
        file_size=128,
        mime_type="audio/wav",
    )


def test_valid_server_issued_upload_token_binds_identity():
    key = "recordings/7/abcd1234/hello.wav"
    payload = _verify_upload_token(
        _token(), organization_id=7, recording_id="abcd1234", storage_key=key
    )
    assert payload["file_size"] == 128


@pytest.mark.parametrize(
    "org,recording_id,key",
    [
        (8, "abcd1234", "recordings/7/abcd1234/hello.wav"),
        (7, "other123", "recordings/7/other123/hello.wav"),
        (7, "abcd1234", "recordings/7/abcd1234/unrelated.wav"),
    ],
)
def test_upload_token_rejects_cross_org_or_arbitrary_key(org, recording_id, key):
    with pytest.raises(HTTPException) as excinfo:
        _verify_upload_token(
            _token(),
            organization_id=org,
            recording_id=recording_id,
            storage_key=key,
        )
    assert excinfo.value.status_code == 409
