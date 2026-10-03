"""Per-workflow, per-slot model settings: draft, validate, publish, rollback.

A workflow's four slots (llm, stt, tts, embeddings) are versioned independently.
Runtime only ever reads the *published* version of a slot; a workflow with no
published slot rows resolves exactly as before (organization configuration plus
any legacy workflow override), so nothing changes until a slot is published.

Secrets are never part of slot config. A slot pins an exact
``(credential_ref, version)`` and the secret is decrypted in memory only when a
service configuration is built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from pydantic import ValidationError

from api.db.workflow_slot_client import (
    CredentialInUseError,
    CredentialRevokedError,
    SlotRevisionConflict,
    SlotStateError,
)
from api.services.configuration import secret_store
from api.services.configuration.masking import SERVICE_SECRET_FIELDS
from api.services.configuration.options.google_vertex_catalog import (
    check_vertex_config,
)
from api.services.configuration.registry import REGISTRY, ServiceProviders, ServiceType
from api.services.configuration.safe_errors import (
    redact_text,
    summarize_validation_error,
    validation_error_text,
)

SLOTS = ("llm", "stt", "tts", "embeddings")
_SLOT_TYPE = {
    "llm": ServiceType.LLM,
    "stt": ServiceType.STT,
    "tts": ServiceType.TTS,
    "embeddings": ServiceType.EMBEDDINGS,
}

# The knowledge-base chunk column is Vector(1536). Every embedding service in
# this repo emits 1536 dimensions; anything else cannot be indexed.
EMBEDDING_INDEX_DIMENSION = 1536

_PLACEHOLDER = "placeholder-not-a-secret"
TEMPLATE_PENDING = "pending"


class SlotSettingsError(Exception):
    """Expected, user-facing failure. ``message`` never contains secret material."""

    def __init__(self, status_code: int, code: str, message: str, **extra: Any):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.extra = extra

    def detail(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, **self.extra}


class SlotResolutionError(RuntimeError):
    """A published slot cannot be turned into a usable service configuration.

    Runtime must fail the call rather than fall back to another provider.
    ``code`` is a stable machine-readable reason; the message never carries
    secret material.
    """

    def __init__(self, message: str, code: str = "slot_resolution_failed"):
        super().__init__(message)
        self.code = code


@dataclass
class SlotIssue:
    """Why one slot of a workflow cannot be resolved, for readback and runtime."""

    slot: str
    code: str
    message: str
    provider: Optional[str] = None
    model: Optional[str] = None


def _config_class(slot: str, provider: str):
    registry = REGISTRY.get(_SLOT_TYPE[slot], {})
    cls = registry.get(provider)
    if cls is None:
        raise SlotSettingsError(
            422,
            "unknown_provider",
            f"provider '{provider}' is not available for the {slot} slot; "
            f"available: {', '.join(sorted(registry))}",
        )
    return cls


# Providers whose config class carries a field that is documented as unused, so
# reflection over the class would accept a credential the provider ignores.
_KIND_OVERRIDES: dict[tuple[str, str], tuple[set[str], bool]] = {
    ("stt", ServiceProviders.GOOGLE.value): ({"service_account_json"}, False),
    ("tts", ServiceProviders.GOOGLE.value): ({"service_account_json"}, False),
    ("llm", ServiceProviders.AWS_BEDROCK.value): ({"aws_iam"}, True),
}


def accepted_credential_kinds(slot: str, provider: str) -> tuple[set[str], bool]:
    """(kinds the provider's config can carry, whether one is mandatory)."""
    fields = _config_class(slot, provider).model_fields
    if (slot, provider) in _KIND_OVERRIDES:
        kinds, required = _KIND_OVERRIDES[(slot, provider)]
        return set(kinds), required
    kinds: set[str] = set()
    if "api_key" in fields:
        kinds.add("api_key")
    if "credentials" in fields:
        kinds.add("service_account_json")
    if "aws_access_key" in fields:
        kinds.add("aws_iam")
    required = "api_key" in fields and fields["api_key"].is_required()
    if "aws_access_key" in fields and fields["aws_access_key"].is_required():
        required = True
    return kinds, required


def normalize_slot_config(slot: str, config: dict[str, Any]) -> dict[str, Any]:
    """Validate the non-secret config and return it with defaults made explicit.

    Making defaults explicit is deliberate: a later change to a registry default
    must not silently change a published slot.
    """
    if slot not in SLOTS:
        raise SlotSettingsError(422, "unknown_slot", f"slot must be one of {SLOTS}")
    leaked = sorted(k for k in config if k in SERVICE_SECRET_FIELDS)
    if leaked:
        raise SlotSettingsError(
            422,
            "secret_in_config",
            f"config must not contain secrets ({', '.join(leaked)}); "
            "create a credential and reference it instead",
        )
    provider = config.get("provider")
    if not isinstance(provider, str) or not provider:
        raise SlotSettingsError(422, "provider_required", "config.provider is required")
    cls = _config_class(slot, provider)
    kinds, required = accepted_credential_kinds(slot, provider)
    probe = dict(config)
    if "api_key" in kinds and cls.model_fields["api_key"].is_required():
        probe["api_key"] = _PLACEHOLDER
    if (
        "aws_iam" in kinds
        and "aws_access_key" in cls.model_fields
        and cls.model_fields["aws_access_key"].is_required()
    ):
        probe.setdefault("aws_access_key", _PLACEHOLDER)
        probe.setdefault("aws_secret_key", _PLACEHOLDER)
    try:
        instance = cls(**probe)
    except ValidationError as exc:
        raise SlotSettingsError(
            422,
            "invalid_config",
            validation_error_text(exc),
            errors=summarize_validation_error(exc),
        ) from None
    normalized = instance.model_dump(mode="json", exclude_none=True)
    for secret in SERVICE_SECRET_FIELDS:
        normalized.pop(secret, None)
    return normalized


def check_credential_compatibility(
    slot: str, config: dict[str, Any], kind: Optional[str]
) -> None:
    """Explicit auth rules, from the registry and the Vertex catalogue."""
    provider = config["provider"]
    kinds, required = accepted_credential_kinds(slot, provider)
    if provider == ServiceProviders.GOOGLE_VERTEX.value:
        if kind not in (None, "api_key", "service_account_json"):
            raise SlotSettingsError(
                422,
                "credential_kind_mismatch",
                f"Google Vertex cannot use a '{kind}' credential",
            )
        error = check_vertex_config(
            slot,
            model=config.get("model", ""),
            location=config.get("location"),
            has_api_key=kind == "api_key",
            has_credentials=kind == "service_account_json",
            project_id=config.get("project_id"),
            voice=config.get("voice"),
        )
        if error:
            raise SlotSettingsError(422, "vertex_auth_not_supported", error)
        return
    if kind is None:
        if required:
            raise SlotSettingsError(
                422, "credential_required", f"provider '{provider}' needs a credential"
            )
        return
    if kind not in kinds:
        raise SlotSettingsError(
            422,
            "credential_kind_mismatch",
            f"provider '{provider}' accepts credential kinds "
            f"{sorted(kinds) or 'none'}, not '{kind}'",
        )


def build_service_config(
    slot: str, config: dict[str, Any], secret: Optional[dict[str, Any]]
):
    cls = _config_class(slot, config["provider"])
    try:
        return cls(**{**config, **(secret or {})})
    except ValidationError as exc:
        # Never str(exc): its input_value repr would carry the secret.
        raise SlotResolutionError(
            f"slot configuration is invalid: {validation_error_text(exc)}",
            "slot_config_invalid",
        ) from None


def _secret_values(secret: Optional[dict[str, Any]]) -> list[str]:
    values: list[str] = []
    for value in (secret or {}).values():
        values.extend(value if isinstance(value, list) else [value])
    return [v for v in values if isinstance(v, str)]


@dataclass
class ResolvedSlot:
    slot: str
    version: int
    service: Any


async def resolve_credential_secret(
    repo, organization_id: int, ref: Optional[str], version: Optional[int]
) -> tuple[Optional[str], Optional[dict[str, Any]]]:
    """Decrypt the pinned credential version. Fails closed on any problem."""
    if ref is None:
        return None, None
    row = await repo.get_provider_credential(organization_id, ref, version or 0)
    if row is None:
        raise SlotResolutionError(
            f"credential {ref} v{version} not found", "credential_not_found"
        )
    if row.revoked_at is not None:
        raise SlotResolutionError(
            f"credential {ref} v{version} is revoked", "credential_revoked"
        )
    payload = secret_store.decrypt_credential(
        organization_id=organization_id,
        credential_ref=ref,
        version=row.version,
        ciphertext=row.ciphertext,
    )
    return row.kind, payload


async def load_published_overlay_with_issues(
    *, repo, workflow_id: int, organization_id: int
) -> tuple[dict[str, ResolvedSlot], dict[str, SlotIssue]]:
    """Resolve every published slot, collecting a per-slot issue for each failure.

    ``issues`` is keyed by slot. A workflow whose organization-default template
    copy is still pending has an issue on *every* slot: it must not run on
    inherited defaults. Runtime treats any issue as fatal (see
    ``load_published_overlay``); readback reports them per slot.
    """
    if await repo.get_workflow_slot_template_status(workflow_id) == TEMPLATE_PENDING:
        message = (
            f"workflow {workflow_id}: model configuration template has not been "
            "applied; POST /workflow/{id}/model-slots/apply-template to retry"
        )
        return {}, {
            slot: SlotIssue(slot, "template_pending", message) for slot in SLOTS
        }
    overlay: dict[str, ResolvedSlot] = {}
    issues: dict[str, SlotIssue] = {}
    for row in await repo.list_published_slot_settings(workflow_id):
        config = dict(row.config)
        try:
            _, secret = await resolve_credential_secret(
                repo, organization_id, row.credential_ref, row.credential_version
            )
            service = build_service_config(row.slot, config, secret)
        except secret_store.SecretStoreUnavailable as exc:
            code, reason = "secret_store_unavailable", str(exc)
        except secret_store.SecretStoreCorrupt as exc:
            code, reason = "secret_store_corrupt", str(exc)
        except SlotResolutionError as exc:
            code, reason = exc.code, str(exc)
        else:
            overlay[row.slot] = ResolvedSlot(row.slot, row.version, service)
            continue
        issues[row.slot] = SlotIssue(
            row.slot,
            code,
            f"{row.slot} slot v{row.version} of workflow {workflow_id}: {reason}",
            provider=config.get("provider"),
            model=config.get("model"),
        )
    return overlay, issues


async def load_published_overlay(
    *, repo, workflow_id: int, organization_id: int
) -> dict[str, ResolvedSlot]:
    """Service configs for the workflow's published slots. Empty for legacy workflows.

    Raises on the first unresolvable slot (runtime fails the call; it never
    falls back to another provider) and while the template copy is pending.
    """
    overlay, issues = await load_published_overlay_with_issues(
        repo=repo, workflow_id=workflow_id, organization_id=organization_id
    )
    for slot in SLOTS:
        if slot in issues:
            raise SlotResolutionError(issues[slot].message, issues[slot].code)
    return overlay


async def check_embedding_compatibility(
    *, repo, organization_id: int, config: dict[str, Any]
) -> None:
    """Block an embeddings change that would silently orphan indexed chunks.

    The knowledge base is scoped to the organization and has a single fixed-size
    vector column; there is no per-index provider/model record. Until index-level
    configuration exists, a model may only be used if it matches every model that
    already has chunks (or if none exist).
    """
    from api.services.configuration.options.google_vertex_catalog import (
        get_vertex_model,
    )

    if config.get("provider") == ServiceProviders.GOOGLE_VERTEX.value:
        entry = get_vertex_model("embeddings", config.get("model", ""))
        dimension = entry.dimensions if entry else None
    else:
        dimension = EMBEDDING_INDEX_DIMENSION
    if dimension != EMBEDDING_INDEX_DIMENSION:
        raise SlotSettingsError(
            409,
            "embedding_dimension_unsupported",
            f"the knowledge-base index stores {EMBEDDING_INDEX_DIMENSION}-dimensional "
            f"vectors; model '{config.get('model')}' does not produce that size",
        )
    await _check_matches_organization_index_model(organization_id, config, dimension)
    signatures = await repo.get_organization_embedding_signatures(organization_id)
    conflicting = [
        s
        for s in signatures
        if s["model"] != config.get("model") or s["dimension"] != dimension
    ]
    if conflicting:
        raise SlotSettingsError(
            409,
            "embedding_index_incompatible",
            "the organization's knowledge base is already indexed with "
            + ", ".join(f"'{s['model']}' ({s['chunks']} chunks)" for s in conflicting)
            + f"; switching to '{config.get('model')}' would make retrieval return "
            "nothing. Re-ingest the documents first (per-index embedding "
            "configuration is not implemented yet).",
            indexed_models=[
                {
                    "model": s["model"],
                    "dimension": s["dimension"],
                    "chunks": s["chunks"],
                }
                for s in signatures
            ],
        )


async def _check_matches_organization_index_model(
    organization_id: int, config: dict[str, Any], dimension: Optional[int]
) -> None:
    """A workflow's embeddings slot must use the organization's index model.

    Ingestion and the knowledge-base index are organization-wide and always use
    the organization's embeddings configuration; only retrieval runs inside a
    workflow. A slot whose model differs would query a different vector space
    than the one documents are written to and return nothing, so it is rejected
    here instead of being accepted and silently not working. (Project, location
    and credential may differ: the same model produces the same vectors.)
    """
    from api.services.configuration.ai_model_configuration import (
        get_resolved_ai_model_configuration,
    )

    org = (
        await get_resolved_ai_model_configuration(organization_id=organization_id)
    ).effective.embeddings
    org_model = getattr(org, "model", None)
    if org_model and config.get("model") and org_model != config.get("model"):
        raise SlotSettingsError(
            409,
            "embedding_model_not_organization_index",
            f"documents are indexed with the organization's embeddings model "
            f"'{org_model}' (ingestion is organization-wide); a workflow slot "
            f"using '{config.get('model')}' would search a different vector space "
            "and find nothing. Use the organization's model, or change the "
            "organization's embeddings configuration and re-ingest.",
            organization_model=org_model,
        )


def credential_summary(row) -> dict[str, Any]:
    return {
        "credential_ref": row.credential_ref,
        "version": row.version,
        "kind": row.kind,
        "label": row.label,
        "source_ref": row.source_ref,
        "key_id": row.key_id,
        "created_at": row.created_at,
        "revoked_at": row.revoked_at,
    }


def version_view(row) -> dict[str, Any]:
    return {
        "version": row.version,
        "state": row.state,
        "config": row.config,
        "credential": (
            {"credential_ref": row.credential_ref, "version": row.credential_version}
            if row.credential_ref
            else None
        ),
        "validation_status": row.validation_status,
        "validated_at": row.validated_at,
        "based_on_version": row.based_on_version,
        "origin": row.origin,
        "change_note": row.change_note,
        "created_by": row.created_by,
        "created_at": row.created_at,
        "published_at": row.published_at,
    }


def slot_source(
    slot: str, published_version: Optional[int], released_configs: dict | None
) -> str:
    """Where a slot's live value comes from, for readback.

    ``organization_default_inherited`` is the legacy behaviour: the workflow has
    no snapshot of its own for this slot, so it follows the organization
    default at call time.
    """
    if published_version is not None:
        return "workflow_slot"
    configs = released_configs or {}
    if configs.get("model_configuration_v2_override"):
        return "legacy_workflow_override"
    override = (configs.get("model_overrides") or {}).get(slot)
    if override:
        return "legacy_workflow_override"
    return "organization_default_inherited"


class WorkflowSlotService:
    def __init__(self, repo):
        self.repo = repo

    async def _workflow(self, organization_id: int, workflow_id: int):
        workflow = await self.repo.get_workflow(
            workflow_id, organization_id=organization_id
        )
        if workflow is None:
            raise SlotSettingsError(404, "workflow_not_found", "workflow not found")
        return workflow

    async def _credential_row(
        self, organization_id: int, ref: str, version: Optional[int]
    ):
        row = (
            await self.repo.get_provider_credential(organization_id, ref, version)
            if version
            else await self.repo.get_latest_provider_credential(organization_id, ref)
        )
        if row is None:
            raise SlotSettingsError(404, "credential_not_found", "credential not found")
        if row.revoked_at is not None:
            raise SlotSettingsError(
                422, "credential_revoked", "credential version is revoked"
            )
        return row

    # ---- credentials ------------------------------------------------------

    async def create_credential(
        self,
        *,
        organization_id: int,
        kind: str,
        secret: dict[str, Any],
        credential_ref: Optional[str],
        label: Optional[str],
        source_ref: Optional[str],
        created_by: Optional[str],
    ):
        try:
            secret_store.validate_payload(kind, secret)
        except ValueError as exc:
            raise SlotSettingsError(422, "invalid_credential", str(exc)) from None

        def encrypt(ref: str, version: int) -> tuple[str, str]:
            return secret_store.encrypt_credential(
                organization_id=organization_id,
                credential_ref=ref,
                version=version,
                payload=secret,
            )

        try:
            return await self.repo.create_provider_credential_version(
                organization_id=organization_id,
                credential_ref=credential_ref,
                kind=kind,
                label=label,
                source_ref=source_ref,
                created_by=created_by,
                encrypt=encrypt,
            )
        except secret_store.SecretStoreUnavailable as exc:
            raise SlotSettingsError(503, "secret_store_unavailable", str(exc)) from None
        except SlotStateError as exc:
            raise SlotSettingsError(404, "credential_not_found", str(exc)) from None

    async def revoke_credential(
        self, *, organization_id: int, credential_ref: str, version: int
    ):
        try:
            row = await self.repo.revoke_provider_credential_version(
                organization_id, credential_ref, version
            )
        except CredentialInUseError as exc:
            raise SlotSettingsError(
                409,
                "credential_in_use",
                f"{exc.in_use} published slot(s) still use this credential version; "
                "publish a replacement first",
            ) from None
        if row is None:
            raise SlotSettingsError(404, "credential_not_found", "credential not found")
        return row

    # ---- drafts -----------------------------------------------------------

    async def save_draft(
        self,
        *,
        organization_id: int,
        workflow_id: int,
        slot: str,
        config: dict[str, Any],
        credential: Optional[dict[str, Any]],
        expected_revision: int,
        change_note: Optional[str],
        created_by: Optional[str],
    ):
        await self._workflow(organization_id, workflow_id)
        normalized = normalize_slot_config(slot, config)
        cred_row = None
        if credential is not None:
            cred_row = await self._credential_row(
                organization_id, credential["credential_ref"], credential.get("version")
            )
        check_credential_compatibility(
            slot, normalized, cred_row.kind if cred_row else None
        )
        try:
            return await self.repo.save_slot_draft(
                workflow_id=workflow_id,
                slot=slot,
                config=normalized,
                credential_ref=cred_row.credential_ref if cred_row else None,
                credential_version=cred_row.version if cred_row else None,
                expected_revision=expected_revision,
                created_by=created_by,
                change_note=change_note,
            )
        except SlotRevisionConflict as exc:
            raise self._conflict(exc) from None

    @staticmethod
    def _conflict(exc: SlotRevisionConflict) -> SlotSettingsError:
        return SlotSettingsError(
            409,
            "stale_revision",
            "the slot changed since you read it; re-read and retry",
            current_revision=exc.current_revision,
        )

    async def validate_draft(
        self,
        *,
        organization_id: int,
        workflow_id: int,
        slot: str,
        version: int,
        validator,
        created_by: Optional[str],
    ) -> tuple[str, list[str]]:
        await self._workflow(organization_id, workflow_id)
        row = await self.repo.get_slot_setting(workflow_id, slot, version)
        if row is None:
            raise SlotSettingsError(404, "version_not_found", "slot version not found")
        if row.state != "draft":
            raise SlotSettingsError(409, "not_a_draft", "only a draft can be validated")
        errors: list[str] = []
        secret: Optional[dict[str, Any]] = None
        try:
            kind, secret = await resolve_credential_secret(
                self.repo, organization_id, row.credential_ref, row.credential_version
            )
            check_credential_compatibility(slot, dict(row.config), kind)
            service = build_service_config(slot, dict(row.config), secret)
            if slot == "embeddings":
                await check_embedding_compatibility(
                    repo=self.repo,
                    organization_id=organization_id,
                    config=dict(row.config),
                )
            problems = await validator.validate_single(
                service,
                slot,
                organization_id=organization_id,
                created_by=created_by,
            )
            errors.extend(problems)
        except SlotSettingsError as exc:
            errors.append(exc.message)
        except SlotResolutionError as exc:
            errors.append(str(exc))
        except (
            secret_store.SecretStoreUnavailable,
            secret_store.SecretStoreCorrupt,
        ) as exc:
            errors.append(str(exc))
        errors = [redact_text(e, _secret_values(secret)) for e in errors]
        status = "invalid" if errors else "valid"
        await self.repo.mark_slot_validation(
            workflow_id=workflow_id, slot=slot, version=version, status=status
        )
        return status, errors

    async def publish(
        self,
        *,
        organization_id: int,
        workflow_id: int,
        slot: str,
        version: int,
        expected_revision: int,
    ):
        await self._workflow(organization_id, workflow_id)
        row = await self.repo.get_slot_setting(workflow_id, slot, version)
        if row is None:
            raise SlotSettingsError(404, "version_not_found", "slot version not found")
        await self._pre_publish_checks(organization_id, slot, row)
        try:
            return await self.repo.publish_slot(
                workflow_id=workflow_id,
                slot=slot,
                version=version,
                expected_revision=expected_revision,
            )
        except SlotRevisionConflict as exc:
            raise self._conflict(exc) from None
        except CredentialRevokedError:
            raise self._revoked() from None
        except SlotStateError as exc:
            raise SlotSettingsError(409, "invalid_transition", str(exc)) from None

    @staticmethod
    def _revoked() -> SlotSettingsError:
        return SlotSettingsError(
            422, "credential_revoked", "credential version is revoked"
        )

    async def rollback(
        self,
        *,
        organization_id: int,
        workflow_id: int,
        slot: str,
        to_version: int,
        expected_revision: int,
        change_note: Optional[str],
        created_by: Optional[str],
    ):
        await self._workflow(organization_id, workflow_id)
        target = await self.repo.get_slot_setting(workflow_id, slot, to_version)
        if target is None:
            raise SlotSettingsError(404, "version_not_found", "slot version not found")
        await self._pre_publish_checks(organization_id, slot, target)
        try:
            return await self.repo.rollback_slot(
                workflow_id=workflow_id,
                slot=slot,
                to_version=to_version,
                expected_revision=expected_revision,
                created_by=created_by,
                change_note=change_note or f"rollback to v{to_version}",
            )
        except SlotRevisionConflict as exc:
            raise self._conflict(exc) from None
        except CredentialRevokedError:
            raise self._revoked() from None
        except SlotStateError as exc:
            raise SlotSettingsError(409, "invalid_transition", str(exc)) from None

    async def _pre_publish_checks(self, organization_id: int, slot: str, row) -> None:
        if row.credential_ref is not None:
            await self._credential_row(
                organization_id, row.credential_ref, row.credential_version
            )
        if slot == "embeddings":
            await check_embedding_compatibility(
                repo=self.repo, organization_id=organization_id, config=dict(row.config)
            )

    async def retry_template(
        self, *, organization_id: int, workflow_id: int, created_by: Optional[str]
    ) -> list[str]:
        await self._workflow(organization_id, workflow_id)
        if (
            await self.repo.get_workflow_slot_template_status(workflow_id)
            != TEMPLATE_PENDING
        ):
            raise SlotSettingsError(
                409,
                "template_not_pending",
                "this workflow has no pending template copy",
            )
        return await apply_organization_template(
            repo=self.repo,
            organization_id=organization_id,
            workflow_id=workflow_id,
            created_by=created_by,
        )

    # ---- reads ------------------------------------------------------------

    async def read_slots(
        self, *, organization_id: int, workflow_id: int, effective_describer=None
    ) -> dict[str, Any]:
        workflow = await self._workflow(organization_id, workflow_id)
        states = {s.slot: s for s in await self.repo.list_slot_states(workflow_id)}
        released = getattr(workflow, "released_definition", None)
        released_configs = getattr(released, "workflow_configurations", None)
        effective = await effective_describer(workflow) if effective_describer else {}
        slots = []
        for slot in SLOTS:
            state = states.get(slot)
            published = draft = None
            if state and state.published_version:
                row = await self.repo.get_slot_setting(
                    workflow_id, slot, state.published_version
                )
                published = version_view(row) if row else None
            if state and state.draft_version:
                row = await self.repo.get_slot_setting(
                    workflow_id, slot, state.draft_version
                )
                draft = version_view(row) if row else None
            slots.append(
                {
                    "slot": slot,
                    "revision": state.revision if state else 0,
                    "source": (
                        "unconfigured"
                        if not (state and state.published_version)
                        and effective_describer is not None
                        and effective.get(slot) is None
                        else slot_source(
                            slot,
                            state.published_version if state else None,
                            released_configs,
                        )
                    ),  # a slot with an ``effective.error`` keeps its real source
                    "published_version": state.published_version if state else None,
                    "draft_version": state.draft_version if state else None,
                    "published": published,
                    "draft": draft,
                    "effective": effective.get(slot),
                }
            )
        return {
            "workflow_id": workflow.id,
            "workflow_uuid": workflow.workflow_uuid,
            "template_status": workflow.slot_template_status,
            "slots": slots,
        }

    async def history(
        self, *, organization_id: int, workflow_id: int, slot: str, limit: int = 50
    ) -> dict[str, Any]:
        await self._workflow(organization_id, workflow_id)
        state = await self.repo.get_slot_state(workflow_id, slot)
        rows = await self.repo.list_slot_settings(workflow_id, slot, limit)
        return {
            "workflow_id": workflow_id,
            "slot": slot,
            "revision": state.revision if state else 0,
            "versions": [version_view(r) for r in rows],
        }

    # ---- template / backfill ---------------------------------------------

    async def plan_snapshot(
        self, *, effective, slots: tuple[str, ...] = SLOTS
    ) -> list[dict[str, Any]]:
        """Split an effective configuration into (secret-free config, secret payload)."""
        plan = []
        for slot in slots:
            service = getattr(effective, slot, None)
            if service is None:
                continue
            data = service.model_dump(mode="json", exclude_none=True)
            data["api_key"] = service.get_all_api_keys() or None
            if data["api_key"] and len(data["api_key"]) == 1:
                data["api_key"] = data["api_key"][0]
            secret = {
                k: data[k]
                for k in SERVICE_SECRET_FIELDS
                if data.get(k) not in (None, "", [])
            }
            config = {k: v for k, v in data.items() if k not in SERVICE_SECRET_FIELDS}
            kind = None
            note = None
            if "credentials" in secret and "api_key" in secret:
                note = "both api_key and credentials set; cannot snapshot"
            elif "credentials" in secret:
                kind = "service_account_json"
            elif "aws_access_key" in secret:
                kind = "aws_iam"
            elif "api_key" in secret:
                kind = "api_key"
            plan.append(
                {
                    "slot": slot,
                    "config": config,
                    "kind": kind,
                    "secret": secret,
                    "blocked": note,
                }
            )
        return plan

    async def seed_workflow_from_effective(
        self,
        *,
        organization_id: int,
        workflow_id: int,
        effective,
        origin: str,
        created_by: Optional[str],
    ) -> list[str]:
        """Snapshot a workflow's current effective config into published slot v1s."""
        await self._workflow(organization_id, workflow_id)
        touched = {
            s.slot
            for s in await self.repo.list_slot_states(workflow_id)
            if s.last_version
        }
        plan = [
            item
            for item in await self.plan_snapshot(effective=effective)
            if item["slot"] not in touched
        ]
        slots: dict[str, dict[str, Any]] = {}
        created: dict[str, Any] = {}  # slot -> credential row made for this seed
        try:
            for item in plan:
                if item["blocked"]:
                    raise SlotSettingsError(
                        422, "snapshot_blocked", f"{item['slot']}: {item['blocked']}"
                    )
                normalized = normalize_slot_config(item["slot"], item["config"])
                check_credential_compatibility(item["slot"], normalized, item["kind"])
                ref = version = None
                if item["kind"]:
                    cred = await self.create_credential(
                        organization_id=organization_id,
                        kind=item["kind"],
                        secret=item["secret"],
                        credential_ref=None,
                        label=f"{origin} snapshot: workflow {workflow_id} {item['slot']}",
                        source_ref=None,
                        created_by=created_by,
                    )
                    created[item["slot"]] = cred
                    ref, version = cred.credential_ref, cred.version
                slots[item["slot"]] = {
                    "config": normalized,
                    "credential_ref": ref,
                    "credential_version": version,
                }
            seeded = await self.repo.seed_published_slots(
                workflow_id=workflow_id,
                slots=slots,
                origin=origin,
                created_by=created_by,
            )
        except Exception:
            await self._discard_unused_credentials(organization_id, created.values())
            raise
        # A slot that gained history between planning and seeding was skipped;
        # its freshly made copy of the secret is unreferenced.
        await self._discard_unused_credentials(
            organization_id, [c for slot, c in created.items() if slot not in seeded]
        )
        return seeded

    async def _discard_unused_credentials(self, organization_id: int, creds) -> None:
        """Revoke credential copies no slot pins, so a failed or repeated template
        copy does not leave encrypted duplicates of the organization's keys behind.
        Best effort: it never masks the error being handled."""
        for cred in list(creds):
            try:
                await self.repo.revoke_provider_credential_version(
                    organization_id, cred.credential_ref, cred.version
                )
            except Exception:  # noqa: BLE001
                pass


async def apply_organization_template(
    *, repo, organization_id: int, workflow_id: int, created_by: Optional[str]
) -> list[str]:
    """Copy the organization's current configuration into a workflow's own slots.

    Used for a brand-new workflow and to retry a ``pending`` one. On success the
    workflow's ``pending`` marker is cleared; on any failure the marker stays, so
    the workflow keeps refusing to run. Raises ``SlotSettingsError`` whose message
    and code never carry secret material (only the exception type is surfaced for
    unexpected errors).
    """
    from api.services.configuration.ai_model_configuration import (
        get_resolved_ai_model_configuration,
    )

    try:
        resolved = await get_resolved_ai_model_configuration(
            organization_id=organization_id
        )
        effective = resolved.effective
        if not any(getattr(effective, slot, None) for slot in SLOTS):
            raise SlotSettingsError(
                422,
                "template_empty",
                "the organization has no model configuration to copy into the workflow",
            )
        seeded = await WorkflowSlotService(repo).seed_workflow_from_effective(
            organization_id=organization_id,
            workflow_id=workflow_id,
            effective=effective,
            origin="template",
            created_by=created_by,
        )
        await repo.set_workflow_slot_template_status(workflow_id, None)
        return seeded
    except SlotSettingsError:
        raise
    except secret_store.SecretStoreUnavailable as exc:
        raise SlotSettingsError(503, "secret_store_unavailable", str(exc)) from None
    except Exception as exc:
        # Type only: the message of a DB or validation error may carry input.
        raise SlotSettingsError(
            503, "template_seed_failed", f"template copy failed ({type(exc).__name__})"
        ) from None
