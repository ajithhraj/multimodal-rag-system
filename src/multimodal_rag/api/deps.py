from __future__ import annotations

import secrets

from fastapi import Depends, HTTPException, Request, status

from multimodal_rag.engine import MultimodalRAG

_engine: MultimodalRAG | None = None


def get_engine() -> MultimodalRAG:
    global _engine
    if _engine is None:
        _engine = MultimodalRAG.from_settings()
    return _engine


def _get_header(headers, key: str) -> str | None:
    return headers.get(key) or headers.get(key.lower()) or headers.get(key.upper())


def resolve_tenant_id(
    request: Request,
    engine: MultimodalRAG = Depends(get_engine),
) -> str:
    settings = engine.settings
    headers = request.headers

    tenant_header = settings.auth_tenant_header
    api_key_header = settings.auth_api_key_header
    requested_tenant_raw = _get_header(headers, tenant_header)

    if not settings.auth_enabled:
        if requested_tenant_raw:
            return settings.normalize_tenant_id(requested_tenant_raw)
        return settings.normalize_tenant_id(settings.default_tenant)

    tenant_key_map = settings.parse_tenant_key_map()
    if not tenant_key_map:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Auth is enabled but no tenant API key map is configured.",
        )

    provided_key = _get_header(headers, api_key_header)
    if not provided_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Missing required header: {api_key_header}",
        )

    authenticated_tenant: str | None = None
    for tenant, key in tenant_key_map.items():
        if secrets.compare_digest(provided_key, key):
            authenticated_tenant = tenant
            break

    if authenticated_tenant is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid API key.",
        )

    if requested_tenant_raw:
        requested_tenant = settings.normalize_tenant_id(requested_tenant_raw)
        if requested_tenant != authenticated_tenant:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="API key is not authorized for requested tenant.",
            )

    return authenticated_tenant
