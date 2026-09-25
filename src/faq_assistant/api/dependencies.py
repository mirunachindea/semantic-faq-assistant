"""FastAPI dependencies: authentication and access to application services."""

import secrets
from collections.abc import Sequence
from typing import Annotated

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import SecretStr

from faq_assistant.config import Settings
from faq_assistant.container import Container
from faq_assistant.services.assistant import FAQAssistant

_bearer = HTTPBearer(auto_error=False, description="Static API token")


def get_container(request: Request) -> Container:
    """The application's dependency container (created in the lifespan handler)."""
    container: Container = request.app.state.container
    return container


def get_settings_dep(container: Annotated[Container, Depends(get_container)]) -> Settings:
    """Settings bound to the running application."""
    return container.settings


def get_assistant(container: Annotated[Container, Depends(get_container)]) -> FAQAssistant:
    """The FAQ assistant service."""
    return container.assistant


def _matches_any(candidate: str, tokens: Sequence[SecretStr]) -> bool:
    # Compare against every token without short-circuiting to avoid timing side channels.
    matched = False
    for token in tokens:
        matched |= secrets.compare_digest(candidate.encode(), token.get_secret_value().encode())
    return matched


def _unauthorized(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


def get_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> str:
    """Require a valid bearer token for regular API access.

    Admin tokens are accepted too, so operators can use the public endpoints.
    """
    if credentials is None:
        raise _unauthorized("Missing bearer token")
    if not _matches_any(credentials.credentials, [*settings.api_tokens, *settings.admin_tokens]):
        raise _unauthorized("Invalid bearer token")
    return credentials.credentials


def require_admin(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
    settings: Annotated[Settings, Depends(get_settings_dep)],
) -> str:
    """Require an admin bearer token."""
    if credentials is None:
        raise _unauthorized("Missing bearer token")
    if not _matches_any(credentials.credentials, settings.admin_tokens):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin token required")
    return credentials.credentials


TokenDep = Annotated[str, Depends(get_token)]
AdminDep = Annotated[str, Depends(require_admin)]
AssistantDep = Annotated[FAQAssistant, Depends(get_assistant)]
ContainerDep = Annotated[Container, Depends(get_container)]
