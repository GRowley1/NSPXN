import json
import re
from urllib.parse import parse_qs

from fastapi import HTTPException

from auth_phase1 import auth_app, init_auth_db, require_auth_from_authorization_header
from main_comprehensive_locked import app as comprehensive_app
from main_photos_only_locked import app as photos_only_app


init_auth_db()


_ALLOWED_CORS_ORIGINS = {
    "https://nspxn.com",
    "https://www.nspxn.com",
    "http://nspxn.com",
    "http://www.nspxn.com",
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
}


def _extract_ai_intent_from_body(body: bytes, content_type: str) -> str:
    ct = (content_type or "").lower()

    if "application/x-www-form-urlencoded" in ct:
        try:
            parsed = parse_qs(body.decode("utf-8", "ignore"))
            return str((parsed.get("ai_intent") or [""])[0]).strip().lower()
        except Exception:
            return ""

    if "multipart/form-data" in ct:
        try:
            m = re.search(
                rb'name="ai_intent"\r\n(?:[^\r\n]*\r\n)*\r\n([^\r\n]+)',
                body,
                flags=re.IGNORECASE,
            )
            if m:
                return m.group(1).decode("utf-8", "ignore").strip().lower()
        except Exception:
            return ""

    return ""


def _cors_headers(scope):
    request_headers = {
        k.decode("latin1").lower(): v.decode("latin1")
        for k, v in scope.get("headers", [])
    }
    origin = request_headers.get("origin", "")

    allow_origin = "*"
    if origin in _ALLOWED_CORS_ORIGINS or re.match(r"^https://.*\.nspxn\.com$", origin or ""):
        allow_origin = origin

    return [
        (b"access-control-allow-origin", allow_origin.encode("latin1")),
        (b"access-control-allow-methods", b"GET,POST,OPTIONS"),
        (b"access-control-allow-headers", b"Authorization,Content-Type,Accept,Origin"),
        (b"access-control-allow-credentials", b"true"),
        (b"access-control-max-age", b"86400"),
    ]


async def _send_json(send, status_code: int, payload: dict, scope=None):
    body = json.dumps(payload).encode("utf-8")
    headers = [
        (b"content-type", b"application/json"),
        (b"content-length", str(len(body)).encode("latin1")),
    ]
    if scope is not None:
        headers.extend(_cors_headers(scope))
    await send({"type": "http.response.start", "status": status_code, "headers": headers})
    await send({"type": "http.response.body", "body": body})


async def _send_options(send, scope):
    await send({"type": "http.response.start", "status": 204, "headers": _cors_headers(scope)})
    await send({"type": "http.response.body", "body": b""})


def _header_dict(scope):
    return {
        k.decode("latin1").lower(): v.decode("latin1")
        for k, v in scope.get("headers", [])
    }


def _replace_or_add_form_field(body: bytes, content_type: str, field_name: str, value: str) -> bytes:
    ct = (content_type or "").lower()
    safe_value = (value or "").encode("utf-8")

    if "application/x-www-form-urlencoded" in ct:
        try:
            parsed = parse_qs(body.decode("utf-8", "ignore"), keep_blank_values=True)
            parsed[field_name] = [value or ""]
            from urllib.parse import urlencode
            return urlencode(parsed, doseq=True).encode("utf-8")
        except Exception:
            return body

    if "multipart/form-data" not in ct:
        return body

    try:
        # Replace existing simple text field value. The frontend already sends appraiser_id.
        pattern = (
            rb'(name="' + re.escape(field_name.encode("utf-8")) +
            rb'"\r\n(?:[^\r\n]*\r\n)*\r\n)([^\r\n]*)'
        )
        if re.search(pattern, body, flags=re.IGNORECASE):
            return re.sub(pattern, rb'\g<1>' + safe_value, body, count=1, flags=re.IGNORECASE)

        # If the field is missing, add it before the closing multipart boundary.
        boundary_match = re.search(r"boundary=([^;]+)", content_type or "", flags=re.IGNORECASE)
        if not boundary_match:
            return body

        boundary = boundary_match.group(1).strip().strip('"')
        closing = ("\r\n--" + boundary + "--").encode("utf-8")
        if closing not in body:
            return body

        insertion = (
            "\r\n--" + boundary +
            f"\r\nContent-Disposition: form-data; name=\"{field_name}\"\r\n\r\n" +
            (value or "")
        ).encode("utf-8")
        return body.replace(closing, insertion + closing, 1)
    except Exception:
        return body


def _scope_with_updated_content_length(scope, body: bytes):
    headers = []
    replaced = False
    for k, v in scope.get("headers", []):
        if k.lower() == b"content-length":
            headers.append((k, str(len(body)).encode("latin1")))
            replaced = True
        else:
            headers.append((k, v))
    if not replaced:
        headers.append((b"content-length", str(len(body)).encode("latin1")))

    new_scope = dict(scope)
    new_scope["headers"] = headers
    return new_scope


class IntentRouterApp:
    def __init__(self, comprehensive, photos_only, auth):
        self.comprehensive = comprehensive
        self.photos_only = photos_only
        self.auth = auth

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.comprehensive(scope, receive, send)
            return

        path = scope.get("path", "")
        method = scope.get("method", "").upper()

        if method == "OPTIONS":
            await _send_options(send, scope)
            return

        if path in {"/login", "/me"}:
            await self.auth(scope, receive, send)
            return

        # Keep all non-/vision-review and non-/download-pdf routes on the comprehensive app.
        if path not in {"/vision-review", "/download-pdf"}:
            await self.comprehensive(scope, receive, send)
            return

        headers = _header_dict(scope)
        try:
            current_user = require_auth_from_authorization_header(headers.get("authorization"))
        except HTTPException as exc:
            detail = exc.detail if isinstance(exc.detail, str) else "Unauthorized."
            await _send_json(send, exc.status_code, {"detail": detail, "error": detail}, scope=scope)
            return
        except Exception:
            await _send_json(send, 401, {"detail": "Invalid login token.", "error": "Invalid login token."}, scope=scope)
            return

        if path == "/download-pdf":
            await self.comprehensive(scope, receive, send)
            return

        body = b""
        while True:
            message = await receive()
            if message["type"] != "http.request":
                continue
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break

        content_type = headers.get("content-type", "")
        body = _replace_or_add_form_field(body, content_type, "appraiser_id", current_user.nspxn_id)
        scope = _scope_with_updated_content_length(scope, body)
        ai_intent = _extract_ai_intent_from_body(body, content_type)

        target_app = (
            self.photos_only
            if ai_intent == "damage_report_from_photos"
            else self.comprehensive
        )

        sent = False

        async def replay_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {
                    "type": "http.request",
                    "body": body,
                    "more_body": False,
                }
            return {
                "type": "http.request",
                "body": b"",
                "more_body": False,
            }

        await target_app(scope, replay_receive, send)


app = IntentRouterApp(comprehensive_app, photos_only_app, auth_app)
