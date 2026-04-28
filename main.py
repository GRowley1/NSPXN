import json
import re
from urllib.parse import parse_qs

from fastapi import HTTPException

from auth_phase1 import (
    auth_app,
    enforce_account_for_upload,
    init_auth_db,
    log_usage_event,
    require_auth_from_authorization_header,
)
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


def _extract_simple_form_field(body: bytes, content_type: str, field_name: str) -> str:
    """Extract a simple text field from urlencoded or multipart body."""
    ct = (content_type or "").lower()

    if "application/x-www-form-urlencoded" in ct:
        try:
            parsed = parse_qs(body.decode("utf-8", "ignore"), keep_blank_values=True)
            return str((parsed.get(field_name) or [""])[0]).strip()
        except Exception:
            return ""

    if "multipart/form-data" in ct:
        try:
            escaped = re.escape(field_name.encode("utf-8"))
            patterns = [
                rb'name="' + escaped + rb'"(?:;[^\r\n]*)?\r?\n(?:[^\r\n]*\r?\n)*\r?\n([^\r\n-][^\r\n]*)',
                rb'name=' + escaped + rb'(?:;[^\r\n]*)?\r?\n(?:[^\r\n]*\r?\n)*\r?\n([^\r\n-][^\r\n]*)',
            ]
            for pattern in patterns:
                m = re.search(pattern, body, flags=re.IGNORECASE)
                if m:
                    return m.group(1).decode("utf-8", "ignore").strip()

            decoded = body.decode("utf-8", "ignore")
            m = re.search(
                r'name=["\']?' + re.escape(field_name) + r'["\']?.*?\r?\n\r?\n([^\r\n-]+)',
                decoded,
                flags=re.IGNORECASE | re.DOTALL,
            )
            if m:
                return m.group(1).strip()
        except Exception:
            return ""

    return ""


def _extract_ai_intent_from_body(body: bytes, content_type: str) -> str:
    """Robustly extract ai_intent. Do not let photos-only silently fall into comprehensive."""
    return _extract_simple_form_field(body, content_type, "ai_intent").lower()


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
        (b"access-control-allow-methods", b"GET,POST,PATCH,OPTIONS"),
        (b"access-control-allow-headers", b"Authorization,Content-Type,Accept,Origin,X-NSPXN-AI-Intent"),
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


def _blocked_payload_from_exception(exc: HTTPException) -> tuple[int, dict]:
    detail = exc.detail

    if isinstance(detail, dict):
        payload = dict(detail)
        payload.setdefault("status", "blocked")
        payload.setdefault("error", "Account blocked.")
        payload.setdefault("detail", payload.get("error", "Account blocked."))
        return exc.status_code, payload

    message = detail if isinstance(detail, str) else "Unauthorized."
    return exc.status_code, {"detail": message, "error": message}


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


def _is_completed_response(status_code: int, body: bytes) -> bool:
    """Count only successful, usable report responses."""
    if status_code < 200 or status_code >= 300:
        return False

    if not body:
        return True

    try:
        payload = json.loads(body.decode("utf-8", "ignore"))
        if isinstance(payload, dict) and payload.get("status") == "blocked":
            return False
    except Exception:
        pass

    return True


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

        if path in {"/login", "/me"} or path.startswith("/admin"):
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
            status_code, payload = _blocked_payload_from_exception(exc)
            await _send_json(send, status_code, payload, scope=scope)
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

        # Prefer explicit frontend header, then fall back to multipart body parsing.
        # Never silently default to comprehensive when the user selected photos-only.
        ai_intent = (headers.get("x-nspxn-ai-intent") or "").strip().lower()
        if not ai_intent:
            ai_intent = _extract_ai_intent_from_body(body, content_type)

        if ai_intent == "damage_report_from_photos":
            target_app = self.photos_only
        elif ai_intent == "comprehensive":
            target_app = self.comprehensive
        else:
            await _send_json(
                send,
                400,
                {
                    "error": "Missing or invalid AI Review request type.",
                    "detail": "Missing or invalid ai_intent. Select Comprehensive or Create a Condition/Damage Report from Photos and resubmit.",
                    "ai_intent_received": ai_intent,
                },
                scope=scope,
            )
            return

        try:
            enforce_account_for_upload(current_user)
        except HTTPException as exc:
            status_code, payload = _blocked_payload_from_exception(exc)
            await _send_json(send, status_code, payload, scope=scope)
            return

        file_number = (
            _extract_simple_form_field(body, content_type, "file_number")
            or _extract_simple_form_field(body, content_type, "file-number")
            or _extract_simple_form_field(body, content_type, "fileNumber")
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

        response_state = {"status": 500, "body": bytearray()}

        async def tracking_send(message):
            if message["type"] == "http.response.start":
                response_state["status"] = int(message.get("status", 500))
            elif message["type"] == "http.response.body":
                response_state["body"].extend(message.get("body", b"") or b"")
            await send(message)

        await target_app(scope, replay_receive, tracking_send)

        if _is_completed_response(response_state["status"], bytes(response_state["body"])):
            log_usage_event(
                current_user=current_user,
                ai_intent=ai_intent,
                file_number=file_number,
                status="completed",
            )


app = IntentRouterApp(comprehensive_app, photos_only_app, auth_app)
