import json
import logging
import re
from urllib.parse import parse_qs, quote

from fastapi import HTTPException

from auth_phase1 import (
    auth_app,
    enforce_account_for_upload,
    init_auth_db,
    log_usage_event,
    require_auth_from_authorization_header,
)
# Heavy report apps are lazy-loaded below. Do NOT import them at startup.
# Importing both comprehensive and photos-only modules loads duplicate Presidio/spaCy
# stacks and can exceed Render memory before a request even completes.
comprehensive_app = None
photos_only_app = None


def _get_comprehensive_app():
    global comprehensive_app
    if comprehensive_app is None:
        logging.warning("NSPXN LAZY LOAD comprehensive_app")
        from main_comprehensive_locked import app as loaded_app
        comprehensive_app = loaded_app
    return comprehensive_app


def _get_photos_only_app():
    global photos_only_app
    if photos_only_app is None:
        logging.warning("NSPXN LAZY LOAD photos_only_app")
        from main_photos_only_locked import app as loaded_app
        photos_only_app = loaded_app
    return photos_only_app


init_auth_db()
logging.warning("NSPXN ROUTER LAZY PDF-ONLY RESPONSE LOCK ACTIVE 2026-05-27")


_ALLOWED_CORS_ORIGINS = {
    "https://nspxn.com",
    "https://www.nspxn.com",
    "http://nspxn.com",
    "http://www.nspxn.com",
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
}

MAX_UPLOAD_BODY_BYTES = 25 * 1024 * 1024  # allow uploads up to 25 MB


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
        # Prefer boundary-based parsing. This avoids regex drift where file_number
        # can accidentally be read as appraiser_id/NSPXN ID from a later field.
        try:
            boundary_match = re.search(r"boundary=([^;]+)", content_type or "", flags=re.IGNORECASE)
            if boundary_match:
                boundary = boundary_match.group(1).strip().strip('"')
                marker = ("--" + boundary).encode("utf-8")
                for part in body.split(marker):
                    if not part or part in (b"--", b"--\r\n"):
                        continue
                    if b"\r\n\r\n" in part:
                        header_blob, value_blob = part.split(b"\r\n\r\n", 1)
                    elif b"\n\n" in part:
                        header_blob, value_blob = part.split(b"\n\n", 1)
                    else:
                        continue
                    headers_txt = header_blob.decode("utf-8", "ignore")
                    if re.search(r'name="' + re.escape(field_name) + r'"', headers_txt, flags=re.IGNORECASE):
                        if re.search(r'filename="', headers_txt, flags=re.IGNORECASE):
                            continue
                        value_blob = value_blob.replace(b"\r\n--", b"--")
                        return value_blob.strip().rstrip(b"-").strip().decode("utf-8", "ignore").strip()
        except Exception:
            pass

        # Legacy fallback regexes.
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
    def __init__(self, auth):
        self.auth = auth

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await _get_comprehensive_app()(scope, receive, send)
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
            await _get_comprehensive_app()(scope, receive, send)
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
            await _get_comprehensive_app()(scope, receive, send)
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
        if len(body) > MAX_UPLOAD_BODY_BYTES:
            await _send_json(
                send,
                413,
                {
                    "error": "Upload too large.",
                    "detail": "Maximum upload size is 25 MB. Please reduce the file size and resubmit.",
                    "max_upload_mb": 25,
                },
                scope=scope,
            )
            return
        body = _replace_or_add_form_field(body, content_type, "appraiser_id", current_user.nspxn_id)
        scope = _scope_with_updated_content_length(scope, body)

        # Prefer explicit frontend header, then fall back to multipart body parsing.
        # Never silently default to comprehensive when the user selected photos-only.
        ai_intent = (headers.get("x-nspxn-ai-intent") or "").strip().lower()
        if not ai_intent:
            ai_intent = _extract_ai_intent_from_body(body, content_type)

        if ai_intent == "damage_report_from_photos":
            target_app = _get_photos_only_app()
        elif ai_intent == "comprehensive":
            target_app = _get_comprehensive_app()
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

        response_state = {
            "status": 500,
            "headers": {},
            "started": False,
            "completed": False,
            "sent_to_client": False,
            "body_bytes": 0,
        }
        captured_chunks = []
        CAPTURE_LIMIT = 64 * 1024  # enough for blocked/error JSON; success bodies are discarded

        def _compact_download_url() -> str:
            safe_file = quote(str(file_number or ""), safe="")
            return f"/download-pdf?file_number={safe_file}"

        async def _send_compact_response(raw_body: bytes = b""):
            status_code = int(response_state.get("status", 500))
            payload = None
            if raw_body:
                try:
                    payload = json.loads(raw_body.decode("utf-8", "ignore"))
                except Exception:
                    payload = None

            if 200 <= status_code < 300:
                if isinstance(payload, dict) and payload.get("status") == "blocked":
                    response_state["sent_to_client"] = True
                    await _send_json(send, 200, payload, scope=scope)
                    return

                download_url = _compact_download_url()
                compact_payload = {
                    "status": "success",
                    "message": "Completed. Download the PDF below.",
                    "file_number": file_number,
                    "download_url": download_url,
                    "pdf_url": download_url,
                }
                logging.warning(
                    "NSPXN ROUTER PDF-ONLY RESPONSE returning compact success file_number=%s mounted_body_bytes=%s",
                    file_number,
                    response_state.get("body_bytes"),
                )
                response_state["sent_to_client"] = True
                await _send_json(send, 200, compact_payload, scope=scope)
                return

            error_payload = payload if isinstance(payload, dict) else {
                "status": "error" if status_code >= 500 else "blocked",
                "error": "Report processing failed." if status_code >= 500 else "Report was not completed.",
                "detail": (raw_body.decode("utf-8", "ignore")[:1000] if raw_body else "No response body returned."),
                "file_number": file_number,
            }
            response_state["sent_to_client"] = True
            await _send_json(send, status_code, error_payload, scope=scope)

        async def tracking_send(message):
            # TRUE SWALLOW sender for mounted /vision-review app.
            # Never forward http.response.start or http.response.body from the mounted app.
            # We collect headers/status and at most CAPTURE_LIMIT bytes only so blocked/error
            # JSON can still be surfaced. The browser receives one compact response after
            # the mounted app fully returns.
            if message["type"] == "http.response.start":
                response_state["started"] = True
                response_state["status"] = int(message.get("status", 500))
                response_state["headers"] = {
                    k.decode("latin1").lower(): v.decode("latin1")
                    for k, v in message.get("headers", [])
                }
                return

            if message["type"] == "http.response.body":
                chunk = message.get("body", b"") or b""
                response_state["body_bytes"] += len(chunk)
                current_capture = sum(len(c) for c in captured_chunks)
                if current_capture < CAPTURE_LIMIT:
                    captured_chunks.append(chunk[: max(0, CAPTURE_LIMIT - current_capture)])
                if not message.get("more_body", False):
                    response_state["completed"] = True
                return

        try:
            logging.warning("NSPXN ROUTER TRUE-SWALLOW PDF-ONLY RESPONSE intercepting /vision-review file_number=%s ai_intent=%s", file_number, ai_intent)
            await target_app(scope, replay_receive, tracking_send)
            if not response_state.get("sent_to_client"):
                await _send_compact_response(b"".join(captured_chunks))
        except Exception as exc:
            logging.exception("NSPXN Comprehensive/Photos router exception", exc_info=exc)
            if response_state.get("sent_to_client"):
                return

            detail = str(exc)
            exc_name = exc.__class__.__name__
            is_quota_error = (
                "insufficient_quota" in detail
                or "exceeded your current quota" in detail.lower()
                or "ratelimiterror" in exc_name.lower()
                or "rate limit" in detail.lower()
            )
            if is_quota_error:
                await _send_json(
                    send,
                    429,
                    {
                        "status": "blocked",
                        "error": "OpenAI quota/rate limit reached.",
                        "detail": "The AI report could not be completed because the OpenAI API quota or rate limit was reached. Check the OpenAI billing/usage limit, then resubmit.",
                        "ai_intent": ai_intent,
                        "file_number": file_number,
                    },
                    scope=scope,
                )
                return

            await _send_json(
                send,
                500,
                {
                    "error": "Report processing failed before completion.",
                    "detail": detail,
                    "ai_intent": ai_intent,
                    "file_number": file_number,
                },
                scope=scope,
            )
            return

        if 200 <= int(response_state.get("status", 500)) < 300:
            response_headers = response_state.get("headers", {}) or {}
            report_completed = str(response_headers.get("x-nspxn-report-completed") or "").strip().lower() == "true"

            # Prefer the completed report's own File # header. The multipart request body also
            # contains appraiser_id/NSPXN ID fields, and those can be confused with the report
            # file number when logging analytics.
            report_file_number = str(response_headers.get("x-nspxn-file-number") or "").strip()
            logged_file_number = report_file_number or file_number
            if logged_file_number == current_user.nspxn_id and report_file_number:
                logged_file_number = report_file_number

            compliance_score = None
            score_source = None
            if ai_intent == "comprehensive":
                score_header = str(response_headers.get("x-nspxn-compliance-score") or "").strip()
                if score_header:
                    try:
                        parsed_score = float(score_header)
                        if 0 <= parsed_score <= 100:
                            compliance_score = round(parsed_score, 1)
                            score_source = str(response_headers.get("x-nspxn-score-source") or "response_header").strip() or "response_header"
                    except Exception:
                        compliance_score = None
                        score_source = None

            # For Comprehensive reports, only count fully completed report responses that set
            # the analytics header. This avoids counting blocked comprehensive quality-gate output.
            if ai_intent != "comprehensive" or report_completed:
                log_usage_event(
                    current_user=current_user,
                    ai_intent=ai_intent,
                    file_number=logged_file_number,
                    status="completed",
                    compliance_score=compliance_score,
                    score_source=score_source,
                )


app = IntentRouterApp(auth_app)
