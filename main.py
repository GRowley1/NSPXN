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

        response_state = {"status": 500, "headers": {}, "started": False, "completed": False, "sent_to_client": False}
        compact_vision_response = path == "/vision-review"
        compact_body_chunks = []
        compact_body_bytes = 0
        MAX_COMPACT_CAPTURE_BYTES = 1024 * 1024  # enough for blocked JSON / current report JSON; prevents browser-heavy responses

        async def _send_compact_or_passthrough_body():
            """For /vision-review, never forward a huge report JSON body to the browser.
            The PDF/email are the durable outputs; the browser only needs a status and download URL.
            This router-level guard prevents older mounted modules from returning 200KB+ JSON payloads.
            """
            nonlocal compact_body_chunks, compact_body_bytes

            status_code = int(response_state.get("status", 500))
            raw_body = b"".join(compact_body_chunks) if compact_body_chunks else b""
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

                pdf_url = ""
                if isinstance(payload, dict):
                    pdf_url = str(
                        payload.get("absolute_pdf_url")
                        or payload.get("download_url")
                        or payload.get("pdf_url")
                        or payload.get("report_url")
                        or ""
                    ).strip()

                if not pdf_url:
                    pdf_url = f"/download-pdf?file_number={quote(str(file_number or ''))}"

                compact_payload = {
                    "status": "success",
                    "file_number": file_number,
                    "ai_intent": ai_intent,
                    "download_url": pdf_url,
                    "pdf_url": pdf_url,
                    "absolute_pdf_url": pdf_url,
                    "gpt_output": "Report complete. Click Download PDF.",
                    "summary_brief": "Report complete. Click Download PDF.",
                }

                # Preserve a few lightweight headers/fields if the mounted app returned them.
                if isinstance(payload, dict):
                    for key in ("claim_number", "vin", "vehicle", "compliance_score", "redaction_status", "pdf_filename"):
                        if payload.get(key) not in (None, ""):
                            compact_payload[key] = payload.get(key)

                response_state["sent_to_client"] = True
                await _send_json(send, 200, compact_payload, scope=scope)
                return

            # Non-success: forward a compact error instead of a large/invalid body.
            error_payload = payload if isinstance(payload, dict) else {
                "status": "blocked" if status_code < 500 else "error",
                "error": "Report processing failed." if status_code >= 500 else "Report was not completed.",
                "detail": (raw_body.decode("utf-8", "ignore")[:1000] if raw_body else "No response body returned."),
                "file_number": file_number,
                "ai_intent": ai_intent,
            }
            response_state["sent_to_client"] = True
            await _send_json(send, status_code, error_payload, scope=scope)

        async def tracking_send(message):
            # For /vision-review, intercept the mounted app response and replace any large
            # report JSON with a compact success/block JSON. For all other routes, pass through.
            if not compact_vision_response:
                if message["type"] == "http.response.start":
                    response_state["started"] = True
                    response_state["status"] = int(message.get("status", 500))
                    response_state["headers"] = {
                        k.decode("latin1").lower(): v.decode("latin1")
                        for k, v in message.get("headers", [])
                    }
                elif message["type"] == "http.response.body" and not message.get("more_body", False):
                    response_state["completed"] = True
                response_state["sent_to_client"] = True
                await send(message)
                return

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
                if compact_body_bytes + len(chunk) <= MAX_COMPACT_CAPTURE_BYTES:
                    compact_body_chunks.append(chunk)
                compact_body_bytes += len(chunk)
                if not message.get("more_body", False):
                    response_state["completed"] = True
                    await _send_compact_or_passthrough_body()
                return

        try:
            await target_app(scope, replay_receive, tracking_send)
        except Exception as exc:
            logging.exception("NSPXN Comprehensive/Photos router exception", exc_info=exc)
            # If a response was already sent to the client, do not send a second ASGI response.
            # In compact /vision-review mode, headers may be captured but not yet sent, so
            # started/completed alone must not suppress the router's error JSON.
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


app = IntentRouterApp(comprehensive_app, photos_only_app, auth_app)
