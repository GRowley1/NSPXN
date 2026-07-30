import json
import logging
import os
import re
import secrets
from datetime import datetime
from urllib.parse import parse_qs, quote

from fastapi import HTTPException

from auth_phase1 import (
    auth_app,
    enforce_account_for_upload,
    init_auth_db,
    log_usage_event,
    require_auth_from_authorization_header,
    validate_public_report_token,
    mark_public_report_started,
    mark_public_report_completed,
    mark_public_report_downloaded,
)
# Heavy report apps are lazy-loaded below. Do NOT import them at startup.
# Importing all report modules loads duplicate Presidio/spaCy stacks and can
# exceed Render memory before a request even completes.
comprehensive_app = None
photos_only_app = None
diminished_value_app = None

DV_INTENTS = {
    "preliminary_diminished_value_screening",
    "preliminary_diminished_value",
    "diminished_value_screening",
    "dv_screening",
}


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


def _get_diminished_value_app():
    global diminished_value_app
    if diminished_value_app is None:
        logging.warning("NSPXN LAZY LOAD diminished_value_app")
        from main_diminished_value_locked import app as loaded_app
        diminished_value_app = loaded_app
    return diminished_value_app


def _normalize_ai_intent_value(value: str) -> str:
    s = str(value or "").strip().lower()
    s = re.sub(r"[\s\-]+", "_", s)
    return s


init_auth_db()
logging.warning("NSPXN ROUTER LAZY TRUE-SWALLOW DV ROUTE ACTIVE 2026-05-27")


_ALLOWED_CORS_ORIGINS = {
    "https://nspxn.com",
    "https://www.nspxn.com",
    "http://nspxn.com",
    "http://www.nspxn.com",
    "http://localhost",
    "http://localhost:3000",
    "http://127.0.0.1:5500",
}

MAX_UPLOAD_BODY_BYTES = int(os.getenv("NSPXN_MAX_UPLOAD_MB", "25")) * 1024 * 1024


def _extract_simple_form_field(body: bytes, content_type: str, field_name: str) -> str:
    """Extract one exact simple text field from urlencoded or multipart body.

    This intentionally does NOT use loose regex scanning for multipart fields.
    Loose regex drift was reading later fields such as appraiser_id/NSPXN ID as
    file_number. Multipart extraction must be boundary-based and exact-name only.
    """
    ct = (content_type or "").lower()
    target_name = str(field_name or "").strip()

    if "application/x-www-form-urlencoded" in ct:
        try:
            parsed = parse_qs(body.decode("utf-8", "ignore"), keep_blank_values=True)
            return str((parsed.get(target_name) or [""])[0]).strip()
        except Exception:
            return ""

    if "multipart/form-data" in ct:
        try:
            boundary_match = re.search(r"boundary=([^;]+)", content_type or "", flags=re.IGNORECASE)
            if not boundary_match:
                return ""
            boundary = boundary_match.group(1).strip().strip('"')
            marker = ("--" + boundary).encode("utf-8")

            for part in body.split(marker):
                if not part or part in (b"--", b"--\r\n", b"--\n"):
                    continue

                if b"\r\n\r\n" in part:
                    header_blob, value_blob = part.split(b"\r\n\r\n", 1)
                    line_ending = b"\r\n"
                elif b"\n\n" in part:
                    header_blob, value_blob = part.split(b"\n\n", 1)
                    line_ending = b"\n"
                else:
                    continue

                headers_txt = header_blob.decode("utf-8", "ignore")
                name_match = re.search(r'(?:^|;\s*)name="([^"]+)"', headers_txt, flags=re.IGNORECASE)
                if not name_match:
                    name_match = re.search(r"(?:^|;\s*)name=([^;\r\n]+)", headers_txt, flags=re.IGNORECASE)
                if not name_match:
                    continue

                part_name = name_match.group(1).strip().strip('"')
                if part_name != target_name:
                    continue

                # Never treat upload file parts as form text.
                if re.search(r'filename="', headers_txt, flags=re.IGNORECASE):
                    return ""

                # Remove multipart terminator/footer and surrounding CRLF only.
                value_blob = value_blob.strip()
                if value_blob.endswith(b"--"):
                    value_blob = value_blob[:-2].strip()
                value_blob = value_blob.rstrip(line_ending).strip()
                return value_blob.decode("utf-8", "ignore").strip()
        except Exception:
            return ""

    return ""


def _extract_simple_form_keys(body: bytes, content_type: str) -> list[str]:
    """Return non-file multipart/urlencoded field names for safe diagnostics."""
    ct = (content_type or "").lower()
    out = []
    if "application/x-www-form-urlencoded" in ct:
        try:
            parsed = parse_qs(body.decode("utf-8", "ignore"), keep_blank_values=True)
            return [str(k) for k in parsed.keys()]
        except Exception:
            return out

    if "multipart/form-data" in ct:
        try:
            boundary_match = re.search(r"boundary=([^;]+)", content_type or "", flags=re.IGNORECASE)
            if not boundary_match:
                return out
            boundary = boundary_match.group(1).strip().strip('"')
            marker = ("--" + boundary).encode("utf-8")
            for part in body.split(marker):
                if not part or part in (b"--", b"--\r\n", b"--\n"):
                    continue
                if b"\r\n\r\n" in part:
                    header_blob = part.split(b"\r\n\r\n", 1)[0]
                elif b"\n\n" in part:
                    header_blob = part.split(b"\n\n", 1)[0]
                else:
                    continue
                headers_txt = header_blob.decode("utf-8", "ignore")
                if re.search(r'filename="', headers_txt, flags=re.IGNORECASE):
                    continue
                name_match = re.search(r'(?:^|;\s*)name="([^"]+)"', headers_txt, flags=re.IGNORECASE)
                if not name_match:
                    name_match = re.search(r"(?:^|;\s*)name=([^;\r\n]+)", headers_txt, flags=re.IGNORECASE)
                if name_match:
                    out.append(name_match.group(1).strip().strip('"'))
        except Exception:
            return out
    return out


def _extract_ai_intent_from_body(body: bytes, content_type: str) -> str:
    """Robustly extract ai_intent. Do not let photos-only silently fall into comprehensive."""
    return _normalize_ai_intent_value(_extract_simple_form_field(body, content_type, "ai_intent"))


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
        (b"access-control-allow-headers", b"Authorization,Content-Type,Accept,Origin,X-NSPXN-AI-Intent,X-NSPXN-Public-Token"),
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

    async def _handle_public_instant_report(self, scope, receive, send):
        path = scope.get("path", "")
        method = scope.get("method", "").upper()
        headers = _header_dict(scope)
        public_token = (headers.get("x-nspxn-public-token") or "").strip()

        # Browser downloads may carry the token in the query string because a normal
        # download link cannot attach a custom header.
        if not public_token:
            try:
                query = parse_qs((scope.get("query_string") or b"").decode("utf-8", "ignore"), keep_blank_values=True)
                public_token = str((query.get("token") or [""])[0]).strip()
            except Exception:
                public_token = ""

        try:
            purchase = validate_public_report_token(public_token, allow_completed=True)
        except HTTPException as exc:
            status_code, payload = _blocked_payload_from_exception(exc)
            await _send_json(send, status_code, payload, scope=scope)
            return

        if path == "/public/download-pdf":
            if purchase.status != "report_completed":
                await _send_json(send, 403, {"error": "Report is not ready for download."}, scope=scope)
                return
            target_app = _get_photos_only_app() if purchase.report_type == "damage_report_from_photos" else _get_diminished_value_app()
            file_number = (purchase.file_number or "").strip()
            if not file_number:
                await _send_json(send, 404, {"error": "Report file number is missing."}, scope=scope)
                return
            new_scope = dict(scope)
            new_scope["path"] = "/download-pdf"
            new_scope["raw_path"] = b"/download-pdf"
            new_scope["query_string"] = ("file_number=" + quote(file_number, safe="")).encode("utf-8")
            response_status = {"code": 500}

            async def download_send(message):
                if message["type"] == "http.response.start":
                    response_status["code"] = int(message.get("status", 500))
                await send(message)
                if message["type"] == "http.response.body" and not message.get("more_body", False):
                    if 200 <= response_status["code"] < 300:
                        mark_public_report_downloaded(public_token)

            await target_app(new_scope, receive, download_send)
            return

        if method != "POST":
            await _send_json(send, 405, {"error": "Method not allowed."}, scope=scope)
            return
        if purchase.status not in {"paid", "report_started"}:
            await _send_json(send, 403, {"error": "This Instant Report has already been completed."}, scope=scope)
            return

        body = b""
        while True:
            message = await receive()
            if message["type"] != "http.request":
                continue
            body += message.get("body", b"")
            if not message.get("more_body", False):
                break

        if len(body) > MAX_UPLOAD_BODY_BYTES:
            await _send_json(send, 413, {"error": "Upload too large.", "max_upload_mb": int(MAX_UPLOAD_BODY_BYTES / 1024 / 1024)}, scope=scope)
            return

        content_type = headers.get("content-type", "")
        ai_intent = _extract_ai_intent_from_body(body, content_type)
        if ai_intent != purchase.report_type:
            await _send_json(send, 403, {"error": "Purchased report type does not match the submitted report type."}, scope=scope)
            return

        file_number = (
            _extract_simple_form_field(body, content_type, "file_number")
            or _extract_simple_form_field(body, content_type, "file-number")
            or _extract_simple_form_field(body, content_type, "fileNumber")
        ).strip()
        if not file_number:
            file_number = "PPR-" + datetime.utcnow().strftime("%Y%m%d-") + secrets.token_hex(3).upper()
            body = _replace_or_add_form_field(body, content_type, "file_number", file_number)
        body = _replace_or_add_form_field(body, content_type, "appraiser_id", "NSPXN Instant Reports")
        if not _extract_simple_form_field(body, content_type, "ia_company"):
            body = _replace_or_add_form_field(body, content_type, "ia_company", "NSPXN Instant Reports")
        new_scope = _scope_with_updated_content_length(scope, body)
        new_scope["path"] = "/vision-review"
        new_scope["raw_path"] = b"/vision-review"

        mark_public_report_started(public_token, file_number)
        target_app = _get_photos_only_app() if ai_intent == "damage_report_from_photos" else _get_diminished_value_app()
        sent = False
        async def replay_receive():
            nonlocal sent
            if not sent:
                sent = True
                return {"type": "http.request", "body": body, "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        state = {"status": 500, "headers": {}, "body": bytearray()}
        async def swallow(message):
            if message["type"] == "http.response.start":
                state["status"] = int(message.get("status", 500))
                state["headers"] = {k.decode("latin1").lower(): v.decode("latin1") for k, v in message.get("headers", [])}
            elif message["type"] == "http.response.body" and len(state["body"]) < 65536:
                state["body"].extend((message.get("body", b"") or b"")[:65536-len(state["body"])])

        try:
            await target_app(new_scope, replay_receive, swallow)
        except Exception as exc:
            logging.exception("NSPXN Instant Report processing failed", exc_info=exc)
            await _send_json(send, 500, {"error": "Instant Report processing failed.", "detail": str(exc)}, scope=scope)
            return

        payload = None
        try:
            payload = json.loads(bytes(state["body"]).decode("utf-8", "ignore"))
        except Exception:
            payload = None
        if 200 <= state["status"] < 300 and not (isinstance(payload, dict) and payload.get("status") == "blocked"):
            pdf_filename = str((payload or {}).get("pdf_filename") or "").strip() if isinstance(payload, dict) else ""
            mark_public_report_completed(public_token, file_number, pdf_filename or None)
            download_url = f"/public/download-pdf?token={quote(public_token, safe='')}"
            await _send_json(send, 200, {"status": "success", "message": "Completed. Download the PDF below.", "file_number": file_number, "download_url": download_url}, scope=scope)
            return

        error_payload = payload if isinstance(payload, dict) else {"error": "Instant Report was not completed."}
        await _send_json(send, state["status"], error_payload, scope=scope)

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

        # PayPal/config/token validation routes live on the auth/database app.
        if path.startswith("/public/paypal/") or path in {"/public/instant-reports/config", "/public/instant-reports/validate"}:
            await self.auth(scope, receive, send)
            return

        if path in {"/public/vision-review", "/public/download-pdf"}:
            await self._handle_public_instant_report(scope, receive, send)
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
        ai_intent = _normalize_ai_intent_value(headers.get("x-nspxn-ai-intent") or "")
        if not ai_intent:
            ai_intent = _extract_ai_intent_from_body(body, content_type)

        if ai_intent == "damage_report_from_photos":
            target_app = _get_photos_only_app()
        elif ai_intent == "comprehensive":
            target_app = _get_comprehensive_app()
        elif ai_intent in DV_INTENTS:
            target_app = _get_diminished_value_app()
        else:
            await _send_json(
                send,
                400,
                {
                    "error": "Missing or invalid AI Review request type.",
                    "detail": "Missing or invalid ai_intent. Select Comprehensive, Preliminary Diminished Value Screening, or Create a Condition/Damage Report from Photos and resubmit.",
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

        # Guard against multipart parser drift: file_number must never be the
        # authenticated NSPXN User ID/appraiser_id.
        if re.fullmatch(r"(?i)NSPXN\d+", str(file_number or "").strip()):
            logging.error(
                "NSPXN ROUTER FIELD ERROR: parsed file_number was NSPXN user id; clearing value. ai_intent=%s",
                ai_intent,
            )
            file_number = ""

        if ai_intent in DV_INTENTS:
            try:
                _form_keys = _extract_simple_form_keys(body, content_type)
                _state_keys = [k for k in _form_keys if any(tok in str(k).lower() for tok in ("state", "jurisdiction", "location"))]
                logging.warning(
                    "NSPXN DV ROUTER FIELD DEBUG file_number_present=%s state_keys=%s form_keys=%s",
                    bool(str(file_number or "").strip()),
                    _state_keys[:20],
                    _form_keys[:80],
                )
            except Exception:
                pass

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
                    "NSPXN ROUTER PDF-ONLY RESPONSE returning compact success file_number=%s ai_intent=%s mounted_body_bytes=%s",
                    file_number,
                    ai_intent,
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
            logging.warning(
                "NSPXN ROUTER TRUE-SWALLOW PDF-ONLY RESPONSE intercepting /vision-review file_number=%s ai_intent=%s",
                file_number,
                ai_intent,
            )
            await target_app(scope, replay_receive, tracking_send)
            if not response_state.get("sent_to_client"):
                await _send_compact_response(b"".join(captured_chunks))
        except Exception as exc:
            logging.exception("NSPXN Comprehensive/Photos/DV router exception", exc_info=exc)
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

            # For Comprehensive and Diminished Value reports, only count fully completed
            # report responses that set the analytics header. This avoids counting blocked
            # quality-gate output as completed usage.
            if (ai_intent not in ({"comprehensive"} | DV_INTENTS)) or report_completed:
                log_usage_event(
                    current_user=current_user,
                    ai_intent=ai_intent,
                    file_number=logged_file_number,
                    status="completed",
                    compliance_score=compliance_score,
                    score_source=score_source,
                )


app = IntentRouterApp(auth_app)
