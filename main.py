import re
from urllib.parse import parse_qs

from main_comprehensive_locked import app as comprehensive_app
from main_photos_only_locked import app as photos_only_app


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


class IntentRouterApp:
    def __init__(self, comprehensive, photos_only):
        self.comprehensive = comprehensive
        self.photos_only = photos_only

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.comprehensive(scope, receive, send)
            return

        path = scope.get("path", "")

        # Keep all non-/vision-review routes on the comprehensive app
        if path != "/vision-review":
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

        headers = {
            k.decode("latin1").lower(): v.decode("latin1")
            for k, v in scope.get("headers", [])
        }
        content_type = headers.get("content-type", "")
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


app = IntentRouterApp(comprehensive_app, photos_only_app)
