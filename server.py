#!/usr/bin/env python3
"""
VisualReq backend — zero-dependency proxy server.

Serves the static HTML app AND proxies AI requests to Claude / OpenAI / Azure
so your API key never lives in the browser source. Uses only the Python
standard library — no pip install needed.

Run:
    python3 server.py
Then open:
    http://localhost:5173/visualreq_web_app_v2.html
"""

import base64
import hmac
import re
import json
import os
import ssl
import time
import urllib.request
import urllib.error
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

PORT = int(os.environ.get("PORT", "5173"))
# HOST: "127.0.0.1" = local only (safe default, use behind a reverse proxy).
# Set HOST=0.0.0.0 to accept connections from the network (e.g. direct/Docker).
HOST = os.environ.get("HOST", "127.0.0.1")

# ---------------------------------------------------------------------------
# Security settings (all optional; local use works exactly as before)
#   APP_PASSWORD        If set, every request needs HTTP Basic auth with this
#                       password (browser shows a login box). Set this when
#                       the app is reachable from the internet (e.g. Render).
#   APP_USER            Optional username for Basic auth (any username if unset).
#   JIRA_ALLOWED_HOSTS  Extra Jira hosts allowed, comma separated
#                       (*.atlassian.net is always allowed).
#   AI_ALLOWED_HOSTS    Extra AI endpoint hosts allowed, comma separated.
#   ALLOW_INSECURE_TLS  "1" = allow the unverified-TLS fallback. Only ever
#                       allowed automatically when running on localhost.
#   MAX_BODY_BYTES      Max request body size (default 5 MB).
# ---------------------------------------------------------------------------
APP_DIR = os.path.dirname(os.path.abspath(__file__))
APP_FILE = "visualreq_web_app_v2.html"
APP_USER = os.environ.get("APP_USER", "")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "")
MAX_BODY_BYTES = int(os.environ.get("MAX_BODY_BYTES", str(5 * 1024 * 1024)))
IS_LOCAL = HOST in ("127.0.0.1", "localhost", "::1")
ALLOW_INSECURE_TLS = IS_LOCAL or os.environ.get("ALLOW_INSECURE_TLS") == "1"


def _host_list(env_name):
    return [h.strip().lower() for h in os.environ.get(env_name, "").split(",") if h.strip()]


JIRA_EXTRA_HOSTS = _host_list("JIRA_ALLOWED_HOSTS")
AI_ALLOWED_SUFFIXES = (".openai.azure.com", ".cognitiveservices.azure.com", ".services.ai.azure.com")
AI_ALLOWED_EXACT = ["api.openai.com"] + _host_list("AI_ALLOWED_HOSTS")


def _url_host(url):
    m = re.match(r"^https://([^/:?#]+)(?::443)?(?:[/?#]|$)", url or "", re.I)
    return m.group(1).lower() if m else None


def check_ai_endpoint(url):
    """Only allow https calls to known OpenAI / Azure OpenAI hosts."""
    if not url:
        return None
    host = _url_host(url)
    if not host:
        raise RuntimeError("AI endpoint ต้องขึ้นต้นด้วย https:// เท่านั้น")
    if host in AI_ALLOWED_EXACT or host.endswith(AI_ALLOWED_SUFFIXES):
        return url
    raise RuntimeError(f"AI endpoint host '{host}' ไม่ได้รับอนุญาต "
                       "(อนุญาต api.openai.com และ *.openai.azure.com — เพิ่มได้ด้วย env AI_ALLOWED_HOSTS)")

# Build a TLS context that works across macOS Python installs.
# Many python.org builds ship without a usable CA bundle ("certificate verify
# failed"). Prefer certifi if available, then the system default. If both fail
# to verify (common on a fresh mac), fall back to unverified with a warning —
# acceptable for a local dev tool calling official, hardcoded API hosts.
def _build_ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where()), False
    except Exception:
        pass
    return ssl.create_default_context(), False

SSL_CTX, _ = _build_ssl_context()
INSECURE_CTX = ssl._create_unverified_context()


def _open_once(req):
    """Open a request, falling back to an unverified context if the CA bundle
    on this machine can't validate the certificate."""
    try:
        return urllib.request.urlopen(req, timeout=120, context=SSL_CTX)
    except urllib.error.URLError as e:
        if ALLOW_INSECURE_TLS and "CERTIFICATE_VERIFY_FAILED" in str(e.reason):
            print("[warn] TLS verification failed — retrying without verification "
                  "(local dev). To fix permanently, run the 'Install Certificates.command' "
                  "in your Python folder, or `pip install certifi`.")
            return urllib.request.urlopen(req, timeout=120, context=INSECURE_CTX)
        raise


def _urlopen(req):
    """Open a request, retrying on transient overload (503 / 529) with backoff.
    The request body is read fresh from `req` each attempt, so retries are safe."""
    delays = [1.5, 3, 6]  # up to 4 attempts total
    attempt = 0
    while True:
        try:
            return _open_once(req)
        except urllib.error.HTTPError as e:
            if e.code in (503, 529) and attempt < len(delays):
                print(f"[info] {e.code} overload from provider — retrying in {delays[attempt]}s "
                      f"(attempt {attempt + 1}/{len(delays)})")
                time.sleep(delays[attempt])
                attempt += 1
                continue
            raise


def call_claude(api_key, model, system, prompt):
    """Anthropic Messages API. Default model: claude-opus-4-8."""
    url = "https://api.anthropic.com/v1/messages"
    body = {
        "model": model or "claude-opus-4-8",
        "max_tokens": 16000,  # headroom for long requirements (non-streaming safe ceiling)
        "system": system,
        "messages": [{"role": "user", "content": prompt}],
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    with _urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    # Concatenate all text blocks
    parts = [b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"]
    return "".join(parts)


def call_openai(api_key, model, system, prompt, base_url=None):
    """OpenAI Chat Completions API (or Azure with base_url).

    Adapts automatically to model quirks: newer models (gpt-5, o1, o3, ...)
    require `max_completion_tokens` instead of `max_tokens`, and some reject
    `response_format` or `temperature`. We retry, stripping/swapping the
    offending field based on the 400 message, up to a few times.
    """
    url = check_ai_endpoint(base_url) or "https://api.openai.com/v1/chat/completions"
    headers = {"content-type": "application/json"}
    # Azure uses api-key header; OpenAI uses Authorization: Bearer
    if base_url and "azure" in base_url.lower():
        headers["api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    body = {
        "model": model or "gpt-4o",
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "response_format": {"type": "json_object"},
        "max_tokens": 16000,
    }

    for _ in range(4):  # at most a few adaptive retries
        req = urllib.request.Request(
            url, data=json.dumps(body).encode("utf-8"), headers=headers, method="POST"
        )
        try:
            with _urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            if e.code != 400:
                raise
            msg = e.read().decode("utf-8", "replace")
            low = msg.lower()
            changed = False
            # max_tokens -> max_completion_tokens (newer models)
            if "max_tokens" in low and "max_completion_tokens" in low and "max_tokens" in body:
                body["max_completion_tokens"] = body.pop("max_tokens")
                changed = True
            # some models reject response_format json_object
            elif "response_format" in low and "response_format" in body:
                body.pop("response_format")
                changed = True
            # some reasoning models reject custom temperature / other params
            elif "temperature" in low and "temperature" in body:
                body.pop("temperature")
                changed = True
            if not changed:
                raise  # genuine 400 we can't fix — surface it
    # exhausted retries
    raise RuntimeError("OpenAI request could not be adapted to this model.")


def call_gemini(api_key, model, system, prompt):
    """Google Gemini (Generative Language API). Default: gemini-2.5-flash."""
    model = model or "gemini-2.5-flash"
    if not re.fullmatch(r"[A-Za-z0-9._-]{1,80}", model):
        raise RuntimeError("ชื่อโมเดล Gemini ไม่ถูกต้อง")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    body = {
        "system_instruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseMimeType": "application/json",
            "maxOutputTokens": 32768,        # big headroom so long JSON isn't truncated
            # Gemini 2.5 "thinking" consumes the output budget and can starve the
            # answer (→ truncated JSON). Disable it for this structured-output task.
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    data = None
    for _ in range(2):  # retry once without thinkingConfig if the model rejects it
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"content-type": "application/json", "x-goog-api-key": api_key},
            method="POST",
        )
        try:
            with _urlopen(req) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            if e.code == 400 and "thinking" in detail.lower() and "thinkingConfig" in body["generationConfig"]:
                body["generationConfig"].pop("thinkingConfig")  # older model → drop it
                continue
            # body already consumed → raise with the detail preserved
            raise RuntimeError(f"Gemini API error {e.code}: {detail[:300]}")
    candidates = (data or {}).get("candidates", [])
    if not candidates:
        raise RuntimeError(f"Gemini returned no content: {json.dumps(data)[:300]}")
    cand = candidates[0]
    parts = cand.get("content", {}).get("parts", [])
    text = "".join(p.get("text", "") for p in parts)
    # If the model hit the token ceiling the JSON will be incomplete → tell the caller clearly.
    if cand.get("finishReason") == "MAX_TOKENS":
        raise RuntimeError("Gemini output ยาวเกินขีดจำกัด (MAX_TOKENS) — requirement ใหญ่เกินไป "
                           "ลองย่อ requirement หรือใช้ gemini-2.5-pro")
    return text


def adf_to_text(node):
    """Flatten Jira Cloud's Atlassian Document Format (ADF) JSON into readable plain text.
    Handles text + links, headings, lists, tables, cards, mentions, emoji, dates, panels, rules."""
    if node is None:
        return ""
    if isinstance(node, list):
        return "".join(adf_to_text(n) for n in node)
    if isinstance(node, str):
        return node

    ntype = node.get("type")
    attrs = node.get("attrs") or {}

    # Leaf / inline nodes that carry data outside of `content`
    if ntype == "text":
        return node.get("text", "")  # plain text only — links are dropped
    if ntype == "hardBreak":
        return "\n"
    if ntype == "rule":
        return "\n---\n"
    if ntype in ("inlineCard", "blockCard", "embedCard"):
        return ""  # drop link/card URLs — user wants text only
    if ntype == "mention":
        return attrs.get("text") or ("@" + str(attrs.get("id", "")))
    if ntype == "emoji":
        return attrs.get("text") or attrs.get("shortName") or ""
    if ntype == "date":
        return attrs.get("timestamp", "")
    if ntype == "status":
        return attrs.get("text", "")

    children = node.get("content") or []
    inner = "".join(adf_to_text(c) for c in children)

    # Block-level nodes — add structure / spacing
    if ntype == "listItem":
        return "• " + inner.strip() + "\n"
    if ntype in ("paragraph", "heading", "blockquote", "codeBlock", "panel", "expand", "nestedExpand"):
        return inner.strip() + "\n"
    if ntype in ("tableHeader", "tableCell"):
        return inner.strip() + " | "
    if ntype == "tableRow":
        return inner.rstrip(" |").rstrip() + "\n"
    if ntype in ("bulletList", "orderedList", "table"):
        return inner.strip() + "\n"
    return inner


def extract_issue_key(s):
    """Pull an issue key (e.g. REQ-123) from a full Jira URL or a raw key."""
    import re
    s = (s or "").strip()
    m = re.search(r"([A-Z][A-Z0-9]+-\d+)", s.upper())
    return m.group(1) if m else s.upper()


def normalize_base_url(base_url):
    """Add https:// if missing and strip any path → scheme://host only."""
    import re as _re
    base_url = (base_url or "").strip().rstrip("/")
    if base_url and not base_url.startswith(("http://", "https://")):
        base_url = "https://" + base_url
    m = _re.match(r"(https?://[^/]+)", base_url)
    base_url = m.group(1) if m else base_url
    if not base_url:
        return base_url
    host = _url_host(base_url + "/")
    if not host:
        raise RuntimeError("Jira Base URL ต้องเป็น https:// เท่านั้น เช่น https://yoursite.atlassian.net")
    if not (host.endswith(".atlassian.net") or host in JIRA_EXTRA_HOSTS):
        raise RuntimeError(f"Jira host '{host}' ไม่ได้รับอนุญาต — ใช้ได้เฉพาะ *.atlassian.net "
                           "(เพิ่ม host อื่นได้ด้วย env JIRA_ALLOWED_HOSTS)")
    return base_url


def jira_search(base_url, auth, jql, fields, max_total=100):
    """Run a JQL search, collecting up to `max_total` issues across pages, using
    the enhanced /search/jql endpoint (nextPageToken paging). Returns
    {"issues": [...]}. The legacy /search endpoint has been retired by Atlassian
    (it now answers every POST with a 400 "Invalid request payload"), so there is
    no fallback — a real JQL/payload error propagates instead of being masked.
    For changelog, use _jira_bulkfetch; the enhanced search can't return it."""
    headers = {"Authorization": f"Basic {auth}", "Accept": "application/json", "content-type": "application/json"}

    def _post(extra):
        body = {"jql": jql, "fields": fields, "maxResults": min(100, max_total)}
        body.update(extra)
        req = urllib.request.Request(base_url + "/rest/api/3/search/jql",
                                     data=json.dumps(body).encode("utf-8"),
                                     headers=headers, method="POST")
        with _urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))

    issues, token = [], None
    while len(issues) < max_total:
        data = _post({"nextPageToken": token} if token else {})
        issues.extend(data.get("issues", []) or [])
        token = data.get("nextPageToken")
        if data.get("isLast") or not token:
            break
    return {"issues": issues[:max_total]}


def call_jira_bulk(base_url, email, token, keys):
    """Phase-1 cheap check: one JQL search per 50 keys returning only
    updated/summary/status. Returns {key: {updated, summary, status}}."""
    base_url = normalize_base_url(base_url)
    if not base_url:
        raise RuntimeError("Missing Jira Base URL")
    keys = [extract_issue_key(k) for k in keys if k]
    keys = list(dict.fromkeys(keys))  # de-dup, preserve order
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")

    out = {}
    CHUNK = 50
    for i in range(0, len(keys), CHUNK):
        chunk = keys[i:i + CHUNK]
        jql = "key in (" + ",".join(chunk) + ")"
        data = jira_search(base_url, auth, jql, ["updated", "summary", "status"])
        for issue in data.get("issues", []) or []:
            k = issue.get("key")
            fs = issue.get("fields", {}) or {}
            out[k] = {
                "updated": fs.get("updated", "") or "",
                "summary": fs.get("summary", "") or "",
                "status": ((fs.get("status") or {}).get("name", "")) or "",
            }
    return out


def call_jira_search(base_url, email, token, query):
    """Typeahead search: find issues by summary/text. Returns up to 10
    {key, summary, status, type} for the auto-suggest dropdown."""
    base_url = normalize_base_url(base_url)
    if not base_url:
        raise RuntimeError("Missing Jira Base URL")
    q = (query or "").strip()
    if len(q) < 2:
        return []
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")
    safe = q.replace("\\", "\\\\").replace('"', '\\"')
    jql = f'(summary ~ "{safe}*" OR text ~ "{safe}") ORDER BY updated DESC'
    data = jira_search(base_url, auth, jql, ["summary", "status", "issuetype"])
    out = []
    for issue in (data.get("issues", []) or [])[:10]:
        fs = issue.get("fields", {}) or {}
        out.append({
            "key": issue.get("key"),
            "summary": fs.get("summary", "") or "",
            "status": ((fs.get("status") or {}).get("name", "")) or "",
            "type": ((fs.get("issuetype") or {}).get("name", "")) or "",
        })
    return out


def call_jira(base_url, email, token, key_or_link):
    """Fetch a Jira Cloud issue and return its requirement fields as plain text."""
    import re as _re
    base_url = normalize_base_url(base_url)
    key = extract_issue_key(key_or_link)
    if not base_url:
        raise RuntimeError("Missing Jira Base URL (e.g. https://yoursite.atlassian.net)")
    if not key:
        raise RuntimeError("Could not find an issue key (e.g. REQ-123) in the input")

    # expand=names → human labels for custom fields (e.g. "User Story", "Acceptance Criteria")
    # No fields= restriction so ALL fields (incl. custom rich-text) are returned.
    url = f"{base_url}/rest/api/3/issue/{key}?expand=names"
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Basic {auth}", "Accept": "application/json"},
        method="GET",
    )
    with _urlopen(req) as resp:
        raw = resp.read().decode("utf-8", "replace")
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError("Jira ตอบกลับไม่ใช่ JSON — ตรวจ Base URL ว่าเป็น https://yoursite.atlassian.net (ไม่ใส่ path อื่น)")

    f = data.get("fields", {}) or {}
    names = data.get("names", {}) or {}
    real_key = data.get("key", key)
    summary = f.get("summary", "") or ""
    status = ((f.get("status") or {}).get("name", "")) or ""
    issuetype = ((f.get("issuetype") or {}).get("name", "")) or ""
    updated = f.get("updated", "") or ""

    def is_adf(v):
        return isinstance(v, dict) and v.get("type") == "doc"

    # Collect every rich-text field (description + custom fields that hold ADF docs),
    # labelled with its display name. This captures "User Story", "Acceptance Criteria",
    # etc. even when they are custom fields rather than the description.
    #
    # Order follows Jira: custom fields are emitted in custom-field-id order
    # (which matches their template/creation order), with the system `description`
    # field first if it has content. No manual re-ordering is imposed.
    def order_key(fid):
        m = _re.search(r"customfield_(\d+)", fid)
        # description (system field) first, then custom fields by numeric id
        return (0, 0) if fid == "description" else (1, int(m.group(1)) if m else 0)

    rich_ids = sorted((k for k in f if is_adf(f.get(k))), key=order_key)
    ordered_ids = rich_ids

    sections = []
    for fid in ordered_ids:
        text = adf_to_text(f.get(fid)).strip()
        if not text:
            continue
        label = names.get(fid, fid)
        sections.append(f"## {label}\n{text}")

    desc = "\n\n".join(sections).strip()
    requirement = f"[{real_key}] {summary}\n\n{desc}".strip()
    return {
        "key": real_key,
        "summary": summary,
        "description": desc,
        "status": status,
        "type": issuetype,
        "updated": updated,
        "url": f"{base_url}/browse/{real_key}",
        "requirement": requirement,
        "sections": [names.get(fid, fid) for fid in ordered_ids],
    }


def call_jira_projects(base_url, email, token):
    """List Jira projects visible to this account. Returns [{key, name, id}]."""
    base_url = normalize_base_url(base_url)
    if not base_url:
        raise RuntimeError("Missing Jira Base URL")
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")
    url = f"{base_url}/rest/api/3/project/search?maxResults=100&orderBy=name"
    req = urllib.request.Request(
        url, headers={"Authorization": f"Basic {auth}", "Accept": "application/json"}, method="GET"
    )
    with _urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    return [{"key": p.get("key"), "name": p.get("name", ""), "id": p.get("id")} for p in data.get("values", []) or []]


# Field ids we never expose as editable inputs — either handled specially
# (project/issuetype/summary/description) or too complex to be worth a
# generic input (assignee/attachment/links/etc.)
_CREATEMETA_BLOCKED_FIELDS = {
    "summary", "description", "project", "issuetype", "reporter", "assignee",
    "attachment", "comment", "worklog", "timetracking", "subtasks",
    "issuelinks", "votes", "watches", "thumbnail", "duedate",
}


def call_jira_createmeta(base_url, email, token, project_key, issuetype_id=None):
    """Discover the real create-issue fields for a project's issue type. When
    `issuetype_id` is given, use it; otherwise default to the type named
    "Issue" (this Jira uses "Issue" with an "Issue Sub-type" field rather than
    a native "Bug" type), then "Bug", then the first non-subtask type. Also
    returns the full issue-type list so the UI can offer a picker. Uses the
    current Jira Cloud granular createmeta endpoints — nothing hardcoded."""
    base_url = normalize_base_url(base_url)
    if not base_url:
        raise RuntimeError("Missing Jira Base URL")
    if not project_key:
        raise RuntimeError("Missing Jira project key")
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")
    headers = {"Authorization": f"Basic {auth}", "Accept": "application/json"}

    url1 = f"{base_url}/rest/api/3/issue/createmeta/{project_key}/issuetypes?maxResults=50"
    req1 = urllib.request.Request(url1, headers=headers, method="GET")
    with _urlopen(req1) as resp1:
        types_data = json.loads(resp1.read().decode("utf-8", "replace"))
    issue_types = types_data.get("issueTypes", []) or []
    if not issue_types:
        raise RuntimeError(f"ไม่พบ Issue Type ที่สร้างได้ในโปรเจกต์ {project_key}")

    chosen = None
    if issuetype_id:
        chosen = next((t for t in issue_types if str(t.get("id")) == str(issuetype_id)), None)
    if not chosen:
        chosen = next((t for t in issue_types if (t.get("name") or "").lower() == "issue"), None)
    if not chosen:
        chosen = next((t for t in issue_types if (t.get("name") or "").lower() == "bug"), None)
    if not chosen:
        chosen = next((t for t in issue_types if not t.get("subtask")), issue_types[0])
    issuetype_id = chosen.get("id")
    issuetype_name = chosen.get("name", "")

    url2 = f"{base_url}/rest/api/3/issue/createmeta/{project_key}/issuetypes/{issuetype_id}?maxResults=200"
    req2 = urllib.request.Request(url2, headers=headers, method="GET")
    with _urlopen(req2) as resp2:
        fields_data = json.loads(resp2.read().decode("utf-8", "replace"))
    raw_fields = fields_data.get("fields", []) or []

    def opt(v):
        return {
            "id": str(v.get("id", v.get("value", ""))),
            "label": v.get("value") or v.get("name") or str(v.get("id", "")),
        }

    fields = []
    sprint_field_id = None
    for f in raw_fields:
        fid = f.get("fieldId")
        if not fid or fid in _CREATEMETA_BLOCKED_FIELDS:
            continue
        schema = f.get("schema", {}) or {}
        stype = schema.get("type")
        items = schema.get("items")
        allowed = f.get("allowedValues") or []

        # Sprint is a gh-sprint field — its options come from the Agile API
        # (see call_jira_sprints), not createmeta. Capture its id and handle
        # it separately so the UI can offer a real Sprint picker.
        if (schema.get("custom") or "").endswith("gh-sprint"):
            sprint_field_id = fid
            continue

        ftype = None
        options = []
        if stype == "array":
            if allowed and items in ("option", "version", "string"):
                ftype, options = "multiselect", [opt(v) for v in allowed]
            elif items == "string":
                ftype = "tags"
            else:
                continue  # unsupported array item type (e.g. user[])
        elif allowed or stype in ("option", "priority"):
            ftype, options = "select", [opt(v) for v in allowed]
        elif stype == "number":
            ftype = "number"
        elif stype == "date":
            ftype = "date"
        elif stype == "string":
            # 'textarea' custom fields on this site store ADF (rich text), so
            # they must be sent as an ADF doc on create — flag them distinctly.
            ftype = "textarea" if (schema.get("custom") or "").endswith("textarea") else "text"
        else:
            continue  # unsupported type (e.g. user)

        fields.append({
            "id": fid,
            "name": f.get("name", fid),
            "required": bool(f.get("required")),
            "type": ftype,
            "options": options,
        })

    types_list = [{"id": str(t.get("id")), "name": t.get("name", "")} for t in issue_types if not t.get("subtask")]
    return {
        "issuetypeId": issuetype_id,
        "issuetypeName": issuetype_name,
        "issueTypes": types_list,
        "sprintFieldId": sprint_field_id,
        "fields": fields,
    }


def call_jira_sprints(base_url, email, token, project_key):
    """List active + future sprints across the project's scrum boards (Agile
    API). Returns [{id, name, state, board}] — newest/active first."""
    base_url = normalize_base_url(base_url)
    if not base_url:
        raise RuntimeError("Missing Jira Base URL")
    if not project_key:
        raise RuntimeError("Missing Jira project key")
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")
    headers = {"Authorization": f"Basic {auth}", "Accept": "application/json"}

    boards_url = f"{base_url}/rest/agile/1.0/board?projectKeyOrId={project_key}&maxResults=50"
    req = urllib.request.Request(boards_url, headers=headers, method="GET")
    with _urlopen(req) as resp:
        boards = json.loads(resp.read().decode("utf-8", "replace")).get("values", []) or []

    out = []
    seen = set()
    for b in boards:
        if b.get("type") != "scrum":
            continue
        bid = b.get("id")
        s_url = f"{base_url}/rest/agile/1.0/board/{bid}/sprint?state=active,future&maxResults=50"
        try:
            sreq = urllib.request.Request(s_url, headers=headers, method="GET")
            with _urlopen(sreq) as sresp:
                sprints = json.loads(sresp.read().decode("utf-8", "replace")).get("values", []) or []
        except urllib.error.HTTPError:
            continue  # some boards disallow sprint listing — skip
        for s in sprints:
            sid = s.get("id")
            if sid in seen:
                continue
            seen.add(sid)
            out.append({
                "id": sid,
                "name": s.get("name", ""),
                "state": s.get("state", ""),
                "board": b.get("name", ""),
            })
    # active sprints first, then future
    out.sort(key=lambda x: (0 if x["state"] == "active" else 1))
    return out


def text_to_adf(text):
    """Plain text → Atlassian Document Format doc. Blank lines separate
    paragraphs; '- '/'* ' lines become a bullet list; '1. ' lines become a
    numbered list. Mirrors adf_to_text's structure in the opposite direction."""
    import re as _re
    text = (text or "").strip()
    if not text:
        return {"type": "doc", "version": 1, "content": [{"type": "paragraph", "content": []}]}

    content = []
    para_lines = []
    list_items = []
    list_type = None

    def flush_para():
        if para_lines:
            content.append({
                "type": "paragraph",
                "content": [{"type": "text", "text": "\n".join(para_lines)}],
            })
            para_lines.clear()

    def flush_list():
        nonlocal list_type
        if list_items:
            content.append({"type": list_type, "content": list(list_items)})
            list_items.clear()
            list_type = None

    for line in text.split("\n"):
        stripped = line.strip()
        bullet_m = _re.match(r"^[-*•]\s+(.*)", stripped)
        num_m = None if bullet_m else _re.match(r"^\d+[.)]\s+(.*)", stripped)
        if bullet_m or num_m:
            flush_para()
            this_type = "bulletList" if bullet_m else "orderedList"
            item_text = (bullet_m or num_m).group(1)
            if list_type != this_type:
                flush_list()
                list_type = this_type
            list_items.append({
                "type": "listItem",
                "content": [{"type": "paragraph", "content": [{"type": "text", "text": item_text}]}],
            })
        elif stripped == "":
            flush_list()
            flush_para()
        else:
            flush_list()
            para_lines.append(line)

    flush_list()
    flush_para()
    if not content:
        content = [{"type": "paragraph", "content": []}]
    return {"type": "doc", "version": 1, "content": content}


def _build_field_payload(extra_fields):
    """Map normalized {id, type, value} entries to the JSON shape Jira's
    create-issue API expects for that field type."""
    out = {}
    for f in extra_fields or []:
        fid = f.get("id")
        ftype = f.get("type")
        val = f.get("value")
        if not fid or val in (None, "", []):
            continue
        if ftype == "select":
            out[fid] = {"id": str(val)}
        elif ftype == "multiselect":
            out[fid] = [{"id": str(v)} for v in val]
        elif ftype == "tags":
            out[fid] = [s.strip() for s in str(val).split(",") if s.strip()]
        elif ftype == "number":
            try:
                out[fid] = float(val) if "." in str(val) else int(val)
            except (TypeError, ValueError):
                out[fid] = val
        elif ftype == "textarea":
            out[fid] = text_to_adf(str(val))  # ADF rich-text field
        else:  # text, date
            out[fid] = val
    return out


def call_jira_create(base_url, email, token, project_key, issuetype_id, summary, description, extra_fields,
                     sprint_field_id=None, sprint_id=None):
    """Create a Jira issue. Returns {key, url}. If sprint_field_id + sprint_id
    are given, the issue is placed directly into that sprint."""
    base_url = normalize_base_url(base_url)
    if not base_url:
        raise RuntimeError("Missing Jira Base URL")
    if not (project_key and issuetype_id and summary):
        raise RuntimeError("ต้องมี Project, Issue Type และ Summary")
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")

    fields = {
        "project": {"key": project_key},
        "issuetype": {"id": issuetype_id},
        "summary": summary,
    }
    if description:
        fields["description"] = text_to_adf(description)
    fields.update(_build_field_payload(extra_fields))
    # Sprint field takes a raw sprint id (integer), not an object.
    if sprint_field_id and sprint_id:
        try:
            fields[sprint_field_id] = int(sprint_id)
        except (TypeError, ValueError):
            fields[sprint_field_id] = sprint_id

    body = json.dumps({"fields": fields}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/rest/api/3/issue",
        data=body,
        headers={"Authorization": f"Basic {auth}", "Accept": "application/json", "content-type": "application/json"},
        method="POST",
    )
    with _urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    key = data.get("key")
    return {"key": key, "url": f"{base_url}/browse/{key}"}


def call_jira_create_subtask(base_url, email, token, project_key, parent_key, summary,
                             assign_to_me=True, bill_type="Non-Billable", target_status=""):
    """Create a Sub-task under `parent_key`. Everything is discovered live (no
    hardcoded ids): the subtask issue type, the Bill Types field + its option
    matching `bill_type`, and the current token owner's accountId for assignee.
    If `target_status` is given, transition the new sub-task to that status
    (one hop; skipped if unreachable). Returns {key, url, status}."""
    base_url = normalize_base_url(base_url)
    if not base_url:
        raise RuntimeError("Missing Jira Base URL")
    if not (project_key and parent_key and summary):
        raise RuntimeError("ต้องมี Project, Parent และ Summary สำหรับ Subtask")
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")
    headers = {"Authorization": f"Basic {auth}", "Accept": "application/json"}

    def _get(path):
        req = urllib.request.Request(base_url + path, headers=headers, method="GET")
        with _urlopen(req) as resp:
            return json.loads(resp.read().decode("utf-8", "replace"))

    # 1. Pick the subtask issue type (prefer one named "Sub-task").
    types = (_get(f"/rest/api/3/issue/createmeta/{project_key}/issuetypes?maxResults=50")
             .get("issueTypes", []) or [])
    subtasks = [t for t in types if t.get("subtask")]
    if not subtasks:
        raise RuntimeError(f"โปรเจกต์ {project_key} ไม่มี Subtask issue type")
    st = next((t for t in subtasks if (t.get("name") or "").lower() == "sub-task"), subtasks[0])
    subtask_type_id = st.get("id")

    # 2. Find the Bill Types field + the option whose value matches bill_type.
    bill_field_id = bill_option_id = None
    if bill_type:
        meta = _get(f"/rest/api/3/issue/createmeta/{project_key}/issuetypes/{subtask_type_id}?maxResults=200")
        for f in meta.get("fields", []) or []:
            if (f.get("name") or "").strip().lower() == "bill types":
                bill_field_id = f.get("fieldId")
                for v in f.get("allowedValues", []) or []:
                    if (v.get("value") or "").strip().lower() == bill_type.strip().lower():
                        bill_option_id = str(v.get("id"))
                        break
                break

    # 3. Current token owner's accountId (this is "assign to me").
    account_id = None
    if assign_to_me:
        try:
            account_id = _get("/rest/api/3/myself").get("accountId")
        except Exception:
            account_id = None  # non-fatal — create unassigned rather than fail

    fields = {
        "project": {"key": project_key},
        "issuetype": {"id": subtask_type_id},
        "parent": {"key": parent_key},
        "summary": summary,
    }
    if account_id:
        fields["assignee"] = {"id": account_id}
    if bill_field_id and bill_option_id:
        fields[bill_field_id] = {"id": bill_option_id}

    body = json.dumps({"fields": fields}).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/rest/api/3/issue",
        data=body,
        headers={"Authorization": f"Basic {auth}", "Accept": "application/json", "content-type": "application/json"},
        method="POST",
    )
    with _urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    key = data.get("key")

    # 4. Optionally move it to the requested status (one transition hop).
    final_status = None
    if key and (target_status or "").strip():
        want = target_status.strip().lower()
        try:
            trans = (_get(f"/rest/api/3/issue/{key}/transitions").get("transitions", []) or [])
            match = next((t for t in trans if ((t.get("to") or {}).get("name") or "").strip().lower() == want), None)
            if match:
                tbody = json.dumps({"transition": {"id": match.get("id")}}).encode("utf-8")
                treq = urllib.request.Request(
                    f"{base_url}/rest/api/3/issue/{key}/transitions",
                    data=tbody,
                    headers={"Authorization": f"Basic {auth}", "Accept": "application/json", "content-type": "application/json"},
                    method="POST",
                )
                with _urlopen(treq) as resp:
                    resp.read()  # 204 no content
                final_status = target_status.strip()
        except Exception:
            final_status = None  # non-fatal: card is created, status just not changed

    return {"key": key, "url": f"{base_url}/browse/{key}", "status": final_status}


def call_jira_subtask_statuses(base_url, email, token, project_key):
    """List the statuses available for the Sub-task issue type in `project_key`
    (for the create-subtask Status picker). Returns [{id, name}]."""
    base_url = normalize_base_url(base_url)
    if not base_url:
        raise RuntimeError("Missing Jira Base URL")
    if not project_key:
        return []
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")
    req = urllib.request.Request(
        f"{base_url}/rest/api/3/project/{project_key}/statuses",
        headers={"Authorization": f"Basic {auth}", "Accept": "application/json"}, method="GET")
    with _urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8", "replace"))
    entry = None
    for it in (data or []):
        if it.get("subtask") or (it.get("name") or "").strip().lower() == "sub-task":
            entry = it
            break
    if not entry:
        return []
    out, seen = [], set()
    for s in (entry.get("statuses", []) or []):
        name = s.get("name") or ""
        if name and name not in seen:
            seen.add(name)
            out.append({"id": s.get("id"), "name": name})
    return out


def call_jira_search_stories(base_url, email, token, project_key, query):
    """Search Story-type issues in a project by summary text → [{key, summary}]."""
    base_url = normalize_base_url(base_url)
    if not base_url:
        raise RuntimeError("Missing Jira Base URL")
    q = (query or "").strip()
    if not project_key or len(q) < 2:
        return []
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")
    safe = q.replace("\\", "\\\\").replace('"', '\\"')
    jql = f'project = {project_key} AND issuetype = Story AND summary ~ "{safe}" ORDER BY updated DESC'
    data = jira_search(base_url, auth, jql, ["summary"])
    out = []
    for issue in (data.get("issues", []) or [])[:10]:
        out.append({"key": issue.get("key"), "summary": (issue.get("fields", {}) or {}).get("summary", "") or ""})
    return out


def call_jira_link(base_url, email, token, from_key, to_key, link_type="Relates"):
    """Create an issue link (default 'Relates') from `from_key` to `to_key`.
    Returns {ok, from, to}. The Jira endpoint replies 201 with no body."""
    base_url = normalize_base_url(base_url)
    if not base_url:
        raise RuntimeError("Missing Jira Base URL")
    if not (from_key and to_key):
        raise RuntimeError("ต้องมีทั้ง 2 การ์ดเพื่อสร้าง Link")
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")
    body = json.dumps({
        "type": {"name": link_type},
        "inwardIssue": {"key": to_key},     # the Story
        "outwardIssue": {"key": from_key},  # the new Issue
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/rest/api/3/issueLink",
        data=body,
        headers={"Authorization": f"Basic {auth}", "Accept": "application/json", "content-type": "application/json"},
        method="POST",
    )
    with _urlopen(req) as resp:
        resp.read()  # 201, empty body
    return {"ok": True, "from": from_key, "to": to_key}


def _jira_account_id(base_url, auth):
    """Current token owner's accountId (for 'is this change/comment mine')."""
    req = urllib.request.Request(base_url + "/rest/api/3/myself",
                                 headers={"Authorization": f"Basic {auth}", "Accept": "application/json"}, method="GET")
    with _urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8", "replace")).get("accountId")


def _jira_bulkfetch(base_url, auth, keys, fields, expand=None):
    """POST /rest/api/3/issue/bulkfetch — returns full fields (and optional
    changelog/comment, via `expand`) for up to 100 keys per call. This is the
    modern replacement for the retired legacy /search endpoint, which was the
    only JQL search that could return changelog inline. Returns a list of issues
    (each with `fields` and, when expanded, `changelog`)."""
    headers = {"Authorization": f"Basic {auth}", "Accept": "application/json", "content-type": "application/json"}
    out = []
    for i in range(0, len(keys), 100):
        body = {"issueIdsOrKeys": keys[i:i + 100], "fields": fields, "fieldsByKeys": False}
        if expand:
            body["expand"] = expand
        req = urllib.request.Request(base_url + "/rest/api/3/issue/bulkfetch",
                                     data=json.dumps(body).encode("utf-8"), headers=headers, method="POST")
        with _urlopen(req) as resp:
            data = json.loads(resp.read().decode("utf-8", "replace"))
        out.extend(data.get("issues", []) or [])
    return out


def call_jira_report(base_url, email, token, project_key, date_from, date_to, only_mine, include_activity=False):
    """Live JQL query for the daily-summary report view.

    include_activity=False → cards I created in the range (grouped by created day).
    include_activity=True  → ALSO cards I worked on: status I changed, cards
    assigned to me that updated, and cards I commented on — grouped by the day of
    MY action. Candidate keys come from independent searches (so an unsupported
    clause degrades gracefully); changelog+comment come from one bulkfetch."""
    base_url = normalize_base_url(base_url)
    if not base_url:
        raise RuntimeError("Missing Jira Base URL")
    if not project_key:
        raise RuntimeError("กรุณาเลือกโปรเจกต์")
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")

    def in_range(iso):
        d = (iso or "")[:10]
        return bool(d) and (not date_from or d >= date_from) and (not date_to or d <= date_to)

    # ---------- simple mode (created by me) ----------
    if not include_activity:
        parts = [f"project = {project_key}"]
        if date_from:
            parts.append(f'created >= "{date_from} 00:00"')
        if date_to:
            parts.append(f'created <= "{date_to} 23:59"')
        if only_mine:
            parts.append("reporter = currentUser()")
        jql = " AND ".join(parts) + " ORDER BY created ASC"
        data = jira_search(base_url, auth, jql, ["summary", "status", "priority", "created", "issuetype"])
        out = []
        for issue in data.get("issues", []) or []:
            f = issue.get("fields", {}) or {}
            key = issue.get("key")
            out.append({
                "key": key, "url": f"{base_url}/browse/{key}",
                "summary": f.get("summary", "") or "",
                "status": ((f.get("status") or {}).get("name", "")) or "",
                "priority": ((f.get("priority") or {}).get("name", "")) or "",
                "created": f.get("created", "") or "",
                "type": ((f.get("issuetype") or {}).get("name", "")) or "",
                "activityDay": (f.get("created", "") or "")[:10],
                "activityKind": "created", "statusChange": None, "myComment": "",
            })
        return out

    # ---------- activity mode (created / status-changed / assigned-updated / commented) ----------
    me = _jira_account_id(base_url, auth)
    dfrom, dto = f'"{date_from} 00:00"', f'"{date_to} 23:59"'
    # Collect candidate keys from three independent searches rather than one big
    # OR query: the `status CHANGED BY ... DURING` history predicate isn't accepted
    # by the enhanced /search/jql endpoint on every instance (team-managed in
    # particular), and bundling it would 400 the whole report. Run each clause on
    # its own and skip any the instance rejects, so the rest still returns.
    def _safe_keys(where):
        try:
            jql = f"project = {project_key} AND {where}"
            res = jira_search(base_url, auth, jql, ["summary"], max_total=80)
            return [i.get("key") for i in (res.get("issues", []) or []) if i.get("key")]
        except urllib.error.HTTPError:
            return []  # clause unsupported on this instance → ignore it

    key_order = []
    for where in (
        f'reporter = currentUser() AND created >= {dfrom} AND created <= {dto}',
        f'assignee = currentUser() AND updated >= {dfrom} AND updated <= {dto}',
        f'status CHANGED BY currentUser() DURING ({dfrom}, {dto})',
    ):
        for k in _safe_keys(where):
            if k not in key_order:
                key_order.append(k)
    keys = key_order[:80]
    full_fields = ["summary", "status", "priority", "created", "updated", "issuetype", "reporter", "comment"]
    try:
        issues = _jira_bulkfetch(base_url, auth, keys, full_fields, expand=["changelog"]) if keys else []
    except urllib.error.HTTPError:
        # bulkfetch unavailable on this instance → fields only (status-change
        # detail is lost, comments still shown if the search returns them)
        issues = (jira_search(base_url, auth, "key in (" + ",".join(keys) + ")",
                              full_fields, max_total=80).get("issues", []) if keys else [])

    out = []
    for issue in issues:
        f = issue.get("fields", {}) or {}
        key = issue.get("key")

        # my comments in range (latest wins) — shows the retest result
        my_comments = [c for c in ((f.get("comment") or {}).get("comments") or [])
                       if ((c.get("author") or {}).get("accountId") == me) and in_range(c.get("created"))]
        my_comments.sort(key=lambda c: c.get("created", ""))
        comment_day = (my_comments[-1].get("created", "")[:10]) if my_comments else ""
        my_comment_text = adf_to_text(my_comments[-1].get("body")).strip()[:600] if my_comments else ""

        # my status changes in range (latest wins) — from changelog
        my_status = []  # (created, from, to)
        for h in ((issue.get("changelog") or {}).get("histories") or []):
            if ((h.get("author") or {}).get("accountId") != me) or not in_range(h.get("created")):
                continue
            for it in (h.get("items") or []):
                if it.get("field") == "status":
                    my_status.append((h.get("created", ""), it.get("fromString") or "", it.get("toString") or ""))
        my_status.sort(key=lambda x: x[0])
        status_change = {"from": my_status[-1][1], "to": my_status[-1][2]} if my_status else None
        status_day = my_status[-1][0][:10] if my_status else ""

        reporter_me = ((f.get("reporter") or {}).get("accountId") == me)
        created_day = (f.get("created", "") or "")[:10]
        updated_day = (f.get("updated", "") or "")[:10]

        if status_change:
            kind, day = "retest", status_day
        elif my_comment_text:
            kind, day = "comment", comment_day
        elif reporter_me and in_range(f.get("created")):
            kind, day = "created", created_day
        else:
            kind, day = "updated", (updated_day if in_range(f.get("updated")) else (date_to or date_from))

        out.append({
            "key": key, "url": f"{base_url}/browse/{key}",
            "summary": f.get("summary", "") or "",
            "status": ((f.get("status") or {}).get("name", "")) or "",
            "priority": ((f.get("priority") or {}).get("name", "")) or "",
            "created": f.get("created", "") or "",
            "type": ((f.get("issuetype") or {}).get("name", "")) or "",
            "activityDay": day, "activityKind": kind,
            "statusChange": status_change, "myComment": my_comment_text,
        })

    out.sort(key=lambda r: (r["activityDay"], r["key"]))
    return out


def call_jira_my_subtasks(base_url, email, token, project_key=None):
    """List Sub-tasks that are assigned to, or were created by, the current
    token owner — the candidate cards for logging time. Sub-task only (the
    time-logging feature forbids logging against non-subtasks). Optionally
    scoped to one project. Returns [{key, url, summary, status, parentKey,
    parentSummary, timeSpent}] newest-updated first."""
    base_url = normalize_base_url(base_url)
    if not base_url:
        raise RuntimeError("Missing Jira Base URL")
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")

    # NOTE: we deliberately do NOT use the subTaskIssueTypes() JQL function —
    # some Jira instances (team-managed / newer "Spaces" projects) don't support
    # it and silently return nothing. Fetch issues assigned to / reported by the
    # user and filter to sub-tasks client-side via issuetype.subtask.
    parts = ["(assignee = currentUser() OR reporter = currentUser())"]
    if project_key:
        parts.insert(0, f"project = {project_key}")
    jql = " AND ".join(parts) + " ORDER BY updated DESC"

    data = jira_search(base_url, auth, jql,
                       ["summary", "status", "parent", "timetracking", "updated", "issuetype"])
    out = []
    for issue in data.get("issues", []) or []:
        f = issue.get("fields", {}) or {}
        if not ((f.get("issuetype") or {}).get("subtask")):
            continue  # sub-tasks only
        key = issue.get("key")
        parent = f.get("parent") or {}
        parent_fields = parent.get("fields", {}) or {}
        tt = f.get("timetracking") or {}
        out.append({
            "key": key,
            "url": f"{base_url}/browse/{key}",
            "summary": f.get("summary", "") or "",
            "status": ((f.get("status") or {}).get("name", "")) or "",
            "parentKey": parent.get("key", "") or "",
            "parentSummary": parent_fields.get("summary", "") or "",
            "timeSpent": tt.get("timeSpent", "") or "",
        })
    return out


def call_jira_subtasks_of_parent(base_url, email, token, parent_key):
    """List EVERY Sub-task under `parent_key` (any assignee) — for the Log Time
    filter that drills down Project → parent card → its sub-tasks. Returns
    [{key, url, summary, status, parentKey, parentSummary, timeSpent}]."""
    base_url = normalize_base_url(base_url)
    if not base_url:
        raise RuntimeError("Missing Jira Base URL")
    parent_key = extract_issue_key(parent_key or "")
    if not parent_key:
        return []
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")
    jql = f"parent = {parent_key} ORDER BY created DESC"
    data = jira_search(base_url, auth, jql,
                       ["summary", "status", "parent", "timetracking", "issuetype"], max_total=200)
    out = []
    for issue in data.get("issues", []) or []:
        f = issue.get("fields", {}) or {}
        key = issue.get("key")
        parent = f.get("parent") or {}
        parent_fields = parent.get("fields", {}) or {}
        tt = f.get("timetracking") or {}
        out.append({
            "key": key,
            "url": f"{base_url}/browse/{key}",
            "summary": f.get("summary", "") or "",
            "status": ((f.get("status") or {}).get("name", "")) or "",
            "parentKey": parent.get("key", "") or parent_key,
            "parentSummary": parent_fields.get("summary", "") or "",
            "timeSpent": tt.get("timeSpent", "") or "",
        })
    return out


def call_jira_worklog(base_url, email, token, issue_key, time_spent, started_date=None, comment=""):
    """Log work against `issue_key`. HARD RULE: the target must be a Sub-task —
    the issue is fetched first and rejected if issuetype.subtask is not true.
    `time_spent` uses Jira's own format (e.g. "2h 30m", "1d", "45m").
    `started_date` is a plain YYYY-MM-DD; the current local time-of-day and
    timezone offset are attached to form the timestamp Jira requires.
    Returns {key, url, timeSpent, started}."""
    import datetime
    base_url = normalize_base_url(base_url)
    if not base_url:
        raise RuntimeError("Missing Jira Base URL")
    issue_key = extract_issue_key(issue_key or "")
    if not issue_key:
        raise RuntimeError("ต้องระบุการ์ด (Subtask) ที่จะลงเวลา")
    if not (time_spent or "").strip():
        raise RuntimeError("ต้องระบุเวลาที่ใช้ เช่น 2h 30m")
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")
    headers = {"Authorization": f"Basic {auth}", "Accept": "application/json"}

    # 1. Enforce the sub-task-only rule before writing anything.
    req = urllib.request.Request(
        f"{base_url}/rest/api/3/issue/{issue_key}?fields=issuetype,summary",
        headers=headers, method="GET")
    with _urlopen(req) as resp:
        issue = json.loads(resp.read().decode("utf-8", "replace"))
    itype = ((issue.get("fields", {}) or {}).get("issuetype") or {})
    if not itype.get("subtask"):
        raise RuntimeError(
            f"ลงเวลาได้เฉพาะ Subtask เท่านั้น — {issue_key} เป็น "
            f"'{itype.get('name', 'ไม่ทราบชนิด')}' ไม่ใช่ Subtask")

    # 2. Build the `started` timestamp Jira wants: YYYY-MM-DDThh:mm:ss.sss+hhmm.
    now = datetime.datetime.now()
    if started_date and started_date.strip():
        try:
            day = datetime.datetime.strptime(started_date.strip(), "%Y-%m-%d")
        except ValueError:
            raise RuntimeError(f"รูปแบบวันที่ไม่ถูกต้อง: {started_date} (ต้องเป็น YYYY-MM-DD)")
    else:
        day = now
    dt = day.replace(hour=now.hour, minute=now.minute, second=now.second, microsecond=0)
    off_sec = -(time.altzone if time.daylight and time.localtime().tm_isdst else time.timezone)
    sign = "+" if off_sec >= 0 else "-"
    off_sec = abs(off_sec)
    started = dt.strftime("%Y-%m-%dT%H:%M:%S.000") + f"{sign}{off_sec // 3600:02d}{(off_sec % 3600) // 60:02d}"

    # 3. Post the worklog.
    body = {"timeSpent": time_spent.strip(), "started": started}
    if (comment or "").strip():
        body["comment"] = text_to_adf(comment)
    data_bytes = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/rest/api/3/issue/{issue_key}/worklog",
        data=data_bytes,
        headers={"Authorization": f"Basic {auth}", "Accept": "application/json", "content-type": "application/json"},
        method="POST",
    )
    with _urlopen(req) as resp:
        result = json.loads(resp.read().decode("utf-8", "replace"))
    return {
        "key": issue_key,
        "url": f"{base_url}/browse/{issue_key}",
        "timeSpent": result.get("timeSpent", time_spent) or time_spent,
        "started": result.get("started", started) or started,
    }


def call_jira_worklog_summary(base_url, email, token, date_from, date_to):
    """Aggregate the current user's own worklogs between date_from and date_to
    (inclusive, YYYY-MM-DD) into a per-day summary for the timesheet view.
    Finds issues via JQL (worklogAuthor + worklogDate), then reads each issue's
    worklogs and keeps only the entries authored by this user whose `started`
    date falls in range. Returns {days: {YYYY-MM-DD: {seconds, entries:[...]}},
    totalSeconds}."""
    base_url = normalize_base_url(base_url)
    if not base_url:
        raise RuntimeError("Missing Jira Base URL")
    if not (date_from and date_to):
        raise RuntimeError("ต้องระบุช่วงวันที่")
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")
    headers = {"Authorization": f"Basic {auth}", "Accept": "application/json"}

    # Whose worklogs count — the token owner.
    account_id = None
    try:
        req = urllib.request.Request(base_url + "/rest/api/3/myself", headers=headers, method="GET")
        with _urlopen(req) as resp:
            account_id = json.loads(resp.read().decode("utf-8", "replace")).get("accountId")
    except Exception:
        account_id = None  # fall back to keeping all entries if this fails

    # Fetch EVERY issue the user logged work on in range (paginate — a busy
    # month easily exceeds one 100-issue page, and capping there under-counts
    # early-in-month days whose issues fall past the newest 100 by `updated`).
    # Request the `worklog` field inline so most issues need no extra call.
    jql = (f'worklogAuthor = currentUser() AND worklogDate >= "{date_from}" '
           f'AND worklogDate <= "{date_to}" ORDER BY updated DESC')
    data = jira_search(base_url, auth, jql, ["summary", "worklog", "status"], max_total=1000)

    def _fetch_all_worklogs(key):
        try:
            wreq = urllib.request.Request(
                f"{base_url}/rest/api/3/issue/{key}/worklog?maxResults=5000",
                headers=headers, method="GET")
            with _urlopen(wreq) as resp:
                return json.loads(resp.read().decode("utf-8", "replace")).get("worklogs", []) or []
        except Exception:
            return []

    days = {}
    total_seconds = 0
    for issue in data.get("issues", []) or []:
        key = issue.get("key")
        f = issue.get("fields", {}) or {}
        summary = f.get("summary", "") or ""
        status = ((f.get("status") or {}).get("name") or "")
        url = f"{base_url}/browse/{key}"
        wl_field = f.get("worklog") or {}
        worklogs = wl_field.get("worklogs", []) or []
        # The JQL guarantees this issue has ≥1 matching worklog, so an empty or
        # truncated inline list means we must fetch the full list per-issue.
        if not worklogs or wl_field.get("total", len(worklogs)) > len(worklogs):
            worklogs = _fetch_all_worklogs(key)
        for w in worklogs:
            if account_id and ((w.get("author") or {}).get("accountId") != account_id):
                continue
            started = w.get("started", "") or ""
            day = started[:10]
            if not (date_from <= day <= date_to):
                continue
            secs = int(w.get("timeSpentSeconds") or 0)
            bucket = days.setdefault(day, {"seconds": 0, "entries": []})
            bucket["seconds"] += secs
            bucket["entries"].append({
                "id": str(w.get("id") or ""),
                "key": key,
                "summary": summary,
                "status": status,
                "url": url,
                "timeSpent": w.get("timeSpent", "") or "",
                "timeSpentSeconds": secs,
                "started": started,
            })
            total_seconds += secs
    return {"days": days, "totalSeconds": total_seconds}


def call_jira_parent_candidates(base_url, email, token, project_key, query="", issue_type=""):
    """List issues in `project_key` that can be a Sub-task's parent (i.e. any
    non-subtask issue — the category/parent cards like Leave, Admin, Training).
    Optional `query` filters by summary; optional `issue_type` restricts to one
    issue type (e.g. "Test Case"). Returns [{key, summary, status, type}]."""
    base_url = normalize_base_url(base_url)
    if not base_url:
        raise RuntimeError("Missing Jira Base URL")
    if not project_key:
        return []
    auth = base64.b64encode(f"{email}:{token}".encode("utf-8")).decode("ascii")
    # Avoid the subTaskIssueTypes() JQL function (unsupported on some instances);
    # fetch project issues and drop sub-tasks client-side via issuetype.subtask.
    parts = [f"project = {project_key}"]
    it = (issue_type or "").strip()
    if it:
        parts.append(f'issuetype = "{it.replace(chr(34), chr(92) + chr(34))}"')
    q = (query or "").strip()
    if q:
        safe = q.replace("\\", "\\\\").replace('"', '\\"')
        parts.append(f'summary ~ "{safe}*"')
    # Order by key ascending so the foundational low-key category cards (Leave,
    # Admin, ...) surface first when showing the whole list without a search.
    jql = " AND ".join(parts) + " ORDER BY key ASC"
    data = jira_search(base_url, auth, jql, ["summary", "status", "issuetype"], max_total=300)
    out = []
    for issue in (data.get("issues", []) or []):
        f = issue.get("fields", {}) or {}
        if ((f.get("issuetype") or {}).get("subtask")):
            continue  # sub-tasks can't be a parent
        out.append({
            "key": issue.get("key"),
            "summary": f.get("summary", "") or "",
            "status": ((f.get("status") or {}).get("name", "")) or "",
            "type": ((f.get("issuetype") or {}).get("name", "")) or "",
        })
    return out


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=APP_DIR, **kwargs)

    def _authorized(self):
        if not APP_PASSWORD:
            return True
        header = self.headers.get("Authorization", "")
        if not header.startswith("Basic "):
            return False
        try:
            user, _, pwd = base64.b64decode(header[6:]).decode("utf-8").partition(":")
        except Exception:
            return False
        ok_pwd = hmac.compare_digest(pwd.encode(), APP_PASSWORD.encode())
        ok_user = (not APP_USER) or hmac.compare_digest(user.encode(), APP_USER.encode())
        return ok_pwd and ok_user

    def _require_auth(self):
        if self._authorized():
            return True
        body = b"Login required"
        self.send_response(401)
        self.send_header("WWW-Authenticate", 'Basic realm="VisualReq", charset="UTF-8"')
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)
        return False

    def _route_static(self):
        path = self.path.split("?", 1)[0].split("#", 1)[0]
        if path == "/healthz":  # unauthenticated health check for hosting platforms
            body = b"ok"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)
            return False
        if not self._require_auth():
            return False
        if path in ("/", "/index.html"):
            self.send_response(302)
            self.send_header("Location", "/" + APP_FILE)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return False
        if path != "/" + APP_FILE:
            self.send_error(404, "Not found")
            return False
        self.path = "/" + APP_FILE
        return True

    def do_GET(self):
        if self._route_static():
            super().do_GET()

    def do_HEAD(self):
        if self._route_static():
            super().do_HEAD()

    def _send_json(self, status, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path not in ("/api/generate", "/api/jira", "/api/jira-bulk", "/api/jira-search",
                              "/api/jira-projects", "/api/jira-createmeta", "/api/jira-create",
                              "/api/jira-report", "/api/jira-sprints", "/api/jira-subtask",
                              "/api/jira-stories", "/api/jira-link",
                              "/api/jira-my-subtasks", "/api/jira-worklog",
                              "/api/jira-worklog-summary", "/api/jira-parent-candidates",
                              "/api/jira-subtasks-of-parent", "/api/jira-subtask-statuses"):
            self.send_error(404, "Not found")
            return
        if not self._require_auth():
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            if length < 0 or length > MAX_BODY_BYTES:
                self._send_json(413, {"error": f"Request too large (max {MAX_BODY_BYTES} bytes)"})
                return
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as e:
            self._send_json(400, {"error": f"Bad request: {e}"})
            return

        if self.path == "/api/jira":
            self._handle_jira(payload)
            return
        if self.path == "/api/jira-bulk":
            self._handle_jira_bulk(payload)
            return
        if self.path == "/api/jira-search":
            self._handle_jira_search(payload)
            return
        if self.path == "/api/jira-projects":
            self._handle_jira_projects(payload)
            return
        if self.path == "/api/jira-createmeta":
            self._handle_jira_createmeta(payload)
            return
        if self.path == "/api/jira-create":
            self._handle_jira_create(payload)
            return
        if self.path == "/api/jira-report":
            self._handle_jira_report(payload)
            return
        if self.path == "/api/jira-sprints":
            self._handle_jira_sprints(payload)
            return
        if self.path == "/api/jira-subtask":
            self._handle_jira_subtask(payload)
            return
        if self.path == "/api/jira-stories":
            self._handle_jira_stories(payload)
            return
        if self.path == "/api/jira-link":
            self._handle_jira_link(payload)
            return
        if self.path == "/api/jira-my-subtasks":
            self._handle_jira_my_subtasks(payload)
            return
        if self.path == "/api/jira-worklog":
            self._handle_jira_worklog(payload)
            return
        if self.path == "/api/jira-worklog-summary":
            self._handle_jira_worklog_summary(payload)
            return
        if self.path == "/api/jira-parent-candidates":
            self._handle_jira_parent_candidates(payload)
            return
        if self.path == "/api/jira-subtasks-of-parent":
            self._handle_jira_subtasks_of_parent(payload)
            return
        if self.path == "/api/jira-subtask-statuses":
            self._handle_jira_subtask_statuses(payload)
            return

        provider = payload.get("provider", "claude")
        api_key = payload.get("apiKey", "").strip()
        model = payload.get("model", "").strip()
        system = payload.get("system", "")
        prompt = payload.get("prompt", "")
        base_url = payload.get("endpoint", "").strip() or None

        if not api_key:
            self._send_json(400, {"error": "Missing API key. Open ⚙️ Settings and paste your key."})
            return

        # API keys must be plain ASCII. A key with hidden/Unicode characters
        # (from copy-pasting out of a chat app, PDF, or rich text) crashes the
        # HTTP layer with a latin-1 error — catch it here with a clear message.
        try:
            api_key.encode("ascii")
        except UnicodeEncodeError:
            bad = [c for c in api_key if ord(c) > 127]
            self._send_json(400, {
                "error": "API key contains invalid (non-ASCII) characters — it was likely "
                         "copied with hidden symbols. Delete it in ⚙️ Settings, then "
                         "re-copy the key directly from the provider's website and paste it again. "
                         f"(Found {len(bad)} bad character(s).)"
            })
            return

        try:
            if provider == "claude":
                text = call_claude(api_key, model, system, prompt)
            elif provider == "gemini":
                text = call_gemini(api_key, model, system, prompt)
            elif provider in ("openai", "azure"):
                text = call_openai(api_key, model, system, prompt, base_url)
            else:
                self._send_json(400, {"error": f"Unknown provider: {provider}"})
                return
            self._send_json(200, {"text": text})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            self._send_json(502, {"error": f"{provider} API error {e.code}: {detail[:500]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"Network error reaching {provider}: {e.reason}"})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def _handle_jira(self, payload):
        base_url = payload.get("baseUrl", "").strip()
        email = payload.get("email", "").strip()
        token = payload.get("token", "").strip()
        key = payload.get("key", "").strip()

        if not (base_url and email and token):
            self._send_json(400, {"error": "ตั้งค่า Jira ไม่ครบ — เปิด ⚙️ Settings แล้วกรอก Base URL, Email และ API Token"})
            return
        try:
            (email + token).encode("ascii")
        except UnicodeEncodeError:
            self._send_json(400, {"error": "Email/API Token มีตัวอักษรไม่ถูกต้อง (non-ASCII) — re-copy ใหม่"})
            return

        try:
            result = call_jira(base_url, email, token, key)
            self._send_json(200, {"jira": result})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            hint = ""
            if e.code in (401, 403):
                hint = " (ตรวจ email/token หรือสิทธิ์เข้าถึง project)"
            elif e.code == 404:
                hint = " (ไม่พบ issue นี้ หรือ Base URL ผิด)"
            self._send_json(502, {"error": f"Jira API error {e.code}{hint}: {detail[:300]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"เชื่อมต่อ Jira ไม่ได้: {e.reason} (ตรวจ Base URL)"})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def _handle_jira_bulk(self, payload):
        base_url = payload.get("baseUrl", "").strip()
        email = payload.get("email", "").strip()
        token = payload.get("token", "").strip()
        keys = payload.get("keys", []) or []

        if not (base_url and email and token):
            self._send_json(400, {"error": "ตั้งค่า Jira ไม่ครบ"})
            return
        if not keys:
            self._send_json(200, {"issues": {}})
            return
        try:
            (email + token).encode("ascii")
        except UnicodeEncodeError:
            self._send_json(400, {"error": "Email/API Token มีตัวอักษรไม่ถูกต้อง (non-ASCII)"})
            return

        try:
            result = call_jira_bulk(base_url, email, token, keys)
            self._send_json(200, {"issues": result})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            self._send_json(502, {"error": f"Jira bulk error {e.code}: {detail[:300]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"เชื่อมต่อ Jira ไม่ได้: {e.reason}"})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def _handle_jira_search(self, payload):
        base_url = payload.get("baseUrl", "").strip()
        email = payload.get("email", "").strip()
        token = payload.get("token", "").strip()
        query = payload.get("query", "")

        if not (base_url and email and token):
            self._send_json(400, {"error": "ตั้งค่า Jira ไม่ครบ"})
            return
        try:
            (email + token).encode("ascii")
        except UnicodeEncodeError:
            self._send_json(400, {"error": "Email/API Token มีตัวอักษรไม่ถูกต้อง (non-ASCII)"})
            return
        try:
            results = call_jira_search(base_url, email, token, query)
            self._send_json(200, {"results": results})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            self._send_json(502, {"error": f"Jira search error {e.code}: {detail[:300]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"เชื่อมต่อ Jira ไม่ได้: {e.reason}"})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def _handle_jira_projects(self, payload):
        base_url = payload.get("baseUrl", "").strip()
        email = payload.get("email", "").strip()
        token = payload.get("token", "").strip()

        if not (base_url and email and token):
            self._send_json(400, {"error": "ตั้งค่า Jira ไม่ครบ — เปิด ⚙️ Settings แล้วกรอก Base URL, Email และ API Token"})
            return
        try:
            (email + token).encode("ascii")
        except UnicodeEncodeError:
            self._send_json(400, {"error": "Email/API Token มีตัวอักษรไม่ถูกต้อง (non-ASCII) — re-copy ใหม่"})
            return
        try:
            result = call_jira_projects(base_url, email, token)
            self._send_json(200, {"projects": result})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            hint = " (ตรวจ email/token)" if e.code in (401, 403) else ""
            self._send_json(502, {"error": f"Jira API error {e.code}{hint}: {detail[:300]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"เชื่อมต่อ Jira ไม่ได้: {e.reason} (ตรวจ Base URL)"})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def _handle_jira_createmeta(self, payload):
        base_url = payload.get("baseUrl", "").strip()
        email = payload.get("email", "").strip()
        token = payload.get("token", "").strip()
        project_key = payload.get("projectKey", "").strip()
        issuetype_id = payload.get("issuetypeId", "").strip() or None

        if not (base_url and email and token):
            self._send_json(400, {"error": "ตั้งค่า Jira ไม่ครบ"})
            return
        try:
            (email + token).encode("ascii")
        except UnicodeEncodeError:
            self._send_json(400, {"error": "Email/API Token มีตัวอักษรไม่ถูกต้อง (non-ASCII)"})
            return
        try:
            result = call_jira_createmeta(base_url, email, token, project_key, issuetype_id)
            self._send_json(200, result)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            hint = " (ตรวจ email/token หรือสิทธิ์เข้าถึง project)" if e.code in (401, 403) else ""
            self._send_json(502, {"error": f"Jira API error {e.code}{hint}: {detail[:300]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"เชื่อมต่อ Jira ไม่ได้: {e.reason}"})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def _handle_jira_sprints(self, payload):
        base_url = payload.get("baseUrl", "").strip()
        email = payload.get("email", "").strip()
        token = payload.get("token", "").strip()
        project_key = payload.get("projectKey", "").strip()

        if not (base_url and email and token):
            self._send_json(400, {"error": "ตั้งค่า Jira ไม่ครบ"})
            return
        try:
            (email + token).encode("ascii")
        except UnicodeEncodeError:
            self._send_json(400, {"error": "Email/API Token มีตัวอักษรไม่ถูกต้อง (non-ASCII)"})
            return
        try:
            result = call_jira_sprints(base_url, email, token, project_key)
            self._send_json(200, {"sprints": result})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            hint = " (ตรวจ email/token หรือสิทธิ์เข้าถึง board)" if e.code in (401, 403) else ""
            self._send_json(502, {"error": f"Jira API error {e.code}{hint}: {detail[:300]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"เชื่อมต่อ Jira ไม่ได้: {e.reason}"})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def _handle_jira_create(self, payload):
        base_url = payload.get("baseUrl", "").strip()
        email = payload.get("email", "").strip()
        token = payload.get("token", "").strip()
        project_key = payload.get("projectKey", "").strip()
        issuetype_id = payload.get("issuetypeId", "").strip()
        summary = payload.get("summary", "").strip()
        description = payload.get("description", "")
        extra_fields = payload.get("fields", []) or []
        sprint_field_id = (payload.get("sprintFieldId") or "").strip() or None
        sprint_id = payload.get("sprintId") or None

        if not (base_url and email and token):
            self._send_json(400, {"error": "ตั้งค่า Jira ไม่ครบ"})
            return
        try:
            (email + token).encode("ascii")
        except UnicodeEncodeError:
            self._send_json(400, {"error": "Email/API Token มีตัวอักษรไม่ถูกต้อง (non-ASCII)"})
            return
        try:
            result = call_jira_create(base_url, email, token, project_key, issuetype_id, summary,
                                      description, extra_fields, sprint_field_id, sprint_id)
            self._send_json(200, result)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            hint = " (ตรวจค่าที่กรอก — บาง field อาจไม่ตรงตัวเลือกที่ Jira กำหนด)" if e.code == 400 else ""
            self._send_json(502, {"error": f"Jira API error {e.code}{hint}: {detail[:500]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"เชื่อมต่อ Jira ไม่ได้: {e.reason}"})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def _handle_jira_subtask(self, payload):
        base_url = payload.get("baseUrl", "").strip()
        email = payload.get("email", "").strip()
        token = payload.get("token", "").strip()
        project_key = payload.get("projectKey", "").strip()
        parent_key = payload.get("parentKey", "").strip()
        summary = payload.get("summary", "").strip()
        assign_to_me = payload.get("assignToMe", True)
        bill_type = payload.get("billType", "Non-Billable")
        target_status = payload.get("targetStatus", "")

        if not (base_url and email and token):
            self._send_json(400, {"error": "ตั้งค่า Jira ไม่ครบ"})
            return
        try:
            (email + token).encode("ascii")
        except UnicodeEncodeError:
            self._send_json(400, {"error": "Email/API Token มีตัวอักษรไม่ถูกต้อง (non-ASCII)"})
            return
        try:
            result = call_jira_create_subtask(base_url, email, token, project_key, parent_key,
                                              summary, assign_to_me, bill_type, target_status)
            self._send_json(200, result)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            hint = " (ตรวจค่าที่กรอก)" if e.code == 400 else ""
            self._send_json(502, {"error": f"Jira Subtask error {e.code}{hint}: {detail[:500]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"เชื่อมต่อ Jira ไม่ได้: {e.reason}"})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def _handle_jira_subtask_statuses(self, payload):
        base_url = payload.get("baseUrl", "").strip()
        email = payload.get("email", "").strip()
        token = payload.get("token", "").strip()
        project_key = payload.get("projectKey", "").strip()

        if not (base_url and email and token):
            self._send_json(400, {"error": "ตั้งค่า Jira ไม่ครบ"})
            return
        try:
            (email + token).encode("ascii")
        except UnicodeEncodeError:
            self._send_json(400, {"error": "Email/API Token มีตัวอักษรไม่ถูกต้อง (non-ASCII)"})
            return
        try:
            result = call_jira_subtask_statuses(base_url, email, token, project_key)
            self._send_json(200, {"statuses": result})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            self._send_json(502, {"error": f"Jira status error {e.code}: {detail[:300]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"เชื่อมต่อ Jira ไม่ได้: {e.reason}"})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def _handle_jira_stories(self, payload):
        base_url = payload.get("baseUrl", "").strip()
        email = payload.get("email", "").strip()
        token = payload.get("token", "").strip()
        project_key = payload.get("projectKey", "").strip()
        query = payload.get("query", "")

        if not (base_url and email and token):
            self._send_json(400, {"error": "ตั้งค่า Jira ไม่ครบ"})
            return
        try:
            (email + token).encode("ascii")
        except UnicodeEncodeError:
            self._send_json(400, {"error": "Email/API Token มีตัวอักษรไม่ถูกต้อง (non-ASCII)"})
            return
        try:
            result = call_jira_search_stories(base_url, email, token, project_key, query)
            self._send_json(200, {"stories": result})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            self._send_json(502, {"error": f"Jira story search error {e.code}: {detail[:300]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"เชื่อมต่อ Jira ไม่ได้: {e.reason}"})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def _handle_jira_link(self, payload):
        base_url = payload.get("baseUrl", "").strip()
        email = payload.get("email", "").strip()
        token = payload.get("token", "").strip()
        from_key = payload.get("fromKey", "").strip()
        to_key = payload.get("toKey", "").strip()
        link_type = payload.get("linkType", "Relates").strip() or "Relates"

        if not (base_url and email and token):
            self._send_json(400, {"error": "ตั้งค่า Jira ไม่ครบ"})
            return
        try:
            (email + token).encode("ascii")
        except UnicodeEncodeError:
            self._send_json(400, {"error": "Email/API Token มีตัวอักษรไม่ถูกต้อง (non-ASCII)"})
            return
        try:
            result = call_jira_link(base_url, email, token, from_key, to_key, link_type)
            self._send_json(200, result)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            self._send_json(502, {"error": f"Jira link error {e.code}: {detail[:300]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"เชื่อมต่อ Jira ไม่ได้: {e.reason}"})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def _handle_jira_my_subtasks(self, payload):
        base_url = payload.get("baseUrl", "").strip()
        email = payload.get("email", "").strip()
        token = payload.get("token", "").strip()
        project_key = payload.get("projectKey", "").strip()

        if not (base_url and email and token):
            self._send_json(400, {"error": "ตั้งค่า Jira ไม่ครบ"})
            return
        try:
            (email + token).encode("ascii")
        except UnicodeEncodeError:
            self._send_json(400, {"error": "Email/API Token มีตัวอักษรไม่ถูกต้อง (non-ASCII)"})
            return
        try:
            result = call_jira_my_subtasks(base_url, email, token, project_key or None)
            self._send_json(200, {"subtasks": result})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            self._send_json(502, {"error": f"Jira subtask search error {e.code}: {detail[:300]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"เชื่อมต่อ Jira ไม่ได้: {e.reason}"})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def _handle_jira_worklog(self, payload):
        base_url = payload.get("baseUrl", "").strip()
        email = payload.get("email", "").strip()
        token = payload.get("token", "").strip()
        issue_key = payload.get("key", "").strip()
        time_spent = payload.get("timeSpent", "").strip()
        started_date = payload.get("startedDate", "").strip()
        comment = payload.get("comment", "")

        if not (base_url and email and token):
            self._send_json(400, {"error": "ตั้งค่า Jira ไม่ครบ"})
            return
        try:
            (email + token).encode("ascii")
        except UnicodeEncodeError:
            self._send_json(400, {"error": "Email/API Token มีตัวอักษรไม่ถูกต้อง (non-ASCII)"})
            return
        try:
            result = call_jira_worklog(base_url, email, token, issue_key, time_spent, started_date, comment)
            self._send_json(200, result)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            hint = " (ตรวจรูปแบบเวลา เช่น 2h 30m)" if e.code == 400 else ""
            self._send_json(502, {"error": f"Jira worklog error {e.code}{hint}: {detail[:500]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"เชื่อมต่อ Jira ไม่ได้: {e.reason}"})
        except RuntimeError as e:
            self._send_json(400, {"error": str(e)})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def _handle_jira_subtasks_of_parent(self, payload):
        base_url = payload.get("baseUrl", "").strip()
        email = payload.get("email", "").strip()
        token = payload.get("token", "").strip()
        parent_key = payload.get("parentKey", "").strip()

        if not (base_url and email and token):
            self._send_json(400, {"error": "ตั้งค่า Jira ไม่ครบ"})
            return
        try:
            (email + token).encode("ascii")
        except UnicodeEncodeError:
            self._send_json(400, {"error": "Email/API Token มีตัวอักษรไม่ถูกต้อง (non-ASCII)"})
            return
        try:
            result = call_jira_subtasks_of_parent(base_url, email, token, parent_key)
            self._send_json(200, {"subtasks": result})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            self._send_json(502, {"error": f"Jira subtask search error {e.code}: {detail[:300]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"เชื่อมต่อ Jira ไม่ได้: {e.reason}"})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def _handle_jira_parent_candidates(self, payload):
        base_url = payload.get("baseUrl", "").strip()
        email = payload.get("email", "").strip()
        token = payload.get("token", "").strip()
        project_key = payload.get("projectKey", "").strip()
        query = payload.get("query", "")
        issue_type = payload.get("issueType", "")

        if not (base_url and email and token):
            self._send_json(400, {"error": "ตั้งค่า Jira ไม่ครบ"})
            return
        try:
            (email + token).encode("ascii")
        except UnicodeEncodeError:
            self._send_json(400, {"error": "Email/API Token มีตัวอักษรไม่ถูกต้อง (non-ASCII)"})
            return
        try:
            result = call_jira_parent_candidates(base_url, email, token, project_key, query, issue_type)
            self._send_json(200, {"parents": result})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            self._send_json(502, {"error": f"Jira parent search error {e.code}: {detail[:300]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"เชื่อมต่อ Jira ไม่ได้: {e.reason}"})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def _handle_jira_worklog_summary(self, payload):
        base_url = payload.get("baseUrl", "").strip()
        email = payload.get("email", "").strip()
        token = payload.get("token", "").strip()
        date_from = payload.get("dateFrom", "").strip()
        date_to = payload.get("dateTo", "").strip()

        if not (base_url and email and token):
            self._send_json(400, {"error": "ตั้งค่า Jira ไม่ครบ"})
            return
        try:
            (email + token).encode("ascii")
        except UnicodeEncodeError:
            self._send_json(400, {"error": "Email/API Token มีตัวอักษรไม่ถูกต้อง (non-ASCII)"})
            return
        try:
            result = call_jira_worklog_summary(base_url, email, token, date_from, date_to)
            self._send_json(200, result)
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            self._send_json(502, {"error": f"Jira worklog summary error {e.code}: {detail[:300]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"เชื่อมต่อ Jira ไม่ได้: {e.reason}"})
        except RuntimeError as e:
            self._send_json(400, {"error": str(e)})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def _handle_jira_report(self, payload):
        base_url = payload.get("baseUrl", "").strip()
        email = payload.get("email", "").strip()
        token = payload.get("token", "").strip()
        project_key = payload.get("projectKey", "").strip()
        date_from = payload.get("dateFrom", "").strip()
        date_to = payload.get("dateTo", "").strip()
        only_mine = bool(payload.get("onlyMine"))
        include_activity = bool(payload.get("includeActivity"))

        if not (base_url and email and token):
            self._send_json(400, {"error": "ตั้งค่า Jira ไม่ครบ"})
            return
        try:
            (email + token).encode("ascii")
        except UnicodeEncodeError:
            self._send_json(400, {"error": "Email/API Token มีตัวอักษรไม่ถูกต้อง (non-ASCII)"})
            return
        try:
            result = call_jira_report(base_url, email, token, project_key, date_from, date_to, only_mine, include_activity)
            self._send_json(200, {"issues": result})
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")
            self._send_json(502, {"error": f"Jira API error {e.code}: {detail[:300]}"})
        except urllib.error.URLError as e:
            self._send_json(502, {"error": f"เชื่อมต่อ Jira ไม่ได้: {e.reason}"})
        except Exception as e:
            self._send_json(500, {"error": f"Server error: {e}"})

    def end_headers(self):
        # Avoid stale cache while developing
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        super().end_headers()

    def log_message(self, fmt, *args):
        # Quieter logs
        pass


if __name__ == "__main__":
    server = ThreadingHTTPServer((HOST, PORT), Handler)
    print(f"VisualReq running →  http://{HOST}:{PORT}/visualreq_web_app_v2.html")
    if not IS_LOCAL and not APP_PASSWORD:
        print("[warn] HOST is not localhost and APP_PASSWORD is not set — anyone with the URL can use this app.")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
