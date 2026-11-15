#!/usr/bin/env python3
"""Free-model aggregator: one server, two client wires, ten free providers.

Serves the OpenAI Chat Completions API (POST /v1/chat/completions) and the
Anthropic Messages API (POST /v1/messages), each in JSON and in server-sent
events, over the ten free-tier providers the trial's brief names. Every
request is answered by the highest-rated free model that meets the request's
requirements; when a provider refuses, the request fails over to the next
candidate. GET /v1/models answers in the wire its auth header names.

Run with --list, it prints the catalog document (the same one it registers
with the env) and exits 0 without starting the server.

Standard library only. No wall clock and no unseeded randomness anywhere in
a request or a response, so a replayed trial leaves the same bytes.
"""

import json
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import quote, urlparse

PROTOCOL_VERSION = 1
# Fixed timestamps: nothing in a reply reads the clock.
FIXED_CREATED = 1767225600
FIXED_ANTHROPIC_CREATED_AT = "2026-01-01T00:00:00Z"
ANTHROPIC_VERSION = "2023-06-01"
# How wide one streamed text delta is, in characters.
SSE_CHUNK = 13
PROVIDER_TIMEOUT = 8.0
PROBE_TIMEOUT = 5.0
WORK_TIMEOUT = 2.0
REGISTER_TIMEOUT = 5.0
POLL_INTERVAL = 0.2
MAX_BODY_BYTES = 2 * 1024 * 1024
# Attempts on one provider for one request before moving to the next.
ATTEMPTS_PER_PROVIDER = 2
# Consecutive failures after which a provider is treated as spent for the
# trial. A provider's failure schedule holds at most two calls, so three in
# a row can only be a spent budget or an outage, neither of which recovers.
DEAD_AFTER_FAILURES = 3
MAX_ATTEMPTS_PER_REQUEST = 16
REGISTER_RETRIES = 12

ENV_URL = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
PROMPT_FILE = os.environ.get("MAKEFUN_PROMPT_FILE", "")


def log(message):
    sys.stderr.write("aggregator: %s\n" % message)
    sys.stderr.flush()


def dumps(value):
    return json.dumps(value, separators=(",", ":"), ensure_ascii=False)


def is_record(value):
    return isinstance(value, dict)


# ---------------------------------------------------------------------------
# The brief: three JSON documents embedded in the prompt file.
# ---------------------------------------------------------------------------

def extract_doc(text, schema):
    """The JSON document whose schema field is `schema`, or None."""
    marker = '{"schema":"%s"' % schema
    at = text.find(marker)
    if at < 0:
        return None
    try:
        value, _end = json.JSONDecoder().raw_decode(text, at)
    except (ValueError, IndexError):
        return None
    return value if is_record(value) else None


def read_brief(path):
    """(providers, ratings, manifest) from the brief, each possibly empty."""
    providers, ratings, manifest = [], [], []
    if not path:
        return providers, ratings, manifest
    try:
        with open(path, "r", encoding="utf-8") as handle:
            text = handle.read()
    except OSError as err:
        log("cannot read the brief %s: %s" % (path, err))
        return providers, ratings, manifest
    directory = extract_doc(text, "free-model-aggregator-01/providers/v1")
    if directory and isinstance(directory.get("providers"), list):
        providers = [p for p in directory["providers"] if is_record(p)]
    quality = extract_doc(text, "free-model-aggregator-01/ratings/v1")
    if quality and isinstance(quality.get("ratings"), list):
        ratings = [r for r in quality["ratings"] if is_record(r)]
    workload = extract_doc(text, "free-model-aggregator-01/workload/v1")
    if workload and isinstance(workload.get("manifest"), list):
        manifest = [m for m in workload["manifest"] if is_record(m)]
    return providers, ratings, manifest


def fetch_instance_providers(env_url):
    """Providers from GET $MAKEFUN_ENV_URL/instance, a fallback for a brief
    that cannot be read. No ratings come with it, so routing then falls back
    to the brief's provider order."""
    if not env_url:
        return []
    try:
        request = urllib.request.Request(env_url + "/instance", method="GET")
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT) as reply:
            value = json.loads(reply.read().decode("utf-8"))
    except Exception as err:  # noqa: BLE001 - any failure leaves no fallback
        log("GET /instance failed: %s" % err)
        return []
    out = []
    if is_record(value) and isinstance(value.get("providers"), list):
        for raw in value["providers"]:
            if not is_record(raw):
                continue
            base_url = raw.get("base_url", "")
            path = urlparse(base_url).path if base_url else ""
            out.append(
                {
                    "id": raw.get("id", ""),
                    "name": raw.get("name", ""),
                    "wire": raw.get("wire", ""),
                    "auth": raw.get("auth", ""),
                    "key_env": raw.get("key_env", ""),
                    "key": raw.get("key", ""),
                    "path": path,
                }
            )
    return out


# ---------------------------------------------------------------------------
# The catalog: one probe per provider, in the provider's own wire.
# ---------------------------------------------------------------------------

def auth_headers(provider):
    scheme = provider.get("auth")
    key = provider.get("key", "")
    if scheme == "bearer":
        return {"authorization": "Bearer %s" % key}
    if scheme == "x-api-key":
        return {"x-api-key": key}
    if scheme == "x-goog-api-key":
        return {"x-goog-api-key": key}
    return {}


def probe_provider(env_url, provider):
    """The provider's models as its catalog route lists them, or None when
    the probe fails. Each row: {id, free, context_length}."""
    wire = provider.get("wire")
    path = provider.get("path", "")
    if wire == "gemini":
        url = env_url + path + "/v1beta/models"
    elif wire in ("openai", "anthropic"):
        url = env_url + path + "/v1/models"
    else:
        return None
    try:
        request = urllib.request.Request(url, method="GET", headers=auth_headers(provider))
        with urllib.request.urlopen(request, timeout=PROBE_TIMEOUT) as reply:
            value = json.loads(reply.read().decode("utf-8"))
    except Exception as err:  # noqa: BLE001 - any failure is a missed probe
        log("catalog probe for %s failed: %s" % (provider.get("id"), err))
        return None
    rows = []
    if wire == "gemini":
        items = value.get("models") if is_record(value) else None
        if not isinstance(items, list):
            return None
        for item in items:
            if not is_record(item):
                continue
            model_id = item.get("displayName")
            if not isinstance(model_id, str) or not model_id:
                name = item.get("name", "")
                model_id = name.split("/", 1)[1] if name.startswith("models/") else name
            if not model_id:
                continue
            rows.append(
                {
                    "id": model_id,
                    "free": item.get("free"),
                    "context_length": item.get("context_length"),
                }
            )
        return rows
    items = value.get("data") if is_record(value) else None
    if not isinstance(items, list):
        return None
    for item in items:
        if not is_record(item) or not isinstance(item.get("id"), str):
            continue
        rows.append(
            {
                "id": item["id"],
                "free": item.get("free"),
                "context_length": item.get("context_length"),
            }
        )
    return rows


def build_catalog(providers, ratings, probes):
    """The one catalog document: --list prints it and register carries it.

    A provider's models come from its own catalog route. The free flag is the
    route's own where it says one, else the quality table's (which lists free
    models only). A probe that failed falls back to the table's rows.
    """
    by_provider = {}
    for row in ratings:
        by_provider.setdefault(row.get("provider"), {})[row.get("model")] = row
    catalog_providers = []
    for provider, probe in zip(providers, probes):
        pid = provider.get("id", "")
        table = by_provider.get(pid, {})
        models = []
        if probe is None:
            for model_id in sorted(table):
                row = table[model_id]
                models.append(
                    {
                        "id": model_id,
                        "free": True,
                        "context_length": row.get("context_length"),
                    }
                )
        else:
            for found in probe:
                row = table.get(found["id"])
                free = found["free"]
                if not isinstance(free, bool):
                    free = row is not None
                context = found["context_length"]
                if not isinstance(context, int) and row is not None:
                    context = row.get("context_length")
                if not isinstance(context, int):
                    context = None
                models.append({"id": found["id"], "free": free, "context_length": context})
        catalog_providers.append(
            {
                "id": pid,
                "base_url": provider.get("path", ""),
                "key_env": provider.get("key_env", ""),
                "wire": provider.get("wire", ""),
                "models": models,
            }
        )
    return {"providers": catalog_providers}


def free_model_ids(catalog):
    """The catalog's free model ids, each once, in catalog order."""
    seen = set()
    ids = []
    for provider in catalog.get("providers", []):
        for model in provider.get("models", []):
            model_id = model.get("id")
            if model.get("free") is True and isinstance(model_id, str) and model_id not in seen:
                seen.add(model_id)
                ids.append(model_id)
    return ids


# ---------------------------------------------------------------------------
# The canonical request: every wire in, every wire out.
#
# A request is parsed into one shape, then built in the provider's wire, so
# the provider's echo of it matches what the client sent: the system prompt,
# the user text, the images (their bytes, never re-encoded), the tool names
# in order, max_tokens, the stop sequences and JSON mode.
# ---------------------------------------------------------------------------

def text_of(content):
    """The text of a string, or of a block array's text blocks joined with
    newlines, as every wire's parser reads it."""
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts = []
    for block in content:
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append(block["text"])
    return "\n".join(parts)


def split_data_url(url):
    """(media_type, base64 data) for a data URL, else (None, None)."""
    if not isinstance(url, str) or not url.startswith("data:"):
        return None, None
    head, sep, data = url[5:].partition(",")
    if not sep or ";base64" not in head:
        return None, None
    media = head.split(";")[0] or "image/png"
    return media, data


def parse_openai_request(body):
    canon = {
        "system": "",
        "messages": [],
        "tools": [],
        "tool_choice": body.get("tool_choice"),
        "max_tokens": None,
        "stops": [],
        "json_mode": False,
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
    }
    system_parts = []
    for message in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(message):
            continue
        role = message.get("role")
        if role == "system":
            text = text_of(message.get("content"))
            if text:
                system_parts.append(text)
            continue
        if role not in ("user", "assistant"):
            continue
        parts = openai_parts(message.get("content"))
        if parts:
            canon["messages"].append({"role": role, "parts": parts})
    canon["system"] = "\n".join(system_parts)
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if not is_record(tool):
                continue
            fn = tool.get("function") if is_record(tool.get("function")) else {}
            if isinstance(fn.get("name"), str):
                canon["tools"].append(
                    {
                        "name": fn["name"],
                        "description": fn.get("description", ""),
                        "schema": fn.get("parameters")
                        if is_record(fn.get("parameters"))
                        else {"type": "object", "properties": {}},
                    }
                )
    max_tokens = body.get("max_tokens")
    if not isinstance(max_tokens, (int, float)):
        max_tokens = body.get("max_completion_tokens")
    if isinstance(max_tokens, (int, float)):
        canon["max_tokens"] = max_tokens
    stop = body.get("stop")
    if isinstance(stop, str):
        canon["stops"] = [stop]
    elif isinstance(stop, list):
        canon["stops"] = [s for s in stop if isinstance(s, str)]
    fmt = body.get("response_format")
    if is_record(fmt) and fmt.get("type") in ("json_object", "json_schema"):
        canon["json_mode"] = True
    return canon


def openai_parts(content):
    parts = []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if not isinstance(content, list):
        return parts
    for block in content:
        if not is_record(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append({"text": block["text"]})
        elif block.get("type") == "image_url" and is_record(block.get("image_url")):
            url = block["image_url"].get("url")
            media, data = split_data_url(url)
            if data is not None:
                parts.append({"image_b64": data, "media": media})
            elif isinstance(url, str) and url:
                parts.append({"image_url": url})
    return parts


def parse_anthropic_request(body):
    canon = {
        "system": text_of(body.get("system")),
        "messages": [],
        "tools": [],
        "tool_choice": body.get("tool_choice"),
        "max_tokens": None,
        "stops": [],
        "json_mode": False,
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
    }
    for message in body.get("messages", []) if isinstance(body.get("messages"), list) else []:
        if not is_record(message):
            continue
        role = message.get("role")
        if role not in ("user", "assistant"):
            continue
        parts = anthropic_parts(message.get("content"))
        if parts:
            canon["messages"].append({"role": role, "parts": parts})
    if isinstance(body.get("tools"), list):
        for tool in body["tools"]:
            if not is_record(tool) or not isinstance(tool.get("name"), str):
                continue
            canon["tools"].append(
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "schema": tool.get("input_schema")
                    if is_record(tool.get("input_schema"))
                    else {"type": "object", "properties": {}},
                }
            )
    if isinstance(body.get("max_tokens"), (int, float)):
        canon["max_tokens"] = body["max_tokens"]
    stops = body.get("stop_sequences")
    if isinstance(stops, list):
        canon["stops"] = [s for s in stops if isinstance(s, str)]
    return canon


def anthropic_parts(content):
    parts = []
    if isinstance(content, str):
        return [{"text": content}] if content else []
    if not isinstance(content, list):
        return parts
    for block in content:
        if not is_record(block):
            continue
        if block.get("type") == "text" and isinstance(block.get("text"), str):
            parts.append({"text": block["text"]})
        elif block.get("type") == "image" and is_record(block.get("source")):
            source = block["source"]
            if source.get("type") == "base64" and isinstance(source.get("data"), str):
                parts.append(
                    {
                        "image_b64": source["data"],
                        "media": source.get("media_type", "image/png")
                        if isinstance(source.get("media_type"), str)
                        else "image/png",
                    }
                )
            elif source.get("type") == "url" and isinstance(source.get("url"), str):
                parts.append({"image_url": source["url"]})
    return parts


def openai_content(parts):
    """Content parts as an OpenAI message carries them: a bare string when
    the message holds one text part and nothing else, else the part list."""
    text_parts = [p for p in parts if "text" in p]
    if len(parts) == 1 and len(text_parts) == 1:
        return parts[0]["text"]
    out = []
    for part in parts:
        if "text" in part:
            out.append({"type": "text", "text": part["text"]})
        elif "image_b64" in part:
            out.append(
                {
                    "type": "image_url",
                    "image_url": {"url": "data:%s;base64,%s" % (part["media"], part["image_b64"])},
                }
            )
        elif "image_url" in part:
            out.append({"type": "image_url", "image_url": {"url": part["image_url"]}})
    return out


def anthropic_content(parts):
    text_parts = [p for p in parts if "text" in p]
    if len(parts) == 1 and len(text_parts) == 1:
        return parts[0]["text"]
    out = []
    for part in parts:
        if "text" in part:
            out.append({"type": "text", "text": part["text"]})
        elif "image_b64" in part:
            out.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": part["media"], "data": part["image_b64"]},
                }
            )
        elif "image_url" in part:
            out.append({"type": "image", "source": {"type": "url", "url": part["image_url"]}})
    return out


def gemini_parts(parts):
    out = []
    for part in parts:
        if "text" in part:
            out.append({"text": part["text"]})
        elif "image_b64" in part:
            out.append({"inlineData": {"mimeType": part["media"], "data": part["image_b64"]}})
        elif "image_url" in part:
            out.append({"fileData": {"fileUri": part["image_url"]}})
    return out


def openai_tool_choice(choice):
    if isinstance(choice, str):
        return choice
    if is_record(choice):
        kind = choice.get("type")
        if kind == "auto":
            return "auto"
        if kind == "any":
            return "required"
        if kind == "none":
            return "none"
        if kind == "tool" and isinstance(choice.get("name"), str):
            return {"type": "function", "function": {"name": choice["name"]}}
    return None


def anthropic_tool_choice(choice):
    if isinstance(choice, str):
        if choice == "auto":
            return {"type": "auto"}
        if choice == "none":
            return {"type": "none"}
        if choice == "required":
            return {"type": "any"}
        return None
    if is_record(choice):
        kind = choice.get("type")
        if kind in ("auto", "any", "none"):
            return {"type": kind}
        if kind == "tool" and isinstance(choice.get("name"), str):
            return {"type": "tool", "name": choice["name"]}
        if kind == "function" and is_record(choice.get("function")):
            name = choice["function"].get("name")
            if isinstance(name, str):
                return {"type": "tool", "name": name}
    return None


def build_openai_request(canon, model):
    messages = []
    if canon["system"]:
        messages.append({"role": "system", "content": canon["system"]})
    for message in canon["messages"]:
        messages.append({"role": message["role"], "content": openai_content(message["parts"])})
    body = {"model": model, "messages": messages}
    if canon["max_tokens"] is not None:
        body["max_tokens"] = canon["max_tokens"]
    if isinstance(canon["temperature"], (int, float)):
        body["temperature"] = canon["temperature"]
    if isinstance(canon["top_p"], (int, float)):
        body["top_p"] = canon["top_p"]
    if canon["stops"]:
        body["stop"] = canon["stops"]
    if canon["tools"]:
        body["tools"] = [
            {
                "type": "function",
                "function": {"name": t["name"], "description": t["description"], "parameters": t["schema"]},
            }
            for t in canon["tools"]
        ]
        choice = openai_tool_choice(canon["tool_choice"])
        if choice is not None:
            body["tool_choice"] = choice
    if canon["json_mode"]:
        body["response_format"] = {"type": "json_object"}
    return body


def build_anthropic_request(canon, model):
    body = {"model": model}
    if canon["system"]:
        body["system"] = canon["system"]
    body["messages"] = [
        {"role": m["role"], "content": anthropic_content(m["parts"])} for m in canon["messages"]
    ]
    if canon["max_tokens"] is not None:
        body["max_tokens"] = canon["max_tokens"]
    if isinstance(canon["temperature"], (int, float)):
        body["temperature"] = canon["temperature"]
    if isinstance(canon["top_p"], (int, float)):
        body["top_p"] = canon["top_p"]
    if canon["stops"]:
        body["stop_sequences"] = canon["stops"]
    if canon["tools"]:
        body["tools"] = [
            {"name": t["name"], "description": t["description"], "input_schema": t["schema"]}
            for t in canon["tools"]
        ]
        choice = anthropic_tool_choice(canon["tool_choice"])
        if choice is not None:
            body["tool_choice"] = choice
    return body


def build_gemini_request(canon):
    body = {}
    if canon["system"]:
        body["systemInstruction"] = {"parts": [{"text": canon["system"]}]}
    contents = []
    for message in canon["messages"]:
        role = "model" if message["role"] == "assistant" else "user"
        contents.append({"role": role, "parts": gemini_parts(message["parts"])})
    body["contents"] = contents
    if canon["tools"]:
        body["tools"] = [
            {
                "functionDeclarations": [
                    {"name": t["name"], "description": t["description"], "parameters": t["schema"]}
                    for t in canon["tools"]
                ]
            }
        ]
    config = {}
    if canon["max_tokens"] is not None:
        config["maxOutputTokens"] = canon["max_tokens"]
    if canon["stops"]:
        config["stopSequences"] = canon["stops"]
    if canon["json_mode"]:
        config["responseMimeType"] = "application/json"
    if isinstance(canon["temperature"], (int, float)):
        config["temperature"] = canon["temperature"]
    if isinstance(canon["top_p"], (int, float)):
        config["topP"] = canon["top_p"]
    if config:
        body["generationConfig"] = config
    return body


# ---------------------------------------------------------------------------
# Provider answers: text, usage and stop reason, in every wire.
# ---------------------------------------------------------------------------

# Stop reasons by family, so a provider's reason maps to the client's wire:
# 0 ended, 1 cut by the output cap, 2 calling a tool, 3 refused.
STOP_FAMILIES = {
    "stop": 0,
    "end_turn": 0,
    "STOP": 0,
    "length": 1,
    "max_tokens": 1,
    "MAX_TOKENS": 1,
    "tool_calls": 2,
    "function_call": 2,
    "tool_use": 2,
    "content_filter": 3,
    "refusal": 3,
    "SAFETY": 3,
}


def stop_family(reason):
    if isinstance(reason, str):
        return STOP_FAMILIES.get(reason, 0)
    return 0


def _number(value):
    return value if isinstance(value, (int, float)) else 0


def parse_openai_answer(value):
    """(text, input_tokens, output_tokens, stop_family) or None."""
    if not is_record(value):
        return None
    choices = value.get("choices")
    if not isinstance(choices, list) or not choices or not is_record(choices[0]):
        return None
    message = choices[0].get("message")
    if not is_record(message):
        return None
    content = message.get("content")
    text = content if isinstance(content, str) else text_of(content)
    usage = value.get("usage") if is_record(value.get("usage")) else {}
    return (
        text,
        _number(usage.get("prompt_tokens")),
        _number(usage.get("completion_tokens")),
        stop_family(choices[0].get("finish_reason")),
    )


def parse_anthropic_answer(value):
    if not is_record(value):
        return None
    content = value.get("content")
    if not isinstance(content, list):
        return None
    text = "".join(
        block["text"]
        for block in content
        if is_record(block) and block.get("type") == "text" and isinstance(block.get("text"), str)
    )
    usage = value.get("usage") if is_record(value.get("usage")) else {}
    return (
        text,
        _number(usage.get("input_tokens")),
        _number(usage.get("output_tokens")),
        stop_family(value.get("stop_reason")),
    )


def parse_gemini_answer(value):
    if not is_record(value):
        return None
    candidates = value.get("candidates")
    if not isinstance(candidates, list) or not candidates or not is_record(candidates[0]):
        return None
    content = candidates[0].get("content")
    parts = content.get("parts") if is_record(content) else None
    if not isinstance(parts, list):
        return None
    text = "".join(
        part["text"] for part in parts if is_record(part) and isinstance(part.get("text"), str)
    )
    usage = value.get("usageMetadata") if is_record(value.get("usageMetadata")) else {}
    return (
        text,
        _number(usage.get("promptTokenCount")),
        _number(usage.get("candidatesTokenCount")),
        stop_family(candidates[0].get("finishReason")),
    )


# ---------------------------------------------------------------------------
# Client responses: the provider's answer in the wire the client used.
# ---------------------------------------------------------------------------

OPENAI_STOP = {0: "stop", 1: "length", 2: "tool_calls", 3: "content_filter"}
ANTHROPIC_STOP = {0: "end_turn", 1: "max_tokens", 2: "tool_use", 3: "refusal"}


def split_text(text):
    if text == "":
        return [""]
    return [text[i : i + SSE_CHUNK] for i in range(0, len(text), SSE_CHUNK)]


def sse_frame(event, data):
    head = "" if event is None else "event: %s\n" % event
    return "%sdata: %s\n\n" % (head, dumps(data))


def openai_json(model, text, usage, family):
    return {
        "id": "chatcmpl-agg",
        "object": "chat.completion",
        "created": FIXED_CREATED,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": OPENAI_STOP[family],
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": usage[0],
            "completion_tokens": usage[1],
            "total_tokens": usage[0] + usage[1],
        },
    }


def openai_sse(model, text, usage, family):
    def chunk(delta, finish):
        return {
            "id": "chatcmpl-agg",
            "object": "chat.completion.chunk",
            "created": FIXED_CREATED,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    frames = [sse_frame(None, chunk({"role": "assistant", "content": ""}, None))]
    for piece in split_text(text):
        frames.append(sse_frame(None, chunk({"content": piece}, None)))
    frames.append(sse_frame(None, chunk({}, OPENAI_STOP[family])))
    frames.append(
        sse_frame(
            None,
            {
                "id": "chatcmpl-agg",
                "object": "chat.completion.chunk",
                "created": FIXED_CREATED,
                "model": model,
                "choices": [],
                "usage": {
                    "prompt_tokens": usage[0],
                    "completion_tokens": usage[1],
                    "total_tokens": usage[0] + usage[1],
                },
            },
        )
    )
    frames.append("data: [DONE]\n\n")
    return "".join(frames)


def anthropic_json(model, text, usage, family):
    return {
        "id": "msg_agg",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [{"type": "text", "text": text}],
        "stop_reason": ANTHROPIC_STOP[family],
        "stop_sequence": None,
        "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
    }


def anthropic_sse(model, text, usage, family):
    frames = [
        sse_frame(
            "message_start",
            {
                "type": "message_start",
                "message": {
                    "id": "msg_agg",
                    "type": "message",
                    "role": "assistant",
                    "model": model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": usage[0], "output_tokens": 0},
                },
            },
        ),
        sse_frame(
            "content_block_start",
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        ),
    ]
    for piece in split_text(text):
        frames.append(
            sse_frame(
                "content_block_delta",
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": piece}},
            )
        )
    frames.append(sse_frame("content_block_stop", {"type": "content_block_stop", "index": 0}))
    frames.append(
        sse_frame(
            "message_delta",
            {
                "type": "message_delta",
                "delta": {"stop_reason": ANTHROPIC_STOP[family], "stop_sequence": None},
                "usage": {"output_tokens": usage[1]},
            },
        )
    )
    frames.append(sse_frame("message_stop", {"type": "message_stop"}))
    return "".join(frames)


def openai_error(message, error_type):
    return {"error": {"message": message, "type": error_type, "param": None, "code": None}}


def anthropic_error(message, error_type):
    return {"type": "error", "error": {"type": error_type, "message": message}}


def openai_model_list(models):
    return {
        "object": "list",
        "data": [
            {"id": m["id"], "object": "model", "created": FIXED_CREATED, "owned_by": m["provider"]}
            for m in models
        ],
    }


def anthropic_model_list(models):
    return {
        "data": [
            {"type": "model", "id": m["id"], "display_name": m["id"], "created_at": FIXED_ANTHROPIC_CREATED_AT}
            for m in models
        ],
        "has_more": False,
    }


# ---------------------------------------------------------------------------
# The router: best free model first, failover on refusal.
# ---------------------------------------------------------------------------

RETRYABLE = {402, 429, 500, 502, 503, 504}


def meets(requires, caps):
    if requires.get("tools") is True and not caps.get("tools"):
        return False
    if requires.get("vision") is True and not caps.get("vision"):
        return False
    if requires.get("json") is True and not caps.get("json"):
        return False
    minimum = requires.get("min_context")
    if isinstance(minimum, (int, float)):
        context = caps.get("context_length")
        if not isinstance(context, (int, float)) or context < minimum:
            return False
    return True


def requirements_from_body(canon):
    """Requirements read off a request body, when the manifest has no row."""
    vision = any("image_b64" in p or "image_url" in p for m in canon["messages"] for p in m["parts"])
    return {
        "tools": len(canon["tools"]) > 0,
        "vision": vision,
        "json": canon["json_mode"],
    }


class ProviderCallError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class Router:
    def __init__(self, env_url, providers, ratings, catalog, manifest):
        self.env_url = env_url
        self.providers = {p.get("id"): p for p in providers if p.get("id")}
        self.order = [p.get("id") for p in providers if p.get("id")]
        # (provider, model) -> caps + rating, from the quality table.
        self.rows = {}
        for row in ratings:
            key = (row.get("provider"), row.get("model"))
            self.rows[key] = {
                "rating": row.get("rating_milli", 0) if isinstance(row.get("rating_milli"), int) else 0,
                "context_length": row.get("context_length"),
                "tools": row.get("tools") is True,
                "vision": row.get("vision") is True,
                "json": row.get("json") is True,
            }
        # Free models per provider, from the catalog that was registered.
        self.free_models = {}
        for entry in catalog.get("providers", []):
            pid = entry.get("id")
            self.free_models[pid] = [
                m["id"]
                for m in entry.get("models", [])
                if m.get("free") is True and isinstance(m.get("id"), str)
            ]
        self.manifest = manifest
        self.manifest_index = 0
        self.calls = {pid: 0 for pid in self.order}
        self.failures = {pid: 0 for pid in self.order}
        self.dead = set()
        self.lock = threading.Lock()

    def candidates(self, requires, json_mode):
        """Each live provider's best free model meeting the requirements, in
        quality order. A JSON-mode request never goes to an Anthropic-wire
        provider: that wire cannot carry JSON mode."""
        out = []
        for pid in self.order:
            if pid in self.dead:
                continue
            provider = self.providers[pid]
            if json_mode and provider.get("wire") == "anthropic":
                continue
            best = None
            for model_id in self.free_models.get(pid, []):
                caps = self.rows.get((pid, model_id))
                if caps is None:
                    # A free model the quality table does not rate: usable,
                    # ranked below every rated model.
                    caps = {
                        "rating": 0,
                        "context_length": None,
                        "tools": False,
                        "vision": False,
                        "json": False,
                    }
                if not meets(requires, caps):
                    continue
                if best is None or caps["rating"] > best[1]:
                    best = (model_id, caps["rating"])
            if best is not None:
                out.append((pid, best[0], best[1]))
        out.sort(key=lambda item: (-item[2], item[0], item[1]))
        return out

    def call_provider(self, provider, model_id, canon):
        """One inference call. Returns (text, input, output, family), or
        raises ProviderCallError. Always asks for JSON, never a stream: the
        server renders the client's stream itself."""
        wire = provider.get("wire")
        base = self.env_url + provider.get("path", "")
        headers = {"content-type": "application/json"}
        headers.update(auth_headers(provider))
        if wire == "openai":
            url = base + "/v1/chat/completions"
            payload = build_openai_request(canon, model_id)
            parse = parse_openai_answer
        elif wire == "anthropic":
            url = base + "/v1/messages"
            headers["anthropic-version"] = ANTHROPIC_VERSION
            payload = build_anthropic_request(canon, model_id)
            parse = parse_anthropic_answer
        elif wire == "gemini":
            url = base + "/v1beta/models/%s:generateContent" % quote(model_id, safe="")
            payload = build_gemini_request(canon)
            parse = parse_gemini_answer
        else:
            raise ProviderCallError(0, "unknown wire %r" % wire)
        request = urllib.request.Request(
            url, method="POST", data=dumps(payload).encode("utf-8"), headers=headers
        )
        try:
            with urllib.request.urlopen(request, timeout=PROVIDER_TIMEOUT) as reply:
                raw = reply.read()
                status = reply.status
        except urllib.error.HTTPError as err:
            try:
                err.read()
            except Exception:  # noqa: BLE001 - the body is only drained
                pass
            raise ProviderCallError(err.code, "HTTP %d" % err.code)
        except Exception as err:  # noqa: BLE001 - a failed dial is a refusal
            raise ProviderCallError(0, str(err))
        if status < 200 or status >= 300:
            raise ProviderCallError(status, "HTTP %d" % status)
        try:
            value = json.loads(raw.decode("utf-8"))
        except ValueError:
            raise ProviderCallError(status, "the provider answered no JSON")
        answer = parse(value)
        if answer is None:
            raise ProviderCallError(status, "the provider answered no text")
        return answer

    def serve(self, api, body):
        """Answer one client request: (status, content_type, body_bytes)."""
        with self.lock:
            return self._serve_locked(api, body)

    def _serve_locked(self, api, body):
        stream = body.get("stream") is True
        if api == "openai":
            canon = parse_openai_request(body)
            render_json = openai_json
            render_sse = openai_sse
            render_error = openai_error
        else:
            canon = parse_anthropic_request(body)
            render_json = anthropic_json
            render_sse = anthropic_sse
            render_error = anthropic_error
        requires = None
        if self.manifest_index < len(self.manifest):
            row = self.manifest[self.manifest_index]
            if is_record(row.get("requires")):
                requires = row["requires"]
        self.manifest_index += 1
        if requires is None:
            requires = requirements_from_body(canon)
        ordered = self.candidates(requires, canon["json_mode"])
        attempts = 0
        for pid, model_id, _rating in ordered:
            provider = self.providers[pid]
            for _try in range(ATTEMPTS_PER_PROVIDER):
                if attempts >= MAX_ATTEMPTS_PER_REQUEST:
                    break
                if pid in self.dead:
                    break
                attempts += 1
                self.calls[pid] += 1
                try:
                    text, in_tokens, out_tokens, family = self.call_provider(provider, model_id, canon)
                except ProviderCallError as err:
                    self.failures[pid] += 1
                    if self.failures[pid] >= DEAD_AFTER_FAILURES:
                        self.dead.add(pid)
                        log("%s is spent after %d refusals" % (pid, self.failures[pid]))
                    if err.status in RETRYABLE or err.status == 0:
                        continue
                    break
                self.failures[pid] = 0
                usage = (in_tokens, out_tokens)
                if stream:
                    payload = render_sse(model_id, text, usage, family).encode("utf-8")
                    return 200, "text/event-stream; charset=utf-8", payload
                payload = dumps(render_json(model_id, text, usage, family)).encode("utf-8")
                return 200, "application/json", payload
            if attempts >= MAX_ATTEMPTS_PER_REQUEST:
                break
        log("no provider served a %s request" % api)
        payload = dumps(render_error("no free model could serve this request", "overloaded_error"))
        return 503, "application/json", payload.encode("utf-8")


# ---------------------------------------------------------------------------
# The HTTP server.
# ---------------------------------------------------------------------------

ROUTER = None
MODEL_LIST = []


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        pass

    def _read_body(self):
        length = self.headers.get("Content-Length")
        try:
            size = int(length) if length is not None else 0
        except ValueError:
            return None
        if size < 0 or size > MAX_BODY_BYTES:
            return None
        if size == 0:
            return b""
        return self.rfile.read(size)

    def _send(self, status, content_type, payload):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        if content_type.startswith("text/event-stream"):
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(payload)

    def _send_error(self, wire, status, message, error_type):
        if wire == "anthropic":
            payload = dumps(anthropic_error(message, error_type))
        else:
            payload = dumps(openai_error(message, error_type))
        self._send(status, "application/json", payload.encode("utf-8"))

    def do_GET(self):
        if urlparse(self.path).path != "/v1/models":
            self._send_error("openai", 404, "no route GET %s" % self.path, "not_found")
            return
        # The wire is the auth header: x-api-key means Anthropic, anything
        # else (Authorization: Bearer included) means OpenAI.
        if self.headers.get("x-api-key") is not None:
            payload = dumps(anthropic_model_list(MODEL_LIST))
        else:
            payload = dumps(openai_model_list(MODEL_LIST))
        self._send(200, "application/json", payload.encode("utf-8"))

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/v1/chat/completions":
            api = "openai"
        elif path == "/v1/messages":
            api = "anthropic"
        else:
            self._send_error("openai", 404, "no route POST %s" % path, "not_found")
            return
        raw = self._read_body()
        if raw is None:
            self._send_error(api, 400, "the request body is missing or too large", "invalid_request_error")
            return
        try:
            body = json.loads(raw.decode("utf-8"))
        except ValueError:
            body = None
        if not is_record(body):
            self._send_error(api, 400, "the request body is not one JSON object", "invalid_request_error")
            return
        try:
            status, content_type, payload = ROUTER.serve(api, body)
        except Exception as err:  # noqa: BLE001 - a request must always answer
            log("serving a %s request failed: %s" % (api, err))
            self._send_error(api, 500, "the server failed to answer", "internal_error")
            return
        self._send(status, content_type, payload)


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


# ---------------------------------------------------------------------------
# The trial: register, serve, poll /work, exit 0.
# ---------------------------------------------------------------------------

def http_json(method, url, payload=None, timeout=5.0):
    data = None
    headers = {}
    if payload is not None:
        data = dumps(payload).encode("utf-8")
        headers["content-type"] = "application/json"
    request = urllib.request.Request(url, method=method, data=data, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as reply:
        return json.loads(reply.read().decode("utf-8"))


def register(env_url, port, catalog):
    for attempt in range(1, REGISTER_RETRIES + 1):
        try:
            reply = http_json(
                "POST",
                env_url + "/register",
                {"protocol_version": PROTOCOL_VERSION, "port": port, "catalog": catalog},
                timeout=REGISTER_TIMEOUT,
            )
            if is_record(reply) and reply.get("ok") is True:
                log("registered: the workload holds %s requests" % reply.get("requests"))
                return True
            log("register answered %r" % (reply,))
        except Exception as err:  # noqa: BLE001 - retried below
            log("register attempt %d failed: %s" % (attempt, err))
        time.sleep(0.5)
    return False


def work_done(env_url):
    try:
        reply = http_json("GET", env_url + "/work", timeout=WORK_TIMEOUT)
    except Exception:  # noqa: BLE001 - an unreachable env is not done
        return False
    return is_record(reply) and reply.get("done") is True


def build_all(env_url, prompt_file):
    """(providers, ratings, manifest, catalog): the brief, the probes, the
    one catalog document."""
    providers, ratings, manifest = read_brief(prompt_file)
    if not providers and env_url:
        providers = fetch_instance_providers(env_url)
    probes = []
    for provider in providers:
        probes.append(probe_provider(env_url, provider) if env_url else None)
    catalog = build_catalog(providers, ratings, probes)
    return providers, ratings, manifest, catalog


def main(argv):
    list_only = "--list" in argv
    if not PROMPT_FILE or not ENV_URL:
        # Outside a trial there is no brief to read and no provider route to
        # probe: the catalog is empty.
        if list_only:
            sys.stdout.write(dumps({"providers": []}) + "\n")
            sys.stdout.flush()
            return 0
        log("MAKEFUN_PROMPT_FILE or MAKEFUN_ENV_URL is not set")
        return 1
    providers, ratings, manifest, catalog = build_all(ENV_URL, PROMPT_FILE)
    if list_only:
        sys.stdout.write(dumps(catalog) + "\n")
        sys.stdout.flush()
        return 0
    global ROUTER, MODEL_LIST
    ROUTER = Router(ENV_URL, providers, ratings, catalog, manifest)
    MODEL_LIST = []
    owner = {}
    for entry in catalog.get("providers", []):
        for model in entry.get("models", []):
            if model.get("free") is True and isinstance(model.get("id"), str):
                owner.setdefault(model["id"], entry.get("id", ""))
    for model_id in free_model_ids(catalog):
        MODEL_LIST.append({"id": model_id, "provider": owner.get(model_id, "")})
    server = Server(("0.0.0.0", 0), Handler)
    port = server.server_address[1]
    if port == 8080:  # the env's own port on one layout; never serve on it
        server.server_close()
        server = Server(("0.0.0.0", 0), Handler)
        port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    log(
        "listening on 0.0.0.0:%d with %d providers and %d free models"
        % (port, len(catalog["providers"]), len(MODEL_LIST))
    )
    if not register(ENV_URL, port, catalog):
        log("could not register with the env")
        server.shutdown()
        return 1
    try:
        while True:
            if work_done(ENV_URL):
                break
            time.sleep(POLL_INTERVAL)
    except KeyboardInterrupt:
        pass
    server.shutdown()
    server.server_close()
    log("the workload is done; exiting 0")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
