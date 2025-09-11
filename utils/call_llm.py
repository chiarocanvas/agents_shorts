from openai import OpenAI
import os
import time
from typing import Dict, Optional, Any
import json as _json
import  json 
from  .prompts import  _load_prompts


def call_llm(
    prompt_text: str,
    text: str,
    *,
    timeout: float = 30.0,
    retries: int = 2,
    delay: float = 1.0,
    response_format_json: bool = False,
    json_schema: Optional[Dict[str, Any]] = None,
) -> dict:
    model_id = os.getenv("MODEL_ID", "ruadaptqwen3-8b-hybrid")
    base_url = os.getenv("OPENAI_BASE_URL", "http://localhost:1234/v1")
    api_key = os.getenv("OPENAI_API_KEY", "lm-studio")

    last_err: Optional[Exception] = None
    # Ensure content payloads are strings to satisfy OpenAI-compatible servers
    user_content = str(text) if not isinstance(text, str) else text

    for attempt in range(retries + 1):
        try:
            client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
            create_kwargs: Dict[str, Any] = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": prompt_text},
                    {"role": "user", "content": f"Текст: {user_content} /no_think"},
                ],
            }
            # Prefer structured output if server supports it
            if response_format_json:
                # Try OpenAI-style json_object first
                create_kwargs["response_format"] = {"type": "json_object"}
                # If explicit json schema provided and server supports it, pass it
                if json_schema:
                    create_kwargs["response_format"] = {
                        "type": "json_schema",
                        "json_schema": json_schema,
                    }

            try:
                result = client.chat.completions.create(**create_kwargs)
            except Exception:
                # Fallback without response_format if server doesn't support it
                if "response_format" in create_kwargs:
                    create_kwargs.pop("response_format", None)
                    result = client.chat.completions.create(**create_kwargs)
                else:
                    raise
            # Prefer returning parsed JSON if model produced it
            choice0 = result.choices[0]
            msg = getattr(choice0, "message", None)
            content: Any = None
            if msg is not None:
                content = getattr(msg, "content", None)
            if content is None:
                content = getattr(choice0, "text", None)

            text_out = "" if content is None else str(content).strip()
            # If structured mode was requested, do a single best-effort JSON parse; no extra heuristics
            if response_format_json:
                try:
                    return _json.loads(text_out)
                except Exception:
                    return {"raw": text_out}
            # Non-structured mode: return raw text
            return {"raw": text_out}
        except Exception as e:  # keep broad with limited retries
            last_err = e
            if attempt < retries:
                time.sleep(delay * (2 ** attempt))
            else:
                raise


def call_llm_for_tool(tool_name: str, text: str) -> dict:
    prompts = _load_prompts()
    prompt_text = prompts[tool_name]
    system_promt = str(prompt_text) if not isinstance(prompt_text, str) else prompt_text
    response_format_json = False
    json_schema: Optional[Dict[str, Any]] = None

    if tool_name in ("clipper", "analyzer", "make_clips"):
        response_format_json = True
        json_schema = {
            "name": "segments_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "segments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "number"},
                                "end": {"type": "number"},
                                "title": {"type": "string"},
                                "reason": {"type": "string"},
                                "keywords": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["start", "end"],
                            "additionalProperties": True,
                        },
                        "minItems": 1,
                    }
                },
                "required": ["segments"],
                "additionalProperties": False,
            },
            "strict": False,
        }

        # Strengthen the instruction to return JSON only
        prompt_text = (
            system_promt
            + "\n\nВыводи только один JSON-объект строго по схеме: {segments: [{start:number, end:number, title?:string, reason?:string, keywords?:string[]}]}."
        )

    return call_llm(
        prompt_text=prompt_text,
        text=text,
        response_format_json=response_format_json,
        json_schema=json_schema,
    )
