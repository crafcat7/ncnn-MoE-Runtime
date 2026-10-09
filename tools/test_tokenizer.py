#!/usr/bin/env python3
"""Compare the private native tokenizer driver with each model's pinned oracle.

Test-only dependencies (installed in an isolated environment):
    python -m pip install openai-harmony==0.0.8 transformers==5.17.0 tokenizers==0.23.2 jinja2==3.1.6

This script reads only local tokenizer assets. It never downloads model weights
or participates in the production package/runtime dependency graph.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import subprocess
import sys
from collections.abc import Mapping
from numbers import Integral
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        value = tolist()
    while isinstance(value, (list, tuple)) and len(value) == 1:
        first = value[0]
        first_tolist = getattr(first, "tolist", None)
        if callable(first_tolist):
            first = first_tolist()
        if not isinstance(first, (list, tuple)):
            break
        value = first
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("the Qwen chat template did not return token IDs")
    if all(type(token) is int for token in value):
        return list(value)
    ids: list[int] = []
    for token in value:
        item = getattr(token, "item", None)
        if callable(item):
            token = item()
        if isinstance(token, bool) or not isinstance(token, Integral):
            raise ValueError("the Qwen chat template did not return token IDs")
        ids.append(int(token))
    return ids


def identify_family(model_dir: Path) -> tuple[str, str]:
    model_config = read_json(model_dir / "config.json")
    model_type = model_config.get("model_type")
    families = {
        "gpt_oss": "gpt_oss",
        "deepseek_v4": "deepseek_v4",
        "qwen3_5_moe": "qwen",
        "qwen4_exp": "qwen",
    }
    if model_type not in families:
        raise RuntimeError(f"unsupported model_type in {model_dir / 'config.json'}: {model_type!r}")
    return families[model_type], str(model_type)


def gpt_harmony():
    try:
        import openai_harmony
    except ImportError as error:
        raise RuntimeError(
            "openai-harmony==0.0.8 is required for the GPT-OSS oracle; see the pip command at the top of this script"
        ) from error
    return openai_harmony.load_harmony_encoding("HarmonyGptOss"), openai_harmony


def load_deepseek_encoding(model_dir: Path):
    module_path = model_dir / "encoding" / "encoding_dsv4.py"
    if not module_path.exists():
        raise RuntimeError(f"DeepSeek official encoding file is missing: {module_path}")
    module_spec = importlib.util.spec_from_file_location("nmr_encoding_dsv4", module_path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"cannot import DeepSeek official encoding: {module_path}")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def make_oracle(model_dir: Path, family: str) -> dict[str, Any]:
    if family == "gpt_oss":
        harmony, harmony_module = gpt_harmony()
        special_ids: set[int] = set()
        for spelling in harmony.special_tokens_set:
            ids = harmony.encode(spelling, allowed_special="all")
            if len(ids) == 1:
                special_ids.add(int(ids[0]))

        def raw_encode(text: str) -> list[int]:
            return [int(value) for value in harmony.encode(text, allowed_special="all")]

        def chat_encode(messages: list[dict[str, Any]], _thinking: bool) -> list[int]:
            built = []
            for message in messages:
                role_value = message.get("role")
                if not isinstance(role_value, str):
                    raise ValueError("message role must be a string")
                role = getattr(harmony_module.Role, role_value.upper(), None)
                if role is None:
                    raise ValueError(f"unsupported Harmony role: {role_value}")
                content = message.get("content")
                if not isinstance(content, str):
                    raise ValueError("message content must be a string")
                if role_value.lower() == "system" and not content:
                    content = harmony_module.SystemContent.new()
                built.append(harmony_module.Message.from_role_and_content(role, content))
            conversation = harmony_module.Conversation.from_messages(built)
            return [
                int(value)
                for value in harmony.render_conversation_for_completion(
                    conversation, harmony_module.Role.ASSISTANT
                )
            ]

        def decode(ids: list[int]) -> str:
            return harmony.decode(ids)

        stops = [int(value) for value in harmony.stop_tokens_for_assistant_actions()]
        return {
            "raw_encode": raw_encode,
            "chat_encode": chat_encode,
            "decode": decode,
            "stops": stops,
            "special_ids": special_ids,
            "added_token_spellings": sorted(str(value) for value in harmony.special_tokens_set),
            "tokenizer": None,
            "config": {},
        }

    if family == "deepseek_v4":
        try:
            from transformers import AutoTokenizer
        except ImportError as error:
            raise RuntimeError(
                "Transformers is required for this oracle; see the pip command at the top of this script"
            ) from error
        tokenizer_config = read_json(model_dir / "tokenizer_config.json")
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
        official_encoding = load_deepseek_encoding(model_dir)

        def raw_encode(text: str) -> list[int]:
            return [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]

        def decode(ids: list[int]) -> str:
            return str(tokenizer.decode(ids, skip_special_tokens=False))

        def chat_encode(messages: list[dict[str, Any]], thinking: bool) -> list[int]:
            mode = "thinking" if thinking else "chat"
            prompt = official_encoding.encode_messages(messages, thinking_mode=mode)
            return [int(value) for value in tokenizer.encode(prompt)]

        stops = [int(tokenizer.eos_token_id)]
    else:
        try:
            from transformers import PreTrainedTokenizerFast
        except ImportError as error:
            raise RuntimeError(
                "Transformers is required for this oracle; see the pip command at the top of this script"
            ) from error
        tokenizer_config = read_json(model_dir / "tokenizer_config.json")
        tokenizer = PreTrainedTokenizerFast.from_pretrained(str(model_dir), local_files_only=True)

        def raw_encode(text: str) -> list[int]:
            return [int(value) for value in tokenizer.encode(text, add_special_tokens=False)]

        generation_path = model_dir / "generation_config.json"
        generation = read_json(generation_path) if generation_path.exists() else {}
        stops = []
        tokenizer_eos = getattr(tokenizer, "eos_token_id", None)
        if isinstance(tokenizer_eos, int):
            stops.append(tokenizer_eos)
        configured_eos = generation.get("eos_token_id")
        if isinstance(configured_eos, int):
            configured_eos = [configured_eos]
        if isinstance(configured_eos, list):
            for token in configured_eos:
                if isinstance(token, int) and token not in stops:
                    stops.append(token)

        def decode(ids: list[int]) -> str:
            text = str(tokenizer.decode(ids, skip_special_tokens=True))
            if ids and ids[-1] in stops:
                text = text.rstrip().rstrip("\ufffd").rstrip()
            return text

        def chat_encode(messages: list[dict[str, Any]], thinking: bool) -> list[int]:
            try:
                values = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=thinking,
                )
            except TypeError:
                values = tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True
                )
            return normalize_token_ids(values)
    special_ids = {int(value) for value in tokenizer.all_special_ids}
    return {
        "raw_encode": raw_encode,
        "chat_encode": chat_encode,
        "decode": decode,
        "stops": stops,
        "special_ids": special_ids,
        "added_token_spellings": sorted(str(value) for value in tokenizer.get_added_vocab()),
        "tokenizer": tokenizer,
        "config": tokenizer_config,
    }


def byte_encoder() -> dict[int, str]:
    byte_values = list(range(ord("!"), ord("~") + 1))
    byte_values += list(range(ord("¡"), ord("¬") + 1))
    byte_values += list(range(ord("®"), ord("ÿ") + 1))
    codepoints = list(byte_values)
    extra = 0
    for value in range(256):
        if value not in byte_values:
            byte_values.append(value)
            codepoints.append(256 + extra)
            extra += 1
    return {value: chr(codepoint) for value, codepoint in zip(byte_values, codepoints)}


def byte_token_ids(model_dir: Path) -> dict[int, int]:
    tokenizer_json = read_json(model_dir / "tokenizer.json")
    vocab = tokenizer_json.get("model", {}).get("vocab")
    if not isinstance(vocab, dict):
        raise RuntimeError(f"tokenizer.json has no BPE vocabulary: {model_dir}")
    encoded = byte_encoder()
    missing = [value for value, spelling in encoded.items() if spelling not in vocab]
    if missing:
        raise RuntimeError(f"tokenizer is missing byte-level fallback tokens: {missing[:8]}")
    return {value: int(vocab[spelling]) for value, spelling in encoded.items()}


def make_text_cases() -> list[tuple[str, str]]:
    texts = [
        ("empty", ""),
        ("ascii", "Hello, world! I'm testing the tokenizer."),
        ("ascii-punctuation", "foo_bar += a->b; ...?! ——"),
        ("chinese", "我们期望简化架构，提高单会话性能。"),
        ("cjk-english", "CPU x86/ARM，中文English混排！"),
        ("emoji", "中文😀 👩🏽‍💻 🇨🇳 ❤️ end"),
        ("code", "int main() {\n\treturn 1024;\n}\n"),
        ("whitespace", "  leading\tspaces   trailing  \n\n"),
        ("crlf", "A\r\nB\rC\n\nD"),
        ("numbers", "0 1234567890 1.234 -2e10"),
        ("three-digit-boundaries", "99 100 101 999 1,000 10_000"),
        ("unicode-numbers", "０１２３ ١٢٣ १२३"),
        ("contractions", "We're I'M you've he'd it'll isn't CAN'T"),
        ("nfc-accent", "café"),
        ("nfd-accent", "cafe\u0301"),
        ("nfd-multiple", "A\u030a e\u0301 n\u0303 o\u0308"),
        ("nfc-hangul", "한글"),
        ("nfd-hangul", "\u1112\u1161\u11ab\u1100\u1173\u11af"),
        ("greek", "Καλημέρα κόσμε"),
        ("cyrillic", "Привет мир"),
        ("arabic", "مرحبًا بالعالم"),
        ("hebrew", "שָׁלוֹם עולם"),
        ("hindi", "नमस्ते दुनिया"),
        ("tamil", "வணக்கம் உலகம்"),
        ("thai", "สวัสดีชาวโลก"),
        ("japanese", "こんにちは、世界！ カタカナ"),
        ("korean", "안녕하세요 세계"),
        ("cjk-extension", "𠀀𪚥 test"),
        ("unicode-space", "a\u00a0b\u2003c\u202fd\u3000e"),
        ("unicode-line-space", "a\u0085b\u2028c\u2029d\u000be\u000cf"),
        ("combining", "x\u0301\u0327 a\u20dd"),
        ("nfc-reorder", "a\u0315\u0300 b\u0301\u0327"),
        ("nfc-blocking", "A\u0305\u030a A\u030a\u0301"),
        ("nfc-exclusions", "\u0344\u0f73\u212b\u0340\u0341"),
        ("nfc-leading-marks", "\u0315\u0300\u0327A"),
        ("nfc-long-marks", "a" + "\u0315\u0301" * 40),
        ("nfc-new-mark-barrier", "x\U0001e4ec\u0301\u0315 y"),
        ("case-fold-contractions", "'ſ 'S 's 'Ll 'rE I'M WE'VE"),
        ("letter-case-boundaries", "AAA中文BBB Aa中AaA ǅǆαΩÉéK"),
        ("mark-prefix-boundaries", "\u0301A A\u0301AAA !\u0301a /\u0301A"),
        ("whitespace-backtracking", "  A  \r \n\tB\t\tC \r\n \t"),
        ("deepseek-stage-boundaries", "1234中abcあカABC456!xyz/ABC7890"),
        ("special-im-start", "<|im_start|>user\nhello<|im_end|>\n"),
        ("special-thinking", "<think>\n推理😀\n</think>\n\n答案"),
        ("special-overlap", "a<|im_start|><|im_end|><|endoftext|>b"),
        ("harmony-frame", "<|start|>assistant<|channel|>final<|message|>Hello<|return|>"),
        ("harmony-reserved", "<|reserved_200018|><|reserved_200020|><|endofprompt|>"),
        ("recent-unicode-letters", "A\u1c89\u1c8a\U00010d40\U0001e5d0Z"),
        ("recent-unicode-marks", "a\U0001e5ee\U0001e5ef\U00011db0z"),
        ("nul-control", "a\u0000b\u0001c"),
        ("mixed", "CPU x86/ARM，αβ 123\n\t😀 café cafe\u0301"),
        ("repetition", "naïve 😀 中文 " * 3),
        ("long-bpe-chunks", "a" * 300 + "!?" * 150 + "中文" * 150),
        ("tabs-and-spaces", "\t  A   B\t\tC  "),
        ("number-repeat", "123 456 789 1234 5678 9012"),
        ("punctuation-cjk", "中文；English，punctuation？！【test】"),
    ]
    components = ["中文", "English", "123", "😀", "cafe\u0301", "\t", "—"]
    for index in range(12):
        left = components[index % len(components)]
        middle = components[(index * 3 + 1) % len(components)]
        right = components[(index * 5 + 2) % len(components)]
        texts.append((f"composition-{index:02d}", f" {left}{middle}  {right}\n{index * 101 + 99}"))
    return texts


def make_chat_cases() -> list[tuple[str, list[dict[str, Any]], bool]]:
    conversations = [
        ("empty", []),
        ("user-hello", [{"role": "user", "content": "Hello"}]),
        ("role-case", [{"role": "USER", "content": "Hello"}]),
        ("user-unicode", [{"role": "user", "content": "中文😀 cafe\u0301"}]),
        ("user-empty", [{"role": "user", "content": ""}]),
        ("system-user", [{"role": "system", "content": "Be concise."}, {"role": "user", "content": "Hello"}]),
        ("system-empty-user", [{"role": "system", "content": ""}, {"role": "user", "content": "Hello"}]),
        ("trim-whitespace", [{"role": "system", "content": "  Be concise.\n"}, {"role": "user", "content": "\t Hello \n"}]),
        ("multi-turn", [{"role": "user", "content": "第一轮"}, {"role": "assistant", "content": "答复"}, {"role": "user", "content": "第二轮"}]),
        ("assistant-final", [{"role": "user", "content": "1+1?"}, {"role": "assistant", "content": "2"}]),
        ("assistant-history", [{"role": "user", "content": "Reason?"}, {"role": "assistant", "content": "I reasoned."}, {"role": "user", "content": "Answer?"}]),
        ("assistant-empty", [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": ""}, {"role": "user", "content": "Again"}]),
        ("unicode-trim", [{"role": "user", "content": "\u3000中文\u00a0"}]),
        ("unicode-control-trim", [{"role": "user", "content": "\u001c\u001d\u0085\u2003中文\u001e\u001f\u3000"}]),
        ("embedded-tags", [{"role": "user", "content": "Explain <think> and </think>."}]),
        ("literal-special-body", [{"role": "user", "content": "<|start|> <|channel|> <|reserved_200018|> <|reserved_200020|> <|reserved_201088|> <|return|>"}]),
        ("developer-user", [{"role": "developer", "content": "Follow the schema."}, {"role": "user", "content": "Proceed."}]),
        ("developer-latest", [{"role": "user", "content": "Question"}, {"role": "developer", "content": "Be concise"}]),
        ("consecutive-tools", [{"role": "user", "content": "Question"}, {"role": "tool", "content": "one"}, {"role": "tool", "content": "two"}]),
        ("assistant-user", [{"role": "assistant", "content": "Earlier answer."}, {"role": "user", "content": "Next question."}]),
        ("system-assistant-user", [{"role": "system", "content": "System."}, {"role": "assistant", "content": "Prior."}, {"role": "user", "content": "Now?"}]),
        ("system-late", [{"role": "user", "content": "Hello"}, {"role": "system", "content": "Late."}]),
        ("unknown-role", [{"role": "other", "content": "Invalid."}]),
        ("tool-without-name", [{"role": "tool", "content": "result"}]),
        ("two-users", [{"role": "user", "content": "First"}, {"role": "user", "content": "Second"}]),
        ("assistant-thinking-markers", [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "<think>work</think>\nanswer"}, {"role": "user", "content": "Follow-up"}]),
        ("repeated-chat-fragments", [{"role": role, "content": "cafe\u0301 café 中文 😀 café " * 4} for role in ("user", "assistant", "user", "assistant", "user")]),
        ("assistant-thinking-trim", [{"role": "user", "content": "Question"}, {"role": "assistant", "content": "\u3000prefix<think>\n\n\u2003work\u00a0\n</think>ignored</think>\n\nanswer\u3000"}]),
        ("padded-tool-response", [{"role": "user", "content": "Question"}, {"role": "user", "content": "\u3000<tool_response>result</tool_response>\u00a0"}]),
    ]
    return [(name, messages, thinking) for name, messages in conversations for thinking in (False, True)]


def json_line(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def captured(call: Callable[[], list[int]]) -> dict[str, Any]:
    try:
        return {"ok": True, "ids": [int(value) for value in call()]}
    except Exception as error:  # Oracle rejection is part of the tested contract.
        return {"ok": False, "error": f"{type(error).__name__}: {error}"}


def add_case(
    cases: list[dict[str, Any]],
    name: str,
    group: str,
    request_line: str,
    expected: dict[str, Any],
) -> None:
    cases.append({"name": name, "group": group, "request": request_line, "expected": expected})


def make_cases(model_dir: Path, family: str, oracle: dict[str, Any]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    texts = make_text_cases()
    added_spellings = sorted(set(oracle.get("added_token_spellings", [])))
    if added_spellings:
        ordinary = " café😀 中文/ "
        mixed_text = "x" + ordinary.join(added_spellings) + ordinary + "end"
        adjacent_text = "".join(reversed(added_spellings))
        texts.extend(
            (
                ("all-added-tokens-mixed-utf8", mixed_text),
                ("all-added-tokens-reversed-adjacent", adjacent_text),
            )
        )

        miss_separator = "\u241e"
        while any(miss_separator in spelling for spelling in added_spellings):
            miss_separator += "\u241f"
        near_miss_spellings = []
        replacement_chars = ("🧩", "x", "q", "~", "\ue000")
        for spelling in added_spellings:
            if not spelling:
                continue
            for replacement in replacement_chars:
                if replacement == spelling[-1]:
                    continue
                candidate = spelling[:-1] + replacement
                if not any(token in candidate for token in added_spellings):
                    near_miss_spellings.append(candidate)
                    break
        if near_miss_spellings:
            near_miss_text = miss_separator.join(near_miss_spellings) + ordinary
            texts.append(("added-token-final-char-near-misses", near_miss_text))

    for name, text in texts:
        expected = captured(lambda value=text: oracle["raw_encode"](value))
        if expected["ok"]:
            expected["decoded"] = oracle["decode"](expected["ids"])
        add_case(cases, f"raw-{name}", "raw", json_line({"op": "encode", "text": text}), expected)

    for name, messages, thinking in make_chat_cases():
        expected = captured(lambda value=messages, mode=thinking: oracle["chat_encode"](value, mode))
        if expected["ok"]:
            expected["decoded"] = oracle["decode"](expected["ids"])
        request = {"op": "chat", "messages": messages, "enable_thinking": thinking}
        add_case(cases, f"chat-{name}-thinking-{str(thinking).lower()}", "chat", json_line(request), expected)

    byte_ids = byte_token_ids(model_dir)
    emoji_bytes = "😀".encode("utf-8")
    emoji_ids = [byte_ids[value] for value in emoji_bytes]
    complete_ids = emoji_ids + [byte_ids[ord("A")]]
    decoded = oracle["decode"](complete_ids)
    add_case(
        cases,
        "decode-byte-stream-emoji-ascii",
        "decode",
        json_line({"op": "decode", "ids": complete_ids}),
        {"ok": True, "decoded": decoded, "pending_bytes": [], "final": True},
    )
    if family == "gpt_oss":
        for name, token_id in (("channel", 200005), ("reserved-200018", 200018)):
            token_ids = [token_id]
            add_case(
                cases,
                f"harmony-special-{name}-id-{token_id}",
                "decode",
                json_line({"op": "decode", "ids": token_ids}),
                {
                    "ok": True,
                    "decoded": oracle["decode"](token_ids),
                    "pending_bytes": [],
                    "final": True,
                },
            )
    for prefix_length in (1, 2, 3):
        partial_ids = emoji_ids[:prefix_length]
        partial_bytes = list(emoji_bytes[:prefix_length])
        add_case(
            cases,
            f"utf8-pending-prefix-{prefix_length}",
            "utf8_pending",
            json_line({"op": "decode", "ids": partial_ids, "final": False}),
            {"ok": True, "decoded": "", "pending_bytes": partial_bytes, "final": False},
        )
        add_case(
            cases,
            f"utf8-final-truncated-{prefix_length}",
            "utf8_final",
            json_line({"op": "decode", "ids": partial_ids, "final": True}),
            {
                "ok": True,
                "decoded": oracle["decode"](partial_ids),
                "pending_bytes": [],
                "final": True,
            },
        )

    stops_expected = sorted(set(oracle["stops"]))
    add_case(cases, "stop-token-set", "stops", json_line({"op": "stops"}), {"ok": True, "ids": stops_expected})
    prefix_ids = oracle["raw_encode"]("before ")
    suffix_ids = oracle["raw_encode"](" after")
    if not prefix_ids or not suffix_ids or not stops_expected:
        raise RuntimeError(f"cannot construct stop-frame case for {model_dir}")
    frame_ids = prefix_ids + [stops_expected[0]] + suffix_ids
    add_case(
        cases,
        "stop-frame-truncates-after-first-stop",
        "stop_frame",
        json_line({"op": "decode", "ids": frame_ids, "stop_frame": True}),
        {
            "ok": True,
            "decoded": oracle["decode"](prefix_ids),
            "pending_bytes": [],
            "stopped": True,
            "consumed": len(prefix_ids),
        },
    )

    invalid_requests = [
        ("malformed-json-object", "{"),
        ("malformed-chat-array", '{"op":"chat","messages":['),
        ("missing-text", json_line({"op": "encode"})),
        ("non-string-text", json_line({"op": "encode", "text": 42})),
        ("missing-chat-messages", json_line({"op": "chat"})),
        ("non-array-chat-messages", json_line({"op": "chat", "messages": "hello"})),
        ("message-missing-role", json_line({"op": "chat", "messages": [{"content": "x"}]})),
        ("message-missing-content", json_line({"op": "chat", "messages": [{"role": "user"}]})),
        ("message-non-string-content", json_line({"op": "chat", "messages": [{"role": "user", "content": None}]})),
        ("message-extra-field", json_line({"op": "chat", "messages": [{"role": "user", "content": "x", "name": "extra"}]})),
        ("chat-unsupported-role", json_line({"op": "chat", "messages": [{"role": "other", "content": "x"}]})),
        ("decode-non-integer-id", json_line({"op": "decode", "ids": [1, "two"]})),
    ]
    for name, request in invalid_requests:
        add_case(cases, f"validation-{name}", "validation", request, {"ok": False})

    return cases


def evaluate(case: dict[str, Any], actual: dict[str, Any]) -> tuple[bool, str | None]:
    expected = case["expected"]
    if not isinstance(actual, dict):
        return False, "driver response is not a JSON object"
    if case["group"] in ("raw", "chat"):
        if not expected["ok"]:
            return (not actual.get("ok"), None if not actual.get("ok") else "expected official rejection")
        if actual.get("ok") is not True:
            return False, f"driver rejected an accepted oracle case: {actual.get('error')}"
        if actual.get("ids") != expected["ids"]:
            return False, "token IDs differ"
        if actual.get("decoded") != expected["decoded"]:
            return False, "decoded text differs"
        return True, None

    if case["group"] in ("decode", "utf8_pending", "utf8_final"):
        if actual.get("ok") is not True:
            return False, f"driver decode failed: {actual.get('error')}"
        for key in ("decoded", "pending_bytes", "final"):
            if actual.get(key) != expected.get(key):
                return False, f"decode field {key} differs"
        if "pieces" in actual and "flush" in actual:
            if "".join(actual["pieces"]) + actual["flush"] != actual.get("decoded"):
                return False, "incremental pieces do not concatenate to decoded text"
        return True, None

    if case["group"] == "stops":
        if actual.get("ok") is not True:
            return False, f"stop query failed: {actual.get('error')}"
        if sorted(set(actual.get("ids", []))) != expected["ids"]:
            return False, f"stop IDs differ: expected {expected['ids']}, got {actual.get('ids')}"
        return True, None

    if case["group"] == "stop_frame":
        if actual.get("ok") is not True:
            return False, f"stop-frame decode failed: {actual.get('error')}"
        for key in ("decoded", "pending_bytes", "stopped", "consumed"):
            if actual.get(key) != expected.get(key):
                return False, f"stop-frame field {key} differs"
        return True, None

    if case["group"] == "validation":
        if actual.get("ok") is not False:
            return False, "invalid request was accepted"
        return True, None

    return False, f"unknown test group: {case['group']}"


def version_or_none(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("driver", type=Path, help="path to ncnn_moe_tokenizer_test")
    parser.add_argument("model_directory", type=Path, help="local tokenizer asset directory")
    parser.add_argument("--report", type=Path, help="override ignored JSON report path")
    args = parser.parse_args()

    driver = args.driver.resolve()
    model_dir = args.model_directory.resolve()
    family, model_type = identify_family(model_dir)
    oracle = make_oracle(model_dir, family)
    cases = make_cases(model_dir, family, oracle)
    payload = "".join(case["request"] + "\n" for case in cases)
    completed = subprocess.run(
        [str(driver), str(model_dir)],
        input=payload,
        text=True,
        encoding="utf-8",
        errors="strict",
        capture_output=True,
        timeout=120,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"native driver exited {completed.returncode}: {completed.stderr.strip()}"
        )
    output_lines = completed.stdout.split("\n")
    if output_lines[-1] == "":
        output_lines.pop()
    if len(output_lines) != len(cases):
        raise RuntimeError(
            f"driver returned {len(output_lines)} JSONL records for {len(cases)} requests; "
            f"stderr={completed.stderr.strip()!r}"
        )

    summaries: list[dict[str, Any]] = []
    group_counts: dict[str, dict[str, Any]] = {}
    for case, line in zip(cases, output_lines):
        try:
            actual = json.loads(line)
        except json.JSONDecodeError as error:
            actual = {"ok": False, "error": f"invalid driver JSON: {error}", "raw": line}
        passed, failure = evaluate(case, actual)
        counts = group_counts.setdefault(case["group"], {"cases": 0, "passed": 0, "failed": []})
        counts["cases"] += 1
        if passed:
            counts["passed"] += 1
        else:
            counts["failed"].append(case["name"])
        item: dict[str, Any] = {"name": case["name"], "group": case["group"], "passed": passed}
        if failure:
            item["error"] = failure
            item["request"] = case["request"]
            item["expected"] = case["expected"]
            item["actual"] = actual
        summaries.append(item)

    passed_all = all(item["passed"] for item in summaries)
    report_path = args.report or ROOT / "build-review-prompt" / "tokenizer-parity" / f"{family}-{model_dir.name}.json"
    report_path = report_path.resolve()
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "family": family,
        "model_type": model_type,
        "model_directory": str(model_dir),
        "driver": str(driver),
        "versions": {
            "transformers": version_or_none("transformers"),
            "tokenizers": version_or_none("tokenizers"),
            "jinja2": version_or_none("jinja2"),
            "openai-harmony": version_or_none("openai-harmony"),
        },
        "groups": group_counts,
        "cases": summaries,
        "passed": passed_all,
        "scope": "Local tokenizer and chat-template parity only; no model weights, inference, download, or performance measurement.",
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"family": family, "groups": group_counts, "report": str(report_path), "passed": passed_all}, ensure_ascii=False, indent=2))
    return 0 if passed_all else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(f"test_tokenizer.py: {type(error).__name__}: {error}", file=sys.stderr)
        sys.exit(2)
