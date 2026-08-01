#!/usr/bin/env python3

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def configure_standard_streams():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=(
            "Run a text prompt through ncnn_moe using the official "
            "DeepSeek V4 message encoding and tokenizer."
        )
    )
    parser.add_argument("runner", help="Path to ncnn_moe_deepseek_v4")
    parser.add_argument(
        "model",
        help="Path to DeepSeek-V4-Flash or DeepSeek-V4-Flash-DSpark",
    )
    parser.add_argument("prompt", help="User prompt")
    parser.add_argument("--system", default="", help="Optional system message")
    parser.add_argument(
        "--thinking-mode",
        choices=("chat", "thinking"),
        default="thinking",
        help="Generate reasoning plus an answer, or a direct chat answer.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Decode and flush generated text as the native runner emits tokens.",
    )
    parser.add_argument(
        "--stream-final-only",
        action="store_true",
        help="With --stream, suppress reasoning text and print the final answer only.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print encoded prompt IDs and native runner diagnostics.",
    )
    parser.add_argument(
        "--report-throughput",
        action="store_true",
        help="Print the optional aggregate generation rate in token/s.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum reply length; increase it when a long answer reaches the limit.",
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-speculative", action="store_true")
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "hybrid", "hybrid-prefetch"),
        default="auto",
    )
    parser.add_argument("--vulkan-device", type=int)
    parser.add_argument(
        "--expert-memory",
        choices=("auto", "eager", "on-demand"),
        default="on-demand",
    )
    parser.add_argument("--host-memory-mb", type=int, default=0)
    parser.add_argument("--expert-cache-mb", type=int, default=0)
    parser.add_argument(
        "--release-vulkan-dense-host",
        action="store_true",
        help="Drop host copies of dense tensors after Vulkan operators take ownership.",
    )
    parser.add_argument("--expert-gpu-cache-mb", type=int, default=0)
    parser.add_argument("--expert-gpu-victim-cache-mb", type=int, default=0)
    parser.add_argument("--expert-io-workers", type=int, default=4)
    io_group = parser.add_mutually_exclusive_group()
    io_group.add_argument("--mmap-experts", action="store_true")
    io_group.add_argument("--direct-expert-io", action="store_true")
    io_group.add_argument("--buffered-expert-io", action="store_true")
    return parser.parse_args()


def validate_arguments(arguments):
    runner = Path(arguments.runner)
    model = Path(arguments.model)
    if not runner.is_file():
        raise ValueError(f"runner does not exist: {runner}")
    if not model.is_dir():
        raise ValueError(f"model directory does not exist: {model}")
    if not (model / "tokenizer.json").is_file():
        raise ValueError(f"tokenizer.json is missing from: {model}")
    if not (model / "encoding" / "encoding_dsv4.py").is_file():
        raise ValueError(f"encoding/encoding_dsv4.py is missing from: {model}")
    if arguments.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero")
    if arguments.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if arguments.vulkan_device is not None and arguments.vulkan_device < 0:
        raise ValueError("--vulkan-device must be non-negative")
    for option, value in (
        ("--host-memory-mb", arguments.host_memory_mb),
        ("--expert-cache-mb", arguments.expert_cache_mb),
        ("--expert-gpu-cache-mb", arguments.expert_gpu_cache_mb),
        (
            "--expert-gpu-victim-cache-mb",
            arguments.expert_gpu_victim_cache_mb,
        ),
        ("--expert-io-workers", arguments.expert_io_workers),
    ):
        if value < 0:
            raise ValueError(f"{option} must be non-negative")
    if arguments.expert_io_workers > 64:
        raise ValueError("--expert-io-workers must be between 0 and 64")
    return runner.resolve(), model.resolve()


def load_text_components(model):
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        from transformers import AutoTokenizer
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "transformers is required; install it with "
            "'python -m pip install -U transformers'"
        ) from error

    sys.path.insert(0, str(model / "encoding"))
    try:
        from encoding_dsv4 import (
            encode_messages,
            parse_message_from_completion_text,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError("cannot import the model's encoding_dsv4.py") from error

    tokenizer = AutoTokenizer.from_pretrained(
        str(model),
        local_files_only=True,
    )
    if tokenizer.eos_token_id != 1:
        raise RuntimeError(
            f"expected DeepSeek V4 EOS token 1, got {tokenizer.eos_token_id}"
        )
    return tokenizer, encode_messages, parse_message_from_completion_text


def build_runner_command(arguments, runner, model, prompt_file):
    command = [
        str(runner),
        str(model),
        "--prompt-token-file",
        str(prompt_file),
        "--max-new-tokens",
        str(arguments.max_new_tokens),
        "--temperature",
        str(arguments.temperature),
        "--top-k",
        str(arguments.top_k),
        "--top-p",
        str(arguments.top_p),
        "--min-p",
        str(arguments.min_p),
        "--seed",
        str(arguments.seed),
        "--expert-memory",
        arguments.expert_memory,
        "--expert-io-workers",
        str(arguments.expert_io_workers),
    ]
    if arguments.no_speculative:
        command.append("--no-speculative")
    if arguments.backend != "auto":
        command.append(f"--{arguments.backend}")
    if arguments.vulkan_device is not None:
        command.extend(["--vulkan-device", str(arguments.vulkan_device)])
    if arguments.host_memory_mb:
        command.extend(["--host-memory-mb", str(arguments.host_memory_mb)])
    if arguments.expert_cache_mb:
        command.extend(["--expert-cache-mb", str(arguments.expert_cache_mb)])
    if arguments.release_vulkan_dense_host:
        command.append("--release-vulkan-dense-host")
    if arguments.expert_gpu_cache_mb:
        command.extend(
            ["--expert-gpu-cache-mb", str(arguments.expert_gpu_cache_mb)]
        )
    if arguments.expert_gpu_victim_cache_mb:
        command.extend(
            [
                "--expert-gpu-victim-cache-mb",
                str(arguments.expert_gpu_victim_cache_mb),
            ]
        )
    if arguments.mmap_experts:
        command.append("--mmap-experts")
    elif arguments.direct_expert_io:
        command.append("--direct-expert-io")
    elif arguments.buffered_expert_io:
        command.append("--buffered-expert-io")
    if arguments.stream:
        command.append("--stream-token-ids")
    if arguments.report_throughput:
        command.append("--report-throughput")
    return command


def parse_generated_tokens(stdout):
    match = re.search(r"^generated token ids:(.*)$", stdout, re.MULTILINE)
    if not match:
        raise RuntimeError("runner did not report generated token IDs")
    return [int(token) for token in match.group(1).split()]


def print_reported_throughput(output):
    match = re.search(
        r"^(?:Parallel sessions: \d+, aggregate throughput: "
        r"|Aggregate throughput: )([0-9.eE+-]+) token/s$",
        output,
        re.MULTILINE,
    )
    if match:
        print(
            f"Aggregate throughput: {match.group(1)} token/s",
            file=sys.stderr,
        )


def strip_eos(text, tokenizer):
    if tokenizer.eos_token and text.endswith(tokenizer.eos_token):
        return text[: -len(tokenizer.eos_token)]
    return text


def clean_partial_completion(text, tokenizer):
    if text.startswith("</think>"):
        text = text[len("</think>") :]
    return strip_eos(text, tokenizer)


def withhold_partial_marker(text, marker):
    maximum = min(len(text), len(marker) - 1)
    for length in range(maximum, 0, -1):
        if text.endswith(marker[:length]):
            return text[:-length]
    return text


def visible_stream_text(text, arguments, tokenizer):
    text = strip_eos(text, tokenizer)
    if arguments.thinking_mode == "chat":
        return "[answer]\n" + clean_partial_completion(text, tokenizer)

    marker = "</think>"
    if not arguments.stream_final_only:
        if marker in text:
            reasoning, answer = text.split(marker, 1)
            return f"[reasoning]\n{reasoning}\n[answer]\n{answer}"
        return "[reasoning]\n" + withhold_partial_marker(text, marker)

    if marker not in text:
        return ""
    return "[answer]\n" + text.split(marker, 1)[1]


def fallback_deepseek_message(completion_text, arguments, tokenizer):
    text = strip_eos(completion_text, tokenizer)
    if arguments.thinking_mode == "chat":
        return {
            "role": "assistant",
            "reasoning_content": "",
            "content": clean_partial_completion(text, tokenizer),
            "tool_calls": [],
        }

    marker = "</think>"
    if marker in text:
        reasoning, answer = text.split(marker, 1)
    else:
        reasoning = withhold_partial_marker(text, marker)
        answer = ""
    return {
        "role": "assistant",
        "reasoning_content": reasoning,
        "content": answer,
        "tool_calls": [],
    }


def print_deepseek_output(message, final_only=False):
    reasoning = message.get("reasoning_content", "")
    answer = message.get("content", "")
    if reasoning and not final_only:
        print("[reasoning]")
        print(reasoning)
    if answer:
        print("[answer]")
        print(answer)
    return bool((reasoning and not final_only) or answer)


class TextStreamer:
    def __init__(self, tokenizer, arguments):
        self.tokenizer = tokenizer
        self.arguments = arguments
        self.tokens = []
        self.emitted = ""

    def push(self, token):
        self.tokens.append(token)
        decoded = self.tokenizer.decode(
            self.tokens,
            skip_special_tokens=False,
        )
        visible = visible_stream_text(
            decoded,
            self.arguments,
            self.tokenizer,
        )
        replacement = visible.find("\ufffd")
        if replacement >= 0:
            visible = visible[:replacement]
        if not visible.startswith(self.emitted):
            return
        delta = visible[len(self.emitted) :]
        if delta:
            sys.stdout.write(delta)
            sys.stdout.flush()
            self.emitted = visible

    def finish(self):
        if self.emitted:
            sys.stdout.write("\n")
            sys.stdout.flush()


def run_streaming(command, tokenizer, arguments):
    process = subprocess.Popen(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    streamer = TextStreamer(tokenizer, arguments)
    diagnostics = []
    token_expression = re.compile(r"^generated token id: (-?\d+)$")
    for line in process.stdout:
        match = token_expression.match(line.rstrip("\n"))
        if match:
            streamer.push(int(match.group(1)))
        else:
            diagnostics.append(line)
            if arguments.verbose:
                sys.stderr.write(line)
                sys.stderr.flush()

    return_code = process.wait()
    streamer.finish()
    if return_code == 0 and arguments.report_throughput and not arguments.verbose:
        print_reported_throughput("".join(diagnostics))
    if return_code != 0:
        sys.stderr.write("".join(diagnostics))
    return return_code, streamer.tokens, streamer


def main():
    configure_standard_streams()
    arguments = parse_arguments()
    try:
        runner, model = validate_arguments(arguments)
        tokenizer, encode_messages, parse_message = load_text_components(model)
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    messages = []
    if arguments.system:
        messages.append({"role": "system", "content": arguments.system})
    messages.append({"role": "user", "content": arguments.prompt})
    prompt_text = encode_messages(
        messages,
        thinking_mode=arguments.thinking_mode,
    )
    prompt_tokens = tokenizer.encode(prompt_text)
    if arguments.verbose:
        print(f"prompt token IDs: {' '.join(str(token) for token in prompt_tokens)}", file=sys.stderr)

    prompt_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="ascii",
            suffix=".tokens",
            delete=False,
        ) as stream:
            stream.write(" ".join(str(token) for token in prompt_tokens))
            stream.write("\n")
            prompt_file = Path(stream.name)

        command = build_runner_command(
            arguments,
            runner,
            model,
            prompt_file,
        )
        if arguments.stream:
            return_code, completion_tokens, streamer = run_streaming(
                command,
                tokenizer,
                arguments,
            )
            completed = None
        else:
            completed = subprocess.run(
                command,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
            )
            return_code = completed.returncode
            streamer = None
    finally:
        if prompt_file is not None:
            prompt_file.unlink(missing_ok=True)

    if return_code != 0:
        if completed is not None:
            sys.stdout.write(completed.stdout)
            sys.stderr.write(completed.stderr)
        return return_code
    if completed is not None:
        if arguments.verbose:
            sys.stderr.write(completed.stdout)
            sys.stderr.write(completed.stderr)
        elif arguments.report_throughput:
            print_reported_throughput(completed.stdout)
        try:
            completion_tokens = parse_generated_tokens(completed.stdout)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            sys.stderr.write(completed.stdout)
            return 1

    completion_text = tokenizer.decode(
        completion_tokens,
        skip_special_tokens=False,
    )
    if arguments.verbose:
        print(f"completion token IDs: {completion_tokens}", file=sys.stderr)
        print(f"decoded completion: {completion_text!r}", file=sys.stderr)
    try:
        message = parse_message(
            completion_text,
            thinking_mode=arguments.thinking_mode,
        )
    except (AssertionError, ValueError):
        message = fallback_deepseek_message(
            completion_text,
            arguments,
            tokenizer,
        )

    if arguments.verbose:
        print(f"parsed message: {message!r}", file=sys.stderr)
    emitted_text = streamer is not None and bool(streamer.emitted)
    if streamer is None or not streamer.emitted:
        suppress_reasoning = arguments.stream and arguments.stream_final_only
        emitted_text = print_deepseek_output(
            message,
            final_only=suppress_reasoning,
        )
    if not emitted_text:
        print(
            "no reasoning or final answer was generated; increase --max-new-tokens.",
            file=sys.stderr,
        )
    if arguments.verbose and message.get("tool_calls"):
        print(
            json.dumps(message["tool_calls"], ensure_ascii=False),
            file=sys.stderr,
        )
    reached_token_limit = (
        len(completion_tokens) >= arguments.max_new_tokens
        and completion_tokens[-1] != tokenizer.eos_token_id
    )
    if reached_token_limit:
        print(
            (
                f"warning: reply reached --max-new-tokens "
                f"{arguments.max_new_tokens} before EOS; rerun with a larger value"
            ),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
