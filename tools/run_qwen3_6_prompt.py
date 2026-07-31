#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping
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
            "Qwen3.6 tokenizer and chat template."
        )
    )
    parser.add_argument("runner", help="Path to ncnn_moe_qwen3_6")
    parser.add_argument("model", help="Path to Qwen3.6-35B-A3B")
    parser.add_argument("prompt", help="User prompt")
    parser.add_argument("--system", default="", help="Optional system message")
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Ask the official chat template to generate a direct answer.",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Decode and flush text as the native runner emits tokens.",
    )
    parser.add_argument(
        "--stream-final-only",
        action="store_true",
        help="With --stream, suppress reasoning text and print only the final answer.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print encoded token IDs and native runner diagnostics.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--speculative",
        action="store_true",
        help=(
            "Enable the optional Qwen MTP proposal loop. Target-only "
            "generation is the default performance profile."
        ),
    )
    parser.add_argument(
        "--speculative-max-draft",
        type=int,
        default=0,
        help="Limit MTP draft tokens per round; zero uses the model plan.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "hybrid", "hybrid-prefetch"),
        default="auto",
    )
    parser.add_argument("--vulkan-device", type=int)
    parser.add_argument(
        "--host-memory-mb",
        type=int,
        default=0,
        help="Override the host-memory budget; zero uses automatic planning.",
    )
    return parser.parse_args()


def validate_arguments(arguments):
    runner = Path(arguments.runner)
    model = Path(arguments.model)
    if not runner.is_file():
        raise ValueError(f"runner does not exist: {runner}")
    if not model.is_dir():
        raise ValueError(f"model directory does not exist: {model}")
    for required in ("config.json", "tokenizer.json", "chat_template.jinja"):
        if not (model / required).is_file():
            raise ValueError(f"{required} is missing from: {model}")
    if arguments.max_new_tokens <= 0:
        raise ValueError("--max-new-tokens must be greater than zero")
    if arguments.top_k < 0:
        raise ValueError("--top-k must be non-negative")
    if arguments.vulkan_device is not None and arguments.vulkan_device < 0:
        raise ValueError("--vulkan-device must be non-negative")
    if arguments.host_memory_mb < 0:
        raise ValueError("--host-memory-mb must be non-negative")
    if arguments.speculative_max_draft < 0:
        raise ValueError("--speculative-max-draft must be non-negative")
    return runner.resolve(), model.resolve()


def load_tokenizer(model):
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise RuntimeError(
            "transformers and jinja2 are required; install them with "
            "'python -m pip install -U transformers jinja2'"
        ) from error

    try:
        return AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    except ImportError as error:
        raise RuntimeError(
            "the official chat template requires jinja2; install it with "
            "'python -m pip install -U jinja2'"
        ) from error


def encode_prompt(tokenizer, arguments):
    messages = []
    if arguments.system:
        messages.append({"role": "system", "content": arguments.system})
    messages.append({"role": "user", "content": arguments.prompt})
    try:
        encoded = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=not arguments.no_thinking,
        )
    except ImportError as error:
        raise RuntimeError(
            "the official chat template requires jinja2; install it with "
            "'python -m pip install -U jinja2'"
        ) from error
    if isinstance(encoded, Mapping):
        encoded = encoded.get("input_ids")
    if not encoded or not all(isinstance(token, int) for token in encoded):
        raise RuntimeError("the Qwen chat template did not return token IDs")
    return encoded


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
    ]
    if arguments.speculative:
        command.append("--speculative")
    if arguments.speculative_max_draft:
        command.extend(
            [
                "--speculative-max-draft",
                str(arguments.speculative_max_draft),
            ]
        )
    if arguments.backend != "auto":
        command.append(f"--{arguments.backend}")
    if arguments.vulkan_device is not None:
        command.extend(["--vulkan-device", str(arguments.vulkan_device)])
    if arguments.host_memory_mb:
        command.extend(["--host-memory-mb", str(arguments.host_memory_mb)])
    if arguments.stream:
        command.append("--stream-token-ids")
    return command


def parse_generated_tokens(stdout):
    match = re.search(r"^generated token ids:(.*)$", stdout, re.MULTILINE)
    if not match:
        raise RuntimeError("runner did not report generated token IDs")
    return [int(token) for token in match.group(1).split()]


def strip_stop_tokens(tokens, stop_tokens):
    while tokens and tokens[-1] in stop_tokens:
        tokens.pop()
    return tokens


def withhold_partial_marker(text, marker):
    maximum = min(len(text), len(marker) - 1)
    for length in range(maximum, 0, -1):
        if text.endswith(marker[:length]):
            return text[:-length]
    return text


def split_completion(text, thinking):
    if not thinking:
        return "", text
    marker = "</think>"
    if marker in text:
        reasoning, answer = text.split(marker, 1)
        return reasoning, answer.lstrip("\r\n")
    return withhold_partial_marker(text, marker), ""


def visible_text(text, arguments):
    reasoning, answer = split_completion(text, not arguments.no_thinking)
    if arguments.no_thinking:
        return "[answer]\n" + answer
    if arguments.stream_final_only:
        return "[answer]\n" + answer if answer else ""
    if answer:
        return f"[reasoning]\n{reasoning}\n[answer]\n{answer}"
    return "[reasoning]\n" + reasoning


class TextStreamer:
    def __init__(self, tokenizer, arguments, stop_tokens):
        self.tokenizer = tokenizer
        self.arguments = arguments
        self.stop_tokens = stop_tokens
        self.tokens = []
        self.emitted = ""

    def push(self, token):
        self.tokens.append(token)
        decoded_tokens = strip_stop_tokens(self.tokens.copy(), self.stop_tokens)
        decoded = self.tokenizer.decode(
            decoded_tokens,
            skip_special_tokens=False,
        )
        visible = visible_text(decoded, self.arguments)
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


def run_streaming(command, tokenizer, arguments, stop_tokens):
    process = subprocess.Popen(
        command,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    streamer = TextStreamer(tokenizer, arguments, stop_tokens)
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
    if return_code != 0:
        sys.stderr.write("".join(diagnostics))
    return return_code, streamer.tokens, streamer


def print_completion(text, arguments):
    reasoning, answer = split_completion(text, not arguments.no_thinking)
    emitted = False
    if reasoning and not (arguments.stream and arguments.stream_final_only):
        print("[reasoning]")
        print(reasoning)
        emitted = True
    if answer:
        print("[answer]")
        print(answer)
        emitted = True
    return emitted


def main():
    configure_standard_streams()
    arguments = parse_arguments()
    try:
        runner, model = validate_arguments(arguments)
        tokenizer = load_tokenizer(model)
        prompt_tokens = encode_prompt(tokenizer, arguments)
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2

    stop_tokens = {
        token
        for token in (
            tokenizer.eos_token_id,
            tokenizer.convert_tokens_to_ids("<|endoftext|>"),
        )
        if isinstance(token, int) and token >= 0
    }
    if arguments.verbose:
        print(
            f"prompt token IDs: {' '.join(str(token) for token in prompt_tokens)}",
            file=sys.stderr,
        )

    prompt_file = None
    completed = None
    streamer = None
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
                stop_tokens,
            )
        else:
            completed = subprocess.run(
                command,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
            )
            return_code = completed.returncode
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
        try:
            completion_tokens = parse_generated_tokens(completed.stdout)
        except RuntimeError as error:
            print(error, file=sys.stderr)
            sys.stderr.write(completed.stdout)
            return 1

    completion_tokens = strip_stop_tokens(completion_tokens, stop_tokens)
    completion_text = tokenizer.decode(
        completion_tokens,
        skip_special_tokens=False,
    )
    if arguments.verbose:
        print(f"completion token IDs: {completion_tokens}", file=sys.stderr)
        print(f"decoded completion: {completion_text!r}", file=sys.stderr)
    emitted = streamer is not None and bool(streamer.emitted)
    if not emitted:
        emitted = print_completion(completion_text, arguments)
    if not emitted:
        print(
            "no final answer was generated; increase --max-new-tokens.",
            file=sys.stderr,
        )
    if len(completion_tokens) >= arguments.max_new_tokens:
        print(
            (
                f"warning: reply reached --max-new-tokens "
                f"{arguments.max_new_tokens} before a stop token"
            ),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
