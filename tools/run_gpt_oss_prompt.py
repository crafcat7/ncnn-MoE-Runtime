#!/usr/bin/env python3

import atexit
import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    load_harmony_encoding,
)


def configure_standard_streams():
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a text prompt through ncnn_moe using the official Harmony encoding."
    )
    parser.add_argument("runner", help="Path to ncnn_moe_gpt_oss")
    parser.add_argument("model", help="Path to the official GPT-OSS model directory")
    parser.add_argument("prompt", help="User prompt")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
        help="Maximum reply length; increase it when a long answer reaches the limit.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-speculative", action="store_true")
    parser.add_argument(
        "--expert-memory",
        choices=("auto", "eager", "on-demand"),
        default="auto",
        help="Select expert residency; auto switches large GPT-OSS models to file-backed experts.",
    )
    parser.add_argument(
        "--host-memory-mb",
        type=int,
        default=0,
        help="Override the host-memory budget; zero uses the runtime's detected-RAM policy.",
    )
    parser.add_argument(
        "--expert-cache-mb",
        type=int,
        default=0,
        help="Override the file-backed MXFP4 cache; zero lets auto mode size it.",
    )
    parser.add_argument(
        "--expert-io-workers",
        type=int,
        default=0,
        help="Fixed expert-storage worker count; zero chooses a hardware-derived default.",
    )
    parser.add_argument(
        "--expert-gpu-cache-mb",
        type=int,
        default=0,
        help="Optional native Vulkan executable Expert cache.",
    )
    parser.add_argument(
        "--expert-gpu-victim-cache-mb",
        type=int,
        default=0,
        help="Optional device-local compressed-weight victim cache for evicted MXFP4 Expert pairs.",
    )
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "hybrid", "hybrid-prefetch"),
        default="auto",
        help="Select the runner backend; auto uses Vulkan/CPU mix when available.",
    )
    parser.add_argument("--vulkan-device", type=int)
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Flush decoded Harmony assistant messages as the runner emits them.",
    )
    parser.add_argument(
        "--stream-final-only",
        action="store_true",
        help="With --stream, suppress analysis-channel text and print final-channel text only.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print native runner diagnostics.",
    )
    io_group = parser.add_mutually_exclusive_group()
    io_group.add_argument("--mmap-experts", action="store_true")
    io_group.add_argument("--direct-expert-io", action="store_true")
    io_group.add_argument("--buffered-expert-io", action="store_true")
    return parser.parse_args()


def decode_tokens(encoding, tokens):
    try:
        text = encoding.decode(tokens)
        return text.decode("utf-8", errors="ignore") if isinstance(text, bytes) else text
    except (AttributeError, UnicodeDecodeError, ValueError):
        return ""


def stream_harmony_token(encoding, token, stream_state):
    text = decode_tokens(encoding, [token])
    if text == "<|channel|>":
        stream_state["expect_channel"] = True
        stream_state["in_message"] = False
        return ""
    if stream_state["expect_channel"]:
        stream_state["channel"] = text
        stream_state["expect_channel"] = False
        return ""
    if text == "<|message|>":
        stream_state["in_message"] = True
        stream_state["message_header_printed"] = False
        stream_state["message_tokens"] = []
        stream_state["emitted_text"] = ""
        return ""
    if text.startswith("<|") and text.endswith("|>"):
        stream_state["in_message"] = False
        return ""
    if not stream_state["in_message"]:
        return ""

    stream_state["message_tokens"].append(token)
    message_text = decode_tokens(encoding, stream_state["message_tokens"])
    emitted_text = stream_state["emitted_text"]
    if not message_text.startswith(emitted_text):
        stream_state["emitted_text"] = message_text
        return ""
    stream_state["emitted_text"] = message_text
    if stream_state["channel"] not in ("analysis", "final"):
        return ""
    if stream_state["final_only"] and stream_state["channel"] != "final":
        return ""
    return message_text[len(emitted_text) :]


def harmony_message_text(message):
    parts = []
    for content in message.content:
        text = getattr(content, "text", None)
        if text:
            parts.append(text)
    return "".join(parts)


def collect_harmony_output(messages):
    reasoning = []
    answers = []
    for message in messages:
        text = harmony_message_text(message)
        if not text:
            continue
        if message.channel == "analysis":
            reasoning.append(text)
        elif message.channel == "final":
            answers.append(text)
    return "\n".join(reasoning), "\n".join(answers)


def print_harmony_output(messages, final_only=False):
    reasoning, answer = collect_harmony_output(messages)
    if reasoning and not final_only:
        print("[reasoning]")
        print(reasoning)
    if answer:
        print("[answer]")
        print(answer)
    return bool((reasoning and not final_only) or answer)


def main():
    configure_standard_streams()
    arguments = parse_arguments()
    if arguments.max_new_tokens <= 0:
        print("--max-new-tokens must be greater than zero", file=sys.stderr)
        return 2
    if arguments.top_k < 0:
        print("--top-k must be non-negative", file=sys.stderr)
        return 2
    if arguments.vulkan_device is not None and arguments.vulkan_device < 0:
        print("--vulkan-device must be non-negative", file=sys.stderr)
        return 2
    if arguments.host_memory_mb < 0:
        print("--host-memory-mb must be non-negative", file=sys.stderr)
        return 2
    if arguments.expert_cache_mb < 0:
        print("--expert-cache-mb must be non-negative", file=sys.stderr)
        return 2
    if arguments.expert_io_workers < 0 or arguments.expert_io_workers > 64:
        print("--expert-io-workers must be between 0 and 64", file=sys.stderr)
        return 2
    if arguments.expert_gpu_cache_mb < 0:
        print("--expert-gpu-cache-mb must be non-negative", file=sys.stderr)
        return 2
    if arguments.expert_gpu_victim_cache_mb < 0:
        print("--expert-gpu-victim-cache-mb must be non-negative", file=sys.stderr)
        return 2
    runner = Path(arguments.runner)
    model = Path(arguments.model)
    if not runner.is_file():
        print(f"runner does not exist: {runner}", file=sys.stderr)
        return 2
    if not model.is_dir():
        print(f"model directory does not exist: {model}", file=sys.stderr)
        return 2
    runner = runner.resolve()
    model = model.resolve()
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    conversation = Conversation.from_messages(
        [
            Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
            Message.from_role_and_content(Role.USER, arguments.prompt),
        ]
    )
    prompt_tokens = encoding.render_conversation_for_completion(
        conversation, Role.ASSISTANT
    )
    stop_tokens = list(encoding.stop_tokens_for_assistant_actions())
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="ascii",
        suffix=".tokens",
        delete=False,
    ) as stream:
        stream.write(" ".join(str(token) for token in prompt_tokens))
        stream.write("\n")
        prompt_file = Path(stream.name)
    atexit.register(prompt_file.unlink, missing_ok=True)

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
    if arguments.expert_memory != "auto":
        command.extend(["--expert-memory", arguments.expert_memory])
    if arguments.host_memory_mb:
        command.extend(["--host-memory-mb", str(arguments.host_memory_mb)])
    if arguments.expert_cache_mb:
        command.extend(["--expert-cache-mb", str(arguments.expert_cache_mb)])
    if arguments.expert_io_workers:
        command.extend(["--expert-io-workers", str(arguments.expert_io_workers)])
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
    if arguments.no_speculative:
        command.append("--no-speculative")
    if arguments.vulkan_device is not None:
        command.extend(["--vulkan-device", str(arguments.vulkan_device)])
    if arguments.mmap_experts:
        command.append("--mmap-experts")
    elif arguments.direct_expert_io:
        command.append("--direct-expert-io")
    elif arguments.buffered_expert_io:
        command.append("--buffered-expert-io")
    for token in stop_tokens:
        command.extend(["--stop-token", str(token)])
    if arguments.backend != "auto":
        command.append(f"--{arguments.backend}")

    if arguments.stream:
        command.append("--stream-token-ids")
        process = subprocess.Popen(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        output = []
        diagnostics = []
        stream_state = {
            "channel": "",
            "emitted_text": "",
            "expect_channel": False,
            "final_only": arguments.stream_final_only,
            "in_message": False,
            "message_header_printed": False,
            "message_tokens": [],
        }
        emitted_text = False
        for line in process.stdout:
            output.append(line)
            match = re.match(r"^generated token id: (-?\d+)$", line.rstrip("\n"))
            if match:
                text = stream_harmony_token(encoding, int(match.group(1)), stream_state)
                if text:
                    if not stream_state["message_header_printed"]:
                        if emitted_text:
                            sys.stdout.write("\n")
                        label = (
                            "reasoning"
                            if stream_state["channel"] == "analysis"
                            else "answer"
                        )
                        sys.stdout.write(f"[{label}]\n")
                        stream_state["message_header_printed"] = True
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    emitted_text = True
            else:
                diagnostics.append(line)
                if arguments.verbose:
                    sys.stderr.write(line)
                    sys.stderr.flush()
        completed_stdout = "".join(output)
        return_code = process.wait()
        if emitted_text:
            print()
    else:
        completed = subprocess.run(
            command,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
        )
        completed_stdout = completed.stdout
        diagnostics = [completed.stdout, completed.stderr]
        return_code = completed.returncode
        emitted_text = False
        if arguments.verbose:
            sys.stderr.write(completed.stdout)
            sys.stderr.write(completed.stderr)

    if return_code != 0:
        if not arguments.verbose:
            sys.stderr.write("".join(diagnostics))
        return return_code

    match = re.search(r"^generated token ids:(.*)$", completed_stdout, re.MULTILINE)
    if not match:
        print("runner did not report generated token IDs", file=sys.stderr)
        return 1
    completion_tokens = [int(token) for token in match.group(1).split()]
    if not arguments.stream or not emitted_text:
        messages = encoding.parse_messages_from_completion_tokens(
            completion_tokens,
            Role.ASSISTANT,
            strict=False,
        )
        emitted_text = print_harmony_output(
            messages,
            final_only=arguments.stream and arguments.stream_final_only,
        )
    if not emitted_text:
        print(
            "no reasoning or final answer was generated; increase --max-new-tokens.",
            file=sys.stderr,
        )
    reached_token_limit = (
        len(completion_tokens) >= arguments.max_new_tokens
        and completion_tokens[-1] not in stop_tokens
    )
    if reached_token_limit:
        print(
            (
                f"warning: reply reached --max-new-tokens "
                f"{arguments.max_new_tokens} before a Harmony stop token; "
                "rerun with a larger value"
            ),
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
