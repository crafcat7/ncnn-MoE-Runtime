#!/usr/bin/env python3

import argparse
import json
import re
import subprocess
import sys

from openai_harmony import (
    Conversation,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    load_harmony_encoding,
)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a text prompt through ncnn_moe using the official Harmony encoding."
    )
    parser.add_argument("runner", help="Path to ncnn_moe_gpt_oss")
    parser.add_argument("model", help="Path to the official GPT-OSS model directory")
    parser.add_argument("prompt", help="User prompt")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--backend",
        choices=("auto", "cpu", "hybrid", "hybrid-prefetch"),
        default="auto",
        help="Select the runner backend; auto uses Vulkan/CPU mix when available.",
    )
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
    if stream_state["final_only"] and stream_state["channel"] != "final":
        return ""
    return message_text[len(emitted_text) :]


def main():
    arguments = parse_arguments()
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

    command = [
        arguments.runner,
        arguments.model,
        *[str(token) for token in prompt_tokens],
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
    for token in stop_tokens:
        command.extend(["--stop-token", str(token)])
    if arguments.backend != "auto":
        command.append(f"--{arguments.backend}")

    if arguments.stream:
        command.append("--stream-token-ids")
        print("loading model and preparing prompt...", file=sys.stderr, flush=True)
        process = subprocess.Popen(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=1,
        )
        output = []
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
                        sys.stdout.write(f"[{stream_state['channel']}] ")
                        stream_state["message_header_printed"] = True
                    sys.stdout.write(text)
                    sys.stdout.flush()
                    emitted_text = True
            else:
                sys.stderr.write(line)
                sys.stderr.flush()
        completed_stdout = "".join(output)
        completed_stderr = ""
        return_code = process.wait()
        if emitted_text:
            print()
        if not emitted_text:
            print(
                "no selected Harmony text was generated; increase --max-new-tokens to allow completion.",
                file=sys.stderr,
            )
    else:
        completed = subprocess.run(command, text=True, capture_output=True)
        completed_stdout = completed.stdout
        completed_stderr = completed.stderr
        return_code = completed.returncode

    sys.stderr.write(completed_stderr)
    if return_code != 0:
        return return_code

    match = re.search(r"^generated token ids:(.*)$", completed_stdout, re.MULTILINE)
    if not match:
        print("runner did not report generated token IDs", file=sys.stderr)
        return 1
    completion_tokens = [int(token) for token in match.group(1).split()]
    messages = encoding.parse_messages_from_completion_tokens(
        completion_tokens, Role.ASSISTANT
    )
    print("decoded Harmony messages:")
    for message in messages:
        print(json.dumps(message.to_dict(), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
