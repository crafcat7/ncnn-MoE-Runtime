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
    return parser.parse_args()


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
    ]
    for token in stop_tokens:
        command.extend(["--stop-token", str(token)])

    completed = subprocess.run(command, text=True, capture_output=True)
    sys.stdout.write(completed.stdout)
    sys.stderr.write(completed.stderr)
    if completed.returncode != 0:
        return completed.returncode

    match = re.search(r"^generated token ids:(.*)$", completed.stdout, re.MULTILINE)
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
