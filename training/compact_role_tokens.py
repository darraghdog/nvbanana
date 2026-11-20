#!/usr/bin/env python3
"""Restore compact ARC role IDs omitted by the r0.7 tokenizer path."""

from __future__ import annotations

from typing import Any


USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
IM_START_ID = 14
NEWLINE_ID = 10


def _as_list(token_ids: Any) -> list[int]:
    values = token_ids.tolist() if hasattr(token_ids, "tolist") else token_ids
    return [int(value) for value in values]


def _restore_type(original: Any, values: list[int]) -> Any:
    if isinstance(original, list):
        return values
    if isinstance(original, tuple):
        return tuple(values)
    if hasattr(original, "new_tensor"):
        return original.new_tensor(values)
    raise TypeError(f"Unsupported token_ids type: {type(original)!r}")


def restore_compact_role_tokens(message_log: list[dict[str, Any]]) -> dict[str, int]:
    """Restore `[14, role, 10]` headers in NeMo RL 0.7 ARC message chunks.

    NeMo-RL's incremental chat formatting places both the user header and the
    assistant generation header in each user chunk. Assistant chunks contain
    only the answer continuation and EOS. The affected NeMo RL 0.7 formatter realizes both
    headers as `[14, 10]`; NeMo RL 0.4 and the Kaggle inference path realize them as `[14, 11, 10]` and
    `[14, 12, 10]`.

    The operation is idempotent for already-roleful chunks and fails closed on
    any other shape.
    """

    inserted = 0
    already_roleful = 0
    user_messages = 0

    for message_index, message in enumerate(message_log):
        role = message["role"]
        original = message["token_ids"]
        values = _as_list(original)

        if role == "assistant":
            if USER_TOKEN_ID in values or ASSISTANT_TOKEN_ID in values:
                raise ValueError(
                    f"Assistant chunk {message_index} unexpectedly contains role IDs"
                )
            continue
        if role != "user":
            raise ValueError(f"Unsupported ARC message role: {role!r}")

        user_messages += 1
        starts = [index for index, value in enumerate(values) if value == IM_START_ID]
        if len(starts) != 2:
            raise ValueError(
                f"User chunk {message_index} expected two im_start IDs, found {starts}"
            )
        followers = [
            values[index + 1] if index + 1 < len(values) else None for index in starts
        ]
        if followers == [USER_TOKEN_ID, ASSISTANT_TOKEN_ID]:
            already_roleful += 2
            continue
        if followers != [NEWLINE_ID, NEWLINE_ID]:
            raise ValueError(
                f"User chunk {message_index} has unexpected header followers {followers}"
            )

        restored = list(values)
        for index, token_id in reversed(
            list(zip(starts, [USER_TOKEN_ID, ASSISTANT_TOKEN_ID]))
        ):
            restored.insert(index + 1, token_id)
        message["token_ids"] = _restore_type(original, restored)
        inserted += 2

    if not user_messages:
        raise ValueError("ARC conversation contains no user messages")
    return {
        "already_roleful": already_roleful,
        "inserted": inserted,
        "user_messages": user_messages,
    }


def require_compact_role_tokens(message_log: list[dict[str, Any]]) -> dict[str, int]:
    """Verify roleful ARC headers without modifying the realized token stream."""

    user_messages = 0
    verified_headers = 0
    for message_index, message in enumerate(message_log):
        role = message["role"]
        values = _as_list(message["token_ids"])
        if role == "assistant":
            if USER_TOKEN_ID in values or ASSISTANT_TOKEN_ID in values:
                raise ValueError(
                    f"Assistant chunk {message_index} unexpectedly contains role IDs"
                )
            continue
        if role != "user":
            raise ValueError(f"Unsupported ARC message role: {role!r}")

        user_messages += 1
        starts = [index for index, value in enumerate(values) if value == IM_START_ID]
        if len(starts) != 2:
            raise ValueError(
                f"User chunk {message_index} expected two im_start IDs, found {starts}"
            )
        followers = [
            values[index + 1] if index + 1 < len(values) else None for index in starts
        ]
        if followers != [USER_TOKEN_ID, ASSISTANT_TOKEN_ID]:
            raise ValueError(
                f"User chunk {message_index} is not roleful: header followers {followers}"
            )
        verified_headers += 2

    if not user_messages:
        raise ValueError("ARC conversation contains no user messages")
    return {"verified_headers": verified_headers, "user_messages": user_messages}
