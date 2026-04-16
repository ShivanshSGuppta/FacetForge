"""Project-wide constants."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SAMPLE_DATA_DIR = DATA_DIR / "samples"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PREDICTIONS_DIR = OUTPUT_DIR / "predictions"
REPORTS_DIR = OUTPUT_DIR / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"
SCREENSHOTS_DIR = DOCS_DIR / "screenshots"

DEFAULT_SCORE_LABELS = [10, 25, 50, 75, 90]
DEFAULT_CATEGORY_ORDER = ["linguistic", "pragmatics", "safety", "emotion"]
DEFAULT_CONTEXT_RADIUS = 1
DEFAULT_BATCH_MIN_SIZE = 16
DEFAULT_BATCH_MAX_SIZE = 24
DEFAULT_ENCODING = "utf-8"

COLUMN_CANDIDATES = {
    "conversation_id": [
        "conversation_id",
        "conversationid",
        "conversation",
        "thread_id",
        "thread",
        "chat_id",
        "chat",
        "dialogue_id",
        "session_id",
    ],
    "turn_id": [
        "turn_id",
        "turnid",
        "message_id",
        "utterance_id",
        "id",
        "row_id",
    ],
    "speaker": [
        "speaker",
        "speaker_role",
        "role",
        "author",
        "sender",
        "participant",
    ],
    "text": [
        "text",
        "message",
        "content",
        "utterance",
        "response",
        "body",
    ],
    "turn_index": [
        "turn_index",
        "turn_number",
        "message_index",
        "order",
        "position",
        "seq",
        "sequence",
    ],
    "timestamp": [
        "timestamp",
        "created_at",
        "createdat",
        "time",
        "datetime",
        "sent_at",
    ],
    "parent_turn_id": [
        "parent_turn_id",
        "parent_id",
        "reply_to",
        "in_reply_to",
    ],
}

SPEAKER_ALIASES = {
    "assistant": "assistant",
    "agent": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "system": "system",
    "user": "user",
    "customer": "user",
    "human": "user",
    "client": "user",
    "reviewer": "reviewer",
}

