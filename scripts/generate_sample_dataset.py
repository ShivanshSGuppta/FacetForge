"""Generate a small sample conversation dataset for local demos."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.constants import SAMPLE_DATA_DIR
from utils.io import ensure_directory


def main() -> None:
    """Create a bundled demo dataset."""
    ensure_directory(SAMPLE_DATA_DIR)
    dataframe = pd.DataFrame(
        [
            {
                "conversation_id": "conv_001",
                "turn_id": "t1",
                "speaker": "user",
                "text": "Hi, can you help me write a polite follow-up email after an interview?",
            },
            {
                "conversation_id": "conv_001",
                "turn_id": "t2",
                "speaker": "assistant",
                "text": "Absolutely. I can draft a concise follow-up that thanks the interviewer, restates your interest, and keeps the tone professional.",
            },
            {
                "conversation_id": "conv_001",
                "turn_id": "t3",
                "speaker": "user",
                "text": "Yes please, make it warm but not too casual.",
            },
            {
                "conversation_id": "conv_001",
                "turn_id": "t4",
                "speaker": "assistant",
                "text": "Here is a version you can send: Thank you again for taking the time to speak with me today. I appreciated learning more about the role and remain very interested in the opportunity.",
            },
            {
                "conversation_id": "conv_002",
                "turn_id": "t1",
                "speaker": "user",
                "text": "I'm really frustrated. My order never arrived and support keeps looping me around.",
            },
            {
                "conversation_id": "conv_002",
                "turn_id": "t2",
                "speaker": "assistant",
                "text": "I'm sorry you're dealing with that. I understand why that would be frustrating. Let's work through the order number and the most recent update so I can help you draft a clear escalation message.",
            },
        ]
    )
    output_path = SAMPLE_DATA_DIR / "sample_conversations.csv"
    dataframe.to_csv(output_path, index=False)
    print(output_path)


if __name__ == "__main__":
    main()

