"""
Entry point for the PDF E-mail Reader Agent.

Usage
-----
    # Run against a live mailbox (credentials from .env):
    python main.py

    # Test with a local PDF file without touching a mailbox:
    python main.py --test-pdf path/to/order.pdf

Environment variables (see .env.example)
-----------------------------------------
    ANTHROPIC_API_KEY   – required
    IMAP_HOST           – required for live mode
    IMAP_USER           – required for live mode
    IMAP_PASSWORD       – required for live mode
    IMAP_MAILBOX        – optional, default: INBOX
    MARK_EMAILS_AS_SEEN – optional, default: true
    MAX_EMAILS_PER_RUN  – optional, default: 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from dotenv import load_dotenv

from src.agents.pdf_email_agent import PdfEmailReaderAgent

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        logger.error("Missing required environment variable: %s", name)
        sys.exit(1)
    return value


def build_agent() -> PdfEmailReaderAgent:
    return PdfEmailReaderAgent(
        imap_host=_require_env("IMAP_HOST"),
        imap_user=_require_env("IMAP_USER"),
        imap_password=_require_env("IMAP_PASSWORD"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        mailbox=os.getenv("IMAP_MAILBOX", "INBOX"),
        mark_as_seen=os.getenv("MARK_EMAILS_AS_SEEN", "true").lower() == "true",
        max_emails_per_run=int(os.getenv("MAX_EMAILS_PER_RUN", "20")),
    )


def run_live() -> None:
    """Full pipeline: connect to mailbox → extract PDFs → print results."""
    agent = build_agent()
    orders = agent.run()

    if not orders:
        print("No orders extracted.")
        return

    print(f"\n{'═' * 60}")
    print(f"  Extracted {len(orders)} sales order(s)")
    print(f"{'═' * 60}\n")
    for i, order in enumerate(orders, 1):
        print(f"[{i}] {agent.summarise(order)}")
        print(json.dumps(order.model_dump(mode="json"), indent=2, ensure_ascii=False))
        print()

    # ── Hand off to Agent 2 ────────────────────────────────────────────────
    # At this point `orders` is a list[ExtractedSalesOrder].  Pass it to the
    # ERP order-creator agent, e.g.:
    #
    #   from src.agents.erp_order_creator import ErpOrderCreatorAgent
    #   creator = ErpOrderCreatorAgent(...)
    #   creator.create_orders(orders)


def run_test_pdf(pdf_path: str) -> None:
    """Extract a single local PDF without connecting to a mailbox."""
    if not os.path.exists(pdf_path):
        logger.error("File not found: %s", pdf_path)
        sys.exit(1)

    with open(pdf_path, "rb") as fh:
        pdf_bytes = fh.read()

    # Build a minimal agent (no IMAP credentials needed for this path)
    agent = PdfEmailReaderAgent(
        imap_host="",
        imap_user="",
        imap_password="",
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
    )
    order = agent.extract_from_pdf_bytes(
        pdf_data=pdf_bytes,
        filename=os.path.basename(pdf_path),
    )

    if order is None:
        print("Extraction failed. Check logs for details.")
        sys.exit(1)

    print(agent.summarise(order))
    print()
    print(json.dumps(order.model_dump(mode="json"), indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="PDF E-mail Reader Agent")
    parser.add_argument(
        "--test-pdf",
        metavar="PATH",
        help="Path to a local PDF file to test extraction (skips IMAP connection)",
    )
    args = parser.parse_args()

    if args.test_pdf:
        run_test_pdf(args.test_pdf)
    else:
        run_live()


if __name__ == "__main__":
    main()
