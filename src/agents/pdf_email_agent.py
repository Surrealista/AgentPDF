"""
Agent 1 – PDF E-mail Reader Agent
==================================
Reads emails from an IMAP mailbox, finds PDF attachments, and uses Claude
to extract structured sales-order information from each PDF.

The structured output (ExtractedSalesOrder) is the interface contract with
Agent 2 (ERP order creator), which will consume these objects to create
sales orders in the target ERP system.

Flow
----
1.  Connect to IMAP mailbox.
2.  Fetch unread emails that contain PDF attachments.
3.  For each PDF:
    a. Encode the PDF as base64 and send it to Claude claude-opus-4-6
       together with a structured-output schema.
    b. Claude reads the PDF natively and returns a JSON payload
       matching the ExtractedSalesOrder schema.
    c. Validate the JSON with Pydantic.
4.  Return the list of validated ExtractedSalesOrder objects.
"""

from __future__ import annotations

import base64
import json
import logging
from typing import Optional

import anthropic

from src.models.sales_order import (
    ExtractionConfidence,
    ExtractedSalesOrder,
)
from src.utils.email_reader import EmailMessage, ImapEmailReader, PdfAttachment

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# System prompt – instructs Claude on the extraction task
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """\
You are a specialist in reading purchase-order and sales-order PDF documents.

Your task is to extract ALL relevant order information from the provided PDF
and return it as a single, well-formed JSON object that strictly matches the
schema described below.

EXTRACTION RULES
────────────────
1.  Extract every field you can find in the document.  Leave optional fields
    as null if the information is genuinely absent.
2.  Normalise dates to ISO 8601 format (YYYY-MM-DD).
3.  Normalise monetary amounts to plain decimal numbers (no currency symbols).
4.  Use the ISO 4217 three-letter code for currency (e.g. EUR, USD, GBP).
5.  For payment_terms, map the document wording to one of these values:
    immediate | net_15 | net_30 | net_60 | net_90 | other
    If you use "other", populate payment_terms_raw with the verbatim text.
6.  For each order line, capture product references exactly as they appear.
7.  Set confidence to:
    - "high"   → you found all key fields (customer, ≥1 line, totals)
    - "medium" → most fields present, a few are inferred or missing
    - "low"    → document is unclear or many key fields are absent
8.  Use extraction_notes to flag any ambiguities, assumptions, or issues.

Return ONLY the JSON object – no markdown fences, no prose, no explanation.
"""

# JSON schema sent to Claude as output_config so the response is guaranteed
# to match the ExtractedSalesOrder structure.
_OUTPUT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "source_email_id": {"type": ["string", "null"]},
        "source_email_subject": {"type": ["string", "null"]},
        "source_email_sender": {"type": ["string", "null"]},
        "source_pdf_filename": {"type": ["string", "null"]},
        "order_reference": {"type": ["string", "null"]},
        "order_date": {"type": ["string", "null"]},
        "customer": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "tax_id": {"type": ["string", "null"]},
                "billing_address": {
                    "type": ["object", "null"],
                    "properties": {
                        "street": {"type": ["string", "null"]},
                        "city": {"type": ["string", "null"]},
                        "postal_code": {"type": ["string", "null"]},
                        "country": {"type": ["string", "null"]},
                        "state_province": {"type": ["string", "null"]},
                    },
                    "additionalProperties": False,
                },
                "shipping_address": {
                    "type": ["object", "null"],
                    "properties": {
                        "street": {"type": ["string", "null"]},
                        "city": {"type": ["string", "null"]},
                        "postal_code": {"type": ["string", "null"]},
                        "country": {"type": ["string", "null"]},
                        "state_province": {"type": ["string", "null"]},
                    },
                    "additionalProperties": False,
                },
                "contact": {
                    "type": ["object", "null"],
                    "properties": {
                        "name": {"type": ["string", "null"]},
                        "email": {"type": ["string", "null"]},
                        "phone": {"type": ["string", "null"]},
                    },
                    "additionalProperties": False,
                },
            },
            "required": ["name"],
            "additionalProperties": False,
        },
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "line_number": {"type": ["integer", "null"]},
                    "product_reference": {"type": ["string", "null"]},
                    "description": {"type": "string"},
                    "quantity": {"type": "number"},
                    "unit": {"type": ["string", "null"]},
                    "unit_price": {"type": ["number", "null"]},
                    "discount_pct": {"type": ["number", "null"]},
                    "tax_rate": {"type": ["number", "null"]},
                    "line_total": {"type": ["number", "null"]},
                },
                "required": ["description", "quantity"],
                "additionalProperties": False,
            },
        },
        "delivery": {
            "type": ["object", "null"],
            "properties": {
                "requested_date": {"type": ["string", "null"]},
                "address": {
                    "type": ["object", "null"],
                    "properties": {
                        "street": {"type": ["string", "null"]},
                        "city": {"type": ["string", "null"]},
                        "postal_code": {"type": ["string", "null"]},
                        "country": {"type": ["string", "null"]},
                        "state_province": {"type": ["string", "null"]},
                    },
                    "additionalProperties": False,
                },
                "instructions": {"type": ["string", "null"]},
                "incoterms": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        },
        "payment_terms": {
            "type": ["string", "null"],
            "enum": ["immediate", "net_15", "net_30", "net_60", "net_90", "other", None],
        },
        "payment_terms_raw": {"type": ["string", "null"]},
        "totals": {
            "type": ["object", "null"],
            "properties": {
                "subtotal": {"type": ["number", "null"]},
                "total_discount": {"type": ["number", "null"]},
                "total_tax": {"type": ["number", "null"]},
                "grand_total": {"type": ["number", "null"]},
                "currency": {"type": ["string", "null"]},
            },
            "additionalProperties": False,
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
        },
        "extraction_notes": {"type": ["string", "null"]},
    },
    "required": ["customer", "lines", "confidence"],
    "additionalProperties": False,
}


class PdfEmailReaderAgent:
    """
    Agent that reads emails, extracts PDFs, and returns structured
    ExtractedSalesOrder objects ready for the ERP order creator agent.

    Parameters
    ----------
    imap_host, imap_user, imap_password:
        IMAP credentials.
    anthropic_api_key:
        Anthropic API key (falls back to ANTHROPIC_API_KEY env var if None).
    mailbox:
        IMAP folder to monitor (default "INBOX").
    mark_as_seen:
        Mark processed emails as read so they are not re-processed.
    max_emails_per_run:
        Safety cap on the number of emails processed per call to run().
    """

    def __init__(
        self,
        imap_host: str,
        imap_user: str,
        imap_password: str,
        anthropic_api_key: Optional[str] = None,
        mailbox: str = "INBOX",
        mark_as_seen: bool = True,
        max_emails_per_run: int = 20,
    ) -> None:
        self.imap_host = imap_host
        self.imap_user = imap_user
        self.imap_password = imap_password
        self.mailbox = mailbox
        self.mark_as_seen = mark_as_seen
        self.max_emails_per_run = max_emails_per_run

        self._claude = anthropic.Anthropic(
            api_key=anthropic_api_key  # None → reads ANTHROPIC_API_KEY env var
        )

    # ── Public interface ───────────────────────────────────────────────────

    def run(self) -> list[ExtractedSalesOrder]:
        """
        Connect to the mailbox, process all unread emails with PDF
        attachments, and return validated ExtractedSalesOrder objects.
        """
        reader = ImapEmailReader(
            host=self.imap_host,
            user=self.imap_user,
            password=self.imap_password,
        )
        with reader:
            emails = reader.fetch_unread_with_pdfs(
                mailbox=self.mailbox,
                mark_as_seen=self.mark_as_seen,
                max_emails=self.max_emails_per_run,
            )

        orders: list[ExtractedSalesOrder] = []
        for email_msg in emails:
            for pdf in email_msg.pdf_attachments:
                order = self._extract_from_pdf(email_msg, pdf)
                if order:
                    orders.append(order)

        logger.info(
            "Agent run complete. Processed %d email(s), extracted %d order(s).",
            len(emails),
            len(orders),
        )
        return orders

    def extract_from_pdf_bytes(
        self,
        pdf_data: bytes,
        filename: str = "document.pdf",
        email_id: Optional[str] = None,
        email_subject: Optional[str] = None,
        email_sender: Optional[str] = None,
    ) -> Optional[ExtractedSalesOrder]:
        """
        Convenience method: extract a sales order from raw PDF bytes without
        going through the IMAP flow.  Useful for testing or batch processing.
        """
        pdf_attachment = PdfAttachment(filename=filename, data=pdf_data)
        fake_email = EmailMessage(
            uid=email_id or "manual",
            subject=email_subject or "",
            sender=email_sender or "",
            date="",
            body_text="",
            pdf_attachments=[pdf_attachment],
        )
        return self._extract_from_pdf(fake_email, pdf_attachment)

    # ── Private helpers ────────────────────────────────────────────────────

    def _extract_from_pdf(
        self, email_msg: EmailMessage, pdf: PdfAttachment
    ) -> Optional[ExtractedSalesOrder]:
        """Send a single PDF to Claude and parse the structured response."""
        logger.info(
            "Extracting order from '%s' (email uid=%s)", pdf.filename, email_msg.uid
        )
        try:
            raw_json = self._call_claude(email_msg, pdf)
            return self._parse_response(raw_json, email_msg, pdf)
        except Exception as exc:
            logger.error(
                "Failed to extract order from '%s': %s", pdf.filename, exc
            )
            return None

    def _call_claude(self, email_msg: EmailMessage, pdf: PdfAttachment) -> str:
        """
        Send the PDF to Claude claude-opus-4-6 with a structured-output constraint
        and return the raw JSON string from the response.

        Claude supports PDF input natively via the document content block,
        which gives better extraction quality than pre-converting to text.
        """
        pdf_b64 = base64.standard_b64encode(pdf.data).decode("utf-8")

        user_message_content: list[dict] = [
            {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": pdf_b64,
                },
                "title": pdf.filename,
            },
            {
                "type": "text",
                "text": (
                    f"Email subject: {email_msg.subject}\n"
                    f"Email sender: {email_msg.sender}\n"
                    f"Email date: {email_msg.date}\n\n"
                    "Please extract the sales order information from this PDF "
                    "and return it as the structured JSON object described in "
                    "your system instructions."
                ),
            },
        ]

        with self._claude.messages.stream(
            model="claude-opus-4-6",
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message_content}],
            output_config={
                "format": {
                    "type": "json_schema",
                    "schema": _OUTPUT_SCHEMA,
                }
            },
        ) as stream:
            final = stream.get_final_message()

        # The structured-output constraint guarantees the first content block
        # is a text block containing valid JSON.
        for block in final.content:
            if block.type == "text":
                return block.text

        raise ValueError("Claude returned no text block in the response.")

    @staticmethod
    def _parse_response(
        raw_json: str,
        email_msg: EmailMessage,
        pdf: PdfAttachment,
    ) -> ExtractedSalesOrder:
        """Parse and validate the JSON returned by Claude."""
        data: dict = json.loads(raw_json)

        # Inject source metadata (Claude cannot know these from the PDF alone)
        data.setdefault("source_email_id", email_msg.uid)
        data.setdefault("source_email_subject", email_msg.subject)
        data.setdefault("source_email_sender", email_msg.sender)
        data.setdefault("source_pdf_filename", pdf.filename)

        order = ExtractedSalesOrder.model_validate(data)
        logger.info(
            "Successfully extracted order for customer '%s' "
            "(confidence=%s, lines=%d)",
            order.customer.name,
            order.confidence.value,
            len(order.lines),
        )
        return order

    # ── Confidence helper exposed for testing ──────────────────────────────

    @staticmethod
    def summarise(order: ExtractedSalesOrder) -> str:
        """Return a human-readable one-liner summary of an extracted order."""
        lines_count = len(order.lines)
        total = (
            f"{order.totals.grand_total} {order.totals.currency}"
            if order.totals and order.totals.grand_total
            else "total unknown"
        )
        return (
            f"[{order.confidence.value.upper()}] "
            f"Customer: {order.customer.name} | "
            f"Ref: {order.order_reference or 'n/a'} | "
            f"Lines: {lines_count} | "
            f"Total: {total} | "
            f"PDF: {order.source_pdf_filename}"
        )
