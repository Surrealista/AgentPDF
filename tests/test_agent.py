"""
Tests for the PDF E-mail Reader Agent.

Run:
    pip install pytest
    pytest tests/ -v
"""

import json
from decimal import Decimal
from unittest.mock import MagicMock, patch

import pytest

from src.agents.pdf_email_agent import PdfEmailReaderAgent
from src.models.sales_order import ExtractionConfidence, ExtractedSalesOrder
from src.utils.email_reader import EmailMessage, PdfAttachment


# ── Fixtures ───────────────────────────────────────────────────────────────

SAMPLE_CLAUDE_RESPONSE = json.dumps(
    {
        "source_email_id": "42",
        "source_email_subject": "Comanda núm. 2024-0099",
        "source_email_sender": "compres@client.com",
        "source_pdf_filename": "comanda_2024-0099.pdf",
        "order_reference": "2024-0099",
        "order_date": "2024-11-15",
        "customer": {
            "name": "Distribucions Client, SL",
            "tax_id": "B12345678",
            "billing_address": {
                "street": "Carrer Major, 10",
                "city": "Barcelona",
                "postal_code": "08001",
                "country": "ES",
                "state_province": None,
            },
            "shipping_address": None,
            "contact": {
                "name": "Marta Puig",
                "email": "compres@client.com",
                "phone": "+34 93 123 45 67",
            },
        },
        "lines": [
            {
                "line_number": 1,
                "product_reference": "REF-001",
                "description": "Widget Premium Model A",
                "quantity": 50,
                "unit": "pcs",
                "unit_price": 12.50,
                "discount_pct": 5.0,
                "tax_rate": 21.0,
                "line_total": 752.81,
            },
            {
                "line_number": 2,
                "product_reference": "REF-002",
                "description": "Caixa d'embalatge reforçada",
                "quantity": 10,
                "unit": "box",
                "unit_price": 3.00,
                "discount_pct": None,
                "tax_rate": 21.0,
                "line_total": 36.30,
            },
        ],
        "delivery": {
            "requested_date": "2024-11-30",
            "address": {
                "street": "Polígon Industrial, Nau 5",
                "city": "Hospitalet de Llobregat",
                "postal_code": "08901",
                "country": "ES",
                "state_province": None,
            },
            "instructions": "Entregar en horari de matí (8h-13h)",
            "incoterms": "DDP",
        },
        "payment_terms": "net_30",
        "payment_terms_raw": None,
        "totals": {
            "subtotal": 662.50,
            "total_discount": 31.25,
            "total_tax": 157.63,
            "grand_total": 789.11,
            "currency": "EUR",
        },
        "confidence": "high",
        "extraction_notes": None,
    }
)


@pytest.fixture
def fake_pdf() -> PdfAttachment:
    return PdfAttachment(filename="comanda_2024-0099.pdf", data=b"%PDF-fake-content")


@pytest.fixture
def fake_email(fake_pdf) -> EmailMessage:
    return EmailMessage(
        uid="42",
        subject="Comanda núm. 2024-0099",
        sender="compres@client.com",
        date="Fri, 15 Nov 2024 09:00:00 +0100",
        body_text="Adjuntem la nostra comanda.",
        pdf_attachments=[fake_pdf],
    )


@pytest.fixture
def agent() -> PdfEmailReaderAgent:
    """Agent instance with dummy IMAP credentials (not used in unit tests)."""
    return PdfEmailReaderAgent(
        imap_host="imap.example.com",
        imap_user="test@example.com",
        imap_password="secret",
        anthropic_api_key="sk-ant-fake",
    )


# ── Tests ──────────────────────────────────────────────────────────────────


class TestParseResponse:
    """_parse_response converts Claude's JSON into a validated Pydantic model."""

    def test_valid_response_returns_order(self, fake_email, fake_pdf):
        order = PdfEmailReaderAgent._parse_response(
            SAMPLE_CLAUDE_RESPONSE, fake_email, fake_pdf
        )
        assert isinstance(order, ExtractedSalesOrder)

    def test_customer_name(self, fake_email, fake_pdf):
        order = PdfEmailReaderAgent._parse_response(
            SAMPLE_CLAUDE_RESPONSE, fake_email, fake_pdf
        )
        assert order.customer.name == "Distribucions Client, SL"

    def test_customer_tax_id(self, fake_email, fake_pdf):
        order = PdfEmailReaderAgent._parse_response(
            SAMPLE_CLAUDE_RESPONSE, fake_email, fake_pdf
        )
        assert order.customer.tax_id == "B12345678"

    def test_order_lines_count(self, fake_email, fake_pdf):
        order = PdfEmailReaderAgent._parse_response(
            SAMPLE_CLAUDE_RESPONSE, fake_email, fake_pdf
        )
        assert len(order.lines) == 2

    def test_first_line_quantity(self, fake_email, fake_pdf):
        order = PdfEmailReaderAgent._parse_response(
            SAMPLE_CLAUDE_RESPONSE, fake_email, fake_pdf
        )
        assert order.lines[0].quantity == Decimal("50")

    def test_grand_total(self, fake_email, fake_pdf):
        order = PdfEmailReaderAgent._parse_response(
            SAMPLE_CLAUDE_RESPONSE, fake_email, fake_pdf
        )
        assert order.totals.grand_total == Decimal("789.11")

    def test_currency(self, fake_email, fake_pdf):
        order = PdfEmailReaderAgent._parse_response(
            SAMPLE_CLAUDE_RESPONSE, fake_email, fake_pdf
        )
        assert order.totals.currency == "EUR"

    def test_confidence_high(self, fake_email, fake_pdf):
        order = PdfEmailReaderAgent._parse_response(
            SAMPLE_CLAUDE_RESPONSE, fake_email, fake_pdf
        )
        assert order.confidence == ExtractionConfidence.HIGH

    def test_source_metadata_injected(self, fake_email, fake_pdf):
        order = PdfEmailReaderAgent._parse_response(
            SAMPLE_CLAUDE_RESPONSE, fake_email, fake_pdf
        )
        assert order.source_email_id == "42"
        assert order.source_pdf_filename == "comanda_2024-0099.pdf"

    def test_invalid_json_raises(self, fake_email, fake_pdf):
        with pytest.raises(Exception):
            PdfEmailReaderAgent._parse_response("not-json", fake_email, fake_pdf)


class TestExtractFromPdf:
    """_extract_from_pdf calls Claude and returns a validated order."""

    def test_calls_claude_and_returns_order(self, agent, fake_email, fake_pdf):
        with patch.object(agent, "_call_claude", return_value=SAMPLE_CLAUDE_RESPONSE):
            order = agent._extract_from_pdf(fake_email, fake_pdf)

        assert order is not None
        assert order.customer.name == "Distribucions Client, SL"

    def test_returns_none_on_claude_error(self, agent, fake_email, fake_pdf):
        with patch.object(agent, "_call_claude", side_effect=RuntimeError("API down")):
            order = agent._extract_from_pdf(fake_email, fake_pdf)

        assert order is None


class TestSummarise:
    """summarise produces a human-readable one-liner."""

    def test_summarise_contains_customer(self, fake_email, fake_pdf):
        order = PdfEmailReaderAgent._parse_response(
            SAMPLE_CLAUDE_RESPONSE, fake_email, fake_pdf
        )
        summary = PdfEmailReaderAgent.summarise(order)
        assert "Distribucions Client, SL" in summary

    def test_summarise_contains_total(self, fake_email, fake_pdf):
        order = PdfEmailReaderAgent._parse_response(
            SAMPLE_CLAUDE_RESPONSE, fake_email, fake_pdf
        )
        summary = PdfEmailReaderAgent.summarise(order)
        assert "789.11" in summary

    def test_summarise_contains_confidence(self, fake_email, fake_pdf):
        order = PdfEmailReaderAgent._parse_response(
            SAMPLE_CLAUDE_RESPONSE, fake_email, fake_pdf
        )
        summary = PdfEmailReaderAgent.summarise(order)
        assert "HIGH" in summary


class TestRun:
    """agent.run() fetches emails and extracts orders end-to-end."""

    def test_run_returns_orders_for_pdfs(self, agent, fake_email):
        with (
            patch("src.agents.pdf_email_agent.ImapEmailReader") as MockImap,
            patch.object(agent, "_call_claude", return_value=SAMPLE_CLAUDE_RESPONSE),
        ):
            mock_reader = MagicMock()
            mock_reader.__enter__ = MagicMock(return_value=mock_reader)
            mock_reader.__exit__ = MagicMock(return_value=False)
            mock_reader.fetch_unread_with_pdfs.return_value = [fake_email]
            MockImap.return_value = mock_reader

            orders = agent.run()

        assert len(orders) == 1
        assert orders[0].customer.name == "Distribucions Client, SL"

    def test_run_returns_empty_when_no_emails(self, agent):
        with patch("src.agents.pdf_email_agent.ImapEmailReader") as MockImap:
            mock_reader = MagicMock()
            mock_reader.__enter__ = MagicMock(return_value=mock_reader)
            mock_reader.__exit__ = MagicMock(return_value=False)
            mock_reader.fetch_unread_with_pdfs.return_value = []
            MockImap.return_value = mock_reader

            orders = agent.run()

        assert orders == []
