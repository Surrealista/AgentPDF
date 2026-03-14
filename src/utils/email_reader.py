"""
IMAP email reader utility.

Connects to a mailbox, fetches unseen emails, and extracts PDF attachments.
Designed to be the input stage of the PDF e-mail reader agent.
"""

from __future__ import annotations

import email
import imaplib
import logging
from dataclasses import dataclass, field
from email.message import Message

logger = logging.getLogger(__name__)


@dataclass
class EmailMessage:
    uid: str
    subject: str
    sender: str
    date: str
    body_text: str
    pdf_attachments: list[PdfAttachment] = field(default_factory=list)


@dataclass
class PdfAttachment:
    filename: str
    data: bytes  # raw PDF bytes


class ImapEmailReader:
    """
    Thin wrapper around Python's imaplib that retrieves emails with PDF
    attachments from an IMAP mailbox.

    Usage
    -----
    >>> reader = ImapEmailReader(host="imap.gmail.com", user="...", password="...")
    >>> emails = reader.fetch_unread_with_pdfs(mailbox="INBOX")
    >>> reader.close()
    """

    def __init__(
        self,
        host: str,
        user: str,
        password: str,
        port: int = 993,
        use_ssl: bool = True,
    ) -> None:
        self.host = host
        self.user = user
        self.password = password
        self.port = port
        self.use_ssl = use_ssl
        self._conn: imaplib.IMAP4 | imaplib.IMAP4_SSL | None = None

    # ── Connection lifecycle ───────────────────────────────────────────────

    def connect(self) -> None:
        logger.info("Connecting to %s:%s", self.host, self.port)
        if self.use_ssl:
            self._conn = imaplib.IMAP4_SSL(self.host, self.port)
        else:
            self._conn = imaplib.IMAP4(self.host, self.port)
        self._conn.login(self.user, self.password)
        logger.info("Logged in as %s", self.user)

    def close(self) -> None:
        if self._conn:
            try:
                self._conn.logout()
            except Exception:
                pass
            self._conn = None

    def __enter__(self) -> "ImapEmailReader":
        self.connect()
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── Fetching ───────────────────────────────────────────────────────────

    def fetch_unread_with_pdfs(
        self,
        mailbox: str = "INBOX",
        mark_as_seen: bool = False,
        max_emails: int = 50,
    ) -> list[EmailMessage]:
        """
        Return a list of EmailMessage objects for every unseen email in
        *mailbox* that contains at least one PDF attachment.

        Parameters
        ----------
        mailbox:
            IMAP folder to search (default: INBOX).
        mark_as_seen:
            If True, messages are marked \\Seen after fetching.
        max_emails:
            Safety cap; stop after processing this many emails.
        """
        if not self._conn:
            raise RuntimeError("Not connected. Call connect() first.")

        self._conn.select(mailbox, readonly=not mark_as_seen)
        _, data = self._conn.search(None, "UNSEEN")
        uids: list[str] = data[0].split()

        if not uids:
            logger.info("No unread messages in %s", mailbox)
            return []

        results: list[EmailMessage] = []
        for uid_bytes in uids[:max_emails]:
            uid = uid_bytes.decode()
            try:
                msg = self._fetch_message(uid)
                if msg is None:
                    continue
                pdfs = self._extract_pdfs(msg)
                if not pdfs:
                    continue  # skip emails without PDF attachments
                email_obj = EmailMessage(
                    uid=uid,
                    subject=self._decode_header(msg.get("Subject", "")),
                    sender=self._decode_header(msg.get("From", "")),
                    date=msg.get("Date", ""),
                    body_text=self._extract_body_text(msg),
                    pdf_attachments=pdfs,
                )
                results.append(email_obj)
                logger.info(
                    "Found email uid=%s with %d PDF(s): %s",
                    uid,
                    len(pdfs),
                    email_obj.subject,
                )
            except Exception as exc:
                logger.warning("Failed to process email uid=%s: %s", uid, exc)

        return results

    # ── Private helpers ────────────────────────────────────────────────────

    def _fetch_message(self, uid: str) -> Message | None:
        _, data = self._conn.fetch(uid, "(RFC822)")  # type: ignore[union-attr]
        for part in data:
            if isinstance(part, tuple):
                return email.message_from_bytes(part[1])
        return None

    @staticmethod
    def _decode_header(value: str) -> str:
        from email.header import decode_header

        parts = decode_header(value)
        decoded = []
        for text, charset in parts:
            if isinstance(text, bytes):
                decoded.append(text.decode(charset or "utf-8", errors="replace"))
            else:
                decoded.append(text)
        return "".join(decoded)

    @staticmethod
    def _extract_pdfs(msg: Message) -> list[PdfAttachment]:
        pdfs: list[PdfAttachment] = []
        for part in msg.walk():
            content_type = part.get_content_type()
            disposition = part.get("Content-Disposition", "")
            filename = part.get_filename() or ""

            is_pdf = (
                content_type == "application/pdf"
                or filename.lower().endswith(".pdf")
                or "attachment" in disposition.lower()
                and filename.lower().endswith(".pdf")
            )
            if is_pdf and part.get_payload(decode=True):
                pdfs.append(
                    PdfAttachment(
                        filename=filename or "attachment.pdf",
                        data=part.get_payload(decode=True),  # type: ignore[arg-type]
                    )
                )
        return pdfs

    @staticmethod
    def _extract_body_text(msg: Message) -> str:
        """Return the plain-text body, falling back to an empty string."""
        body_parts: list[str] = []
        for part in msg.walk():
            if part.get_content_type() == "text/plain":
                payload = part.get_payload(decode=True)
                if payload:
                    charset = part.get_content_charset() or "utf-8"
                    body_parts.append(payload.decode(charset, errors="replace"))
        return "\n".join(body_parts)
