"""
Pydantic models for the structured sales order data extracted from PDFs.
These models define exactly what the PDF reader agent extracts so the
second agent (ERP order creator) can consume them without ambiguity.
"""

from __future__ import annotations

from datetime import date
from decimal import Decimal
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class AddressModel(BaseModel):
    street: Optional[str] = None
    city: Optional[str] = None
    postal_code: Optional[str] = None
    country: Optional[str] = None
    state_province: Optional[str] = None


class ContactModel(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None


class CustomerModel(BaseModel):
    name: str = Field(..., description="Full name or company name of the customer")
    tax_id: Optional[str] = Field(None, description="VAT / NIF / CIF / tax identifier")
    billing_address: Optional[AddressModel] = None
    shipping_address: Optional[AddressModel] = None
    contact: Optional[ContactModel] = None


class OrderLineModel(BaseModel):
    line_number: Optional[int] = Field(None, description="Line position in the order")
    product_reference: Optional[str] = Field(
        None, description="Supplier or customer product code / SKU"
    )
    description: str = Field(..., description="Product or service description")
    quantity: Decimal = Field(..., description="Quantity ordered", gt=0)
    unit: Optional[str] = Field(None, description="Unit of measure, e.g. 'kg', 'pcs', 'box'")
    unit_price: Optional[Decimal] = Field(None, description="Price per unit before taxes")
    discount_pct: Optional[Decimal] = Field(
        None, description="Discount percentage (0-100)", ge=0, le=100
    )
    tax_rate: Optional[Decimal] = Field(None, description="Tax / VAT rate percentage")
    line_total: Optional[Decimal] = Field(
        None, description="Total for this line (quantity × unit_price - discount + tax)"
    )


class PaymentTerms(str, Enum):
    IMMEDIATE = "immediate"
    NET_15 = "net_15"
    NET_30 = "net_30"
    NET_60 = "net_60"
    NET_90 = "net_90"
    OTHER = "other"


class DeliveryModel(BaseModel):
    requested_date: Optional[date] = None
    address: Optional[AddressModel] = None
    instructions: Optional[str] = None
    incoterms: Optional[str] = Field(None, description="e.g. EXW, FOB, CIF, DDP")


class OrderTotalsModel(BaseModel):
    subtotal: Optional[Decimal] = Field(None, description="Sum before taxes and discounts")
    total_discount: Optional[Decimal] = None
    total_tax: Optional[Decimal] = None
    grand_total: Optional[Decimal] = Field(None, description="Final amount to be invoiced")
    currency: Optional[str] = Field(None, description="ISO 4217 currency code, e.g. EUR, USD")


class ExtractionConfidence(str, Enum):
    HIGH = "high"        # All key fields clearly present
    MEDIUM = "medium"    # Most fields found, some inferred
    LOW = "low"          # Document unclear or many fields missing


class ExtractedSalesOrder(BaseModel):
    """
    Complete structured representation of a purchase/sales order extracted
    from a PDF attachment.  This is the contract between Agent 1 (extractor)
    and Agent 2 (ERP order creator).
    """

    # ── Source metadata ────────────────────────────────────────────────────
    source_email_id: Optional[str] = Field(
        None, description="UID of the email the PDF was attached to"
    )
    source_email_subject: Optional[str] = None
    source_email_sender: Optional[str] = None
    source_pdf_filename: Optional[str] = None

    # ── Order header ───────────────────────────────────────────────────────
    order_reference: Optional[str] = Field(
        None, description="Customer's own purchase order number"
    )
    order_date: Optional[date] = None
    customer: CustomerModel

    # ── Lines ──────────────────────────────────────────────────────────────
    lines: list[OrderLineModel] = Field(
        default_factory=list, description="One entry per product / service line"
    )

    # ── Delivery & payment ─────────────────────────────────────────────────
    delivery: Optional[DeliveryModel] = None
    payment_terms: Optional[PaymentTerms] = None
    payment_terms_raw: Optional[str] = Field(
        None, description="Raw text if payment terms don't map to enum"
    )

    # ── Totals ─────────────────────────────────────────────────────────────
    totals: Optional[OrderTotalsModel] = None

    # ── Extraction quality ─────────────────────────────────────────────────
    confidence: ExtractionConfidence = ExtractionConfidence.MEDIUM
    extraction_notes: Optional[str] = Field(
        None,
        description="Free-text notes from the agent about ambiguities or assumptions",
    )
