import os


DOCS_DIR = os.path.join(
    "support_assistant",
    "docs"
)

os.makedirs(
    DOCS_DIR,
    exist_ok=True
)


documents = {
    "delivery_policy.txt": """
Delivery Policy

Standard delivery is available for eligible orders placed during normal service hours.
Customers can view the estimated delivery time on the order tracking page.

Delivery times can vary depending on store availability, order volume, traffic, weather,
and delivery partner availability.

If an order is delayed beyond the displayed estimate, the customer should first check
the tracking status. For continued delays, the customer can contact support with the
order ID.

Customers should ensure that the delivery address and contact number are correct before
placing an order.

Delivery ETA shown in the application is an estimate and may change after an order is
placed.
""",

    "return_policy.txt": """
Return Policy

Eligible products may be returned according to the product-specific return conditions.

Customers should initiate a return request through the support process and provide the
order ID and the reason for the return.

Products must generally be unused and in acceptable condition unless the return is for
a damaged, incorrect, or defective item.

Some product categories may have different return eligibility rules. Perishable or
consumable products may not be eligible for standard returns unless they arrived damaged
or incorrect.

After a return request is submitted, support may review the request before approving
the return.
""",

    "refund_policy.txt": """
Refund Policy

Refunds for eligible orders are processed after the return or refund request is approved.

Once approved, the refund is normally initiated within 10 working days.

The original payment method is used whenever supported by the payment provider.

Customers should keep the original payment receipt or transaction reference when
requesting a refund.

If a refund has been approved but has not appeared after the stated processing period,
the customer should contact support with the order ID and refund reference.
""",

    "membership_policy.txt": """
Membership Policy

The membership program provides eligible members with benefits described in the
current membership plan.

Membership benefits may include promotional offers, delivery-related benefits, or
member-only discounts when applicable.

Benefits can vary by plan and may be subject to minimum order values or product
eligibility conditions.

Customers can review their current membership status and available benefits from their
account.

Membership fees and renewal conditions should be checked in the membership section
before subscribing or renewing.
""",

    "order_tracking.txt": """
Order Tracking

Customers can track an active order using the order tracking section of the application.

Tracking information may include order confirmation, preparation, dispatch, and
delivery status.

The displayed delivery estimate may change as the order progresses.

If tracking has not updated for an extended period, the customer should verify the
order ID and contact support when necessary.

Support can use the order ID to investigate an order whose tracking status appears
incorrect or delayed.
""",

    "cancellation_policy.txt": """
Order Cancellation Policy

An order can generally be cancelled only while it is still in a cancellable stage.

Once an order has been prepared, dispatched, or handed to a delivery partner, cancellation
may no longer be available.

Customers should use the cancellation option shown for the active order.

When cancellation is unavailable in the application, the customer should contact
support with the order ID.

Any refund resulting from an approved cancellation follows the applicable refund
processing rules.
""",

    "gift_card_policy.txt": """
Gift Card Policy

Gift cards can be used according to the terms associated with the specific gift card.

Customers should keep the gift card code or reference available when asking support
about a gift card.

A gift card may have validity, usage, or promotional restrictions.

If a gift card code is not working, the customer should check that the code was entered
correctly and that the gift card is still valid.

Support may request the gift card reference and relevant transaction details when
investigating a gift card issue.
""",

    "support_hours.txt": """
Customer Support Hours

Customer support is available during the service hours displayed in the application.

For an active order issue, customers should provide the order ID when contacting support
so the issue can be investigated more efficiently.

For delivery delays, tracking problems, cancellations, returns, and refunds, customers
should provide the relevant order information.

Support response time can vary depending on request volume and issue complexity.

The latest support availability should be checked in the application because service
hours may change.
"""
}


# ---------------------------------------------------------
# Create exactly eight documents
# ---------------------------------------------------------

for filename, content in documents.items():

    file_path = os.path.join(
        DOCS_DIR,
        filename
    )

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            content.strip() + "\n"
        )

    print(
        f"Created: {file_path}"
    )


# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------

created_files = sorted(
    os.listdir(DOCS_DIR)
)

print("\n" + "=" * 70)
print("SUPPORT DOCUMENT VERIFICATION")
print("=" * 70)

print(
    f"Documents created: {len(created_files)}"
)

for filename in created_files:
    print(
        f"- {filename}"
    )

if len(created_files) == 8:
    print("\nExactly 8 support documents created successfully.")
else:
    print(
        f"\nWARNING: Expected 8 documents, found "
        f"{len(created_files)}."
    )