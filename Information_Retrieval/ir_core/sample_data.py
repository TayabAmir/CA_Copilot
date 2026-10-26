"""MOCK data for the CA Copilot IR module. Illustrative paraphrases shaped like Pakistani tax law -
NOT legal text, no real rates. Replace with the real Ordinance / SRO text for the final demo."""

_HDR = """ID: {id}
Title: {title}
Law: {law}
Section: {sec}
Source Type: {src}
Tax Year: 2026
Effective Date: 01-Jul-2025
"""


def _h(id, title, law, sec, src):
    return _HDR.format(id=id, title=title, law=law, sec=sec, src=src)


ITO = "Income Tax Ordinance, 2001"

SAMPLE_DOCS = [
    _h("ita-149", f"{ITO} — Section 149: Salary", ITO, "149", "Ordinance") + """
Section 149 Salary
(1) The person responsible for paying any salary to an employee shall deduct tax from the amount of salary at the time of payment, computed at the rates applicable to the employee for the tax year.
(2) The employer shall compute the tax on the estimated annual salary and deduct it in equal monthly instalments.
(a) Where the employee receives salary from more than one employer, the employee may elect which employer shall make the deduction.
(b) Any tax deducted shall be paid to the credit of the Federal Government within the prescribed time.
Provided that no deduction shall be made where the salary of the employee is below the taxable limit.
""",
    _h("ita-153", f"{ITO} — Section 153: Payments for goods, services and contracts", ITO, "153", "Ordinance") + """
Section 153 Payments for goods, services and contracts
(1) Every prescribed person making a payment to a resident person shall deduct withholding tax at the time of payment at the rate specified in the First Schedule.
(a) For the sale of goods, the withholding tax applies to the gross amount of the payment.
(b) For the rendering of or providing services, and for the execution of a contract, the withholding tax applies to the payment made to the contractor, with a rate for filers and a higher rate for non-filers.
(2) A contractor who is a non-filer shall be subject to the higher rate of withholding tax prescribed for the contract payment.
(3) The withholding agent shall deposit the tax and furnish a statement as required under section 165.
Provided that where an exemption certificate is produced by the payee, tax shall be deducted as stated in the certificate.
""",
    _h("ita-155", f"{ITO} — Section 155: Income from property", ITO, "155", "Ordinance") + """
Section 155 Income from property
(1) Every person paying rent of immovable property to a resident person shall deduct withholding tax from the gross amount of rent at the rate specified in the First Schedule.
(2) Where the rent is paid for a building used for business, the payer shall also report the payment in the statement required under section 165.
Provided that no tax shall be deducted where the annual rent is below the threshold.
""",
    _h("ita-165", f"{ITO} — Section 165: Statements", ITO, "165", "Ordinance") + """
Section 165 Statements
(1) Every person who deducts or collects tax under section 149, section 153 or section 155 shall furnish a statement of withholding to the Commissioner in the prescribed form and within the prescribed time.
(2) The statement shall show the name of the payee, the amount of payment, the section under which tax was deducted, the rate applied and the amount of tax withheld.
(a) A schedule of vendors and payments shall accompany the statement.
Provided that a correction statement may be filed to correct an error in a statement already furnished.
""",
    _h("sta-3", "Sales Tax Act, 1990 — Section 3: Scope of tax (services)", "Sales Tax Act, 1990", "3", "Act") + """
Section 3 Scope of tax
(1) Sales tax shall be charged on the supply of goods and on the services rendered or provided by a registered person.
(a) A contractor who provides services shall charge sales tax on the invoice at the rate applicable to the service.
(b) The registered person shall issue a tax invoice and account for output tax in the return.
Provided that provincial sales tax on services applies where the service is rendered in the province.
""",
    _h("sro-100", "SRO 100(I)/2025 (illustrative) — Withholding rates for contractors", ITO, "153", "SRO") + """
Notification
This notification amends the rates of withholding tax for payments to contractors under section 153.
(1) The withholding tax rate for a contractor who is a filer shall be the rate stated in the table of this notification.
(2) The withholding tax rate for a contractor who is a non-filer shall be higher than the filer rate.
Provided that the notification applies to payments made on or after the effective date.
""",
    _h("fbr-filer", "FBR Guidance (illustrative) — Filer and non-filer status for withholding", ITO, "153", "Circular") + """
# Filer and non-filer status
A person whose name appears on the active taxpayers list is treated as a filer for withholding purposes. A person not on the list is a non-filer and is subject to the higher withholding rates.
## Checking status
The withholding agent should verify the status of the payee on the active taxpayers list before making a payment.
## Consequences
Where a non-filer is paid, the higher rate applies as provided in section 153 and section 155.
""",
]

SAMPLE_QUERIES = [
    "withholding tax contractor payment",
    "section 153",
    "salary tax deduction employer",
    "rent of immovable property withholding",
    "non-filer higher rate",
    "sales tax on services by contractor",
    "statement of withholding to commissioner",
]

# ground truth (which documents are truly relevant) for the precision / recall demo
SAMPLE_JUDGMENTS = {
    "withholding tax contractor payment": ["ita-153", "sro-100"],
    "section 153": ["ita-153", "sro-100"],
    "salary tax deduction employer": ["ita-149"],
    "rent of immovable property withholding": ["ita-155"],
    "non-filer higher rate": ["fbr-filer", "sro-100", "ita-153"],
    "sales tax on services by contractor": ["sta-3"],
    "statement of withholding to commissioner": ["ita-165"],
}
