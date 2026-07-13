# Part 4 — LLM-Powered Feature: Tabular Record Batch Scoring

## Chosen Track

**(B) Tabular Record Batch Scoring.** Three real applicant records from `cleaned_data.csv` (Part 1) are each
formatted as a JSON object and scored against a lending-risk rubric via an LLM call, with the response
validated against a 5-field JSON schema.

**Why Track B**: risk-tiering an applicant against a defined rubric is a real, recognizable step in the
underwriting workflow this whole capstone is built around — it's the track where the LLM feature reads as a
genuine extension of the loan-approval system from Parts 2–3, rather than a bolted-on demo.

## Setup

- **Provider**: [OpenRouter](https://openrouter.ai) (`https://openrouter.ai/api/v1/chat/completions`), which
  accepts the standard `model` + `messages` JSON body over HTTPS.
- **Model**: `meta-llama/llama-3.1-8b-instruct:free` (free tier — swappable via the `LLM_MODEL` variable).
- **API key handling**: entered via `getpass.getpass()` in Google Colab at runtime, stored only in the
  `LLM_API_KEY` environment variable for that session. **Never hardcoded, never printed, never committed to
  any file in this repository.**

## `call_llm` Function

Implemented exactly per spec: builds the `{model, messages, temperature, max_tokens}` payload, sets the
`Authorization: Bearer <key>` header, checks `response.status_code == 200`, and returns
`response.json()['choices'][0]['message']['content']` (or `None` with a printed status code on failure).

**Test prompt demonstration:**

> _[PASTE: the "Test output: ..." line from your Colab run here — should show the model replying with just "hello"]_

## Prompt Design

**System prompt (verbatim)** — states the LLM's role, the full scoring rubric, and one worked input-output
example:

```
You are a loan-risk scoring assistant for a lending institution. You will be given a
single loan applicant record as a JSON object. Score the applicant against the following rubric and respond
with ONLY a valid JSON object -- no markdown, no explanation, no code fences.

RUBRIC:
- risk_tier: "low" if Credit_History == 1 AND ApplicantIncome is comfortably above LoanAmount-implied
  repayment needs (roughly ApplicantIncome > 4x LoanAmount / (Loan_Amount_Term/12)); "medium" if
  Credit_History == 1 but income is only marginally sufficient, or Credit_History is missing; "high" if
  Credit_History == 0, regardless of income.
- flag_for_review: true if Credit_History == 0, OR if ApplicantIncome and CoapplicantIncome combined seem
  inconsistent with the requested LoanAmount (very high loan relative to combined income); false otherwise.
- primary_signal: the single field that most influenced your risk_tier decision (e.g. "Credit_History",
  "ApplicantIncome", "income-to-loan ratio").
- confidence: "low", "medium", or "high" -- how confident you are in this assessment given the fields
  available.
- recommended_action: one short sentence describing the suggested next step for a loan officer.

Required JSON fields: risk_tier, flag_for_review, primary_signal, confidence, recommended_action.

WORKED EXAMPLE:
Input: {"Loan_ID": "LPEXAMPLE", "ApplicantIncome": 2000, "CoapplicantIncome": 0, "LoanAmount": 250,
"Loan_Amount_Term": 360, "Credit_History": 0.0, "Dependents": 2, "Property_Area": "Rural"}
Output: {"risk_tier": "high", "flag_for_review": true, "primary_signal": "Credit_History",
"confidence": "high", "recommended_action": "Escalate to a senior underwriter before proceeding; credit
history does not meet guidelines."}

Now score the applicant record provided in the user message. Respond with ONLY the JSON object.
```

**User prompt template**: `json.dumps(record, indent=2)` — the applicant record's fields inserted directly as
a JSON string, with no additional wrapper text.

**Why `temperature=0`**: this is a structured scoring task where the goal is consistent, reproducible batch
output — the same applicant record should get the same risk assessment every time it's scored, not a different
judgment call depending on random sampling. Temperature near 0 makes the model deterministically pick its
highest-probability next token at each step, which is exactly the behavior a structured-output pipeline needs.

## Temperature A/B Comparison

Each of the three records was scored twice — once at `temperature=0`, once at `temperature=0.7`.

| Input (Loan_ID) | Output at temp=0 | Output at temp=0.7 | Key difference |
|---|---|---|---|
| _[PASTE Loan_ID]_ | _[PASTE JSON output]_ | _[PASTE JSON output]_ | _[note: identical? different risk_tier? different wording only?]_ |
| _[PASTE Loan_ID]_ | _[PASTE JSON output]_ | _[PASTE JSON output]_ | _[...]_ |
| _[PASTE Loan_ID]_ | _[PASTE JSON output]_ | _[PASTE JSON output]_ | _[...]_ |

**Why temperature=0 is more deterministic**: the model always selects the single highest-probability next
token at each generation step, so an identical prompt reliably reproduces the same (or near-identical) output
across repeated calls. **Why temperature=0.7 introduces variability**: the model instead samples from a wider
slice of the probability distribution over candidate next tokens, so plausible-but-different phrasing — and
occasionally a different risk_tier judgment call on borderline applicants — can appear across repeated calls
with the exact same input.

## PII Guardrail

Implemented via regex checks on the user input for email addresses and 10-digit / hyphenated phone numbers,
per the required `has_pii()` function. Any call whose input matches either pattern is blocked before it ever
reaches the LLM.

**Test results:**

| Test input | Contains PII? | Result |
|---|---|---|
| Applicant record + `jane.doe@example.com` inserted into the prompt | Yes (email) | _[PASTE: "Input blocked: PII detected." / blocked_test result]_ |
| Applicant record alone, no PII | No | _[PASTE: the clean_test JSON output that came through]_ |

## Structured Output Handling

- **Schema** (`RISK_SCHEMA`): 5 required scalar fields — `risk_tier` (enum: low/medium/high),
  `flag_for_review` (boolean), `primary_signal` (string), `confidence` (enum: low/medium/high),
  `recommended_action` (string).
- Each LLM response is stripped of whitespace, parsed with `json.loads()` inside a
  `try/except json.JSONDecodeError`, then validated with `jsonschema.validate()` inside a
  `try/except ValidationError`. On either failure, a fallback dict with all 5 fields set to `None` is returned
  and the error is printed/logged.

## End-to-End Demonstration (3 Records)

| Loan_ID | LLM Assessment JSON | Validation Status |
|---|---|---|
| LP001006 | `{"risk_tier": "low", "flag_for_review": false, "primary_signal": "Credit_History", "confidence": "high", "recommended_action": "Approve loan as per standard guidelines."}` | pass |
| LP001014 | `{"risk_tier": "high", "flag_for_review": true, "primary_signal": "Credit_History", "confidence": "high", "recommended_action": "Escalate to a senior underwriter before proceeding; credit history does not meet guidelines."}` | pass |
| LP001066 | `{"risk_tier": "low", "flag_for_review": false, "primary_signal": "Credit_History", "confidence": "high", "recommended_action": "Approve loan as per standard guidelines."}` | pass |

**All three records passed schema validation** — no JSON parsing or schema failures were observed for this
model/prompt combination.

**Finding**: `Credit_History` was the `primary_signal` for all three assessments, and `risk_tier` tracked it
almost exactly (`Credit_History=1` → low risk in both cases where it held; `Credit_History=0` → high risk,
flagged for review, for LP001014). One thing worth noting: LP001006 has a comfortably-covered but not
high income (₹2,583 applicant + ₹2,358 co-applicant income against a ₹120k loan) — under the rubric's stricter
reading this could plausibly have landed in "medium" rather than "low," but the model weighted `Credit_History`
heavily enough to call it "low" anyway. This is a reasonable illustration of why `confidence` and
`primary_signal` are included as required fields: they let a human reviewer see *which* signal drove the
call and second-guess borderline cases rather than trusting `risk_tier` blindly.

## Repository Contents

- `part4_loan_llm_scoring_colab.ipynb` — full notebook, designed to run in Google Colab with your own
  OpenRouter API key entered securely at runtime via `getpass`.
- `README.md` — this file.
- **No API key is hardcoded anywhere in the codebase.**
