"""
Module A Exercise: Agent Observability with LangSmith
-------------------------------------------------------
Set up LangSmith tracing, run queries, and inspect the trace tree.

Segments covered:
  2. LangSmith setup & first traces
  3. Trace anatomy — debugging tool calls, latency, token counts
"""
 
import os
import sys
from pathlib import Path
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

# ---------------------------------------------------------------------------
# TODO 1: Enable LangSmith tracing
#
# Set these environment variables (or add them to your .env file):
#   LANGCHAIN_TRACING_V2 = "true"
#   LANGCHAIN_API_KEY    = "<your LangSmith API key>"
#
# Verify tracing is enabled by checking the env var.
# Free Developer plan: https://smith.langchain.com (1 seat, 5K traces/month)
# ---------------------------------------------------------------------------
# YOUR CODE HERE — set the env var
# os.environ["LANGCHAIN_TRACING_V2"] = ...

# Verify:
tracing_enabled = os.environ.get("LANGCHAIN_TRACING_V2", "false") == "true"
print(f"LangSmith tracing enabled: {tracing_enabled}")
if not tracing_enabled:
    print("WARNING: Set LANGCHAIN_TRACING_V2=true to enable tracing!")


sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import build_support_agent, ask

# ---------------------------------------------------------------------------
# TODO 2: Build the multi-agent pipeline
#
# Use build_support_agent() with collection_name="observability_exercise"
# ---------------------------------------------------------------------------
# YOUR CODE HERE
agent = build_support_agent(
    collection_name = "observability_exercise"
)
app = agent["app"]
print(app)
print("Pipeline ready. All runs will be traced to LangSmith.\n")


# ---------------------------------------------------------------------------
# TODO 3: Run 3 different queries — one for each agent type
#
# Run these queries through the agent and print the results:
#   a) A policy question (e.g., "What is the overdraft fee?")
#   b) An account lookup (e.g., "What is the balance on ACC-12345?")
#   c) An escalation (e.g., "I need to speak to a manager about fraud!")
#
# For each, print: intent, response (first 200 chars), retrieved_sources
# ---------------------------------------------------------------------------
"""
print("\n" + "=" * 60)
print("SEGMENT 2: FIRST TRACES")
print("=" * 60)

queries = [
    "What is the overdraft fee?",
    "What is the balance on ACC-12345?",
    "I need to speak to a manager about fraud on my account!",
]

if app is not None:
    for query in queries:
        print(f"\nQuery: {query}")
        response = ask(app, query)
        print(f"Response: {response.keys()}")
        for k, v in response.items():
            print(f"{k}: {v}")
        print()
else:
    print("Complete TODO 2 first.")
"""

# ---------------------------------------------------------------------------
# TODO 4: Inspect traces in LangSmith
#
# Open https://smith.langchain.com and find your traces.
# For EACH of the 3 traces above, answer these questions:
#
# a) Which agent did the supervisor route to?
# b) How many total LLM calls were made? (supervisor + agent)
# c) How many prompt tokens were used in the most expensive call?
# d) What was the total latency of the trace?
# e) For the policy query: which documents were retrieved?
#
# Write your answers as comments below:
# ---------------------------------------------------------------------------

# ==============================================================================

# Trace 1
# Input:   "I need to speak to a manager about fraud on my account!"
# Output:  "escalation"
# Tokens:  359 | Latency: 2.05s | Cost: $0.0001056

# Trace 2
# Input:   "What is the balance on ACC-12345?"
# Output:  {"account_id": "ACC-12345", "name": "Alice Johnson", ...}
# Tokens:  464 | Latency: 2.10s | Cost: $0.00008490

# Trace 3
# Input:   "What is the overdraft fee?"
# Output:  "[account_fees.md] --- ## Overdraft Fees - Overdraft fee: $35 per transaction..."
# Tokens:  1,001 | Latency: 2.60s | Cost: $0.0001781


# ---------------------------------------------------------------------------
# TODO 5: Inject a deliberate failure and trace the error path
#
# Run a query with a non-existent account number (e.g., ACC-99999).
# Then find the trace in LangSmith and answer:
#   - Did the supervisor route correctly?
#   - Where in the trace tree does the "not found" response originate?
#   - How does error information flow through the run tree?
# ---------------------------------------------------------------------------
"""
print("\n" + "=" * 60)
print("SEGMENT 3: ERROR TRACING")
print("=" * 60)

if app is not None:
    error_query = "What is the balance on ACC-99999?"
    print(f"\nQuery: {error_query}")
    response = ask(app, error_query)
    print(f"Response: {response}")
else:
    print("Complete TODO 2 first.")
sys.exit()
"""
# Input:   "What is the balance on ACC-99999?"
# Output:  "I couldn't find account ACC-99999 in our system. Please double-check or contact support@securebank.com."
# Intent:  account_status
# Tokens:  120 | Latency: 1.16s | Cost: $0.00001890


# ---------------------------------------------------------------------------
# TODO 6 (BONUS): Tag your runs for monitoring
#
# Re-run the 3 queries from TODO 3, but this time add tags
# using the config parameter:
#   result = app.invoke(inputs, config={"tags": ["your-tag"]})
#
# Then filter by your tag in the LangSmith dashboard.
# ---------------------------------------------------------------------------
"""
print("\n" + "=" * 60)
print("BONUS: TAGGED RUNS")
print("=" * 60)

queries = [
    "What is the overdraft fee?",
    "What is the balance on ACC-12345?",
    "I need to speak to a manager about fraud on my account!",
]

if app is not None:
    for query in queries:
        print(f"\nQuery: {query}")
        input = {
            "query": query,
            "intent": "",
            "response": "",
            "context": "",
            "retrieved_sources": [],
        }
        response = app.invoke(input, config={"tags": ["ex1-tag"]})
        print(f"Response: {response.keys()}")
        for k, v in response.items():
            print(f"{k}: {v}")
        print()
else:
    print("Complete TODO 2 first.")
"""