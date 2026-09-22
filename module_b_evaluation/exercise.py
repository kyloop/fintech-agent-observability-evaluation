"""
Module B Exercise: Evaluation with LangSmith + DeepEval
---------------------------------------------------------
Implement evaluators, compute MRR, use DeepEval metrics, build
a G-Eval custom metric, enhance the dataset, and hill-climb.

Segments covered:
  6.  LangSmith evaluators (custom + LLM-as-judge)
  8.  MRR (Mean Reciprocal Rank)
  9.  DeepEval (faithfulness, hallucination, answer relevancy)
  10. G-Eval (custom criteria — empathy)
  11. Dataset enhancement (add edge-case examples)
  12. Hill climbing (top_k=1 vs top_k=5)
"""

import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langsmith.evaluation import evaluate
from langsmith import Client

from eval_dataset import EXERCISE_DATASET_NAME, EVAL_EXAMPLES, ensure_exercise_dataset
from eval_dataset import EXERCISE_HC_DATASET_NAME, ensure_exercise_hc_dataset

# from eval_dataset import SOLUTION_DATASET_NAME, EVAL_EXAMPLES, ensure_solution_dataset
# from eval_dataset import SOLUTION_HC_DATASET_NAME, ensure_solution_hc_dataset


load_dotenv()

os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")

sys.path.insert(0, str(Path(__file__).parent.parent / "project"))
from fintech_support_agent import build_support_agent, ask

# --- Build pipeline (provided) ---
print("Building FinTech support agent...")
agent = build_support_agent(collection_name="eval_exercise")
app = agent["app"]
retriever = agent["retriever"]
judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
print("Pipeline ready.\n")

# --- Ensure exercise dataset exists in LangSmith ---
ensure_exercise_dataset()

# ===================================================================
# SEGMENT 6: LangSmith Evaluators
# ===================================================================

# ---------------------------------------------------------------------------
# TODO 1: Implement run_agent
#
# Run the multi-agent graph and return:
#   {"answer": ..., "intent": ..., "context": ..., "retrieved_sources": [...]}
# ---------------------------------------------------------------------------
def run_agent(inputs):
    question = inputs["question"]
    result = ask(app, inputs["question"])
    return {
        "answer": result["response"],
        "intent": result["intent"],
        "retrieved_sources": result["retrieved_sources"],
        "context": result["context"],
    }
# ---------------------------------------------------------------------------
# TODO 2: Implement routing accuracy evaluator
#
# Compare run.outputs["intent"] to example.outputs["intent"]
# Score 1.0 if match, 0.0 if not.
# Return: {"key": "routing_accuracy", "score": float}
# ---------------------------------------------------------------------------
def routing_evaluator(run, example):
    """Check if the supervisor routed to the correct agent."""
    predicted = run.outputs.get("intent", "")
    expected = example.outputs.get("intent", "")
    score = 1.0 if predicted == expected else 0.0
    print(f"  [Routing] expected={expected}, predicted={predicted}, score={score}")
    return {"key": "routing_accuracy", "score": score}

# ---------------------------------------------------------------------------
# TODO 3: Implement LLM-as-judge faithfulness evaluator
#
# Use judge_llm to assess whether the answer is faithful to the context.
# Score: 1.0 = fully faithful, 0.5 = partial, 0.0 = not faithful
# For escalation responses (empty context), score 1.0 if it's a general
# empathetic handoff without specific policy claims.
#
# Return: {"key": "faithfulness", "score": float}
#
# Hint: Build a ChatPromptTemplate, call judge_llm, parse JSON response.
# ---------------------------------------------------------------------------
def faithfulness_evaluator(run, example): 
    answer = run.outputs.get("answer", "")
    context = run.outputs.get("context", "")
    question = example.inputs.get("question", "")

    if not answer:
        return {"key": "faithfulness", "score": 0.0}

    FAITHFULNESS_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
            "You are an expert evaluator. Assess whether the answer is faithful "
            "to the provided context.\n\n"
            "Score 1.0 = fully faithful: every claim is supported by the context\n"
            "Score 0.5 = partially faithful: some claims are unsupported\n"
            "Score 0.0 = not faithful: contains claims contradicting or absent from context\n\n"
            "If the context is empty (escalation response), score 1.0 if the response "
            "is a general empathetic handoff without specific policy claims, else 0.5.\n\n"
            'Respond ONLY with JSON: {{"score": <float>, "reason": "<one sentence>"}}'
        ),
        ("human",
            "Context:\n{context}\n\nQuestion: {question}\n\nAnswer to evaluate:\n{answer}"),
        ])

    messages = FAITHFULNESS_PROMPT.format_messages(
        context=context[:2000] if context else "(no context — escalation response)",
        question=question,
        answer=answer,
    )

    response = judge_llm.invoke(messages).content.strip()

    try:
        start, end = response.find("{"), response.rfind("}") + 1
        parsed = json.loads(response[start:end])
        score = float(parsed.get("score", 0.5))
        reason = parsed.get("reason", "")
        return {"key": "faithfulness", "score": score, "comment": reason}
    except (json.JSONDecodeError, ValueError):
        return {"key": "faithfulness", "score": 0.5}

# ---------------------------------------------------------------------------
# TODO 4: Implement LLM-as-judge correctness evaluator
#
# Compare actual answer to expected answer using judge_llm.
# Focus on factual accuracy, not exact wording.
# Score: 1.0 = all key facts correct, 0.5 = partial, 0.0 = wrong
#
# Return: {"key": "correctness", "score": float}
# ---------------------------------------------------------------------------
def correctness_evaluator(run, example):
    actual = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")
    question = example.inputs.get("question", "")

    if not actual or not expected:
        return {"key": "correctness", "score": 0.0}

    CORRECTNESS_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
            "You are an expert evaluator. Compare the AI's answer to the expected answer.\n\n"
            "Score 1.0 = all key facts correct\n"
            "Score 0.5 = partially correct\n"
            "Score 0.0 = key facts wrong or missing\n\n"
            "Focus on factual accuracy, not exact wording. "
            "For escalation responses, check that empathy and contact info are present.\n\n"
            'Respond ONLY with JSON: {{"score": <float>, "reason": "<one sentence>"}}'
        ),
        ("human",
            "Question: {question}\n\nExpected: {expected}\n\nActual: {actual}"
        ),
    ])

    messages = CORRECTNESS_PROMPT.format_messages(
        question=question, expected=expected, actual=actual
    )

    response = judge_llm.invoke(messages).content.strip()

    try:
        start, end = response.find("{"), response.rfind("}") + 1
        parsed = json.loads(response[start:end])
        score = float(parsed.get("score", 0.5))
        reason = parsed.get("reason", "")
        return {"key": "correctness", "score": score, "comment": reason}
    except (json.JSONDecodeError, ValueError):
        return {"key": "correctness", "score": 0.5}
    


# ---------------------------------------------------------------------------
# TODO 5: Run evaluate() with all evaluators
# Use data=EXERCISE_DATASET_NAME, experiment_prefix="exercise-eval-student"
# ---------------------------------------------------------------------------
"""
print("Complete TODOs 1-4, then run evaluate() here.\n")

results = evaluate(
    run_agent,
    data=EXERCISE_DATASET_NAME,
    evaluators=[routing_evaluator, faithfulness_evaluator, correctness_evaluator],
    experiment_prefix="exercise-eval",
    metadata={"model": "gpt-4o-mini"},
)
print(results)
print("\nEvaluation complete. View results in LangSmith.\n")
"""
# ===================================================================
# SEGMENT 8: MRR (Mean Reciprocal Rank)
# ===================================================================

# ---------------------------------------------------------------------------
# TODO 6: Compute MRR for the retriever
#
# For each query below, run it through the retriever and find
# the rank (1-based) of the first relevant document.
#
# Expected relevant sources are provided.
# If no relevant doc is found in results, reciprocal rank = 0.
#
# MRR = average of all reciprocal ranks
# ---------------------------------------------------------------------------
"""
mrr_queries = [
    # Easy — clearly maps to one document
    {"query": "What credit score do I need for a personal loan?", "relevant_source": "loan_policy.md"},
    {"query": "How do I report identity theft?", "relevant_source": "fraud_policy.md"},
    # Ambiguous — wire transfer fees appear in BOTH account_fees.md and transfer_policy.md
    {"query": "How much does an international wire transfer cost?", "relevant_source": "transfer_policy.md"},
    {"query": "What are the wire transfer fees?", "relevant_source": "account_fees.md"},
    # Cross-domain — "interest rate" matches savings APY AND loan APR
    {"query": "What interest rate will I get?", "relevant_source": "account_fees.md"},
    {"query": "What is the APR on a used car loan?", "relevant_source": "loan_policy.md"},
    # Confusing — "late fee" could match overdraft fee OR loan late payment fee
    {"query": "What happens if I'm late on a payment?", "relevant_source": "loan_policy.md"},
    # Overlapping term — "card replacement" is in account_fees.md AND fraud_policy.md
    {"query": "How much does a replacement debit card cost?", "relevant_source": "account_fees.md"},
    # Vague — "daily limit" is only in transfer_policy.md but could match account_fees.md
    {"query": "What are the daily transaction limits?", "relevant_source": "transfer_policy.md"},
    # Specific but tricky — "two-factor authentication" in both fraud_policy.md and transfer_policy.md
    {"query": "When is two-factor authentication required?", "relevant_source": "fraud_policy.md"},
]

print("=" * 60)
print("SEGMENT 8: MRR COMPUTATION")
print("=" * 60)
"""
# YOUR CODE HERE
# For each query:
#   1. docs = retriever.invoke(query["query"])
#   2. Find rank of first doc where metadata["source"] == query["relevant_source"]
#   3. reciprocal_rank = 1/rank if found, else 0
#   4. Print query, rank, reciprocal rank
# Then compute MRR = mean of all reciprocal ranks
"""
rr = [] # Reciprocal Rank list
for idx, query in enumerate(mrr_queries):
    print(f"idx: {idx}")
    print(query['query'])
    print(query['relevant_source'])
    print()
    docs_lst = retriever.invoke(query['query'])
    returned_ranked_doc_lst = [doc.metadata['source'] for doc in docs_lst]
    print(returned_ranked_doc_lst)
    ranked_idx = (returned_ranked_doc_lst.index(query['relevant_source']))+1 if query['relevant_source'] in returned_ranked_doc_lst else 0 
    print(f"ranked_idx: {ranked_idx}")
    rr.append(1/ranked_idx if ranked_idx >0  else 0)
print(f"MRR Score: {sum(rr) / len(rr)}")


print("Complete TODO 6 to compute MRR.\n")
"""
# ===================================================================
# SEGMENT 9: DeepEval
# ===================================================================

# ---------------------------------------------------------------------------
# TODO 7: Run DeepEval metrics
#
# pip install deepeval
#
# Create 5 test cases from the agent's actual outputs.
# Run: FaithfulnessMetric, AnswerRelevancyMetric, HallucinationMetric
#
# Example:
#   from deepeval.test_case import LLMTestCase
#   from deepeval.metrics import FaithfulnessMetric
#   from deepeval import assert_test
#
#   test_case = LLMTestCase(
#       input="What is the overdraft fee?",
#       actual_output="The overdraft fee is $35.",
#       retrieval_context=["...retrieved doc content..."]
#   )
#   faithfulness = FaithfulnessMetric(threshold=0.7)
#   assert_test(test_case, [faithfulness])
# ---------------------------------------------------------------------------
print("=" * 60)
print("SEGMENT 9: DEEPEVAL METRICS")
print("=" * 60)

# YOUR CODE HERE
# Step 1: Run 5 queries through the agent to get actual outputs + context
# Step 2: Create LLMTestCase objects
# Step 3: Run FaithfulnessMetric, AnswerRelevancyMetric, HallucinationMetric
# Step 4: Print results
"""
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric
from deepeval import assert_test

# 1. Define a list of test questions you want to evaluate
eval_queries = [
    "What is the overdraft fee?",
]

test_case_results = []

# 2. Iterate over the string queries properly
for query in eval_queries:
    # Get the live response from your FinTech multi-agent system
    result = ask(app, query)
    
    # Establish context fallback structures cleanly
    ctx = [result["context"]] if result["context"] else ["No context retrieved."]
    
    # Construct the dynamic LLMTestCase using the live pipeline outputs
    dynamic_tc = LLMTestCase(
        input=query,
        actual_output=result["response"],
        retrieval_context=ctx
    )
    
    test_case_results.append({
        "tc": dynamic_tc,
        "response": result["response"],
    })

    # 3. Instantiate your metric with strict penalization turned on
    faithfulness = FaithfulnessMetric(
        threshold=0.7, 
        model="gpt-4o-mini",
        penalize_ambiguous_claims=True # Catches empty contexts/fabrications correctly!
    )
    
    # Measure the dynamic test case containing the live context
    faithfulness.measure(dynamic_tc)
    
    # 4. Print clean broken down outputs
    print(f"\n================ QUERY: {query} ================")
    print(f"Query: {query}")
    print(f"Agent actual output:  {dynamic_tc.actual_output}")
    print(f"Final Score:       {faithfulness.score}")
    print(f"Passed Threshold: {faithfulness.is_successful()}")
    print(f"Reasoning:        {faithfulness.reason}")  

    print("\n--- Detailed Claims Breakdown ---")
    for verdict in faithfulness.verdicts:
        print(f"Verdict classification: {verdict.verdict.upper()}")
        print(f"Verdict Reason:         {verdict.reason}\n")


# print(assert_test(test_case, [faithfulness]))

print("Complete TODO 7 to run DeepEval metrics.\n")

sys.exit()
"""
# ===================================================================
# SEGMENT 10: G-Eval
# ===================================================================

# ---------------------------------------------------------------------------
# TODO 8: Build a G-Eval metric for empathy in escalation responses
#
# from deepeval.metrics import GEval
# from deepeval.test_case import LLMTestCaseParams
#b
# Define a GEval metric with:
#   name: "Empathy"
#   criteria: describe what empathetic support looks like
#   evaluation_params: [LLMTestCaseParams.ACTUAL_OUTPUT]
#   threshold: 0.7
#
# Run on escalation test cases (frustrated customer queries).
# ---------------------------------------------------------------------------
# ===================================================================
# SEGMENT 10: G-Eval
# ===================================================================
"""
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
from deepeval.test_case import LLMTestCase
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

print("=" * 60)
print("SEGMENT 10: G-EVAL (EMPATHY)")
print("=" * 60)

escalation_queries = [
    "This is ridiculous! Someone withdrew \$15,000 from my savings without my permission!",
    "I've been waiting 3 weeks for my fraud dispute to be resolved! This is unacceptable!",
    "What is the wire transfer fee?",
]

# 1. Define the G-Eval Empathy Metric as requested in TODO 8
empathy_metric = GEval(
    name="Empathy",
    criteria=(
        "Assess whether the response shows empathy toward a frustrated customer. "
        "An empathetic response acknowledges the customer's feelings or problem, "
        "maintains a supportive and reassuring tone, summarizes the issue briefly, "
        "and outlines clear next steps or handoff instructions without quoting cold policies."
    ),
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    threshold=0.7,
    model="gpt-4o-mini"
)

for query in escalation_queries:
    # FIX: Pass the loop variable `query`, NOT the string literal "query"
    result = ask(app, query)

    ctx = [result["context"]] if result["context"] else ["No context retrieved."]

    # Construct the test case with the live output
    dynamic_tc = LLMTestCase(
        input=query,
        actual_output=result["response"],
        retrieval_context=ctx
    )

    # Measure using the empathy metric
    empathy_metric.measure(dynamic_tc)

    print(f"\n================ QUERY: {query} ================")
    print(f"Agent actual output: {dynamic_tc.actual_output}")
    print(f"Empathy Score:       {empathy_metric.score}")
    print(f"Passed Threshold:    {empathy_metric.is_successful()}")
    print(f"Reasoning:           {empathy_metric.reason}")


print("Complete TODO 8 to run G-Eval empathy metric.")

sys.exit()
"""
# ===================================================================
# SEGMENT 11: Dataset Enhancement
# ===================================================================

# ---------------------------------------------------------------------------
# TODO 9: Add new evaluation examples to improve dataset coverage
#
# The base dataset (EVAL_EXAMPLES) has 15 examples. Real-world evaluation
# datasets grow over time as you discover failure modes.
#
# Add 3 new examples that test edge cases the current dataset misses:
#   a) A multi-part question (asks two things at once)
#   b) A question with a wrong/misspelled account number
#   c) A boundary-case policy question (e.g., exact threshold amounts)
#
# Steps:
#   1. Define the 3 new examples in the same format as EVAL_EXAMPLES
#   2. Upload them to the existing LangSmith dataset using client.create_examples()
#   3. Print confirmation
#
# Hint: Each example needs {"inputs": {"question": ...}, "outputs": {"answer": ..., "intent": ...}}
# ---------------------------------------------------------------------------
"""
print("=" * 60)
print("SEGMENT 11: DATASET ENHANCEMENT")
print("=" * 60)

client = Client()

# Step 1: Define the 3 new edge-case evaluation examples
new_examples = [
    {
        # a) Multi-part question: Combines personal loan credit score and loan fees
        "inputs": {"question": "What credit score is needed for a personal loan, and what is the late payment fee if I miss the deadline?"},
        "outputs": {
            "answer": "You need a credit score of 620 or higher. The late payment fee is $39 or 5% of the payment amount, whichever is greater, charged after a 15-day grace period.",
            "intent": "policy",
        },
    },
    {
        # b) Wrong/misspelled account number format (Tests fallback string logic in Account Agent)
        "inputs": {"question": "Can you check the balance for my account AC-12345?"},
        "outputs": {
            "answer": "I'd be happy to help with your account! Could you please provide your account number? It starts with 'ACC-' followed by digits (e.g., ACC-12345).",
            "intent": "account_status",
        },
    },
    {
        # c) Boundary-case policy question: Exact fee waiver threshold limit ($1,500 balance boundary)
        "inputs": {"question": "If my Premium Checking account balance falls exactly to $1,500, will the monthly fee be waived?"},
        "outputs": {
            "answer": "Yes, the $12.99 monthly fee is waived if the daily balance stays above $1,500 or with a direct deposit of $500 or more per month.",
            "intent": "policy",
        },
    },
]

# Step 2: Look up the destination dataset and push the new evaluation vectors
existing = list(client.list_datasets(dataset_name=EXERCISE_DATASET_NAME))
if existing:
    client.create_examples(
        inputs=[e["inputs"] for e in new_examples],
        outputs=[e["outputs"] for e in new_examples],
        dataset_id=existing[0].id,
    )
    # Step 3: Print validation confirmation to standard output
    print(f"Successfully uploaded {len(new_examples)} edge-case examples to dataset: '{EXERCISE_DATASET_NAME}'")
else:
    print(f"Error: Dataset '{EXERCISE_DATASET_NAME}' could not be located in LangSmith. Run ensure_exercise_dataset() first.")


print("Complete TODO 9 to add new examples to the dataset.\n")

sys.exit()
"""

# ===================================================================
# SEGMENT 12: Hill Climbing
# ===================================================================

# ---------------------------------------------------------------------------
# TODO 10: Hill climbing — improve correctness by changing ONE variable
#
# The demo showed hill climbing on keyword_correctness by changing chunk_size.
# Now you'll hill-climb on a CORRECTNESS evaluator by changing top_k.
#
# This uses a separate LangSmith dataset (fintech-exercise-hill-climb) with
# 8 policy questions that require precise factual answers.
#
# IMPORTANT: Both agents use chunk_size=200 (tiny fragments) so that
# top_k actually matters. With large chunks, even top_k=1 has enough info.
# With tiny chunks, top_k=1 gets one incomplete fragment while top_k=5
# assembles a more complete picture.
#
# Steps:
#   1. Create the hill climbing dataset (provided — just call the helper)
#   2. Implement a correctness_evaluator (LLM-as-judge): compare the agent's
#      answer to the expected answer. Score 1.0 = all key facts correct,
#      0.5 = partial, 0.0 = wrong/missing facts.
#      Return: {"key": "correctness", "score": float}
#      Hint: Use judge_llm with a ChatPromptTemplate, parse JSON response.
#   3. Build a baseline agent with chunk_size=200, top_k=1
#   4. Run evaluate() with [routing_evaluator, keyword_correctness,
#      correctness_evaluator] on the hill-climb dataset.
#      Use experiment_prefix="exercise-hc-topk1".
#   5. Build an improved agent with chunk_size=200, top_k=5
#   6. Run evaluate() with the same evaluators.
#      Use experiment_prefix="exercise-hc-topk5".
#   7. Compare in LangSmith: Datasets → fintech-exercise-hill-climb → Compare
#
# Why this works:
#   chunk_size=200: documents are split into tiny fragments (incomplete facts)
#   top_k=1 + small chunks: only ONE tiny fragment → LLM misses key details
#   top_k=5 + small chunks: FIVE fragments → more context → better correctness
# ---------------------------------------------------------------------------
print("=" * 60)
print("SEGMENT 12: HILL CLIMBING")
print("=" * 60)

# --- Create the hill climbing dataset ---
ensure_exercise_hc_dataset()
judge_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# --- Evaluators from the demo (provided) ---
def routing_evaluator_hc(run, example):
    predicted = run.outputs.get("intent", "")
    expected = example.outputs.get("intent", "")
    return {"key": "routing_accuracy", "score": 1.0 if predicted == expected else 0.0}

def keyword_correctness_hc(run, example):
    import re
    actual = run.outputs.get("answer", "").lower()
    expected = example.outputs.get("answer", "").lower()
    key_terms = re.findall(r"\$[\d,.]+|\d+(?:\.\d+)?%?|acc-\d+", expected)
    if not key_terms:
        return {"key": "keyword_correctness", "score": 0.5}
    matches = sum(1 for term in key_terms if term in actual)
    return {"key": "keyword_correctness", "score": round(matches / len(key_terms), 4)}

def correctness_evaluator_hc(run, example):
    actual = run.outputs.get("answer", "")
    expected = example.outputs.get("answer", "")
    question = example.inputs.get("question", "")

    # Handle edge case if either response value is missing completely
    if not actual or not expected:
        return {"key": "correctness", "score": 0.0, "comment": "Missing response text to evaluate."}

    # 1. Build a ChatPromptTemplate that asks judge_llm to compare actual vs expected
    CORRECTNESS_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
            "You are an objective evaluation judge. Compare the AI's actual answer to the ground-truth expected answer.\n\n"
            "Rules:\n"
            "- Ignore exact wording, formatting, or stylistic variations.\n"
            "- Focus entirely on factual accuracy and completeness regarding numerical values, criteria, or policy thresholds.\n\n"
            "Scoring Guide:\n"
            "Score 1.0 = All key facts, limits, rates, or numbers match the expected answer correctly.\n"
            "Score 0.5 = The answer is partially correct but misses critical constraints or details.\n"
            "Score 0.0 = The key facts are flat-out wrong, missing, or contradictory.\n\n"
            'Respond ONLY with valid JSON inside a codeblock matching this layout: {{"score": <float>, "reason": "<one sentence>"}}'
        ),
        ("human",
            "Customer Question: {question}\n\n"
            "Expected Ground Truth: {expected}\n\n"
            "AI Actual Output: {actual}"
        ),
    ])

    # Format values into prompt structure
    messages = CORRECTNESS_PROMPT.format_messages(
        question=question, expected=expected, actual=actual
    )

    # 2. Invoke the deterministic judge_llm
    response = judge_llm.invoke(messages).content.strip()

    # 3. Parse JSON response and safely pull variables out
    try:
        start, end = response.find("{"), response.rfind("}") + 1
        parsed = json.loads(response[start:end])
        score = float(parsed.get("score", 0.0))
        reason = parsed.get("reason", "")
        
        # 4. Return the standard LangSmith result vector
        return {"key": "correctness", "score": score, "comment": reason}
    except (json.JSONDecodeError, ValueError, IndexError):
        # Fallback security blanket if the LLM output corrupts structural boundaries
        return {"key": "correctness", "score": 0.0, "comment": "Failed to parse evaluator JSON structure."}


# --- YOUR CODE: Build agents and run experiments ---
# Step 3: Build baseline (chunk_size=200, top_k=1)
agent_v1 = build_support_agent(collection_name="hill_climb_v1", chunk_size=200, chunk_overlap=20, top_k=1)
app_v1 = agent_v1["app"]

def run_agent_v1(inputs):
    result = ask(app_v1, inputs["question"])
    return {"answer": result["response"], "intent": result["intent"],
            "context": result["context"], "retrieved_sources": result["retrieved_sources"]}
#
# Step 4: Run baseline evaluate()
evaluate(run_agent_v1, data=EXERCISE_HC_DATASET_NAME,
         evaluators=[routing_evaluator_hc, keyword_correctness_hc, correctness_evaluator_hc],
         experiment_prefix="exercise-hc-topk1",
         metadata={"chunk_size": 200, "top_k": 1})
#
# Step 5: Build improved (chunk_size=200, top_k=5)
agent_v2 = build_support_agent(collection_name="hill_climb_v2", chunk_size=200, chunk_overlap=20, top_k=5)
app_v2 = agent_v2["app"]
#
def run_agent_v2(inputs): 
    result = ask(app_v2, inputs["question"])
    return {
        "answer": result["response"], 
        "intent": result["intent"],
        "context": result["context"], 
        "retrieved_sources": result["retrieved_sources"]
    }
#
# Step 6: Run improved evaluate()
evaluate(run_agent_v1, data=EXERCISE_HC_DATASET_NAME,
         evaluators=[routing_evaluator_hc, keyword_correctness_hc, correctness_evaluator_hc],
         experiment_prefix="exercise-hc-topk5",
         metadata={"chunk_size": 200, "top_k": 5})

print("Complete TODO 10 to run hill climbing experiment.")
