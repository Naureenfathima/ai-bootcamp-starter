"""
anti_rag/structured_knowledge.py — SQL / API lookups instead of vector search.

The core insight: if your data is already structured (a database, a REST API,
a spreadsheet), embedding it into a vector store is often the wrong move.

Instead, use Text-to-SQL or Text-to-API: let the LLM write a precise query
against the source of truth, then execute it and return the results.

RAG limitation this solves:
  RAG: "Retrieve chunks similar to 'total revenue Q4'" → maybe gets it, maybe not
  SQL: "SELECT SUM(amount) FROM orders WHERE quarter = 'Q4'" → always correct

Patterns covered here:
  1. Text-to-SQL  — LLM writes SQL, you execute it, LLM explains results
  2. Text-to-API  — LLM writes an API call spec, you execute it
  3. Schema-first — give LLM the schema upfront for accurate query generation

STUDENT TODO:
  - Connect _execute_sql() to a real database (SQLite for dev, PostgreSQL for prod).
  - Add query validation: check for dangerous SQL (DROP, DELETE) before executing.
  - Add result caching: same schema + same question = cached SQL + cached results.
  - Try Vanna AI (https://vanna.ai) for production Text-to-SQL with auto-training.
"""

import logging
import json
import re
from dataclasses import dataclass, field
from typing import Any

import anthropic
from app.config import settings

logger = logging.getLogger(__name__)


@dataclass
class StructuredQueryResult:
    """Result from a structured knowledge query."""
    answer: str
    query_generated: str      # the SQL or API call that was produced
    raw_results: Any          # raw data before LLM interpretation
    query_type: str           # "sql" | "api"
    input_tokens: int
    output_tokens: int


class StructuredKnowledgePipeline:
    """
    Text-to-SQL pipeline: translate natural language questions into SQL,
    execute against a schema, and return an LLM-interpreted answer.

    This is particularly powerful for:
      - Business intelligence ("What was our best-selling product in Q3?")
      - Operational queries ("Which users have not logged in for 30 days?")
      - Aggregations ("What is the average response time per endpoint?")

    Usage:
        schema = \"\"\"
        CREATE TABLE orders (
            id INTEGER, customer_id INTEGER, amount FLOAT,
            status TEXT, created_at TIMESTAMP
        );
        \"\"\"
        pipeline = StructuredKnowledgePipeline(schema=schema)
        result = pipeline.query("How many orders were placed this month?")
        print(result.answer)
    """

    TEXT_TO_SQL_PROMPT = """
You are an expert SQL query writer. Given a database schema and a natural language question,
write a single valid SQL SELECT query that answers the question.

Rules:
- Write ONLY the SQL query, no explanation
- Use standard SQL syntax (compatible with SQLite and PostgreSQL)
- Never use DROP, DELETE, UPDATE, INSERT, or any data-modifying statements
- If the question cannot be answered from the schema, write: -- CANNOT_ANSWER

Schema:
{schema}

Question: {question}

SQL Query:"""

    INTERPRET_RESULTS_PROMPT = """
You are a data analyst. Given the original question, the SQL query used, and the results,
provide a clear, concise answer in plain English.

Question: {question}
SQL Query: {sql}
Results: {results}

Answer:"""

    def __init__(self, schema: str = ""):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._schema = schema or self._default_demo_schema()
        logger.info("StructuredKnowledgePipeline initialised")

    def _default_demo_schema(self) -> str:
        """A sample schema for demonstration purposes."""
        return """
CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    category TEXT,
    price FLOAT,
    stock_count INTEGER
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    product_id INTEGER REFERENCES products(id),
    customer_email TEXT,
    quantity INTEGER,
    total_amount FLOAT,
    status TEXT,  -- 'pending' | 'shipped' | 'delivered' | 'cancelled'
    created_at TIMESTAMP
);

CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    email TEXT UNIQUE NOT NULL,
    name TEXT,
    plan TEXT,   -- 'free' | 'pro' | 'enterprise'
    created_at TIMESTAMP,
    last_active_at TIMESTAMP
);
"""

    def update_schema(self, schema: str) -> None:
        """Update the database schema the pipeline reasons over."""
        self._schema = schema
        logger.info("StructuredKnowledgePipeline: schema updated (%d chars)", len(schema))

    def generate_sql(self, question: str) -> str:
        """
        Use Claude to translate a natural language question into SQL.

        Returns:
            A SQL SELECT query string.
        """
        prompt = self.TEXT_TO_SQL_PROMPT.format(
            schema=self._schema,
            question=question
        )

        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )

        sql = response.content[0].text.strip()

        # Strip markdown code fences if present
        sql = re.sub(r"```sql\s*", "", sql)
        sql = re.sub(r"```\s*", "", sql)

        logger.info("Text-to-SQL: '%s...' → %s", question[:50], sql[:80])
        return sql

    def _execute_sql(self, sql: str) -> Any:
        """
        Execute a SQL query and return results.

        STUDENT TODO: Replace this stub with a real database connection.

        Example with SQLite:
            import sqlite3
            conn = sqlite3.connect("data/demo.db")
            cursor = conn.execute(sql)
            return cursor.fetchall()

        Example with PostgreSQL (psycopg2):
            import psycopg2
            conn = psycopg2.connect(settings.database_url)
            cursor = conn.cursor()
            cursor.execute(sql)
            return cursor.fetchall()
        """
        # Demo stub — returns simulated data based on keywords in the SQL
        sql_lower = sql.lower()
        if "-- cannot_answer" in sql_lower:
            return {"error": "This question cannot be answered from the available schema."}
        if "count" in sql_lower and "order" in sql_lower:
            return [{"count": 142}]
        if "sum" in sql_lower and "amount" in sql_lower:
            return [{"total": 28450.75}]
        if "avg" in sql_lower:
            return [{"average": 187.32}]
        if "select" in sql_lower and "product" in sql_lower:
            return [
                {"name": "AI Course Pro", "category": "education", "price": 299.0},
                {"name": "API Starter Pack", "category": "tools", "price": 49.0},
            ]
        return [{"result": "stub — connect a real database to get live results"}]

    def query(self, question: str) -> StructuredQueryResult:
        """
        Full pipeline: question → SQL → execute → interpret → answer.

        Args:
            question: Natural language question about the data.

        Returns:
            StructuredQueryResult with answer, generated SQL, and raw data.
        """
        # Step 1: Generate SQL
        sql = self.generate_sql(question)

        # Step 2: Execute
        raw_results = self._execute_sql(sql)

        # Step 3: Interpret results in plain English
        interpret_prompt = self.INTERPRET_RESULTS_PROMPT.format(
            question=question,
            sql=sql,
            results=json.dumps(raw_results, indent=2)
        )

        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=512,
            messages=[{"role": "user", "content": interpret_prompt}]
        )

        answer = response.content[0].text.strip()

        return StructuredQueryResult(
            answer=answer,
            query_generated=sql,
            raw_results=raw_results,
            query_type="sql",
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
