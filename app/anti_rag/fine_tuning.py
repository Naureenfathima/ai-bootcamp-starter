"""
anti_rag/fine_tuning.py — Fine-tuning as an alternative to RAG.

Fine-tuning bakes knowledge directly into model weights. It is a different
tradeoff from RAG:

  RAG:         retrieves at inference time → fresh data, higher latency
  Fine-tuning: knowledge baked in at training time → stale but fast

This module covers:
  1. When to fine-tune vs. when to use RAG
  2. Dataset preparation patterns (instruction-tuning format)
  3. LoRA / PEFT concepts (parameter-efficient fine-tuning)
  4. Evaluation of fine-tuned models
  5. A practical dataset builder you can use with Claude to generate training data

Key point for students:
  In this course we cover fine-tuning concepts and KAG (Knowledge Augmented
  Generation) as part of the "anti-RAG" module. We use LoRA/PEFT conceptually
  — the hands-on focus is on when and why to choose each approach, and on
  dataset preparation which is the biggest practical bottleneck.

STUDENT TODO:
  - Generate a fine-tuning dataset from your domain using TrainingDataBuilder.
  - Try fine-tuning a small model (Mistral-7B) on Google Colab using LoRA.
  - Evaluate the fine-tuned model with the harness module (app/harness/).
"""

import logging
import json
from dataclasses import dataclass, field
from typing import Literal

import anthropic
from app.config import settings

logger = logging.getLogger(__name__)


# ── Decision Framework ─────────────────────────────────────────────────────────
FINE_TUNING_VS_RAG = """
Fine-tuning vs. RAG — Decision Framework
=========================================

Use FINE-TUNING when:
  ✓ Knowledge is stable (doesn't change often)
  ✓ You need consistent tone/style/format (brand voice)
  ✓ You want to reduce latency (no retrieval step)
  ✓ You want to reduce cost (smaller model after fine-tuning)
  ✓ Domain is narrow and well-defined (legal, medical Q&A)
  ✓ You have > 1000 high-quality examples

Use RAG when:
  ✓ Data changes frequently (news, product catalog, tickets)
  ✓ You need source citations
  ✓ You want to add knowledge without retraining
  ✓ Data is too large to fit in fine-tuning context
  ✓ You need to answer questions about private docs

Hybrid approach (often best):
  ✓ Fine-tune for tone, format, domain reasoning
  ✓ RAG for current facts and citations
  ✓ KAG for structured multi-hop reasoning

LoRA / PEFT (Parameter-Efficient Fine-Tuning):
  - Full fine-tuning updates ALL weights → expensive
  - LoRA freezes the base model and trains small low-rank adapter matrices
  - Typical LoRA rank: r=8 to r=64 (higher = more capacity, more VRAM)
  - Typical LoRA alpha: 2x rank (e.g. alpha=16 for r=8)
  - Reduces trainable parameters by 10,000x vs full fine-tuning
  - Tools: HuggingFace PEFT, Axolotl, Unsloth (4x faster)
"""


# ── Training Data Structures ──────────────────────────────────────────────────
@dataclass
class TrainingExample:
    """
    A single training example in the instruction-tuning format.
    Compatible with Alpaca, ShareGPT, and HuggingFace datasets.
    """
    instruction: str    # what the model should do
    input: str          # optional context / user input
    output: str         # the ideal model response
    metadata: dict = field(default_factory=dict)

    def to_alpaca_format(self) -> dict:
        """Alpaca format used by most fine-tuning frameworks."""
        return {
            "instruction": self.instruction,
            "input": self.input,
            "output": self.output,
        }

    def to_sharegpt_format(self) -> list[dict]:
        """ShareGPT / ChatML format for multi-turn fine-tuning."""
        messages = []
        if self.instruction:
            messages.append({"from": "system", "value": self.instruction})
        if self.input:
            messages.append({"from": "human", "value": self.input})
        messages.append({"from": "gpt", "value": self.output})
        return messages


@dataclass
class TrainingDataset:
    """A collection of training examples."""
    name: str
    examples: list[TrainingExample] = field(default_factory=list)
    description: str = ""

    def add(self, example: TrainingExample) -> None:
        self.examples.append(example)

    def to_jsonl(self) -> str:
        """Export as JSONL (one JSON object per line) — standard format for fine-tuning."""
        lines = [json.dumps(ex.to_alpaca_format()) for ex in self.examples]
        return "\n".join(lines)

    def split(self, train_ratio: float = 0.8) -> tuple["TrainingDataset", "TrainingDataset"]:
        """Split into train/validation sets."""
        n_train = int(len(self.examples) * train_ratio)
        train_ds = TrainingDataset(name=f"{self.name}_train", examples=self.examples[:n_train])
        val_ds   = TrainingDataset(name=f"{self.name}_val",   examples=self.examples[n_train:])
        return train_ds, val_ds

    def __len__(self) -> int:
        return len(self.examples)


# ── Training Data Builder ─────────────────────────────────────────────────────
class TrainingDataBuilder:
    """
    Use Claude to generate synthetic training data from a domain description
    or existing documents.

    This is a practical tool: generating 500-1000 high-quality examples
    typically takes 1-2 hours and costs ~$2-5 with Claude Haiku.

    Usage:
        builder = TrainingDataBuilder()
        dataset = builder.generate_from_description(
            domain="customer support for a SaaS product",
            instruction="Answer customer support questions helpfully and concisely.",
            n_examples=10
        )
        print(dataset.to_jsonl())
    """

    GENERATION_PROMPT = """
Generate {n} diverse training examples for fine-tuning a language model.

Domain: {domain}
System instruction: {instruction}

For each example, create a realistic user question/request and an ideal response.
Return a JSON array of objects with keys: "input" (user message) and "output" (ideal response).

Requirements:
- Make examples diverse: vary question complexity, style, and topic
- Keep responses concise and accurate
- Include edge cases and clarifying questions
- Do NOT repeat similar examples

Return ONLY the JSON array, no explanation.
"""

    def __init__(self):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def generate_from_description(
        self,
        domain: str,
        instruction: str,
        n_examples: int = 10,
        dataset_name: str = "generated_dataset",
    ) -> TrainingDataset:
        """
        Generate synthetic training examples using Claude.

        Args:
            domain:       Description of the domain (e.g. "medical Q&A").
            instruction:  The system instruction for the fine-tuned model.
            n_examples:   How many examples to generate.
            dataset_name: Name for the resulting dataset.

        Returns:
            TrainingDataset ready to export as JSONL.
        """
        logger.info("Generating %d training examples for domain: %s", n_examples, domain)

        prompt = self.GENERATION_PROMPT.format(
            n=n_examples,
            domain=domain,
            instruction=instruction
        )

        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            raw_examples = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse generated examples: %s", e)
            return TrainingDataset(name=dataset_name)

        dataset = TrainingDataset(name=dataset_name, description=f"Generated for: {domain}")
        for item in raw_examples:
            example = TrainingExample(
                instruction=instruction,
                input=item.get("input", ""),
                output=item.get("output", ""),
            )
            dataset.add(example)

        logger.info("Generated %d training examples", len(dataset))
        return dataset

    def generate_from_document(
        self,
        document: str,
        instruction: str,
        n_examples: int = 10,
        dataset_name: str = "doc_dataset",
    ) -> TrainingDataset:
        """
        Generate Q&A training pairs from an existing document.
        Useful for creating domain-specific fine-tuning data from your docs.
        """
        prompt = f"""
Read the following document and generate {n_examples} question-answer pairs for fine-tuning.
Each pair should test understanding of the document's content.

System instruction for the fine-tuned model: {instruction}

Document:
{document[:4000]}

Return a JSON array of {{"input": "question", "output": "answer"}} objects.
"""
        response = self._client.messages.create(
            model=settings.claude_model,
            max_tokens=4096,
            messages=[{"role": "user", "content": prompt}]
        )

        raw = response.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            raw_examples = json.loads(raw)
        except json.JSONDecodeError:
            return TrainingDataset(name=dataset_name)

        dataset = TrainingDataset(name=dataset_name)
        for item in raw_examples:
            dataset.add(TrainingExample(
                instruction=instruction,
                input=item.get("input", ""),
                output=item.get("output", ""),
            ))

        return dataset


# ── Guide ─────────────────────────────────────────────────────────────────────
class FineTuningGuide:
    """
    A reference class that surfaces fine-tuning decision logic and LoRA concepts.
    Used in the API to answer student questions about when to fine-tune.
    """

    @staticmethod
    def decision_framework() -> str:
        return FINE_TUNING_VS_RAG

    @staticmethod
    def lora_config_recommendation(
        model_size: Literal["7b", "13b", "70b"],
        task: Literal["qa", "summarization", "classification", "chat"],
    ) -> dict:
        """
        Returns a recommended LoRA configuration for a given model size and task.
        Based on community best practices from HuggingFace and Axolotl.
        """
        base_configs = {
            "7b":  {"r": 16, "lora_alpha": 32,  "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj"]},
            "13b": {"r": 32, "lora_alpha": 64,  "lora_dropout": 0.05, "target_modules": ["q_proj", "v_proj", "k_proj"]},
            "70b": {"r": 64, "lora_alpha": 128, "lora_dropout": 0.1,  "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"]},
        }
        task_epochs = {
            "qa": 3,
            "summarization": 2,
            "classification": 5,
            "chat": 3,
        }

        config = base_configs[model_size].copy()
        config["num_train_epochs"] = task_epochs[task]
        config["per_device_train_batch_size"] = 2 if model_size == "70b" else 4
        config["gradient_accumulation_steps"] = 4
        config["learning_rate"] = 2e-4
        config["fp16"] = True
        config["task"] = task
        config["model_size"] = model_size
        config["notes"] = (
            f"Recommended for {task} on a {model_size} parameter model. "
            "Adjust r and lora_alpha based on your task complexity. "
            "Higher r = more capacity but more VRAM."
        )
        return config
