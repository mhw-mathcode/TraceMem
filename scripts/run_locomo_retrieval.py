from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx

from tracemem.add_service import AddService
from tracemem.api_models import AddRequest, SearchRequest
from tracemem.benchmarks.locomo import LocomoQuestion, LocomoSample, parse_samples
from tracemem.config import Settings
from tracemem.db import Database
from tracemem.evidence import EvidencePacker
from tracemem.retrieval import HybridRetriever
from tracemem.runtime import build_model_bundle
from tracemem.search_service import SearchService


@dataclass
class MetricBucket:
    questions: int = 0
    hit5: int = 0
    hit10: int = 0
    hit30: int = 0
    evidence_found: int = 0
    evidence_total: int = 0
    reciprocal_rank_sum: float = 0.0

    def update(self, ranks: list[int], evidence_total: int) -> None:
        self.questions += 1
        self.evidence_total += evidence_total
        self.evidence_found += len(ranks)
        if ranks:
            best = min(ranks)
            self.hit5 += int(best <= 5)
            self.hit10 += int(best <= 10)
            self.hit30 += int(best <= 30)
            self.reciprocal_rank_sum += 1.0 / best

    def summary(self) -> dict[str, float | int]:
        denominator = self.questions or 1
        evidence_denominator = self.evidence_total or 1
        return {
            "questions": self.questions,
            "hit@5": round(self.hit5 / denominator, 4),
            "hit@10": round(self.hit10 / denominator, 4),
            "hit@30": round(self.hit30 / denominator, 4),
            "evidence_recall@30": round(
                self.evidence_found / evidence_denominator,
                4,
            ),
            "mrr@30": round(self.reciprocal_rank_sum / denominator, 4),
        }


def settings_for_model_mode(
    mode: str,
    base: Settings | None = None,
) -> Settings:
    settings = base or Settings()
    if mode == "offline":
        return settings.model_copy(
            update={
                "embedding_mode": "hash",
                "extraction_mode": "disabled",
                "rerank_mode": "disabled",
            }
        )
    if mode == "bailian":
        return settings.model_copy(
            update={
                "embedding_mode": "openai",
                "extraction_mode": "openai",
                "rerank_mode": "disabled",
            }
        )
    raise ValueError(f"unsupported model mode: {mode}")


def model_mode_label(settings: Settings) -> str:
    if (
        settings.embedding_mode == "hash"
        and settings.extraction_mode == "disabled"
        and settings.rerank_mode == "disabled"
    ):
        return "offline/hash-embedding/no-extraction/no-rerank"
    return (
        f"remote/{settings.embedding_model}/"
        f"{settings.llm_model}/no-rerank"
    )


async def evaluate(
    samples: list[LocomoSample],
    *,
    database_path: Path,
    max_questions: int | None,
    settings: Settings,
) -> dict:
    database = Database(database_path)
    database.initialize()
    async with httpx.AsyncClient() as client:
        models = build_model_bundle(settings, client)
        add_service = AddService(
            database=database,
            embedder=models.embedder,
            extractor=models.extractor,
            lease_seconds=settings.add_lease_seconds,
        )
        search_service = SearchService(
            retriever=HybridRetriever(
                database=database,
                embedder=models.embedder,
                cards_enabled=settings.cards_enabled,
                state_enabled=settings.temporal_enabled,
                per_channel_limit=90,
            ),
            reranker=models.reranker,
            packer=EvidencePacker(
                pairing_enabled=settings.evidence_pairing_enabled
            ),
            final_result_limit=settings.final_result_limit,
        )

        overall = MetricBucket()
        categories: dict[str, MetricBucket] = defaultdict(MetricBucket)
        skipped_without_evidence = 0
        processed_questions = 0
        sample_index = 0

        for sample_index, sample in enumerate(samples, start=1):
            for session in sample.sessions:
                request = AddRequest.model_validate(
                    {
                        "request_id": (
                            f"locomo:{sample.sample_id}:session:{session.index}"
                        ),
                        "messages": [
                            {
                                "role": message.role,
                                "timestamp": session.timestamp,
                                "content": message.content,
                            }
                            for message in session.messages
                        ],
                        "user_id": f"locomo:{sample.sample_id}",
                        "session_id": (
                            f"locomo:{sample.sample_id}:session:{session.index}"
                        ),
                    }
                )
                await add_service.add(request)

            for question in sample.questions:
                if (
                    max_questions is not None
                    and processed_questions >= max_questions
                ):
                    break
                if not question.evidence_ids:
                    skipped_without_evidence += 1
                    continue
                processed_questions += 1
                response = await search_service.search(
                    SearchRequest(
                        query=question.question,
                        user_id=f"locomo:{sample.sample_id}",
                        top_k=90,
                    )
                )
                ranks = evidence_ranks(sample, question, response)
                evidence_total = sum(
                    evidence_id in sample.evidence_text
                    for evidence_id in question.evidence_ids
                )
                overall.update(ranks, evidence_total)
                categories[question.category].update(ranks, evidence_total)

            print(
                f"[{sample_index}/{len(samples)}] {sample.sample_id}: "
                f"{len(sample.sessions)} sessions, "
                f"{len(sample.questions)} questions",
                file=sys.stderr,
                flush=True,
            )
            if (
                max_questions is not None
                and processed_questions >= max_questions
            ):
                break

    return {
        "mode": model_mode_label(settings),
        "samples": sample_index,
        "questions_with_evidence": processed_questions,
        "questions_without_evidence_skipped": skipped_without_evidence,
        "overall": overall.summary(),
        "by_category": {
            category: bucket.summary()
            for category, bucket in sorted(categories.items())
        },
    }


def evidence_ranks(sample, question, response) -> list[int]:
    ranks: list[int] = []
    for evidence_id in question.evidence_ids:
        evidence = sample.evidence_text.get(evidence_id)
        if not evidence:
            continue
        for rank, result in enumerate(response.data, start=1):
            if evidence in result.content:
                ranks.append(rank)
                break
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dataset",
        type=Path,
        help="Path to locomo10.json or compatible LoCoMo JSON",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--max-questions", type=int)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--model-mode",
        choices=("offline", "bailian"),
        default="offline",
    )
    args = parser.parse_args()

    payload = json.loads(args.dataset.read_text(encoding="utf-8"))
    samples = parse_samples(payload)[: args.samples]
    settings = settings_for_model_mode(args.model_mode)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (
        Path("outputs")
        / f"locomo_{args.model_mode}_{timestamp}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    database_path = output.with_suffix(".db")
    summary = asyncio.run(
        evaluate(
            samples,
            database_path=database_path,
            max_questions=args.max_questions,
            settings=settings,
        )
    )
    output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary: {output}", file=sys.stderr)
    print(f"database: {database_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
