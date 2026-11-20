"""Answer policy for QA: source verification, scope checks, refusal handling, and multi-image aggregation."""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from .qa_client import ModelOutput, QAInvalidOutputError
from .qa_evidence import EvidenceInput, LoadedImageBundle
from .schemas import QAAlternativeItem, QAEvidenceItem, QALanguage, QAStatus

_UNSUPPORTED_SCOPE_RE = re.compile(
    r"(?:"
    r"tổng\s+cộng\s+bao\s+nhiêu\s+lần|"
    r"trong\s+(?:cả|toàn\s+bộ)\s+(?:ngày|bộ\s+dữ\s+liệu|kho\s+ảnh|tập\s+ảnh|5000\s+ảnh|5\.000\s+ảnh)|"
    r"toàn\s+bộ\s+(?:5000|5\.000)?\s*ảnh|"
    r"across\s+(?:all|the\s+entire|the\s+whole)\s+(?:images|dataset|corpus|day)|"
    r"in\s+the\s+(?:entire|whole)\s+(?:dataset|corpus|collection)|"
    r"total\s+(?:number\s+of\s+)?times\s+in\s+the\s+(?:day|dataset|corpus)|"
    r"never\s+appears\s+in\s+the\s+(?:entire\s+)?(?:dataset|corpus)"
    r")",
    re.IGNORECASE,
)


def is_unsupported_corpus_scope(question: str) -> bool:
    """Detect questions that ask for corpus-wide totals or exhaustive negative conclusions."""
    return bool(_UNSUPPORTED_SCOPE_RE.search(question))


def normalize_answer_key(text: str) -> str:
    """Normalize an answer string for comparing agreement across candidate images."""
    norm = unicodedata.normalize("NFC", text).casefold().strip()
    norm = re.sub(r"^[\s.,;:!?\"'()\[\]\-–—]+|[\s.,;:!?\"'()\[\]\-–—]+$", "", norm)
    norm = re.sub(r"\s+", " ", norm)
    return norm


@dataclass
class PolicyDecision:
    status: QAStatus
    answer: str | None = None
    reason_code: str | None = None
    message: str | None = None
    evidence: list[QAEvidenceItem] = field(default_factory=list)
    alternatives: list[QAAlternativeItem] = field(default_factory=list)
    note: str | None = None
    note_evidence: list[QAEvidenceItem] = field(default_factory=list)


def _reason_message(reason_code: str | None, language: QALanguage) -> str:
    messages_vi = {
        "not_visible": "Ảnh chưa đủ rõ để xác định.",
        "no_relevant_image": "Ảnh chưa đủ rõ để xác định.",
        "unsupported_scope": "Câu hỏi vượt phạm vi ảnh đang xét (không thống kê trên toàn bộ bộ dữ liệu).",
        "multiple_subjects": "Có nhiều đối tượng phù hợp trong ảnh, bạn muốn hỏi về đối tượng nào?",
        "conflicting_candidates": "Nhiều ảnh cho kết quả khác nhau, bạn hãy chọn một ảnh bằng chứng để hỏi tiếp.",
        "missing_context": "Chưa xác định được ảnh cần hỏi. Bạn hãy chọn một ảnh hoặc mô tả rõ hơn ngữ cảnh.",
        "partial_budget_exceeded": "Đã hết thời gian xét thêm ảnh ứng viên; kết quả dưới đây chỉ mang tính tạm thời.",
    }
    messages_en = {
        "not_visible": "The image is not clear enough to determine.",
        "no_relevant_image": "The image is not clear enough to determine.",
        "unsupported_scope": "Question scope exceeds the examined image(s).",
        "multiple_subjects": "Multiple subjects match in the image; which one are you asking about?",
        "conflicting_candidates": "Multiple images give different answers; please pick an evidence image to continue.",
        "missing_context": "Please select an image or describe the scene more specifically.",
        "partial_budget_exceeded": "Time budget expired before checking all candidates; provisional result shown.",
    }
    table = messages_vi if language == "vi" else messages_en
    return table.get(reason_code or "not_visible", table["not_visible"])


class AnswerPolicy:
    """Validates model output against server-controlled evidence IDs and enforces refusal/ambiguity rules."""

    def __init__(self, ground_crops: bool = False):
        # Per QA_SYSTEM_DESIGN.md Section 3.7: if grounding is not deployed, box=null.
        # When ground_crops=True, crop evidence preserves its verified crop bounding box.
        self._ground_crops = ground_crops

    def check_question_scope(
        self,
        question: str,
        language: QALanguage,
        default_image_id: str | None = None,
    ) -> PolicyDecision | None:
        if is_unsupported_corpus_scope(question):
            ev = [QAEvidenceItem(image_id=default_image_id, box=None, source="image")] if default_image_id else []
            return PolicyDecision(
                status="insufficient_evidence",
                answer=None,
                reason_code="unsupported_scope",
                message=_reason_message("unsupported_scope", language),
                evidence=ev,
            )
        return None

    def map_and_verify_evidence(
        self,
        model_output: ModelOutput,
        bundle: LoadedImageBundle,
    ) -> list[QAEvidenceItem]:
        by_id: dict[str, EvidenceInput] = {item.evidence_id: item for item in bundle.items}
        for eid in model_output.evidence_ids:
            if eid not in by_id:
                raise QAInvalidOutputError(f"Mô hình trả về evidence_id không hợp lệ: {eid!r}")

        if model_output.answerability == "answerable" and not model_output.evidence_ids:
            raise QAInvalidOutputError("Đáp án answerable nhưng không có evidence_ids")

        mapped: list[QAEvidenceItem] = []
        seen: set[tuple[str, tuple[float, ...] | None, str]] = set()
        for eid in model_output.evidence_ids:
            src = by_id[eid]
            box = src.box if (self._ground_crops and src.kind == "crop") else None
            source_type = src.source if (self._ground_crops or src.source == "metadata") else "image"
            dedup_key = (src.image_id, tuple(box) if box else None, source_type)
            if dedup_key not in seen:
                seen.add(dedup_key)
                mapped.append(QAEvidenceItem(image_id=src.image_id, box=box, source=source_type))

        if not mapped:
            mapped.append(QAEvidenceItem(image_id=bundle.image_id, box=None, source="image"))
        return mapped

    def evaluate_single(
        self,
        model_output: ModelOutput,
        bundle: LoadedImageBundle,
        language: QALanguage,
    ) -> PolicyDecision:
        mapped_evidence = self.map_and_verify_evidence(model_output, bundle)

        if model_output.context_match != "yes":
            reason = model_output.reason_code or "no_relevant_image"
            return PolicyDecision(
                status="insufficient_evidence",
                answer=None,
                reason_code=reason,
                message=_reason_message(reason, language),
                evidence=[QAEvidenceItem(image_id=bundle.image_id, box=None, source="image")],
            )

        if model_output.answerability == "insufficient":
            reason = model_output.reason_code or "not_visible"
            return PolicyDecision(
                status="insufficient_evidence",
                answer=None,
                reason_code=reason,
                message=_reason_message(reason, language),
                evidence=mapped_evidence,
            )

        if model_output.answerability == "ambiguous":
            reason = model_output.reason_code or "multiple_subjects"
            msg = model_output.clarification or _reason_message(reason, language)
            return PolicyDecision(
                status="ambiguous",
                answer=None,
                reason_code=reason,
                message=msg,
                evidence=mapped_evidence,
                alternatives=[],
            )

        clean_answer = (model_output.answer or "").strip()
        if not clean_answer:
            raise QAInvalidOutputError("Đáp án rỗng khi trạng thái là answerable")

        return PolicyDecision(
            status="answered",
            answer=clean_answer,
            reason_code=None,
            message=None,
            evidence=mapped_evidence,
            alternatives=[],
        )

    def is_confident_answer(self, model_output: ModelOutput) -> bool:
        """True only when context_match is explicitly 'yes' and answerability is 'answerable' with a non-empty answer."""
        if model_output.context_match != "yes" or model_output.answerability != "answerable":
            return False
        clean_ans = (model_output.answer or "").strip()
        return bool(clean_ans and normalize_answer_key(clean_ans))

    def evaluate_early_stop(
        self,
        primary: tuple[LoadedImageBundle, ModelOutput] | None,
        confirmations: list[tuple[LoadedImageBundle, ModelOutput]],
        evaluated: list[tuple[LoadedImageBundle, ModelOutput]],
        language: QALanguage,
        partial: bool = False,
    ) -> PolicyDecision:
        """Verified early-stop policy: always returns 1 concise answer (when a confident candidate exists)
        while checking confirmation candidates to surface conflicts via a compact note."""
        # Validate evidence IDs across all evaluated outputs first (anti-hallucination invariant)
        verified_map: dict[int, list[QAEvidenceItem]] = {}
        for idx, (bundle, out) in enumerate(evaluated):
            verified_map[idx] = self.map_and_verify_evidence(out, bundle)

        if primary is not None:
            prim_bundle, prim_out = primary
            prim_ev = self.map_and_verify_evidence(prim_out, prim_bundle)
            prim_ans = (prim_out.answer or "").strip()
            prim_key = normalize_answer_key(prim_ans)

            conflicting_ev: list[QAEvidenceItem] = []
            seen_conf_ids: set[str] = {e.image_id for e in prim_ev}

            for conf_bundle, conf_out in confirmations:
                conf_ev = self.map_and_verify_evidence(conf_out, conf_bundle)
                if self.is_confident_answer(conf_out):
                    conf_ans = (conf_out.answer or "").strip()
                    conf_key = normalize_answer_key(conf_ans)
                    if conf_key == prim_key:
                        # Agreement: merge supporting evidence
                        for e in conf_ev:
                            if e.image_id not in seen_conf_ids:
                                prim_ev.append(e)
                                seen_conf_ids.add(e.image_id)
                    else:
                        # Conflict detected on confirmation candidate
                        for e in conf_ev:
                            if e.image_id not in seen_conf_ids:
                                conflicting_ev.append(e)
                                seen_conf_ids.add(e.image_id)

            note: str | None = None
            note_ev: list[QAEvidenceItem] = []
            if conflicting_ev:
                return PolicyDecision(status="ambiguous", answer=None,
                    reason_code="conflicting_candidates",
                    message=_reason_message("conflicting_candidates", language),
                    evidence=prim_ev + conflicting_ev)
            if partial and not confirmations:
                note = (
                    "Chưa kịp đối chiếu thêm ảnh khác do giới hạn thời gian."
                    if language == "vi"
                    else "Did not finish checking additional confirmation images due to time budget."
                )

            return PolicyDecision(
                status="answered",
                answer=prim_ans,
                reason_code=None,
                message=None,
                evidence=prim_ev,
                alternatives=[],
                note=note,
                note_evidence=note_ev,
            )

        # No confident primary answer found across evaluated candidates
        for idx, (bundle, out) in enumerate(evaluated):
            if out.context_match == "no":
                continue
            if out.answerability == "ambiguous":
                ev_list = verified_map[idx]
                msg = out.clarification or _reason_message("multiple_subjects", language)
                return PolicyDecision(
                    status="ambiguous",
                    answer=None,
                    reason_code="multiple_subjects",
                    message=msg,
                    evidence=ev_list,
                    alternatives=[],
                )

        for idx, (bundle, out) in enumerate(evaluated):
            if out.context_match != "no":
                return PolicyDecision(
                    status="insufficient_evidence",
                    answer=None,
                    reason_code="not_visible",
                    message=_reason_message("not_visible", language),
                    evidence=verified_map[idx],
                    alternatives=[],
                )

        return PolicyDecision(
            status="insufficient_evidence",
            answer=None,
            reason_code="no_relevant_image",
            message=_reason_message("no_relevant_image", language),
            evidence=[],
            alternatives=[],
        )

    def evaluate_candidates(
        self,
        evaluated: list[tuple[LoadedImageBundle, ModelOutput]],
        language: QALanguage,
        partial: bool = False,
    ) -> PolicyDecision:
        if not evaluated:
            return PolicyDecision(
                status="insufficient_evidence",
                answer=None,
                reason_code="no_relevant_image",
                message=_reason_message("no_relevant_image", language),
                evidence=[],
            )

        answer_groups: dict[str, tuple[str, list[QAEvidenceItem]]] = {}
        ambiguous_items: list[QAAlternativeItem] = []
        ambiguous_evidence: list[QAEvidenceItem] = []
        relevant_insufficient_evidence: list[QAEvidenceItem] = []

        for bundle, out in evaluated:
            ev_list = self.map_and_verify_evidence(out, bundle)
            if out.context_match == "no":
                continue

            if out.answerability == "answerable" and out.context_match == "yes":
                clean_ans = (out.answer or "").strip()
                if not clean_ans:
                    raise QAInvalidOutputError("Đáp án ứng viên rỗng khi answerable")
                key = normalize_answer_key(clean_ans)
                if not key:
                    raise QAInvalidOutputError("Đáp án ứng viên không hợp lệ sau chuẩn hóa")
                if key not in answer_groups:
                    answer_groups[key] = (clean_ans, list(ev_list))
                else:
                    existing_ans, existing_ev = answer_groups[key]
                    seen_ids = {e.image_id for e in existing_ev}
                    for ev in ev_list:
                        if ev.image_id not in seen_ids:
                            existing_ev.append(ev)
                            seen_ids.add(ev.image_id)
            elif out.answerability == "ambiguous":
                ambiguous_evidence.extend(ev_list)
                if out.answer:
                    ambiguous_items.append(
                        QAAlternativeItem(
                            answer=out.answer.strip(),
                            evidence=ev_list,
                            clarification=out.clarification,
                        )
                    )
            else:
                relevant_insufficient_evidence.extend(ev_list)

        # Multiple distinct answers across relevant candidate images -> ambiguous (retrieval score does NOT break ties).
        if len(answer_groups) > 1 or (len(answer_groups) == 1 and ambiguous_evidence):
            alternatives: list[QAAlternativeItem] = []
            all_evidence: list[QAEvidenceItem] = []
            seen_ev_ids: set[str] = set()
            for _, (disp_ans, ev_items) in answer_groups.items():
                img_list = ", ".join(e.image_id for e in ev_items)
                clarification = (
                    f"Theo ảnh {img_list}" if language == "vi" else f"From image(s) {img_list}"
                )
                alternatives.append(
                    QAAlternativeItem(
                        answer=disp_ans,
                        evidence=ev_items,
                        clarification=clarification,
                    )
                )
                for e in ev_items:
                    if e.image_id not in seen_ev_ids:
                        all_evidence.append(e)
                        seen_ev_ids.add(e.image_id)
            for alt in ambiguous_items:
                alternatives.append(alt)
            for e in ambiguous_evidence:
                if e.image_id not in seen_ev_ids:
                    all_evidence.append(e)
                    seen_ev_ids.add(e.image_id)
            return PolicyDecision(
                status="ambiguous",
                answer=None,
                reason_code="conflicting_candidates",
                message=_reason_message("conflicting_candidates", language),
                evidence=all_evidence,
                alternatives=alternatives,
            )

        # Single agreed answer across matching candidate(s)
        if len(answer_groups) == 1:
            agreed_ans, agreed_ev = next(iter(answer_groups.values()))
            if partial:
                # Per QA_SYSTEM_DESIGN.md Section 4.7: downgrade to ambiguous when partial budget could change answer.
                clarification = (
                    "Kết quả tạm thời trước khi hết ngân sách xét ứng viên."
                    if language == "vi"
                    else "Provisional answer before candidate budget was exhausted."
                )
                return PolicyDecision(
                    status="ambiguous",
                    answer=None,
                    reason_code="partial_budget_exceeded",
                    message=_reason_message("partial_budget_exceeded", language),
                    evidence=agreed_ev,
                    alternatives=[
                        QAAlternativeItem(
                            answer=agreed_ans,
                            evidence=agreed_ev,
                            clarification=clarification,
                        )
                    ],
                )
            return PolicyDecision(
                status="answered",
                answer=agreed_ans,
                reason_code=None,
                message=None,
                evidence=agreed_ev,
                alternatives=[],
            )

        if ambiguous_evidence:
            return PolicyDecision(
                status="ambiguous",
                answer=None,
                reason_code="multiple_subjects",
                message=_reason_message("multiple_subjects", language),
                evidence=ambiguous_evidence,
                alternatives=ambiguous_items,
            )

        if relevant_insufficient_evidence:
            return PolicyDecision(
                status="insufficient_evidence",
                answer=None,
                reason_code="not_visible",
                message=_reason_message("not_visible", language),
                evidence=relevant_insufficient_evidence,
                alternatives=[],
            )

        return PolicyDecision(
            status="insufficient_evidence",
            answer=None,
            reason_code="no_relevant_image",
            message=_reason_message("no_relevant_image", language),
            evidence=[],
            alternatives=[],
        )
