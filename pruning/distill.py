"""Distillation for the recovery fine-tune: the unpruned model is the (frozen) teacher of the pruned student.

Loss = alpha * T^2 * KL(teacher || student) + (1 - alpha) * the usual cross-entropy, over non-padded target
positions. A vocabulary-pruned student lives in a smaller id space: its ids are mapped back to original ids
for the teacher, and the teacher's logits are gathered at the kept ids so both distributions match.
"""
import torch
import torch.nn.functional as F
from transformers import Seq2SeqTrainer


def distill_loss(student_logits, teacher_logits, labels, ce, alpha=0.5, temp=2.0):
    mask = labels != -100
    student = F.log_softmax(student_logits[mask] / temp, dim=-1)
    teacher = F.softmax(teacher_logits[mask] / temp, dim=-1)
    kl = F.kl_div(student, teacher, reduction="batchmean")  # mean over the masked positions
    return alpha * temp**2 * kl + (1 - alpha) * ce


class DistillTrainer(Seq2SeqTrainer):
    """Seq2SeqTrainer whose loss also matches `teacher` (already on the student's device)."""

    def __init__(self, *args, teacher, kept=None, alpha=0.5, temp=2.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher.eval().requires_grad_(False)
        self.kept = None if kept is None else torch.as_tensor(kept)
        self.alpha, self.temp = alpha, temp

    def compute_loss(self, model, inputs, return_outputs=False, **_):
        outputs = model(**inputs)
        labels, ids = inputs["labels"], inputs["input_ids"]
        kept = None if self.kept is None else self.kept.to(ids.device)
        old_ids, old_labels = ids, labels
        if kept is not None:
            old_ids, old_labels = kept[ids], torch.where(labels == -100, labels, kept[labels.clamp(min=0)])
        with torch.no_grad():
            teacher = self.teacher(input_ids=old_ids, attention_mask=inputs["attention_mask"], labels=old_labels).logits
        if kept is not None:
            teacher = teacher[..., kept]
        loss = distill_loss(outputs.logits, teacher, labels, outputs.loss, self.alpha, self.temp)
        return (loss, outputs) if return_outputs else loss
