import pytest
import torch
from datasets import Dataset

import distill
import surgery
import train_t5
import vocab


def test_distill_loss_is_plain_ce_share_when_student_matches_teacher():
    logits = torch.randn(2, 3, 5)
    labels = torch.tensor([[1, 2, -100], [3, -100, -100]])
    loss = distill.distill_loss(logits, logits.clone(), labels, ce=torch.tensor(2.0), alpha=0.5, temp=2.0)
    assert float(loss) == pytest.approx(1.0, abs=1e-6)  # KL = 0, so only (1 - alpha) * ce remains


def test_distill_loss_ignores_padding_and_grows_with_disagreement():
    labels = torch.tensor([[1, -100]])
    teacher = torch.tensor([[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]]])
    student = torch.tensor([[[0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]])
    base = distill.distill_loss(student, teacher, labels, torch.tensor(0.0))
    changed_pad = student.clone()
    changed_pad[0, 1] = torch.tensor([9.0, 9.0, 9.0])  # padded position: must not matter
    assert float(distill.distill_loss(changed_pad, teacher, labels, torch.tensor(0.0))) == pytest.approx(float(base))
    assert float(base) > 0


def train_rows(rows):
    return Dataset.from_list([{"spoken": r["spoken"], "target": r["target"]} for r in rows])


def test_trainer_step_with_full_vocab_teacher(model, make_model, tok, rows, tmp_path):
    teacher = make_model(seed=1)
    trainer = train_t5.fit(model, tok, train_rows(rows), tmp_path, 1, 2, 1e-3, 16, 16, save=False,
                           trainer_cls=distill.DistillTrainer, teacher=teacher)
    assert trainer.state.global_step == 2 and not any(p.requires_grad for p in teacher.parameters())


def test_trainer_step_maps_ids_for_a_vocab_pruned_student(model, make_model, tok, rows, tmp_path):
    teacher = make_model(seed=1)
    kept = vocab.kept_ids(tok, rows)
    student_tok = vocab.apply(model, tok, kept)
    trainer = train_t5.fit(model, student_tok, train_rows(rows), tmp_path, 1, 2, 1e-3, 16, 16, save=False,
                           trainer_cls=distill.DistillTrainer, teacher=teacher, kept=kept)
    assert trainer.state.global_step == 2 and surgery.spec_of(model)["vocab"] == len(kept)
    batch = next(iter(trainer.get_train_dataloader()))
    assert torch.isfinite(trainer.compute_loss(model, batch))
    loss, outputs = trainer.compute_loss(model, batch, return_outputs=True)
    assert outputs.logits.shape[-1] == len(kept) and torch.isfinite(loss)
