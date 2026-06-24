from transformers import Trainer
import torch


class CETrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        inputs.pop("image_path", None)
        inputs.pop("path", None)
        inputs.pop("paths", None)

        outputs = model(**inputs)
        logits = outputs.logits

        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(logits, labels, weight=weight)

        return (loss, outputs) if return_outputs else loss


class FocalTrainer(Trainer):
    def __init__(self, *args, focal_loss_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.focal_loss_fn = focal_loss_fn

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        inputs.pop("image_path", None)
        inputs.pop("path", None)
        inputs.pop("paths", None)

        outputs = model(**inputs)
        logits = outputs.logits

        if self.focal_loss_fn is not None:
            loss = self.focal_loss_fn(logits, labels)
        else:
            loss = torch.nn.functional.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss