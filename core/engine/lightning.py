import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from collections import OrderedDict
from core.metrics import CARNetLoss, MetricLogger
from core.utils import get_callable_name


class LightningNet(pl.LightningModule):
    def __init__(
        self,
        cfg,
        dataloaders,
        model,
        criterion,
        loss_weights,
        metrics,
        log_histograms=False,
    ):
        super().__init__()

        self.cfg = cfg
        self.dataloaders = dataloaders

        self.model = model
        self.criterion = criterion
        self.loss_weights = loss_weights

        self.metrics = metrics
        self.log_histograms = log_histograms
        self.metric_logger = MetricLogger()

    def forward(self, inputs):
        return self.model(inputs)

    def training_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        if self.log_histograms:
            self._log_histograms(outputs)
        return self._shared_eval(outputs, targets, "train")

    def validation_step(self, batch, batch_idx):
        inputs, targets = batch
        outputs = self.forward(inputs)
        return self._shared_eval(outputs, targets, "val")

    def validation_end(self, outputs):
        avg_log = {
            k: self.metric_logger[k].global_avg for k in outputs[0]["log"]
        }
        self._reset_logger()
        return {"log": avg_log}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(), **self.cfg.OPTIMIZER.ARGS
        )
        sched = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                opt, **self.cfg.SCHEDULER.ARGS
            ),
            "interval": "step",
            "frequency": 1,
        }
        return [opt], [sched]

    def _shared_eval(self, outputs, targets, prefix):
        loss_dict = self.criterion(outputs, targets)

        loss, loss_logs = self._eval_loss(loss_dict, prefix)
        metrics_logs = self._eval_metrics(outputs, targets, prefix)

        logs = {**loss_logs, **metrics_logs}
        progress_logs = {k: self.metric_logger[k].avg for k in logs}
        results = {"loss": loss, "log": logs, "progress_bar": progress_logs}
        return results

    def _eval_loss(self, loss_dict, prefix):
        loss = sum(
            l * w for l, w in zip(loss_dict.values(), self.loss_weights)
        )
        loss_logs = {
            f"{k}/{prefix}": v.item()
            for k, v in {"total_loss": loss, **loss_dict}.items()
        }
        self.metric_logger.update(**loss_logs)
        return loss, loss_logs

    def _eval_metrics(self, outputs, targets, prefix):
        results = {}
        for output, target, metrics in zip(outputs, targets, self.metrics):
            for name, metric in metrics.items():
                results[f"{name}/{prefix}"] = metric(output, target)

        self.metric_logger.update(**results)
        return results

    def _reset_logger(self):
        self.metric_logger = MetricLogger()

    def _log_histograms(self, outputs):
        step = self.trainer.global_step
        if step % 10:
            return

        for tag, value in self.model.named_parameters():
            tag = tag.replace(".", "/")
            self.logger.experiment.add_histogram(tag, value, step)

            # if value.grad is not None:
            #     print(value.grad.cpu().numpy())
            #     self.logger.experiment.add_histogram(
            #         f"{tag}/grad", value.grad, step
            #     )

        bs = outputs[0].shape[0]
        predictions = [F.softmax(t, dim=1).view(bs, -1) for t in outputs]
        for prediction, labels in zip(
            predictions,
            [
                ["background", "logo"],
                ["single", "collaged"],
                ["normal", "right", "upside-down", "left"],
            ],
        ):
            for idx, name in enumerate(labels):
                self.logger.experiment.add_histogram(
                    f"prediction_confidence/{name}", prediction[:, idx], step
                )

    def train_dataloader(self):
        return self.dataloaders["train"]

    def val_dataloader(self):
        return self.dataloaders["val"]

    def test_dataloader(self):
        return self.dataloaders["test"]
