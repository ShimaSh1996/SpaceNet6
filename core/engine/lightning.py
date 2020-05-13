import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from collections import OrderedDict
from core.metrics import MetricLogger


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
            k: self.metric_logger[k].global_avg for k in outputs[0]["progress_bar"]
        }
        self._reset_logger()
        return {"log": avg_log}

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(), **self.cfg.OPTIMIZER.ARGS
        )
        # sched = {
        #     "scheduler": torch.optim.lr_scheduler.OneCycleLR(
        #         opt, **self.cfg.SCHEDULER.ARGS
        #     ),
        #     "interval": "step",
        #     "frequency": 1,
        # }
        return [opt]#, [sched]

    def _shared_eval(self, outputs, targets, mode):
        loss_dict = self.criterion(outputs, targets, mode=mode)

        loss, loss_logs = self._eval_loss(loss_dict, mode)
        metrics_logs = self._eval_metrics(outputs, targets, mode)

        all_logs = {**loss_logs, **metrics_logs}
        progress_logs = {k: self.metric_logger[k].avg for k in all_logs}

        all_logs = self._change_log_tags(all_logs)
        results = {
            "loss": loss,
            "log": all_logs,
            "progress_bar": progress_logs,
        }
        return results

    def _eval_loss(self, loss_dict, mode):
        loss = sum(
            l * w for l, w in zip(loss_dict.values(), self.loss_weights)
        )
        loss_logs = {
            f"{mode}_{k}": v.item()
            for k, v in {"total_loss": loss, **loss_dict}.items()
        }
        self.metric_logger.update(**loss_logs)
        return loss, loss_logs

    def _eval_metrics(self, outputs, targets, mode):
        results = {}
        for output, target, metrics in zip(outputs, targets, self.metrics):
            for metric in metrics:
                metric_results = {f"{mode}_{k}":v for k,v in metric(output, target).items()}
                results.update(metric_results)

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

        for oid, pred in enumerate(outputs):
            bs, classes = outputs[0].shape[:2]
            pred = F.softmax(pred, dim=1).view(bs, classes, -1)
            for cid in range(classes):
                self.logger.experiment.add_histogram(
                    f"pred_confidence/output_{oid}/class_{cid}",
                    pred[:, cid],
                    step,
                )

    def _change_log_tags(self, logs):
        def tag_from_key(key):
            mode, name = key.split("_", 1)
            return f"{name}/{mode}"

        logs = {tag_from_key(k): v for k, v in logs.items()}
        return logs

    def train_dataloader(self):
        return self.dataloaders["train"]

    def val_dataloader(self):
        return self.dataloaders["val"]

    # def test_dataloader(self):
    #     return self.dataloaders["test"]
