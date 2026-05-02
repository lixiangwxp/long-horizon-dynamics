import warnings

import pytorch_lightning
import torch

from .registry import get_model

warnings.filterwarnings("ignore")


class DynamicsLearning(pytorch_lightning.LightningModule):
    def __init__(self, args, resources_path, experiment_path, input_size, output_size, max_iterations):
        super().__init__()
        self.args = args
        self.resources_path = resources_path
        self.experiment_path = experiment_path
        self.input_size = input_size
        self.output_size = output_size
        self.max_iterations = max_iterations

        self.warmup_lr = args.warmup_lr
        self.cosine_lr = args.cosine_lr
        self.warmup_steps = args.warmup_steps
        self.cosine_steps = args.cosine_steps
        self.adam_beta1 = args.adam_beta1
        self.adam_beta2 = args.adam_beta2
        self.adam_eps = args.adam_eps
        self.weight_decay = args.weight_decay

        self.lambda_p = args.lambda_p
        self.lambda_v = args.lambda_v
        self.lambda_q = args.lambda_q
        self.lambda_omega = args.lambda_omega

        self.model = get_model(args, input_size, output_size)

        self.best_valid_loss = torch.tensor(float("inf"))
        self.validation_step_outputs = []

    def forward(self, z_hist, init_memory):
        return self.model(z_hist, init_memory)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(),
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_eps,
            lr=self.warmup_lr,
            weight_decay=self.weight_decay,
        )
        schedulers = [
            torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=self.warmup_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cosine_steps, eta_min=self.cosine_lr),
            torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=self.cosine_lr / self.warmup_lr,
                total_iters=self.max_iterations,
            ),
        ]
        milestones = [self.warmup_steps, self.warmup_steps + self.cosine_steps]
        scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer, schedulers=schedulers, milestones=milestones)
        return ([optimizer], [{"scheduler": scheduler, "interval": "step", "frequency": 1}])

    def exp_quat(self, dtheta):
        angle = torch.linalg.norm(dtheta, dim=-1, keepdim=True)
        half_angle = 0.5 * angle
        small_scale = 0.5 - (angle * angle) / 48.0
        scale = torch.where(angle > 1e-6, torch.sin(half_angle) / angle, small_scale)
        quat = torch.cat([torch.cos(half_angle), scale * dtheta], dim=-1)
        return self.quat_normalize(quat)

    def quat_multiply(self, q1, q2):
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        return torch.stack(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dim=-1,
        )

    def quat_normalize(self, q):
        return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-12)

    def quat_geodesic_error(self, q_pred, q_true):
        q_pred = self.quat_normalize(q_pred)
        q_true = self.quat_normalize(q_true)
        dot = torch.sum(q_pred * q_true, dim=-1).abs()
        dot = torch.clamp(dot, min=-1.0 + 1e-7, max=1.0 - 1e-7)
        return 2.0 * torch.acos(dot)

    def apply_full_state_update(self, x_t, delta):
        p_t = x_t[:, 0:3]
        v_t = x_t[:, 3:6]
        q_t = x_t[:, 6:10]
        omega_t = x_t[:, 10:13]

        dp = delta[:, 0:3]
        dv = delta[:, 3:6]
        dtheta = delta[:, 6:9]
        domega = delta[:, 9:12]

        p_next = p_t + dp
        v_next = v_t + dv
        q_next = self.quat_multiply(self.exp_quat(dtheta), q_t)
        q_next = self.quat_normalize(q_next)
        omega_next = omega_t + domega

        return torch.cat([p_next, v_next, q_next, omega_next], dim=-1)

    def full_state_loss(self, x_pred, x_true):
        p_loss = torch.mean(torch.abs(x_pred[:, 0:3] - x_true[:, 0:3]))
        v_loss = torch.mean(torch.abs(x_pred[:, 3:6] - x_true[:, 3:6]))
        q_loss = torch.mean(self.quat_geodesic_error(x_pred[:, 6:10], x_true[:, 6:10]))
        omega_loss = torch.mean(torch.abs(x_pred[:, 10:13] - x_true[:, 10:13]))
        loss = (
            self.lambda_p * p_loss
            + self.lambda_v * v_loss
            + self.lambda_q * q_loss
            + self.lambda_omega * omega_loss
        )
        return loss, {
            "p_loss": p_loss,
            "v_loss": v_loss,
            "q_loss": q_loss,
            "omega_loss": omega_loss,
        }

    def full_state_rollout(self, batch):
        x_hist_curr = batch["x_hist"].float()
        u_hist_curr = batch["u_hist"].float()
        u_roll = batch["u_roll"].float()
        y_future = batch["y_future"].float()

        total_loss = 0.0
        metrics = {
            "p_loss": 0.0,
            "v_loss": 0.0,
            "q_loss": 0.0,
            "omega_loss": 0.0,
        }
        preds = []

        F = self.args.unroll_length
        for i in range(F):
            z_hist = torch.cat([x_hist_curr, u_hist_curr], dim=-1)
            delta = self.forward(z_hist, init_memory=(i == 0))
            x_next_pred = self.apply_full_state_update(x_hist_curr[:, -1], delta)

            target = y_future[:, i]
            step_loss, step_metrics = self.full_state_loss(x_next_pred, target)
            total_loss = total_loss + step_loss / F
            for key in metrics:
                metrics[key] = metrics[key] + step_metrics[key] / F

            preds.append(x_next_pred)

            if i < F - 1:
                x_hist_curr = torch.cat([x_hist_curr[:, 1:, :], x_next_pred.unsqueeze(1)], dim=1)
                u_next_for_history = u_roll[:, i + 1, :]
                u_hist_curr = torch.cat([u_hist_curr[:, 1:, :], u_next_for_history.unsqueeze(1)], dim=1)

        return preds, total_loss, metrics

    def training_step(self, train_batch, batch_idx):
        _, loss, metrics = self.full_state_rollout(train_batch)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_p_loss", metrics["p_loss"], on_step=True, on_epoch=True, logger=True)
        self.log("train_v_loss", metrics["v_loss"], on_step=True, on_epoch=True, logger=True)
        self.log("train_q_loss", metrics["q_loss"], on_step=True, on_epoch=True, logger=True)
        self.log("train_omega_loss", metrics["omega_loss"], on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, valid_batch, batch_idx, dataloader_idx=0):
        _, loss, metrics = self.full_state_rollout(valid_batch)
        self.validation_step_outputs.append(loss.detach())
        self.log("valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("valid_p_loss", metrics["p_loss"], on_step=True, on_epoch=True, logger=True)
        self.log("valid_v_loss", metrics["v_loss"], on_step=True, on_epoch=True, logger=True)
        self.log("valid_q_loss", metrics["q_loss"], on_step=True, on_epoch=True, logger=True)
        self.log("valid_omega_loss", metrics["omega_loss"], on_step=True, on_epoch=True, logger=True)
        return loss

    def test_step(self, test_batch, batch_idx, dataloader_idx=0):
        _, loss, metrics = self.full_state_rollout(test_batch)
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_p_loss", metrics["p_loss"], on_step=True, logger=True)
        self.log("test_v_loss", metrics["v_loss"], on_step=True, logger=True)
        self.log("test_q_loss", metrics["q_loss"], on_step=True, logger=True)
        self.log("test_omega_loss", metrics["omega_loss"], on_step=True, logger=True)
        return loss

    def on_validation_epoch_end(self):
        if self.validation_step_outputs:
            avg_loss = torch.stack(self.validation_step_outputs).mean()
            if avg_loss < self.best_valid_loss.to(avg_loss.device):
                self.best_valid_loss = avg_loss.detach()
                self.log("best_valid_loss", self.best_valid_loss, on_epoch=True, prog_bar=True, logger=True)
            self.validation_step_outputs.clear()
