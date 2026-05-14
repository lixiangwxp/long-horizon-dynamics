import warnings

import pytorch_lightning
import torch

from .registry import get_model

warnings.filterwarnings("ignore")


class DynamicsLearning(pytorch_lightning.LightningModule):
    def __init__(
        self,
        args,
        resources_path,
        experiment_path,
        input_size,
        output_size,
        max_iterations,
    ):
        super().__init__()
        if args.predictor_type != "full_state":
            if args.predictor_type in {"velocity", "attitude"}:
                raise ValueError(
                    "velocity/attitude Rao predictors are not compatible with "
                    "the current full-state pipeline. Use --predictor_type full_state."
                )
            raise ValueError(f"Unsupported predictor_type: {args.predictor_type}")

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

        self.lambda_p = getattr(args, "lambda_p", 1.0)
        self.lambda_v = getattr(args, "lambda_v", 1.0)
        self.lambda_q = getattr(args, "lambda_q", 1.0)
        self.lambda_omega = getattr(args, "lambda_omega", 1.0)
        self.input_noise_std = getattr(args, "input_noise_std", 0.0)
        self.input_noise_loss_weight = getattr(args, "input_noise_loss_weight", 0.0)
        self.feedback_noise_std = getattr(args, "feedback_noise_std", 0.0)
        self.rollout_loss_tail_weight = getattr(args, "rollout_loss_tail_weight", 1.0)
        self.physics_loss_weight = getattr(args, "physics_loss_weight", 0.0)
        self.physics_kinematic_weight = getattr(args, "physics_kinematic_weight", 1.0)
        self.physics_quat_norm_weight = getattr(args, "physics_quat_norm_weight", 0.01)
        self.physics_v_smooth_weight = getattr(args, "physics_v_smooth_weight", 0.0)
        self.physics_omega_smooth_weight = getattr(
            args, "physics_omega_smooth_weight", 0.0
        )
        self.physics_reliability_scale = getattr(
            args, "physics_reliability_scale", 10.0
        )
        self.physics_slack_margin = getattr(args, "physics_slack_margin", 0.0)

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
            torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=1.0, total_iters=self.warmup_steps
            ),
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.cosine_steps, eta_min=self.cosine_lr
            ),
            torch.optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=self.cosine_lr / self.warmup_lr,
                total_iters=self.max_iterations,
            ),
        ]
        milestones = [self.warmup_steps, self.warmup_steps + self.cosine_steps]
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer, schedulers=schedulers, milestones=milestones
        )
        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step", "frequency": 1}],
        )

    def exp_quat(self, dtheta):
        angle = torch.linalg.norm(dtheta, dim=-1, keepdim=True)
        half_angle = 0.5 * angle
        small_scale = 0.5 - (angle * angle) / 48.0
        scale = torch.where(
            angle > 1e-6,
            torch.sin(half_angle) / angle.clamp_min(1e-6),
            small_scale,
        )
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
        return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)

    def orientation_error(self, q_pred, q_true):
        q_pred = self.quat_normalize(q_pred)
        q_true = self.quat_normalize(q_true)
        q_true_inv = torch.cat([q_true[..., :1], -q_true[..., 1:]], dim=-1)
        q_rel = self.quat_multiply(q_true_inv, q_pred)
        q_rel = self.quat_normalize(q_rel)
        vec_norm = torch.linalg.norm(q_rel[..., 1:], dim=-1)
        scalar = q_rel[..., 0].abs()
        return 2.0 * torch.atan2(vec_norm, scalar.clamp_min(1e-12))

    def apply_full_state_update(self, x_t, delta):
        delta_p = delta[:, 0:3]
        delta_v = delta[:, 3:6]
        dtheta = delta[:, 6:9]
        delta_omega = delta[:, 9:12]

        p_next = x_t[:, 0:3] + delta_p
        v_next = x_t[:, 3:6] + delta_v
        q_t = self.quat_normalize(x_t[:, 6:10])

        # dtheta is a local/body-frame rotation increment.
        # q is q_WB in wxyz order.
        # Therefore q_next = q_t ⊗ Exp(dtheta).
        q_next = self.quat_multiply(q_t, self.exp_quat(dtheta))
        q_next = self.quat_normalize(q_next)

        omega_next = x_t[:, 10:13] + delta_omega
        return torch.cat([p_next, v_next, q_next, omega_next], dim=-1)

    def full_state_loss(self, pred, target):
        p_loss = torch.linalg.norm(pred[:, 0:3] - target[:, 0:3], dim=-1).mean()
        v_loss = torch.linalg.norm(pred[:, 3:6] - target[:, 3:6], dim=-1).mean()
        q_loss = self.orientation_error(pred[:, 6:10], target[:, 6:10]).mean()
        omega_loss = torch.linalg.norm(pred[:, 10:13] - target[:, 10:13], dim=-1).mean()
        state_mse = ((pred - target) ** 2).sum(dim=-1).mean()

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
            "state_mse": state_mse,
        }

    def normalize_state_quaternion(self, state):
        q = self.quat_normalize(state[..., 6:10])
        return torch.cat([state[..., :6], q, state[..., 10:]], dim=-1)

    def add_state_noise(self, state, noise_std):
        noisy_state = state + torch.randn_like(state) * noise_std
        return self.normalize_state_quaternion(noisy_state)

    def rollout_loss_weights(self, horizon, device, dtype, tail_weight):
        weights = torch.linspace(1.0, tail_weight, horizon, device=device, dtype=dtype)
        return weights / weights.sum()

    def physics_regularization(self, x_t, x_next_pred, target):
        dt = x_next_pred.new_tensor(1.0 / float(self.args.sampling_frequency))

        pred_step = x_next_pred[:, 0:3] - x_t[:, 0:3]
        pred_trap_step = 0.5 * dt * (x_t[:, 3:6] + x_next_pred[:, 3:6])
        pred_kinematic = ((pred_step - pred_trap_step) ** 2).sum(dim=-1)

        target_step = target[:, 0:3] - x_t[:, 0:3]
        target_trap_step = 0.5 * dt * (x_t[:, 3:6] + target[:, 3:6])
        target_kinematic = ((target_step - target_trap_step) ** 2).sum(dim=-1).detach()

        slack = target_kinematic + self.physics_slack_margin
        reliability = torch.exp(
            -self.physics_reliability_scale * target_kinematic
        ).detach()
        kinematic_loss = (reliability * torch.relu(pred_kinematic - slack)).mean()

        quat_norm_loss = (x_next_pred[:, 6:10].norm(dim=-1) - 1.0).square().mean()
        v_smooth_loss = ((x_next_pred[:, 3:6] - x_t[:, 3:6]) ** 2).sum(dim=-1).mean()
        omega_smooth_loss = (
            (x_next_pred[:, 10:13] - x_t[:, 10:13]) ** 2
        ).sum(dim=-1).mean()

        physics_loss = (
            self.physics_kinematic_weight * kinematic_loss
            + self.physics_quat_norm_weight * quat_norm_loss
            + self.physics_v_smooth_weight * v_smooth_loss
            + self.physics_omega_smooth_weight * omega_smooth_loss
        )
        return physics_loss, {
            "physics_kinematic_loss": kinematic_loss,
            "physics_quat_norm_loss": quat_norm_loss,
            "physics_v_smooth_loss": v_smooth_loss,
            "physics_omega_smooth_loss": omega_smooth_loss,
            "physics_reliability": reliability.mean(),
        }

    def full_state_rollout(
        self,
        batch,
        feedback_noise_std=0.0,
        rollout_loss_tail_weight=1.0,
        physics_regularization=False,
    ):
        x_hist_curr = batch["x_hist"].float()
        u_hist_curr = batch["u_hist"].float()
        u_roll = batch["u_roll"].float()
        y_future = batch["y_future"].float()
        loss_weights = self.rollout_loss_weights(
            self.args.unroll_length,
            y_future.device,
            y_future.dtype,
            rollout_loss_tail_weight,
        )

        total_loss = 0.0
        metrics = {
            "p_loss": 0.0,
            "v_loss": 0.0,
            "q_loss": 0.0,
            "omega_loss": 0.0,
            "state_mse": 0.0,
        }
        if physics_regularization:
            metrics.update(
                {
                    "physics_loss": 0.0,
                    "physics_kinematic_loss": 0.0,
                    "physics_quat_norm_loss": 0.0,
                    "physics_v_smooth_loss": 0.0,
                    "physics_omega_smooth_loss": 0.0,
                    "physics_reliability": 0.0,
                }
            )
        preds = []

        horizon = self.args.unroll_length
        for step in range(horizon):
            z_hist = torch.cat([x_hist_curr, u_hist_curr], dim=-1)
            delta = self.forward(z_hist, init_memory=(step == 0))
            x_next_pred = self.apply_full_state_update(x_hist_curr[:, -1], delta)

            target = y_future[:, step]
            step_loss, step_metrics = self.full_state_loss(x_next_pred, target)
            step_weight = loss_weights[step]
            total_loss = total_loss + step_weight * step_loss
            for key in metrics:
                if key in step_metrics:
                    metrics[key] = metrics[key] + step_weight * step_metrics[key]

            if physics_regularization:
                physics_loss, physics_metrics = self.physics_regularization(
                    x_hist_curr[:, -1], x_next_pred, target
                )
                metrics["physics_loss"] = (
                    metrics["physics_loss"] + step_weight * physics_loss
                )
                for key, value in physics_metrics.items():
                    metrics[key] = metrics[key] + step_weight * value

            preds.append(x_next_pred)

            if step < horizon - 1:
                x_next_history = x_next_pred
                if feedback_noise_std > 0.0:
                    x_next_history = self.add_state_noise(
                        x_next_history, feedback_noise_std
                    )
                x_hist_curr = torch.cat(
                    [x_hist_curr[:, 1:, :], x_next_history.unsqueeze(1)], dim=1
                )
                u_next_for_history = u_roll[:, step + 1, :]
                u_hist_curr = torch.cat(
                    [u_hist_curr[:, 1:, :], u_next_for_history.unsqueeze(1)],
                    dim=1,
                )

        pred_future = torch.stack(preds, dim=1)
        return pred_future, total_loss, metrics

    def add_input_noise(self, batch):
        noisy_batch = dict(batch)
        noisy_batch["x_hist"] = self.add_state_noise(
            batch["x_hist"], self.input_noise_std
        )
        noisy_batch["u_hist"] = batch["u_hist"] + torch.randn_like(
            batch["u_hist"]
        ) * self.input_noise_std
        return noisy_batch

    def training_step(self, train_batch, batch_idx):
        _, loss, metrics = self.full_state_rollout(
            train_batch,
            feedback_noise_std=self.feedback_noise_std,
            rollout_loss_tail_weight=self.rollout_loss_tail_weight,
            physics_regularization=self.physics_loss_weight > 0.0,
        )
        train_objective = loss
        log_train_objective = False
        if self.physics_loss_weight > 0.0:
            train_objective = train_objective + (
                self.physics_loss_weight * metrics["physics_loss"]
            )
            log_train_objective = True
            self.log(
                "train_physics_loss",
                metrics["physics_loss"],
                on_step=True,
                on_epoch=True,
                logger=True,
            )
            self.log(
                "train_physics_kinematic_loss",
                metrics["physics_kinematic_loss"],
                on_step=True,
                on_epoch=True,
                logger=True,
            )
            self.log(
                "train_physics_reliability",
                metrics["physics_reliability"],
                on_step=True,
                on_epoch=True,
                logger=True,
            )
        if self.input_noise_std > 0.0 and self.input_noise_loss_weight > 0.0:
            noisy_batch = self.add_input_noise(train_batch)
            _, noisy_loss, _ = self.full_state_rollout(noisy_batch)
            train_objective = train_objective + self.input_noise_loss_weight * noisy_loss
            log_train_objective = True
            self.log(
                "train_noisy_loss",
                noisy_loss,
                on_step=True,
                on_epoch=True,
                logger=True,
            )
        if log_train_objective:
            self.log(
                "train_objective",
                train_objective,
                on_step=True,
                on_epoch=True,
                logger=True,
            )
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_p_loss", metrics["p_loss"], on_step=True, on_epoch=True, logger=True
        )
        self.log(
            "train_v_loss", metrics["v_loss"], on_step=True, on_epoch=True, logger=True
        )
        self.log(
            "train_q_loss", metrics["q_loss"], on_step=True, on_epoch=True, logger=True
        )
        self.log(
            "train_omega_loss",
            metrics["omega_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "train_state_mse",
            metrics["state_mse"],
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return train_objective

    def validation_step(self, valid_batch, batch_idx, dataloader_idx=0):
        _, loss, metrics = self.full_state_rollout(valid_batch)
        self.validation_step_outputs.append(loss.detach())
        self.log(
            "valid_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "valid_p_loss", metrics["p_loss"], on_step=True, on_epoch=True, logger=True
        )
        self.log(
            "valid_v_loss", metrics["v_loss"], on_step=True, on_epoch=True, logger=True
        )
        self.log(
            "valid_q_loss", metrics["q_loss"], on_step=True, on_epoch=True, logger=True
        )
        self.log(
            "valid_omega_loss",
            metrics["omega_loss"],
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "valid_state_mse",
            metrics["state_mse"],
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return loss

    def test_step(self, test_batch, batch_idx, dataloader_idx=0):
        pred_future, loss, metrics = self.full_state_rollout(test_batch)
        self.log("test_loss", loss, on_step=True, prog_bar=True, logger=True)
        self.log("test_p_loss", metrics["p_loss"], on_step=True, logger=True)
        self.log("test_v_loss", metrics["v_loss"], on_step=True, logger=True)
        self.log("test_q_loss", metrics["q_loss"], on_step=True, logger=True)
        self.log("test_omega_loss", metrics["omega_loss"], on_step=True, logger=True)
        self.log("test_state_mse", metrics["state_mse"], on_step=True, logger=True)
        return {
            "pred_future": pred_future.detach(),
            "y_future": test_batch["y_future"].detach(),
            "loss": loss.detach(),
        }

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pred_future, loss, _ = self.full_state_rollout(batch)
        return {
            "pred_future": pred_future.detach(),
            "y_future": batch["y_future"].detach(),
            "loss": loss.detach(),
        }

    def on_train_epoch_end(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return

        avg_loss = torch.stack(self.validation_step_outputs).mean()
        if not torch.isfinite(avg_loss):
            self.best_valid_loss = avg_loss.detach()
        elif avg_loss < self.best_valid_loss.to(avg_loss.device):
            self.best_valid_loss = avg_loss.detach()
        self.log(
            "best_valid_loss",
            self.best_valid_loss.to(avg_loss.device),
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.validation_step_outputs.clear()
