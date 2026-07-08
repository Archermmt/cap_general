"""Genesis grasp behavior-cloning policy wrapper."""

from __future__ import annotations

import os
import pickle
import re
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from cap_general.core.policy import BasePolicy, BasePolicyConfig

if TYPE_CHECKING:
    from logging import Logger


class BehaviorCloning:
    """Multi-task behavior cloning with action prediction and object pose estimation."""

    def __init__(self, env: Any, cfg: dict, teacher: nn.Module | None, device: str = "cpu"):
        self._env = env
        self._cfg = cfg
        self._device = device
        self._teacher = teacher
        self._num_steps_per_env = cfg["num_steps_per_env"]

        rgb_shape = (6, env.image_height, env.image_width)
        action_dim = env.num_actions
        self._policy = Policy(cfg["policy"], action_dim).to(device)
        self._optimizer = torch.optim.Adam(self._policy.parameters(), lr=cfg["learning_rate"])
        self._buffer = ExperienceBuffer(
            num_envs=env.num_envs,
            max_size=self._cfg["buffer_size"],
            img_shape=rgb_shape,
            state_dim=self._cfg["policy"]["action_head"]["state_obs_dim"],
            action_dim=action_dim,
            device=device,
            dtype=self._policy.dtype,
        )
        self._current_iter = 0

    def learn(self, num_learning_iterations: int, log_dir: str) -> None:
        self._rewbuffer = deque(maxlen=100)
        self._cur_reward_sum = torch.zeros(self._env.num_envs, dtype=torch.float, device=self._device)
        self._buffer.clear()

        tf_writer = SummaryWriter(log_dir)
        for it in range(num_learning_iterations):
            start_time = time.time()
            self._collect_with_rl_teacher()
            forward_time = time.time() - start_time

            total_action_loss = 0.0
            total_pose_loss = 0.0
            num_batches = 0

            start_time = time.time()
            generator = self._buffer.get_batches(self._cfg.get("num_mini_batches", 4), self._cfg["num_epochs"])
            for batch in generator:
                pred_action = self._policy(batch["rgb_obs"], batch["robot_pose"])
                pred_left_pose, pred_right_pose = self._policy.predict_pose(batch["rgb_obs"])
                action_loss = F.mse_loss(pred_action, batch["actions"])
                pose_left_loss = self._compute_pose_loss(pred_left_pose, batch["object_poses"])
                pose_right_loss = self._compute_pose_loss(pred_right_pose, batch["object_poses"])
                pose_loss = pose_left_loss + pose_right_loss
                total_loss = action_loss + pose_loss

                self._optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self._policy.parameters(), self._cfg["max_grad_norm"])
                self._optimizer.step()

                total_action_loss += action_loss
                total_pose_loss += pose_loss
                num_batches += 1

            backward_time = time.time() - start_time
            if num_batches == 0:
                raise ValueError("No batches collected")
            avg_action_loss = total_action_loss / num_batches
            avg_pose_loss = total_pose_loss / num_batches

            self._current_iter = it
            fps = (self._num_steps_per_env * self._env.num_envs) / forward_time
            if (it + 1) % self._cfg["log_freq"] == 0:
                current_lr = self._optimizer.param_groups[0]["lr"]
                tf_writer.add_scalar("loss/action_loss", avg_action_loss, it)
                tf_writer.add_scalar("loss/pose_loss", avg_pose_loss, it)
                tf_writer.add_scalar("loss/total_loss", avg_action_loss + avg_pose_loss, it)
                tf_writer.add_scalar("lr", current_lr, it)
                tf_writer.add_scalar("buffer_size", self._buffer.size, it)
                tf_writer.add_scalar("speed/forward", forward_time, it)
                tf_writer.add_scalar("speed/backward", backward_time, it)
                tf_writer.add_scalar("speed/fps", int(fps), it)
                if len(self._rewbuffer) > 0:
                    tf_writer.add_scalar("reward/mean", np.mean(self._rewbuffer), it)

            if (it + 1) % self._cfg["save_freq"] == 0:
                self.save(os.path.join(log_dir, f"checkpoint_{it + 1:04d}.pt"))

        tf_writer.close()

    def _compute_pose_loss(self, pred_poses: torch.Tensor, target_poses: torch.Tensor) -> torch.Tensor:
        pred_pos = pred_poses[:, :3]
        pred_quat = F.normalize(pred_poses[:, 3:7], p=2, dim=1)
        target_pos = target_poses[:, :3]
        target_quat = F.normalize(target_poses[:, 3:7], p=2, dim=1)
        pos_loss = F.mse_loss(pred_pos, target_pos)
        quat_dot = torch.sum(pred_quat * target_quat, dim=1)
        quat_loss = torch.mean(1.0 - torch.abs(quat_dot))
        return pos_loss + quat_loss

    def _collect_with_rl_teacher(self) -> None:
        if self._teacher is None:
            raise ValueError("BehaviorCloning requires a teacher policy for training")
        obs_dict = self._env.get_observations()
        with torch.inference_mode():
            for _ in range(self._num_steps_per_env):
                rgb_obs = self._env.get_stereo_rgb_images(normalize=True)
                teacher_action = self._teacher(obs_dict).detach()
                ee_pose = self._env.robot.ee_pose
                object_pose = torch.cat([self._env.object.get_pos(), self._env.object.get_quat()], dim=-1)
                self._buffer.add(rgb_obs, ee_pose, object_pose, teacher_action)

                student_action = self._policy(rgb_obs.float(), ee_pose.float())
                action_diff = torch.norm(student_action - teacher_action, dim=-1)
                condition = (action_diff < 1.0).unsqueeze(-1).expand_as(student_action)
                action = torch.where(condition, student_action, teacher_action)

                obs_dict, reward, done, _ = self._env.step(action)
                self._cur_reward_sum += reward
                new_ids = (done > 0).nonzero(as_tuple=False)
                self._rewbuffer.extend(self._cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                self._cur_reward_sum[new_ids] = 0

    def save(self, path: str) -> None:
        checkpoint = {
            "model_state_dict": self._policy.state_dict(),
            "optimizer_state_dict": self._optimizer.state_dict(),
            "current_iter": self._current_iter,
            "config": self._cfg,
        }
        torch.save(checkpoint, path)

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self._device, weights_only=False)
        self._policy.load_state_dict(checkpoint["model_state_dict"])
        self._optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self._current_iter = checkpoint["current_iter"]


class ExperienceBuffer:
    """A first-in-first-out buffer for behavior cloning experience."""

    def __init__(
        self,
        num_envs: int,
        max_size: int,
        img_shape: tuple[int, int, int],
        state_dim: int,
        action_dim: int,
        device: str = "cpu",
        dtype: torch.dtype | None = None,
    ):
        self._num_envs = num_envs
        self._max_size = max_size
        self._img_shape = img_shape
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._device = device
        self._ptr = 0
        self._size = 0
        self._rgb_obs = torch.empty(max_size, num_envs, *img_shape, dtype=dtype, device=device)
        self._robot_pose = torch.empty(max_size, num_envs, state_dim, dtype=dtype, device=device)
        self._object_poses = torch.empty(max_size, num_envs, 7, dtype=dtype, device=device)
        self._actions = torch.empty(max_size, num_envs, action_dim, dtype=dtype, device=device)

    def add(
        self,
        rgb_obs: torch.Tensor,
        robot_pose: torch.Tensor,
        object_poses: torch.Tensor,
        actions: torch.Tensor,
    ) -> None:
        self._rgb_obs[self._ptr] = rgb_obs
        self._robot_pose[self._ptr] = robot_pose
        self._object_poses[self._ptr] = object_poses
        self._actions[self._ptr] = actions
        self._ptr = (self._ptr + 1) % self._max_size
        self._size = min(self._size + 1, self._max_size)

    def get_batches(self, num_mini_batches: int, num_epochs: int) -> Iterator[dict[str, torch.Tensor]]:
        batch_size = self._size // num_mini_batches
        for _ in range(num_epochs):
            indices = torch.randperm(self._size)
            for batch_idx in range(0, self._size, batch_size):
                batch_indices = indices[batch_idx : batch_idx + batch_size]
                yield {
                    "rgb_obs": self._rgb_obs[batch_indices].reshape(-1, *self._img_shape),
                    "robot_pose": self._robot_pose[batch_indices].reshape(-1, self._state_dim),
                    "object_poses": self._object_poses[batch_indices].reshape(-1, 7),
                    "actions": self._actions[batch_indices].reshape(-1, self._action_dim),
                }

    def clear(self) -> None:
        self._ptr = 0
        self._size = 0

    @property
    def size(self) -> int:
        return self._size


class Policy(nn.Module):
    """Multi-task behavior cloning policy with shared stereo encoder/decoder."""

    def __init__(self, config: dict, action_dim: int):
        super().__init__()
        self.shared_encoder = self._build_cnn(config["vision_encoder"])
        conv_out_channels = config["vision_encoder"]["conv_layers"][-1]["out_channels"]
        vision_dim = conv_out_channels * 4 * 4
        self.feature_fusion = nn.Sequential(
            nn.Linear(vision_dim * 2, vision_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )
        mlp_cfg = config["action_head"]
        self.state_obs_dim = config["action_head"]["state_obs_dim"]
        mlp_cfg["input_dim"] = vision_dim + self.state_obs_dim if self.state_obs_dim is not None else vision_dim
        mlp_cfg["output_dim"] = action_dim
        self.mlp = self._build_mlp(mlp_cfg)

        pose_mlp_cfg = config["pose_head"]
        pose_mlp_cfg["input_dim"] = vision_dim
        pose_mlp_cfg["output_dim"] = 7
        self.pose_mlp = self._build_mlp(pose_mlp_cfg)

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @staticmethod
    def _build_cnn(config: dict) -> nn.Sequential:
        layers = []
        for conv_config in config["conv_layers"]:
            layers.extend(
                [
                    nn.Conv2d(
                        conv_config["in_channels"],
                        conv_config["out_channels"],
                        kernel_size=conv_config["kernel_size"],
                        stride=conv_config["stride"],
                        padding=conv_config["padding"],
                    ),
                    nn.BatchNorm2d(conv_config["out_channels"]),
                    nn.ReLU(),
                ]
            )
        if config.get("pooling") == "adaptive_avg":
            layers.append(nn.AdaptiveAvgPool2d((4, 4)))
        return nn.Sequential(*layers)

    @staticmethod
    def _build_mlp(config: dict) -> nn.Sequential:
        mlp_input_dim = config["input_dim"]
        layers = []
        for hidden_dim in config["hidden_dims"]:
            layers.extend([nn.Linear(mlp_input_dim, hidden_dim), nn.ReLU()])
            mlp_input_dim = hidden_dim
        layers.append(nn.Linear(mlp_input_dim, config["output_dim"]))
        return nn.Sequential(*layers)

    def get_features(self, rgb_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        left_rgb = rgb_obs[:, 0:3]
        right_rgb = rgb_obs[:, 3:6]
        left_features = self.shared_encoder(left_rgb).flatten(start_dim=1)
        right_features = self.shared_encoder(right_rgb).flatten(start_dim=1)
        return left_features, right_features

    def forward(self, rgb_obs: torch.Tensor, state_obs: torch.Tensor | None = None) -> torch.Tensor:
        left_features, right_features = self.get_features(rgb_obs)
        combined_features = torch.cat([left_features, right_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        if state_obs is not None and self.state_obs_dim is not None:
            fused_features = torch.cat([fused_features, state_obs], dim=-1)
        return self.mlp(fused_features)

    def predict_pose(self, rgb_obs: torch.Tensor) -> torch.Tensor:
        left_features, right_features = self.get_features(rgb_obs)
        left_pose = self.pose_mlp(left_features)
        right_pose = self.pose_mlp(right_features)
        return left_pose, right_pose


@dataclass
class BehaviorCloningPolicyConfig(BasePolicyConfig):
    """Configuration for Genesis manipulation BehaviorCloning checkpoints."""

    log_dir: str | Path = "logs/grasp_bc"
    checkpoint: str | Path | None = None
    cfgs_filename: str = "cfgs.pkl"
    bc_cfg_index: int = 4
    device: str | None = None


@BasePolicy.register()
class BehaviorCloningPolicy(BasePolicy):
    """Load and run Genesis manipulation behavior-cloning policies."""

    name = "Genesis Behavior Cloning Policy"
    config_cls = BehaviorCloningPolicyConfig

    def __init__(self, config: BehaviorCloningPolicyConfig, logger: Logger):
        super().__init__(config=config, logger=logger)
        self._policy = None
        self._loaded_env_id: int | None = None

    policy_type = "genesis_behavior_cloning"

    def reset(self, *args: Any, **kwargs: Any) -> None:
        """Clear loaded checkpoint; policy is restored lazily for the active env."""
        self._policy = None
        self._loaded_env_id = None

    def inference(self, *, env: Any, rgb_obs: Any, ee_pose: Any) -> Any:
        """Run the BC policy on stereo images and end-effector pose."""
        self._ensure_loaded(env)
        self._policy.eval()
        return self._policy(rgb_obs, ee_pose)

    def update(self, *, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Load trained weights into the current policy."""
        self._policy.load_state_dict(state_dict)
        return {}

    def _ensure_loaded(self, env: Any) -> None:
        if self._policy is not None and self._loaded_env_id == id(env):
            return
        try:
            import genesis as gs
        except ImportError as exc:
            raise ImportError("BehaviorCloningPolicy requires genesis") from exc

        log_dir = Path(self._config.log_dir).expanduser()
        with (log_dir / self._config.cfgs_filename).open("rb") as file:
            cfgs = pickle.load(file)
        bc_cfg = cfgs[self._config.bc_cfg_index]
        bc_runner = BehaviorCloning(env, bc_cfg, None, device=self._config.device or gs.device)
        bc_runner.load(str(self._checkpoint_path(log_dir)))
        self._policy = bc_runner._policy
        self._policy.eval()
        self._loaded_env_id = id(env)

    def _checkpoint_path(self, log_dir: Path) -> Path:
        if self._config.checkpoint is not None:
            return Path(self._config.checkpoint).expanduser()
        checkpoint_files = list(log_dir.glob("checkpoint_*.pt"))
        if not checkpoint_files:
            raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
        return max(checkpoint_files, key=self._checkpoint_number)

    @staticmethod
    def _checkpoint_number(path: Path) -> int:
        match = re.search(r"\d+", path.stem)
        return int(match.group()) if match else -1
