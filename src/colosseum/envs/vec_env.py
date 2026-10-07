from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np

from colosseum.envs.base_env import BaseEnv


class VectorEnv:
    """Manages K copies of a BaseEnv, all stepped together.

    Each env is an independent match with N players.
    Envs are stepped sequentially (parallelism is at the worker-process level).
    Auto-resets envs when episodes finish.

    All public methods return stacked numpy arrays with shape
    [num_envs, num_players, ...] for observations and
    [num_envs, num_players] for rewards, plus per-env boolean arrays
    for terminated/truncated flags.
    """

    def __init__(self, env_fn: Callable[[], BaseEnv], num_envs: int) -> None:
        self.envs: list[BaseEnv] = [env_fn() for _ in range(num_envs)]
        self.num_envs: int = num_envs
        self.num_players: int = self.envs[0].num_players
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        from colosseum.core.action_spec import ActionSpec
        self.action_spec: ActionSpec = ActionSpec.from_space(self.action_space)

        # Cache observation shape/dtype for pre-allocation
        _sample = self.observation_space.sample()
        self._obs_shape: tuple = np.asarray(_sample).shape
        self._obs_dtype = np.asarray(_sample).dtype

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _stack_obs(
        self, obs_dicts: list[dict[int, np.ndarray]]
    ) -> np.ndarray:
        """Stack per-player observation dicts into [K, N, *obs_shape]."""
        return np.array(
            [[obs_dict[p] for p in range(self.num_players)] for obs_dict in obs_dicts]
        )

    def _unstack_actions(
        self, actions: np.ndarray
    ) -> list[dict[int, Any]]:
        """Convert [K, N, *act_shape] array to list of per-player action dicts."""
        action_dicts: list[dict[int, Any]] = []
        for k in range(self.num_envs):
            if self.action_spec.is_composite:
                action_dicts.append({
                    p: self.action_spec.decode(actions[k, p])
                    for p in range(self.num_players)
                })
            else:
                action_dicts.append(
                    {p: actions[k, p] for p in range(self.num_players)}
                )
        return action_dicts

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset_all(
        self, seed: Optional[int] = None
    ) -> tuple[np.ndarray, list[dict[int, dict]]]:
        """Reset all envs.

        Args:
            seed: Optional base seed. Each env receives ``seed + i`` if seed
                is not None, otherwise None.

        Returns:
            observations: np.ndarray of shape [num_envs, num_players, *obs_shape]
            infos: list of num_envs info dicts (each mapping player_index -> info)
        """
        obs_dicts: list[dict[int, np.ndarray]] = []
        all_infos: list[dict[int, dict]] = []
        for i, env in enumerate(self.envs):
            env_seed = (seed + i) if seed is not None else None
            obs, info = env.reset(seed=env_seed)
            obs_dicts.append(obs)
            all_infos.append(info)

        observations = self._stack_obs(obs_dicts)
        return observations, all_infos

    def step(
        self, actions: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict[int, dict]]]:
        """Step all envs and auto-reset any that are done.

        Args:
            actions: Array of shape [num_envs, num_players, *act_shape]
                (or [num_envs, num_players] for discrete action spaces).

        Returns:
            obs:        np.ndarray [num_envs, num_players, *obs_shape]
            rewards:    np.ndarray [num_envs, num_players]  (float64)
            terminated: np.ndarray [num_envs]  (bool) -- True if ANY player terminated
            truncated:  np.ndarray [num_envs]  (bool) -- True if ANY player truncated
            infos:      list of num_envs info dicts (player_index -> info).
                        For auto-reset envs the info for each player includes
                        ``"terminal_observation"`` with the final observation and
                        ``"terminal_info"`` with the final info dict.
        """
        action_dicts = self._unstack_actions(actions)

        obs_dicts: list[dict[int, np.ndarray]] = []
        rewards = np.empty((self.num_envs, self.num_players), dtype=np.float64)
        terminated = np.empty(self.num_envs, dtype=bool)
        truncated = np.empty(self.num_envs, dtype=bool)
        all_infos: list[dict[int, dict]] = []

        for k, env in enumerate(self.envs):
            obs_k, rew_k, term_k, trunc_k, info_k = env.step(action_dicts[k])

            # Per-env done flags (any player triggers whole-match done).
            env_terminated = any(term_k[p] for p in range(self.num_players))
            env_truncated = any(trunc_k[p] for p in range(self.num_players))

            terminated[k] = env_terminated
            truncated[k] = env_truncated
            for p in range(self.num_players):
                rewards[k, p] = rew_k[p]

            # Auto-reset if the episode ended.
            if env_terminated or env_truncated:
                # Preserve terminal data in infos before resetting.
                for p in range(self.num_players):
                    original_info = {k: v for k, v in info_k[p].items()}
                    info_k[p]["terminal_observation"] = obs_k[p]
                    info_k[p]["terminal_info"] = original_info

                new_obs, new_info = env.reset()
                obs_dicts.append(new_obs)
                # Merge reset info into the returned infos (terminal data already stored).
                for p in range(self.num_players):
                    info_k[p].update(new_info[p])
                all_infos.append(info_k)
            else:
                obs_dicts.append(obs_k)
                all_infos.append(info_k)

        observations = self._stack_obs(obs_dicts)
        return observations, rewards, terminated, truncated, all_infos

    def reset_done(
        self, terminated: np.ndarray, truncated: np.ndarray
    ) -> tuple[np.ndarray, list[dict[int, dict]]]:
        """Reset only the envs where the episode is done.

        This is useful when the caller wants explicit control over
        auto-reset (e.g. after calling step() without built-in auto-reset).

        Args:
            terminated: bool array [num_envs] -- per-env terminated flags.
            truncated:  bool array [num_envs] -- per-env truncated flags.

        Returns:
            obs:   np.ndarray [num_envs, num_players, *obs_shape]
                   For envs that were NOT done, observations are zeros.
            infos: list of num_envs info dicts.  For envs that were NOT
                   done, the info dict is empty ``{}``.
        """
        obs_array = np.zeros(
            (self.num_envs, self.num_players, *self._obs_shape),
            dtype=self._obs_dtype,
        )
        all_infos: list[dict[int, dict]] = [{} for _ in range(self.num_envs)]

        done = terminated | truncated
        for k in range(self.num_envs):
            if done[k]:
                new_obs, new_info = self.envs[k].reset()
                for p in range(self.num_players):
                    obs_array[k, p] = new_obs[p]
                all_infos[k] = new_info

        return obs_array, all_infos

    def close(self) -> None:
        """Close all environments and release resources."""
        for env in self.envs:
            env.close()
