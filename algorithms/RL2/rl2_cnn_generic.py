"""
Generic RL2-style recurrent PPO for multiple SSD environments with sampled teammate policies.
Supports: coop_mining, cleanup, coin_game, common_harvest, gift, mushrooms, pd_arena, territory
"""

import os
import pickle
from pathlib import Path
from typing import Any, NamedTuple

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import socialmeta
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from socialmeta.wrappers.baselines import LogWrapper


class CNNEncoder(nn.Module):
    activation: str = "relu"

    @nn.compact
    def __call__(self, x):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        x = nn.Conv(
            features=32,
            kernel_size=(5, 5),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            features=64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = activation(x)
        return x


class FrozenActorCritic(nn.Module):
    """Feed-forward teammate policy architecture compatible with SVO checkpoints."""

    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        embedding = CNNEncoder(self.activation, name="CNN_0")(obs)

        actor = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        actor = activation(actor)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)

        critic = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return logits, jnp.squeeze(critic, axis=-1)


class FrozenActor(nn.Module):
    """Actor-only teammate architecture to avoid unnecessary critic compute."""

    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        embedding = CNNEncoder(self.activation, name="CNN_0")(obs)

        actor = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(embedding)
        actor = activation(actor)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)
        return logits


class RecurrentActorCritic(nn.Module):
    action_dim: int
    hidden_size: int = 128
    activation: str = "relu"

    @nn.compact
    def __call__(self, hidden, obs, prev_action, prev_reward, reset):
        embedding = CNNEncoder(self.activation)(obs)
        prev_action_oh = jax.nn.one_hot(prev_action, self.action_dim, dtype=jnp.float32)
        r_in = prev_reward[:, None]
        x = jnp.concatenate([embedding, prev_action_oh, r_in], axis=-1)

        hidden = jnp.where(reset[:, None], jnp.zeros_like(hidden), hidden)
        hidden, rnn_out = nn.GRUCell(features=self.hidden_size)(hidden, x)

        activation = nn.relu if self.activation == "relu" else nn.tanh

        actor = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(rnn_out)
        actor = activation(actor)
        logits = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)

        critic = nn.Dense(
            64,
            kernel_init=orthogonal(np.sqrt(2)),
            bias_init=constant(0.0),
        )(rnn_out)
        critic = activation(critic)
        critic = nn.Dense(
            1,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(critic)

        return hidden, logits, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    reset: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    prev_action: jnp.ndarray
    prev_reward: jnp.ndarray


def load_teammate_policy_bank(policy_dir: str, pattern: str):
    paths = sorted(Path(policy_dir).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No policy files found in {policy_dir} with pattern {pattern}")

    params_list = []
    for path in paths:
        with open(path, "rb") as f:
            params = pickle.load(f)
        if "params" not in params:
            raise ValueError(f"{path} does not contain a 'params' key.")
        params_list.append(jax.tree_util.tree_map(lambda x: jnp.asarray(x), params))

    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *params_list)
    return stacked, [str(p) for p in paths]


def save_params(params, save_path: str):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    params_np = jax.tree_util.tree_map(lambda x: np.array(x), params)
    with open(save_path, "wb") as f:
        pickle.dump(params_np, f)


def gather_by_agent(x: jnp.ndarray, agent_idx: jnp.ndarray):
    env_idx = jnp.arange(x.shape[0])
    return x[env_idx, agent_idx]


def sample_eval_assignments(
    rng: jnp.ndarray, num_envs: int, num_agents: int, num_teammate_policies: int
):
    rng_a, rng_p = jax.random.split(rng)
    controlled_agent = jax.random.randint(
        rng_a, shape=(num_envs,), minval=0, maxval=num_agents, dtype=jnp.int32
    )
    teammate_policy_ids = jax.random.randint(
        rng_p,
        shape=(num_envs, num_agents),
        minval=0,
        maxval=num_teammate_policies,
        dtype=jnp.int32,
    )
    return controlled_agent, teammate_policy_ids


def make_policy_evaluator(config: dict):
    """
    Build and JIT-compile an evaluator once, then reuse it for multiple checkpoints.
    """
    env_kwargs = dict(config["ENV_KWARGS"])
    env_kwargs["num_outer_steps"] = int(config["TRIAL_EPISODES"])
    env = socialmeta.make(config["ENV_NAME"], **env_kwargs)
    env = LogWrapper(env, replace_info=False)

    eval_num = int(config["EVAL_NUM"])
    test_seed = int(config["TEST_SEED"])
    rnn_hidden_size = int(config["RNN_HIDDEN_SIZE"])
    last_episode_idx = int(config["TRIAL_EPISODES"]) - 1

    num_agents = env.num_agents
    action_dim = env.action_space().n
    eval_steps = int(env_kwargs["num_inner_steps"]) * int(config["TRIAL_EPISODES"])

    teammate_bank, _ = load_teammate_policy_bank(
        config["TEAMMATE_POLICY_DIR"], config["TEAMMATE_POLICY_GLOB"]
    )
    num_teammate_policies = jax.tree_util.tree_leaves(teammate_bank)[0].shape[0]

    network = RecurrentActorCritic(
        action_dim=action_dim,
        hidden_size=rnn_hidden_size,
        activation=config["ACTIVATION"],
    )
    teammate_network = FrozenActor(action_dim=action_dim, activation=config["ACTIVATION"])

    def _eval_once(policy_params, rng, controlled_agent, teammate_policy_ids):
        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, eval_num)
        last_obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)

        hidden = jnp.zeros((eval_num, rnn_hidden_size), dtype=jnp.float32)
        prev_action = jnp.zeros((eval_num,), dtype=jnp.int32)
        prev_reward = jnp.zeros((eval_num,), dtype=jnp.float32)
        last_done = jnp.ones((eval_num,), dtype=jnp.bool_)

        reward_sum = jnp.zeros((eval_num,), dtype=jnp.float32)
        running_trial_return = jnp.zeros((eval_num,), dtype=jnp.float32)
        episode_return_sum = jnp.array(0.0, dtype=jnp.float32)
        episode_return_sq_sum = jnp.array(0.0, dtype=jnp.float32)
        episode_count = jnp.array(0.0, dtype=jnp.float32)

        def _step(carry, _):
            (
                env_state,
                last_obs,
                hidden,
                prev_action,
                prev_reward,
                last_done,
                reward_sum,
                running_trial_return,
                episode_return_sum,
                episode_return_sq_sum,
                episode_count,
                rng,
            ) = carry

            obs_ctrl = gather_by_agent(last_obs, controlled_agent)
            hidden_next, logits, _ = network.apply(
                policy_params,
                hidden,
                obs_ctrl,
                prev_action,
                prev_reward,
                last_done,
            )

            rng, ctrl_key, teammate_key, env_key = jax.random.split(rng, 4)
            ctrl_dist = distrax.Categorical(logits=logits)
            ctrl_action = ctrl_dist.sample(seed=ctrl_key)

            selected_params = jax.tree_util.tree_map(lambda x: x[teammate_policy_ids], teammate_bank)
            selected_params_flat = jax.tree_util.tree_map(
                lambda x: x.reshape((eval_num * num_agents,) + x.shape[2:]),
                selected_params,
            )
            obs_flat = last_obs[:, :, None, ...].reshape(
                (eval_num * num_agents, 1) + last_obs.shape[2:]
            )
            teammate_logits_flat = jax.vmap(
                teammate_network.apply, in_axes=(0, 0)
            )(selected_params_flat, obs_flat)
            teammate_logits = jnp.squeeze(teammate_logits_flat, axis=1).reshape(
                eval_num, num_agents, action_dim
            )
            actions = jax.random.categorical(teammate_key, teammate_logits, axis=-1).astype(jnp.int32)
            actions = actions.at[jnp.arange(eval_num), controlled_agent].set(ctrl_action)

            step_keys = jax.random.split(env_key, eval_num)
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_keys, env_state, actions)

            done_trial = done["__all__"].astype(jnp.bool_)
            done_trial_f = done_trial.astype(jnp.float32)
            reward_ctrl = gather_by_agent(reward, controlled_agent)
            current_outer_t = env_state.env_state.outer_t
            in_last_episode = current_outer_t == last_episode_idx
            # Eval policy is always measured on the final episode only.
            reward_weight = jnp.where(in_last_episode, 1.0, 0.0)
            reward_ctrl = reward_ctrl * reward_weight

            reward_sum = reward_sum + reward_ctrl
            running_trial_return = running_trial_return + reward_ctrl
            episode_return_sum = episode_return_sum + (running_trial_return * done_trial_f).sum()
            episode_return_sq_sum = episode_return_sq_sum + ((running_trial_return ** 2) * done_trial_f).sum()
            episode_count = episode_count + done_trial_f.sum()
            running_trial_return = jnp.where(done_trial, jnp.zeros_like(running_trial_return), running_trial_return)

            next_prev_action = jnp.where(done_trial, jnp.zeros_like(ctrl_action), ctrl_action)
            next_prev_reward = jnp.where(done_trial, jnp.zeros_like(reward_ctrl), reward_ctrl)

            carry = (
                env_state,
                obsv,
                hidden_next,
                next_prev_action,
                next_prev_reward,
                done_trial,
                reward_sum,
                running_trial_return,
                episode_return_sum,
                episode_return_sq_sum,
                episode_count,
                rng,
            )
            return carry, None

        carry = (
            env_state,
            last_obs,
            hidden,
            prev_action,
            prev_reward,
            last_done,
            reward_sum,
            running_trial_return,
            episode_return_sum,
            episode_return_sq_sum,
            episode_count,
            rng,
        )
        carry, _ = jax.lax.scan(_step, carry, None, eval_steps)
        (
            _env_state,
            _last_obs,
            _hidden,
            _prev_action,
            _prev_reward,
            _last_done,
            reward_sum,
            _running_trial_return,
            episode_return_sum,
            episode_return_sq_sum,
            episode_count,
            _rng,
        ) = carry

        episode_return_mean = jnp.where(episode_count > 0.0, episode_return_sum / episode_count, 0.0)
        episode_return_var = jnp.where(
            episode_count > 0.0,
            jnp.maximum(0.0, episode_return_sq_sum / episode_count - episode_return_mean ** 2),
            0.0,
        )

        return {
            "eval_reward_mean": reward_sum.mean() / float(eval_steps),
            "eval_episode_return_mean": episode_return_mean,
            "eval_episode_return_std": jnp.sqrt(episode_return_var),
        }

    eval_once = jax.jit(_eval_once)

    def evaluate(policy_params, reward_progress: float = 1.0, extra_log: dict | None = None):
        eval_rng = jax.random.PRNGKey(test_seed)
        reward_progress = float(np.clip(reward_progress, 0.0, 1.0))

        eval_rng, assign_rng, rollout_rng = jax.random.split(eval_rng, 3)
        controlled_agent, teammate_policy_ids = sample_eval_assignments(
            assign_rng, eval_num, num_agents, num_teammate_policies
        )
        metrics = eval_once(
            policy_params,
            rollout_rng,
            controlled_agent,
            teammate_policy_ids,
        )
        eval_summary = jax.tree_util.tree_map(lambda x: float(jax.device_get(x)), metrics)
        log_payload = dict(eval_summary)
        if extra_log:
            log_payload.update(extra_log)
        log_payload["reward_progress"] = reward_progress

        if config.get("LOG_WANDB", False) and wandb.run is not None:
            wandb.log(log_payload)

        print(
            "Evaluation "
            f"(seed={test_seed}, eval_num={eval_num}): "
            f"return_mean={eval_summary['eval_episode_return_mean']:.4f}, "
            f"return_std={eval_summary['eval_episode_return_std']:.4f}, "
            f"reward_mean={eval_summary['eval_reward_mean']:.4f}"
        )
        return eval_summary

    return evaluate


def evaluate_policy(
    config: dict, policy_params: Any, reward_progress: float = 1.0, extra_log: dict | None = None
):
    evaluator = make_policy_evaluator(config)
    return evaluator(policy_params, reward_progress=reward_progress, extra_log=extra_log)


def make_train_stepper(config):
    env_kwargs = dict(config["ENV_KWARGS"])
    env_kwargs["num_outer_steps"] = int(config["TRIAL_EPISODES"])
    env = socialmeta.make(config["ENV_NAME"], **env_kwargs)
    env = LogWrapper(env, replace_info=False)

    num_envs = int(config["NUM_ENVS"])
    num_steps = int(config["NUM_STEPS"])
    num_minibatches = int(config["NUM_MINIBATCHES"])
    num_agents = env.num_agents
    action_dim = env.action_space().n
    obs_shape = env.observation_space()[0].shape
    rnn_hidden_size = int(config["RNN_HIDDEN_SIZE"])
    total_updates = int(config["TOTAL_TIMESTEPS"]) // num_steps // num_envs
    trial_episodes = int(config["TRIAL_EPISODES"])
    episode_reward_weights = np.asarray(
        config.get("EPISODE_REWARD_WEIGHTS", [0.2, 0.3, 0.5]),
        dtype=np.float32,
    )
    assert len(episode_reward_weights) == trial_episodes, (
        f"EPISODE_REWARD_WEIGHTS length ({len(episode_reward_weights)}) must match "
        f"TRIAL_EPISODES ({trial_episodes})."
    )
    assert np.isclose(float(episode_reward_weights.sum()), 1.0, atol=1e-6), (
        "EPISODE_REWARD_WEIGHTS must sum to 1.0."
    )
    episode_reward_weights = jnp.asarray(episode_reward_weights, dtype=jnp.float32)
    log_interval_updates = max(1, int(config.get("LOG_INTERVAL_UPDATES", 20)))
    env_scan_unroll = max(1, int(config.get("ENV_SCAN_UNROLL", 2)))
    rnn_scan_unroll = max(1, int(config.get("RNN_SCAN_UNROLL", 4)))
    gae_scan_unroll = max(1, int(config.get("GAE_SCAN_UNROLL", 4)))

    if num_envs % num_minibatches != 0:
        raise ValueError("NUM_ENVS must be divisible by NUM_MINIBATCHES for recurrent minibatching.")

    config["NUM_UPDATES"] = total_updates
    envs_per_minibatch = num_envs // num_minibatches

    teammate_bank, teammate_files = load_teammate_policy_bank(
        config["TEAMMATE_POLICY_DIR"], config["TEAMMATE_POLICY_GLOB"]
    )
    num_teammate_policies = jax.tree_util.tree_leaves(teammate_bank)[0].shape[0]
    print(f"Loaded {num_teammate_policies} teammate policies from {config['TEAMMATE_POLICY_DIR']}")
    for path in teammate_files:
        print(f"  - {path}")

    network = RecurrentActorCritic(
        action_dim=action_dim,
        hidden_size=rnn_hidden_size,
        activation=config["ACTIVATION"],
    )
    teammate_network = FrozenActor(action_dim=action_dim, activation=config["ACTIVATION"])

    def select_teammate_params_flat(teammate_policy_ids):
        selected_params = jax.tree_util.tree_map(lambda x: x[teammate_policy_ids], teammate_bank)
        return jax.tree_util.tree_map(
            lambda x: x.reshape((num_envs * num_agents,) + x.shape[2:]),
            selected_params,
        )

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (num_minibatches * int(config["UPDATE_EPOCHS"])))
            / config["NUM_UPDATES"]
        )
        return float(config["LR"]) * frac

    def sample_trial_assignments(rng):
        rng_a, rng_p = jax.random.split(rng)
        controlled_agent = jax.random.randint(
            rng_a, shape=(num_envs,), minval=0, maxval=num_agents, dtype=jnp.int32
        )
        teammate_policy_ids = jax.random.randint(
            rng_p,
            shape=(num_envs, num_agents),
            minval=0,
            maxval=num_teammate_policies,
            dtype=jnp.int32,
        )
        return controlled_agent, teammate_policy_ids

    def _calculate_gae(traj_batch, last_value):
        gamma = float(config["GAMMA"])
        gae_lambda = float(config["GAE_LAMBDA"])

        def _gae_scan(carry, transition):
            gae, next_value = carry
            done = transition.done.astype(jnp.float32)
            delta = transition.reward + gamma * next_value * (1.0 - done) - transition.value
            gae = delta + gamma * gae_lambda * (1.0 - done) * gae
            return (gae, transition.value), gae

        _, advantages = jax.lax.scan(
            _gae_scan,
            (jnp.zeros_like(last_value), last_value),
            traj_batch,
            reverse=True,
            unroll=gae_scan_unroll,
        )
        targets = advantages + traj_batch.value
        return advantages, targets

    def _loss_fn(params, traj_mb, adv_mb, target_mb, init_hidden_mb):
        def _rnn_forward(carry, inps):
            obs, pa, pr, reset, action = inps
            carry, logits, value = network.apply(params, carry, obs, pa, pr, reset)
            dist = distrax.Categorical(logits=logits)
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return carry, (log_prob, value, entropy)

        _, (new_log_prob, new_value, entropy) = jax.lax.scan(
            _rnn_forward,
            init_hidden_mb,
            (
                traj_mb.obs,
                traj_mb.prev_action,
                traj_mb.prev_reward,
                traj_mb.reset,
                traj_mb.action,
            ),
            unroll=rnn_scan_unroll,
        )

        adv_norm = (adv_mb - adv_mb.mean()) / (adv_mb.std() + 1e-8)
        ratio = jnp.exp(new_log_prob - traj_mb.log_prob)
        loss_actor_1 = ratio * adv_norm
        loss_actor_2 = (
            jnp.clip(
                ratio,
                1.0 - float(config["CLIP_EPS"]),
                1.0 + float(config["CLIP_EPS"]),
            )
            * adv_norm
        )
        loss_actor = -jnp.minimum(loss_actor_1, loss_actor_2).mean()

        value_pred_clipped = traj_mb.value + (
            new_value - traj_mb.value
        ).clip(-float(config["CLIP_EPS"]), float(config["CLIP_EPS"]))
        value_losses = jnp.square(new_value - target_mb)
        value_losses_clipped = jnp.square(value_pred_clipped - target_mb)
        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

        entropy_loss = entropy.mean()
        total_loss = (
            loss_actor
            + float(config["VF_COEF"]) * value_loss
            - float(config["ENT_COEF"]) * entropy_loss
        )
        approx_kl = ((traj_mb.log_prob - new_log_prob) ** 2).mean()

        metrics = {
            "total_loss": total_loss,
            "value_loss": value_loss,
            "policy_loss": loss_actor,
            "entropy": entropy_loss,
            "approx_kl": approx_kl,
        }
        return total_loss, metrics

    def _update_step(runner_state, _):
        (
            train_state,
            env_state,
            last_obs,
            hidden,
            prev_action,
            prev_reward,
            last_done,
            controlled_agent,
            teammate_policy_ids,
            selected_teammate_params_flat,
            update_step,
            rng,
        ) = runner_state

        initial_hidden = hidden

        def _env_step(carry, _):
            (
                train_state,
                env_state,
                last_obs,
                hidden,
                prev_action,
                prev_reward,
                last_done,
                controlled_agent,
                teammate_policy_ids,
                selected_teammate_params_flat,
                update_step,
                rng,
            ) = carry

            obs_ctrl = gather_by_agent(last_obs, controlled_agent)
            hidden_next, logits, value = network.apply(
                train_state.params,
                hidden,
                obs_ctrl,
                prev_action,
                prev_reward,
                last_done,
            )

            rng, ctrl_key, teammate_key, env_key, assign_key = jax.random.split(rng, 5)

            ctrl_dist = distrax.Categorical(logits=logits)
            ctrl_action = ctrl_dist.sample(seed=ctrl_key)
            ctrl_log_prob = ctrl_dist.log_prob(ctrl_action)

            obs_flat = last_obs[:, :, None, ...].reshape(
                (num_envs * num_agents, 1) + last_obs.shape[2:]
            )
            teammate_logits_flat = jax.vmap(
                teammate_network.apply, in_axes=(0, 0)
            )(selected_teammate_params_flat, obs_flat)
            teammate_logits = jnp.squeeze(teammate_logits_flat, axis=1).reshape(
                num_envs, num_agents, action_dim
            )
            actions = jax.random.categorical(teammate_key, teammate_logits, axis=-1).astype(jnp.int32)
            actions = actions.at[jnp.arange(num_envs), controlled_agent].set(ctrl_action)

            step_keys = jax.random.split(env_key, num_envs)
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_keys, env_state, actions)

            done_trial = done["__all__"].astype(jnp.bool_)
            reward_ctrl = gather_by_agent(reward, controlled_agent)
            current_outer_t = env_state.env_state.outer_t.astype(jnp.int32)
            episode_idx = jnp.clip(current_outer_t, 0, trial_episodes - 1)
            reward_weight = episode_reward_weights[episode_idx]
            reward_ctrl = reward_ctrl * reward_weight

            new_controlled_agent, new_teammate_ids = sample_trial_assignments(assign_key)
            controlled_agent = jnp.where(done_trial, new_controlled_agent, controlled_agent)
            teammate_policy_ids = jnp.where(
                done_trial[:, None], new_teammate_ids, teammate_policy_ids
            )
            selected_teammate_params_flat = jax.lax.cond(
                jnp.any(done_trial),
                lambda ids: select_teammate_params_flat(ids),
                lambda _ids: selected_teammate_params_flat,
                teammate_policy_ids,
            )

            next_prev_action = jnp.where(done_trial, jnp.zeros_like(ctrl_action), ctrl_action)
            next_prev_reward = jnp.where(done_trial, jnp.zeros_like(reward_ctrl), reward_ctrl)

            transition = Transition(
                reset=last_done,
                done=done_trial,
                action=ctrl_action,
                value=value,
                reward=reward_ctrl,
                log_prob=ctrl_log_prob,
                obs=obs_ctrl,
                prev_action=prev_action,
                prev_reward=prev_reward,
            )

            carry = (
                train_state,
                env_state,
                obsv,
                hidden_next,
                next_prev_action,
                next_prev_reward,
                done_trial,
                controlled_agent,
                teammate_policy_ids,
                selected_teammate_params_flat,
                update_step,
                rng,
            )
            return carry, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, num_steps, unroll=env_scan_unroll
        )

        (
            train_state,
            env_state,
            last_obs,
            hidden,
            prev_action,
            prev_reward,
            last_done,
            controlled_agent,
            teammate_policy_ids,
            selected_teammate_params_flat,
            update_step,
            rng,
        ) = runner_state

        last_obs_ctrl = gather_by_agent(last_obs, controlled_agent)
        _, _, last_value = network.apply(
            train_state.params,
            hidden,
            last_obs_ctrl,
            prev_action,
            prev_reward,
            last_done,
        )
        advantages, targets = _calculate_gae(traj_batch, last_value)

        def _update_epoch(carry, _):
            train_state, rng = carry
            rng, perm_rng = jax.random.split(rng)
            env_perm = jax.random.permutation(perm_rng, num_envs)
            traj_shuffled = jax.tree_util.tree_map(lambda x: x[:, env_perm], traj_batch)
            adv_shuffled = advantages[:, env_perm]
            target_shuffled = targets[:, env_perm]
            hidden_shuffled = initial_hidden[env_perm]

            traj_minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((num_steps, num_minibatches, envs_per_minibatch) + x.shape[2:]).swapaxes(0, 1),
                traj_shuffled,
            )
            adv_minibatches = adv_shuffled.reshape(
                (num_steps, num_minibatches, envs_per_minibatch)
            ).swapaxes(0, 1)
            target_minibatches = target_shuffled.reshape(
                (num_steps, num_minibatches, envs_per_minibatch)
            ).swapaxes(0, 1)
            hidden_minibatches = hidden_shuffled.reshape(
                (num_minibatches, envs_per_minibatch, rnn_hidden_size)
            )

            def _update_minibatch(ts, minibatch):
                traj_mb, adv_mb, target_mb, init_hidden_mb = minibatch

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                (_, metrics), grads = grad_fn(
                    ts.params, traj_mb, adv_mb, target_mb, init_hidden_mb
                )
                ts = ts.apply_gradients(grads=grads)
                return ts, metrics

            train_state, mb_metrics = jax.lax.scan(
                _update_minibatch,
                train_state,
                (traj_minibatches, adv_minibatches, target_minibatches, hidden_minibatches),
            )
            epoch_metrics = jax.tree_util.tree_map(lambda x: x.mean(), mb_metrics)
            return (train_state, rng), epoch_metrics

        (train_state, rng), epoch_metrics = jax.lax.scan(
            _update_epoch, (train_state, rng), None, int(config["UPDATE_EPOCHS"])
        )
        loss_metrics = jax.tree_util.tree_map(lambda x: x.mean(), epoch_metrics)

        update_step = update_step + 1
        def _episode_stats_scan(carry, transition):
            running_return, total_return, total_count = carry
            running_return = running_return + transition.reward
            done_f = transition.done.astype(jnp.float32)
            total_return = total_return + (running_return * done_f).sum()
            total_count = total_count + done_f.sum()
            running_return = jnp.where(transition.done, jnp.zeros_like(running_return), running_return)
            return (running_return, total_return, total_count), None

        (_, episode_return_sum, episode_count), _ = jax.lax.scan(
            _episode_stats_scan,
            (jnp.zeros((num_envs,), dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32), jnp.array(0.0, dtype=jnp.float32)),
            traj_batch,
        )
        mean_episode_return = jnp.where(episode_count > 0, episode_return_sum / episode_count, 0.0)

        metric = {
            "update_step": update_step.astype(jnp.float32),
            "env_step": (update_step * num_steps * num_envs).astype(jnp.float32),
            "reward_mean": traj_batch.reward.mean(),
            "episode_return_mean": mean_episode_return,
            "episode_count": episode_count,
            "total_loss": loss_metrics["total_loss"],
            "value_loss": loss_metrics["value_loss"],
            "policy_loss": loss_metrics["policy_loss"],
            "entropy": loss_metrics["entropy"],
            "approx_kl": loss_metrics["approx_kl"],
        }

        if config["LOG_WANDB"] and bool(config.get("LOG_IN_JIT", False)):

            def callback(m):
                wandb.log({k: float(v) for k, v in m.items()})

            should_log = ((update_step % log_interval_updates) == 0) | (
                update_step == int(config["NUM_UPDATES"])
            )
            jax.lax.cond(
                should_log,
                lambda m: jax.debug.callback(callback, m),
                lambda _m: None,
                metric,
            )

        runner_state = (
            train_state,
            env_state,
            last_obs,
            hidden,
            prev_action,
            prev_reward,
            last_done,
            controlled_agent,
            teammate_policy_ids,
            selected_teammate_params_flat,
            update_step,
            rng,
        )
        return runner_state, metric

    def init_runner_state(rng):
        rng, init_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((num_envs, *obs_shape), dtype=jnp.float32)
        dummy_prev_action = jnp.zeros((num_envs,), dtype=jnp.int32)
        dummy_prev_reward = jnp.zeros((num_envs,), dtype=jnp.float32)
        dummy_reset = jnp.zeros((num_envs,), dtype=jnp.bool_)
        dummy_hidden = jnp.zeros((num_envs, rnn_hidden_size), dtype=jnp.float32)
        network_params = network.init(
            init_rng,
            dummy_hidden,
            dummy_obs,
            dummy_prev_action,
            dummy_prev_reward,
            dummy_reset,
        )

        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"])),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"])),
                optax.adam(float(config["LR"]), eps=1e-5),
            )
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        rng, reset_rng = jax.random.split(rng)
        env_reset_keys = jax.random.split(reset_rng, num_envs)
        last_obs, env_state = jax.vmap(env.reset, in_axes=(0,))(env_reset_keys)

        rng, assign_rng = jax.random.split(rng)
        controlled_agent, teammate_policy_ids = sample_trial_assignments(assign_rng)
        selected_teammate_params_flat = select_teammate_params_flat(teammate_policy_ids)

        hidden = jnp.zeros((num_envs, rnn_hidden_size), dtype=jnp.float32)
        prev_action = jnp.zeros((num_envs,), dtype=jnp.int32)
        prev_reward = jnp.zeros((num_envs,), dtype=jnp.float32)
        last_done = jnp.ones((num_envs,), dtype=jnp.bool_)

        return (
            train_state,
            env_state,
            last_obs,
            hidden,
            prev_action,
            prev_reward,
            last_done,
            controlled_agent,
            teammate_policy_ids,
            selected_teammate_params_flat,
            0,
            rng,
        )

    def run_updates(runner_state, num_updates):
        metric_init = {
            "update_step": jnp.array(0.0, dtype=jnp.float32),
            "env_step": jnp.array(0.0, dtype=jnp.float32),
            "reward_mean": jnp.array(0.0, dtype=jnp.float32),
            "episode_return_mean": jnp.array(0.0, dtype=jnp.float32),
            "episode_count": jnp.array(0.0, dtype=jnp.float32),
            "total_loss": jnp.array(0.0, dtype=jnp.float32),
            "value_loss": jnp.array(0.0, dtype=jnp.float32),
            "policy_loss": jnp.array(0.0, dtype=jnp.float32),
            "entropy": jnp.array(0.0, dtype=jnp.float32),
            "approx_kl": jnp.array(0.0, dtype=jnp.float32),
        }

        def _fori_body(_, carry):
            state, _last_metric = carry
            state, metric = _update_step(state, None)
            return state, metric

        return jax.lax.fori_loop(
            0,
            num_updates,
            _fori_body,
            (runner_state, metric_init),
        )

    return init_runner_state, run_updates, total_updates


def make_train(config):
    init_runner_state, run_updates, total_updates = make_train_stepper(config)

    def train(rng):
        runner_state = init_runner_state(rng)
        runner_state, metrics = run_updates(runner_state, total_updates)
        return {"runner_state": runner_state, "metrics": metrics}

    return train


def train_and_evaluate(config: dict):
    num_seeds = int(config["NUM_SEEDS"])
    total_updates = int(config["TOTAL_TIMESTEPS"]) // int(config["NUM_STEPS"]) // int(config["NUM_ENVS"])
    config["NUM_UPDATES"] = total_updates
    evaluator = make_policy_evaluator(config)

    if not bool(config.get("EVAL_DURING_TRAIN", False)):
        rng = jax.random.PRNGKey(int(config["SEED"]))
        rngs = jax.random.split(rng, num_seeds)
        train_fn = jax.jit(make_train(config))
        outs = jax.vmap(train_fn)(rngs)
        jax.block_until_ready(outs["metrics"]["env_step"])
        if config["LOG_WANDB"] and not bool(config.get("LOG_IN_JIT", False)):
            last_metric = jax.tree_util.tree_map(
                lambda x: float(jax.device_get(x[0])),
                outs["metrics"],
            )
            wandb.log(last_metric)
        train_state = jax.tree_util.tree_map(lambda x: x[0], outs["runner_state"][0])
        evaluator(
            train_state.params,
            reward_progress=1.0,
            extra_log={
                "eval_update_step": float(total_updates),
                "eval_progress_pct": 100.0,
                "eval_env_step": float(total_updates * int(config["NUM_STEPS"]) * int(config["NUM_ENVS"])),
            },
        )
        return train_state

    if num_seeds != 1:
        raise ValueError("EVAL_DURING_TRAIN=True currently supports NUM_SEEDS=1.")
    eval_times = max(1, int(config.get("EVAL_TIMES", 100)))

    init_runner_state, run_updates, total_updates = make_train_stepper(config)
    init_runner_state_jit = jax.jit(init_runner_state)
    run_chunk_fns = {}
    eval_targets = np.unique(
        np.clip(
            np.rint(np.linspace(1, total_updates, num=eval_times)).astype(np.int32),
            1,
            total_updates,
        )
    )
    eval_targets = eval_targets.tolist()

    def get_run_chunk_fn(chunk_updates: int):
        if chunk_updates not in run_chunk_fns:
            run_chunk_fns[chunk_updates] = jax.jit(
                lambda runner_state, n=chunk_updates: run_updates(runner_state, n),
                donate_argnums=(0,),
            )
        return run_chunk_fns[chunk_updates]

    rng = jax.random.PRNGKey(int(config["SEED"]))
    runner_state = init_runner_state_jit(rng)

    updates_done = 0
    for target_update in eval_targets:
        chunk = int(target_update - updates_done)
        if chunk <= 0:
            continue
        runner_state, chunk_metrics = get_run_chunk_fn(chunk)(runner_state)
        if config["LOG_WANDB"] and not bool(config.get("LOG_IN_JIT", False)):
            last_metric = jax.tree_util.tree_map(lambda x: float(jax.device_get(x)), chunk_metrics)
            wandb.log(last_metric)

        updates_done += chunk
        progress_pct = 100.0 * updates_done / total_updates
        train_state = runner_state[0]
        evaluator(
            train_state.params,
            reward_progress=float(progress_pct / 100.0),
            extra_log={
                "eval_update_step": float(updates_done),
                "eval_progress_pct": float(progress_pct),
                "eval_env_step": float(updates_done * int(config["NUM_STEPS"]) * int(config["NUM_ENVS"])),
            },
        )

    return runner_state[0]


def single_run(config):
    config = OmegaConf.to_container(config, resolve=True)
    config["ENV_KWARGS"]["num_outer_steps"] = int(config["TRIAL_EPISODES"])

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["RL2", "RNN", config["ENV_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
        name=f"rl2_cnn_{config['ENV_NAME']}",
    )
    train_state = train_and_evaluate(config)

    print("** Saving Results **")
    filename = f"{config['ENV_NAME']}_rl2_seed{config['SEED']}"
    save_path = f"./checkpoints/rl2/{filename}.pkl"
    save_params(train_state.params, save_path)
    print(f"Saved params to {save_path}")


def tune(default_config):
    """
    WandB sweep entry for RL2.
    Default behavior only tunes LR and keeps rollout shape fixed.
    """
    import copy

    default_config = OmegaConf.to_container(default_config, resolve=True)
    default_config["ENV_KWARGS"]["num_outer_steps"] = int(default_config["TRIAL_EPISODES"])

    sweep_config = {
        "name": f"rl2_{default_config['ENV_NAME']}_lr_sweep",
        "method": "grid",
        "metric": {
            "name": "eval_episode_return_mean",
            "goal": "maximize",
        },
        "parameters": {
            "LR": {"values": [1e-4, 3e-4, 5e-4, 1e-3]},
        },
    }

    sweep_id = wandb.sweep(sweep_config, project=default_config["PROJECT"])

    def sweep_train():
        run = wandb.init()
        config = copy.deepcopy(default_config)
        config["LR"] = wandb.config.LR
        train_and_evaluate(config)

    wandb.agent(sweep_id, function=sweep_train)


# Store CS for Hydra
store = hydra.core.config_store.ConfigStore.instance()
store.store(name=f"rl2_cnn_generic", node={"defaults": [{"override hydra/help": "rl2_cnn_generic"}]})


# Hydra entrypoint
@hydra.main(version_base=None, config_path="config", config_name="rl2_cnn_generic")
def hydra_main(cfg):
    single_run(cfg)


if __name__ == "__main__":
    hydra_main()
