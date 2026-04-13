"""
Generic MAML (feed-forward) for multiple SSD environments with sampled teammate policies.
Supports: coop_mining, cleanup, coin_game, common_harvest, gift, mushrooms, pd_arena
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


class ActorCritic(nn.Module):
    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        embedding = CNNEncoder(self.activation)(obs)

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
    """Actor-only teammate architecture compatible with SVO checkpoints."""

    action_dim: int
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        activation = nn.relu if self.activation == "relu" else nn.tanh
        # Keep legacy checkpoint module path: /CNN_0/...
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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    mining_gold: jnp.ndarray


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
    trial_episodes = int(config["TRIAL_EPISODES"])
    last_episode_idx = trial_episodes - 1
    inner_lr = float(config.get("INNER_LR", 2e-3))
    inner_steps = max(1, int(config.get("INNER_STEPS", 1)))
    eval_with_inner_update = bool(config.get("EVAL_WITH_INNER_UPDATE", True))
    max_grad_norm = float(config["MAX_GRAD_NORM"])
    gae_scan_unroll = max(1, int(config.get("GAE_SCAN_UNROLL", 4)))
    eval_loss_scan_unroll = max(1, int(config.get("EVAL_LOSS_SCAN_UNROLL", 4)))
    eval_num_minibatches = max(
        1,
        int(config.get("EVAL_NUM_MINIBATCHES", config.get("MAML_NUM_MINIBATCHES", 2))),
    )
    eval_loss_remat = bool(config.get("EVAL_LOSS_REMAT", config.get("MAML_LOSS_REMAT", True)))
    eval_steps_per_episode = int(env_kwargs["num_inner_steps"])
    eval_steps = eval_steps_per_episode * trial_episodes
    default_support_steps = int(config.get("NUM_STEPS", eval_steps_per_episode))
    eval_support_steps = int(config.get("EVAL_SUPPORT_STEPS", default_support_steps))
    eval_support_steps = int(np.clip(eval_support_steps, 1, eval_steps_per_episode))
    eval_tail_steps = eval_steps_per_episode - eval_support_steps
    if eval_num % eval_num_minibatches != 0:
        raise ValueError(
            f"EVAL_NUM ({eval_num}) must be divisible by EVAL_NUM_MINIBATCHES ({eval_num_minibatches})."
        )
    eval_envs_per_minibatch = eval_num // eval_num_minibatches

    num_agents = env.num_agents
    action_dim = env.action_space().n
    obs_shape = env.observation_space()[0].shape
    episode_reward_weights = np.asarray(
        config.get("EPISODE_REWARD_WEIGHTS", [0.2, 0.3, 0.5]),
        dtype=np.float32,
    )
    assert len(episode_reward_weights) == trial_episodes, (
        f"EPISODE_REWARD_WEIGHTS length ({len(episode_reward_weights)}) must match "
        f"TRIAL_EPISODES ({trial_episodes})."
    )
    episode_reward_weights = jnp.asarray(episode_reward_weights, dtype=jnp.float32)

    teammate_bank, _ = load_teammate_policy_bank(
        config["TEAMMATE_POLICY_DIR"], config["TEAMMATE_POLICY_GLOB"]
    )
    num_teammate_policies = jax.tree_util.tree_leaves(teammate_bank)[0].shape[0]

    network = ActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
    teammate_network = FrozenActor(action_dim=action_dim, activation=config["ACTIVATION"])
    policy_apply_for_loss = (
        jax.checkpoint(lambda p, o: network.apply(p, o))
        if eval_loss_remat
        else (lambda p, o: network.apply(p, o))
    )

    def compute_teammate_logits_chunked(last_obs, teammate_policy_ids):
        logits_per_agent = []
        for agent_idx in range(num_agents):
            obs_agent = last_obs[:, agent_idx : agent_idx + 1, ...]
            params_agent = jax.tree_util.tree_map(
                lambda x, i=agent_idx: x[teammate_policy_ids[:, i]],
                teammate_bank,
            )
            logits_agent = jax.vmap(teammate_network.apply, in_axes=(0, 0))(
                params_agent,
                obs_agent,
            )
            logits_per_agent.append(jnp.squeeze(logits_agent, axis=1))
        return jnp.stack(logits_per_agent, axis=1)

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

    def _loss_fn(params, traj_mb, adv_mb, target_mb):
        def _ff_forward(_, inps):
            obs_t, action_t = inps
            logits_t, value_t = policy_apply_for_loss(params, obs_t)
            dist_t = distrax.Categorical(logits=logits_t)
            log_prob_t = dist_t.log_prob(action_t)
            entropy_t = dist_t.entropy()
            return None, (log_prob_t, value_t, entropy_t)

        _, (new_log_prob, new_value, entropy) = jax.lax.scan(
            _ff_forward,
            None,
            (traj_mb.obs, traj_mb.action),
            unroll=eval_loss_scan_unroll,
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
        return total_loss

    def _split_env_minibatches(traj_mb, adv_mb, target_mb):
        traj_minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape(
                (x.shape[0], eval_num_minibatches, eval_envs_per_minibatch) + x.shape[2:]
            ).swapaxes(0, 1),
            traj_mb,
        )
        adv_minibatches = adv_mb.reshape(
            (adv_mb.shape[0], eval_num_minibatches, eval_envs_per_minibatch)
        ).swapaxes(0, 1)
        target_minibatches = target_mb.reshape(
            (target_mb.shape[0], eval_num_minibatches, eval_envs_per_minibatch)
        ).swapaxes(0, 1)
        return traj_minibatches, adv_minibatches, target_minibatches

    def _clip_grads(grads):
        grad_norm = optax.global_norm(grads)
        scale = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-8))
        clipped = jax.tree_util.tree_map(lambda g: g * scale, grads)
        return clipped

    def _adapt_params_eval(params, traj_batch, last_value):
        advantages, targets = _calculate_gae(traj_batch, last_value)
        loss_grad_fn = jax.value_and_grad(_loss_fn)
        traj_minibatches, adv_minibatches, target_minibatches = _split_env_minibatches(
            traj_batch, advantages, targets
        )

        def _inner_body(_, p):
            def _minibatch_body(carry, minibatch):
                grad_sum = carry
                traj_mb, adv_mb, target_mb = minibatch
                _, grads_i = loss_grad_fn(p, traj_mb, adv_mb, target_mb)
                grad_sum = jax.tree_util.tree_map(lambda g, gi: g + gi, grad_sum, grads_i)
                return grad_sum, None

            init_grad_sum = jax.tree_util.tree_map(jnp.zeros_like, p)
            grad_sum, _ = jax.lax.scan(
                _minibatch_body,
                init_grad_sum,
                (traj_minibatches, adv_minibatches, target_minibatches),
            )
            grads = jax.tree_util.tree_map(lambda g: g / float(eval_num_minibatches), grad_sum)
            grads = _clip_grads(grads)
            return jax.tree_util.tree_map(lambda pp, g: pp - inner_lr * g, p, grads)

        return jax.lax.fori_loop(0, inner_steps, _inner_body, params)

    def _eval_once(policy_params, rng, controlled_agent, teammate_policy_ids):
        rng, reset_rng = jax.random.split(rng)
        reset_keys = jax.random.split(reset_rng, eval_num)
        last_obs, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_keys)

        reward_sum = jnp.zeros((eval_num,), dtype=jnp.float32)
        mining_gold_sum = jnp.zeros((eval_num,), dtype=jnp.float32)
        running_trial_return = jnp.zeros((eval_num,), dtype=jnp.float32)
        episode_return_sum = jnp.array(0.0, dtype=jnp.float32)
        episode_return_sq_sum = jnp.array(0.0, dtype=jnp.float32)
        episode_count = jnp.array(0.0, dtype=jnp.float32)
        eval_params = policy_params

        def _episode_body(ep_idx, carry):
            (
                env_state,
                last_obs,
                eval_params,
                reward_sum,
                mining_gold_sum,
                running_trial_return,
                episode_return_sum,
                episode_return_sq_sum,
                episode_count,
                rng,
            ) = carry

            metric_reward_weight = jnp.where(ep_idx == last_episode_idx, 1.0, 0.0)
            support_reward_weight = episode_reward_weights[jnp.minimum(ep_idx, trial_episodes - 1)]

            def _support_step(step_carry, _):
                (
                    env_state,
                    last_obs,
                    reward_sum,
                    mining_gold_sum,
                    running_trial_return,
                    episode_return_sum,
                    episode_return_sq_sum,
                    episode_count,
                    rng,
                ) = step_carry

                obs_ctrl = gather_by_agent(last_obs, controlled_agent)
                logits, value = network.apply(eval_params, obs_ctrl)

                rng, ctrl_key, teammate_key, env_key = jax.random.split(rng, 4)
                ctrl_dist = distrax.Categorical(logits=logits)
                ctrl_action = ctrl_dist.sample(seed=ctrl_key)
                ctrl_log_prob = ctrl_dist.log_prob(ctrl_action)

                teammate_logits = compute_teammate_logits_chunked(last_obs, teammate_policy_ids)
                actions = jax.random.categorical(teammate_key, teammate_logits, axis=-1).astype(
                    jnp.int32
                )
                actions = actions.at[jnp.arange(eval_num), controlled_agent].set(ctrl_action)

                step_keys = jax.random.split(env_key, eval_num)
                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(step_keys, env_state, actions)

                done_trial = done["__all__"].astype(jnp.bool_)
                done_trial_f = done_trial.astype(jnp.float32)
                reward_ctrl_raw = gather_by_agent(reward, controlled_agent)
                mining_gold_raw = gather_by_agent(info["mining_gold"], controlled_agent)

                metric_reward = reward_ctrl_raw * metric_reward_weight
                metric_mining_gold = mining_gold_raw * metric_reward_weight
                support_reward = reward_ctrl_raw * support_reward_weight

                reward_sum = reward_sum + metric_reward
                mining_gold_sum = mining_gold_sum + metric_mining_gold
                running_trial_return = running_trial_return + metric_reward
                episode_return_sum = episode_return_sum + (running_trial_return * done_trial_f).sum()
                episode_return_sq_sum = episode_return_sq_sum + (
                    (running_trial_return ** 2) * done_trial_f
                ).sum()
                episode_count = episode_count + done_trial_f.sum()
                running_trial_return = jnp.where(
                    done_trial, jnp.zeros_like(running_trial_return), running_trial_return
                )

                transition = Transition(
                    done=done_trial,
                    action=ctrl_action,
                    value=value,
                    reward=support_reward,
                    log_prob=ctrl_log_prob,
                    obs=obs_ctrl,
                    mining_gold=metric_mining_gold,
                )

                step_carry = (
                    env_state,
                    obsv,
                    reward_sum,
                    mining_gold_sum,
                    running_trial_return,
                    episode_return_sum,
                    episode_return_sq_sum,
                    episode_count,
                    rng,
                )
                return step_carry, transition

            episode_carry = (
                env_state,
                last_obs,
                reward_sum,
                mining_gold_sum,
                running_trial_return,
                episode_return_sum,
                episode_return_sq_sum,
                episode_count,
                rng,
            )
            episode_carry, support_traj = jax.lax.scan(
                _support_step, episode_carry, None, eval_support_steps
            )

            (
                env_state,
                last_obs_after_support,
                reward_sum,
                mining_gold_sum,
                running_trial_return,
                episode_return_sum,
                episode_return_sq_sum,
                episode_count,
                rng,
            ) = episode_carry

            def _tail_scan_fn(step_carry):
                def _tail_step(inner_carry, _):
                    (
                        env_state,
                        last_obs,
                        reward_sum,
                        mining_gold_sum,
                        running_trial_return,
                        episode_return_sum,
                        episode_return_sq_sum,
                        episode_count,
                        rng,
                    ) = inner_carry

                    obs_ctrl = gather_by_agent(last_obs, controlled_agent)
                    logits, _ = network.apply(eval_params, obs_ctrl)

                    rng, ctrl_key, teammate_key, env_key = jax.random.split(rng, 4)
                    ctrl_dist = distrax.Categorical(logits=logits)
                    ctrl_action = ctrl_dist.sample(seed=ctrl_key)

                    teammate_logits = compute_teammate_logits_chunked(last_obs, teammate_policy_ids)
                    actions = jax.random.categorical(
                        teammate_key, teammate_logits, axis=-1
                    ).astype(jnp.int32)
                    actions = actions.at[jnp.arange(eval_num), controlled_agent].set(ctrl_action)

                    step_keys = jax.random.split(env_key, eval_num)
                    obsv, env_state, reward, done, info = jax.vmap(
                        env.step, in_axes=(0, 0, 0)
                    )(step_keys, env_state, actions)

                    done_trial = done["__all__"].astype(jnp.bool_)
                    done_trial_f = done_trial.astype(jnp.float32)
                    reward_ctrl_raw = gather_by_agent(reward, controlled_agent)
                    mining_gold_raw = gather_by_agent(info["mining_gold"], controlled_agent)
                    metric_reward = reward_ctrl_raw * metric_reward_weight
                    metric_mining_gold = mining_gold_raw * metric_reward_weight

                    reward_sum = reward_sum + metric_reward
                    mining_gold_sum = mining_gold_sum + metric_mining_gold
                    running_trial_return = running_trial_return + metric_reward
                    episode_return_sum = episode_return_sum + (running_trial_return * done_trial_f).sum()
                    episode_return_sq_sum = episode_return_sq_sum + (
                        (running_trial_return ** 2) * done_trial_f
                    ).sum()
                    episode_count = episode_count + done_trial_f.sum()
                    running_trial_return = jnp.where(
                        done_trial, jnp.zeros_like(running_trial_return), running_trial_return
                    )

                    inner_carry = (
                        env_state,
                        obsv,
                        reward_sum,
                        mining_gold_sum,
                        running_trial_return,
                        episode_return_sum,
                        episode_return_sq_sum,
                        episode_count,
                        rng,
                    )
                    return inner_carry, None

                return jax.lax.scan(_tail_step, step_carry, None, eval_tail_steps)[0]

            episode_carry = jax.lax.cond(
                eval_tail_steps > 0,
                _tail_scan_fn,
                lambda x: x,
                episode_carry,
            )

            (
                env_state,
                last_obs,
                reward_sum,
                mining_gold_sum,
                running_trial_return,
                episode_return_sum,
                episode_return_sq_sum,
                episode_count,
                rng,
            ) = episode_carry

            obs_ctrl_after_support = gather_by_agent(last_obs_after_support, controlled_agent)
            _, last_value_support = network.apply(eval_params, obs_ctrl_after_support)

            do_adapt = eval_with_inner_update & (ep_idx < last_episode_idx)
            eval_params = jax.lax.cond(
                do_adapt,
                lambda p: _adapt_params_eval(p, support_traj, last_value_support),
                lambda p: p,
                eval_params,
            )

            carry = (
                env_state,
                last_obs,
                eval_params,
                reward_sum,
                mining_gold_sum,
                running_trial_return,
                episode_return_sum,
                episode_return_sq_sum,
                episode_count,
                rng,
            )
            return carry

        carry = (
            env_state,
            last_obs,
            eval_params,
            reward_sum,
            mining_gold_sum,
            running_trial_return,
            episode_return_sum,
            episode_return_sq_sum,
            episode_count,
            rng,
        )
        carry = jax.lax.fori_loop(0, trial_episodes, _episode_body, carry)
        (
            _env_state,
            _last_obs,
            _eval_params,
            reward_sum,
            mining_gold_sum,
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
            "eval_mining_gold_mean": mining_gold_sum.mean() / float(eval_steps),
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
            f"reward_mean={eval_summary['eval_reward_mean']:.4f}, "
            f"mining_gold_mean={eval_summary['eval_mining_gold_mean']:.4f}"
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
    num_agents = env.num_agents
    action_dim = env.action_space().n
    obs_shape = env.observation_space()[0].shape
    trial_episodes = int(config["TRIAL_EPISODES"])
    episode_steps = int(env_kwargs["num_inner_steps"])
    steps_per_update = episode_steps * trial_episodes
    total_updates = int(config["TOTAL_TIMESTEPS"]) // steps_per_update // num_envs
    inner_steps = max(1, int(config.get("INNER_STEPS", 1)))
    inner_lr = float(config.get("INNER_LR", 2e-3))
    outer_lr = float(config.get("OUTER_LR", 1e-4))
    first_order_maml = bool(config.get("FIRST_ORDER_MAML", True))
    max_grad_norm = float(config["MAX_GRAD_NORM"])

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
    gae_scan_unroll = max(1, int(config.get("GAE_SCAN_UNROLL", 4)))
    maml_loss_scan_unroll = max(1, int(config.get("MAML_LOSS_SCAN_UNROLL", 4)))
    maml_num_minibatches = max(
        1,
        int(config.get("MAML_NUM_MINIBATCHES", config.get("NUM_MINIBATCHES", 2))),
    )
    maml_loss_remat = bool(config.get("MAML_LOSS_REMAT", True))
    if num_envs % maml_num_minibatches != 0:
        raise ValueError(
            f"NUM_ENVS ({num_envs}) must be divisible by MAML_NUM_MINIBATCHES ({maml_num_minibatches})."
        )
    envs_per_maml_minibatch = num_envs // maml_num_minibatches

    config["NUM_UPDATES"] = total_updates
    config["TRAIN_STEPS_PER_UPDATE"] = steps_per_update

    teammate_bank, teammate_files = load_teammate_policy_bank(
        config["TEAMMATE_POLICY_DIR"], config["TEAMMATE_POLICY_GLOB"]
    )
    num_teammate_policies = jax.tree_util.tree_leaves(teammate_bank)[0].shape[0]
    print(f"Loaded {num_teammate_policies} teammate policies from {config['TEAMMATE_POLICY_DIR']}")
    for path in teammate_files:
        print(f"  - {path}")

    network = ActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
    teammate_network = FrozenActor(action_dim=action_dim, activation=config["ACTIVATION"])
    policy_apply_for_loss = (
        jax.checkpoint(lambda p, o: network.apply(p, o))
        if maml_loss_remat
        else (lambda p, o: network.apply(p, o))
    )

    def outer_linear_schedule(count):
        frac = 1.0 - count / jnp.maximum(1.0, float(config["NUM_UPDATES"]))
        return outer_lr * frac

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

    def compute_teammate_logits_chunked(last_obs, teammate_policy_ids):
        logits_per_agent = []
        for agent_idx in range(num_agents):
            obs_agent = last_obs[:, agent_idx : agent_idx + 1, ...]
            params_agent = jax.tree_util.tree_map(
                lambda x, i=agent_idx: x[teammate_policy_ids[:, i]],
                teammate_bank,
            )
            logits_agent = jax.vmap(teammate_network.apply, in_axes=(0, 0))(
                params_agent,
                obs_agent,
            )
            logits_per_agent.append(jnp.squeeze(logits_agent, axis=1))
        return jnp.stack(logits_per_agent, axis=1)

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

    def _loss_fn(params, traj_mb, adv_mb, target_mb):
        def _ff_forward(_, inps):
            obs_t, action_t = inps
            logits_t, value_t = policy_apply_for_loss(params, obs_t)
            dist_t = distrax.Categorical(logits=logits_t)
            log_prob_t = dist_t.log_prob(action_t)
            entropy_t = dist_t.entropy()
            return None, (log_prob_t, value_t, entropy_t)

        _, (new_log_prob, new_value, entropy) = jax.lax.scan(
            _ff_forward,
            None,
            (traj_mb.obs, traj_mb.action),
            unroll=maml_loss_scan_unroll,
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

    def _zero_loss_metrics():
        return {
            "total_loss": jnp.array(0.0, dtype=jnp.float32),
            "value_loss": jnp.array(0.0, dtype=jnp.float32),
            "policy_loss": jnp.array(0.0, dtype=jnp.float32),
            "entropy": jnp.array(0.0, dtype=jnp.float32),
            "approx_kl": jnp.array(0.0, dtype=jnp.float32),
        }

    def _split_env_minibatches(traj_mb, adv_mb, target_mb):
        traj_minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape(
                (x.shape[0], maml_num_minibatches, envs_per_maml_minibatch) + x.shape[2:]
            ).swapaxes(0, 1),
            traj_mb,
        )
        adv_minibatches = adv_mb.reshape(
            (adv_mb.shape[0], maml_num_minibatches, envs_per_maml_minibatch)
        ).swapaxes(0, 1)
        target_minibatches = target_mb.reshape(
            (target_mb.shape[0], maml_num_minibatches, envs_per_maml_minibatch)
        ).swapaxes(0, 1)
        return traj_minibatches, adv_minibatches, target_minibatches

    def _loss_grad_minibatched(params, traj_mb, adv_mb, target_mb):
        grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
        traj_minibatches, adv_minibatches, target_minibatches = _split_env_minibatches(
            traj_mb, adv_mb, target_mb
        )

        def _minibatch_step(carry, minibatch):
            loss_sum, metrics_sum, grads_sum = carry
            traj_i, adv_i, target_i = minibatch
            (loss_i, metrics_i), grads_i = grad_fn(params, traj_i, adv_i, target_i)
            loss_sum = loss_sum + loss_i
            metrics_sum = jax.tree_util.tree_map(lambda a, b: a + b, metrics_sum, metrics_i)
            grads_sum = jax.tree_util.tree_map(lambda a, b: a + b, grads_sum, grads_i)
            return (loss_sum, metrics_sum, grads_sum), None

        init_carry = (
            jnp.array(0.0, dtype=jnp.float32),
            _zero_loss_metrics(),
            jax.tree_util.tree_map(jnp.zeros_like, params),
        )
        (loss_sum, metrics_sum, grads_sum), _ = jax.lax.scan(
            _minibatch_step,
            init_carry,
            (traj_minibatches, adv_minibatches, target_minibatches),
        )
        inv = 1.0 / float(maml_num_minibatches)
        return (
            loss_sum * inv,
            jax.tree_util.tree_map(lambda m: m * inv, metrics_sum),
            jax.tree_util.tree_map(lambda g: g * inv, grads_sum),
        )

    def _clip_grads(grads):
        grad_norm = optax.global_norm(grads)
        scale = jnp.minimum(1.0, max_grad_norm / (grad_norm + 1e-8))
        clipped = jax.tree_util.tree_map(lambda g: g * scale, grads)
        return clipped, grad_norm

    def _adapt_params(base_params, support_traj, support_adv, support_target):
        def _inner_step(carry, _):
            params, _ = carry
            support_loss, _, grads = _loss_grad_minibatched(
                params, support_traj, support_adv, support_target
            )
            grads, grad_norm = _clip_grads(grads)
            if first_order_maml:
                grads = jax.lax.stop_gradient(grads)
            params = jax.tree_util.tree_map(lambda p, g: p - inner_lr * g, params, grads)
            if first_order_maml:
                params = jax.tree_util.tree_map(jax.lax.stop_gradient, params)
            return (params, grad_norm), support_loss

        (adapted_params, grad_norm), support_losses = jax.lax.scan(
            _inner_step,
            (base_params, jnp.array(0.0, dtype=jnp.float32)),
            None,
            length=inner_steps,
        )
        return adapted_params, grad_norm, support_losses[-1]

    def _rollout_episode(carry, policy_params, episode_idx: int):
        episode_reward_weight = episode_reward_weights[min(episode_idx, trial_episodes - 1)]

        def _env_step(carry, _):
            (
                env_state,
                last_obs,
                controlled_agent,
                teammate_policy_ids,
                rng,
            ) = carry

            obs_ctrl = gather_by_agent(last_obs, controlled_agent)
            logits, value = network.apply(policy_params, obs_ctrl)

            rng, ctrl_key, teammate_key, env_key, assign_key = jax.random.split(rng, 5)

            ctrl_dist = distrax.Categorical(logits=logits)
            ctrl_action = ctrl_dist.sample(seed=ctrl_key)
            ctrl_log_prob = ctrl_dist.log_prob(ctrl_action)

            teammate_logits = compute_teammate_logits_chunked(last_obs, teammate_policy_ids)
            actions = jax.random.categorical(teammate_key, teammate_logits, axis=-1).astype(jnp.int32)
            actions = actions.at[jnp.arange(num_envs), controlled_agent].set(ctrl_action)

            step_keys = jax.random.split(env_key, num_envs)
            obsv, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(step_keys, env_state, actions)

            done_trial = done["__all__"].astype(jnp.bool_)
            reward_ctrl = gather_by_agent(reward, controlled_agent)
            mining_gold = gather_by_agent(info["mining_gold"], controlled_agent)
            reward_ctrl = reward_ctrl * episode_reward_weight
            mining_gold = mining_gold * episode_reward_weight

            new_controlled_agent, new_teammate_ids = sample_trial_assignments(assign_key)
            controlled_agent = jnp.where(done_trial, new_controlled_agent, controlled_agent)
            teammate_policy_ids = jnp.where(
                done_trial[:, None], new_teammate_ids, teammate_policy_ids
            )

            transition = Transition(
                done=done_trial,
                action=ctrl_action,
                value=value,
                reward=reward_ctrl,
                log_prob=ctrl_log_prob,
                obs=obs_ctrl,
                mining_gold=mining_gold,
            )

            carry = (
                env_state,
                obsv,
                controlled_agent,
                teammate_policy_ids,
                rng,
            )
            return carry, transition

        carry, traj_batch = jax.lax.scan(
            _env_step, carry, None, episode_steps, unroll=env_scan_unroll
        )
        (
            env_state,
            last_obs,
            controlled_agent,
            teammate_policy_ids,
            rng,
        ) = carry

        last_obs_ctrl = gather_by_agent(last_obs, controlled_agent)
        _, last_value = network.apply(policy_params, last_obs_ctrl)
        advantages, targets = _calculate_gae(traj_batch, last_value)

        carry = (
            env_state,
            last_obs,
            controlled_agent,
            teammate_policy_ids,
            rng,
        )
        return carry, traj_batch, advantages, targets

    def _episode_stats(traj_batch):
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
            (
                jnp.zeros((num_envs,), dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
                jnp.array(0.0, dtype=jnp.float32),
            ),
            traj_batch,
        )
        mean_episode_return = jnp.where(episode_count > 0, episode_return_sum / episode_count, 0.0)
        return mean_episode_return, episode_count

    def _update_step(runner_state, _):
        (
            train_state,
            env_state,
            last_obs,
            controlled_agent,
            teammate_policy_ids,
            update_step,
            rng,
        ) = runner_state

        rollout_carry_init = (
            env_state,
            last_obs,
            controlled_agent,
            teammate_policy_ids,
            rng,
        )

        rollout_carry = rollout_carry_init

        if first_order_maml:
            behavior_params = train_state.params
            query_loss_sum = jnp.array(0.0, dtype=jnp.float32)
            query_metrics_sum = _zero_loss_metrics()
            outer_grads_sum = jax.tree_util.tree_map(jnp.zeros_like, train_state.params)
            query_count = jnp.array(0.0, dtype=jnp.float32)

            support_reward_sum = jnp.array(0.0, dtype=jnp.float32)
            query_reward_sum = jnp.array(0.0, dtype=jnp.float32)
            support_mining_gold_sum = jnp.array(0.0, dtype=jnp.float32)
            query_mining_gold_sum = jnp.array(0.0, dtype=jnp.float32)
            support_episode_count = jnp.array(0.0, dtype=jnp.float32)
            query_episode_metric_count = jnp.array(0.0, dtype=jnp.float32)

            inner_grad_norm_sum = jnp.array(0.0, dtype=jnp.float32)
            support_loss_sum = jnp.array(0.0, dtype=jnp.float32)
            inner_update_count = jnp.array(0.0, dtype=jnp.float32)
            query_episode_return = jnp.array(0.0, dtype=jnp.float32)
            query_episode_count = jnp.array(0.0, dtype=jnp.float32)

            for ep_idx in range(trial_episodes):
                rollout_params = behavior_params
                rollout_carry, ep_traj, ep_adv, ep_target = _rollout_episode(
                    rollout_carry, rollout_params, ep_idx
                )

                if ep_idx == trial_episodes - 1:
                    query_episode_return, query_episode_count = _episode_stats(ep_traj)

                if ep_idx < trial_episodes - 1:
                    support_reward_sum = support_reward_sum + ep_traj.reward.mean()
                    support_mining_gold_sum = support_mining_gold_sum + ep_traj.mining_gold.mean()
                    support_episode_count = support_episode_count + 1.0

                if ep_idx > 0:
                    query_reward_sum = query_reward_sum + ep_traj.reward.mean()
                    query_mining_gold_sum = query_mining_gold_sum + ep_traj.mining_gold.mean()
                    query_episode_metric_count = query_episode_metric_count + 1.0

                    q_loss, q_metrics, q_grads = _loss_grad_minibatched(
                        rollout_params, ep_traj, ep_adv, ep_target
                    )
                    query_loss_sum = query_loss_sum + q_loss
                    query_metrics_sum = jax.tree_util.tree_map(
                        lambda a, b: a + b, query_metrics_sum, q_metrics
                    )
                    outer_grads_sum = jax.tree_util.tree_map(
                        lambda a, b: a + b, outer_grads_sum, q_grads
                    )
                    query_count = query_count + 1.0

                if ep_idx < trial_episodes - 1:
                    behavior_params, inner_grad_norm_i, support_loss_i = _adapt_params(
                        behavior_params, ep_traj, ep_adv, ep_target
                    )
                    inner_grad_norm_sum = inner_grad_norm_sum + inner_grad_norm_i
                    support_loss_sum = support_loss_sum + support_loss_i
                    inner_update_count = inner_update_count + 1.0

            query_denom = jnp.maximum(query_count, 1.0)
            support_denom = jnp.maximum(support_episode_count, 1.0)
            query_metric_denom = jnp.maximum(query_episode_metric_count, 1.0)
            inner_denom = jnp.maximum(inner_update_count, 1.0)

            meta_loss = query_loss_sum / query_denom
            outer_loss_metrics = jax.tree_util.tree_map(
                lambda m: m / query_denom, query_metrics_sum
            )
            outer_grads = jax.tree_util.tree_map(lambda g: g / query_denom, outer_grads_sum)
            support_reward_mean = support_reward_sum / support_denom
            query_reward_mean = query_reward_sum / query_metric_denom
            support_mining_gold_mean = (
                support_mining_gold_sum / support_denom
            ) * env_kwargs["num_inner_steps"]
            query_mining_gold_mean = (
                query_mining_gold_sum / query_metric_denom
            ) * env_kwargs["num_inner_steps"]
            inner_grad_norm = inner_grad_norm_sum / inner_denom
            support_inner_loss = support_loss_sum / inner_denom
        else:
            behavior_params = train_state.params
            episode_trajs = []
            episode_advs = []
            episode_targets = []
            inner_grad_norms = []
            support_losses = []

            for ep_idx in range(trial_episodes):
                rollout_carry, ep_traj, ep_adv, ep_target = _rollout_episode(
                    rollout_carry, behavior_params, ep_idx
                )
                episode_trajs.append(ep_traj)
                episode_advs.append(ep_adv)
                episode_targets.append(ep_target)

                if ep_idx < trial_episodes - 1:
                    behavior_params, inner_grad_norm_i, support_loss_i = _adapt_params(
                        behavior_params, ep_traj, ep_adv, ep_target
                    )
                    inner_grad_norms.append(inner_grad_norm_i)
                    support_losses.append(support_loss_i)

            def _mean_or_zero(xs):
                if len(xs) == 0:
                    return jnp.array(0.0, dtype=jnp.float32)
                return jnp.stack(xs).mean()

            if trial_episodes <= 1:
                meta_loss = jnp.array(0.0, dtype=jnp.float32)
                outer_loss_metrics = _zero_loss_metrics()
                outer_grads = jax.tree_util.tree_map(jnp.zeros_like, train_state.params)
            else:

                def _meta_objective(base_params):
                    params_i = base_params
                    query_losses = []
                    query_metrics = []

                    for ep_idx in range(trial_episodes):
                        if ep_idx > 0:
                            q_loss, q_metrics = _loss_fn(
                                params_i,
                                episode_trajs[ep_idx],
                                episode_advs[ep_idx],
                                episode_targets[ep_idx],
                            )
                            query_losses.append(q_loss)
                            query_metrics.append(q_metrics)

                        if ep_idx < trial_episodes - 1:
                            params_i, _, _ = _adapt_params(
                                params_i,
                                episode_trajs[ep_idx],
                                episode_advs[ep_idx],
                                episode_targets[ep_idx],
                            )

                    outer_loss = jnp.stack(query_losses).mean()
                    outer_metrics = jax.tree_util.tree_map(
                        lambda *xs: jnp.stack(xs).mean(),
                        *query_metrics,
                    )
                    return outer_loss, outer_metrics

                grad_fn = jax.value_and_grad(_meta_objective, has_aux=True)
                (meta_loss, outer_loss_metrics), outer_grads = grad_fn(train_state.params)

            query_episode_return, query_episode_count = _episode_stats(episode_trajs[-1])
            support_reward_mean = _mean_or_zero([t.reward.mean() for t in episode_trajs[:-1]])
            query_reward_mean = _mean_or_zero([t.reward.mean() for t in episode_trajs[1:]])
            support_mining_gold_mean = _mean_or_zero(
                [t.mining_gold.mean() for t in episode_trajs[:-1]]
            ) * env_kwargs["num_inner_steps"]
            query_mining_gold_mean = _mean_or_zero(
                [t.mining_gold.mean() for t in episode_trajs[1:]]
            ) * env_kwargs["num_inner_steps"]
            inner_grad_norm = _mean_or_zero(inner_grad_norms)
            support_inner_loss = _mean_or_zero(support_losses)

        outer_grad_norm = optax.global_norm(outer_grads)
        train_state = train_state.apply_gradients(grads=outer_grads)

        update_step = update_step + 1

        metric = {
            "update_step": update_step.astype(jnp.float32),
            "env_step": (update_step * steps_per_update * num_envs).astype(jnp.float32),
            "support_reward_mean": support_reward_mean,
            "query_reward_mean": query_reward_mean,
            "episode_return_mean": query_episode_return,
            "episode_count": query_episode_count,
            "support_mining_gold_mean": support_mining_gold_mean,
            "query_mining_gold_mean": query_mining_gold_mean,
            "meta_loss": meta_loss,
            "support_total_loss": support_inner_loss,
            "query_total_loss": outer_loss_metrics["total_loss"],
            "query_value_loss": outer_loss_metrics["value_loss"],
            "query_policy_loss": outer_loss_metrics["policy_loss"],
            "query_entropy": outer_loss_metrics["entropy"],
            "query_approx_kl": outer_loss_metrics["approx_kl"],
            "outer_value_loss": outer_loss_metrics["value_loss"],
            "outer_policy_loss": outer_loss_metrics["policy_loss"],
            "outer_entropy": outer_loss_metrics["entropy"],
            "inner_grad_norm": inner_grad_norm,
            "outer_grad_norm": outer_grad_norm,
            "inner_lr": jnp.array(inner_lr, dtype=jnp.float32),
            "outer_lr": jnp.array(outer_lr, dtype=jnp.float32),
            "inner_support_loss": support_inner_loss,
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

        (
            env_state,
            last_obs,
            controlled_agent,
            teammate_policy_ids,
            rng,
        ) = rollout_carry
        runner_state = (
            train_state,
            env_state,
            last_obs,
            controlled_agent,
            teammate_policy_ids,
            update_step,
            rng,
        )
        return runner_state, metric

    def init_runner_state(rng):
        rng, init_rng = jax.random.split(rng)
        dummy_obs = jnp.zeros((1, *obs_shape), dtype=jnp.float32)
        network_params = network.init(init_rng, dummy_obs)

        if bool(config.get("ANNEAL_OUTER_LR", config.get("ANNEAL_LR", False))):
            tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(learning_rate=outer_linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(max_grad_norm),
                optax.adam(outer_lr, eps=1e-5),
            )
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        rng, reset_rng = jax.random.split(rng)
        env_reset_keys = jax.random.split(reset_rng, num_envs)
        last_obs, env_state = jax.vmap(env.reset, in_axes=(0,))(env_reset_keys)

        rng, assign_rng = jax.random.split(rng)
        controlled_agent, teammate_policy_ids = sample_trial_assignments(assign_rng)

        return (
            train_state,
            env_state,
            last_obs,
            controlled_agent,
            teammate_policy_ids,
            0,
            rng,
        )

    def run_updates(runner_state, num_updates):
        metric_init = {
            "update_step": jnp.array(0.0, dtype=jnp.float32),
            "env_step": jnp.array(0.0, dtype=jnp.float32),
            "support_reward_mean": jnp.array(0.0, dtype=jnp.float32),
            "query_reward_mean": jnp.array(0.0, dtype=jnp.float32),
            "episode_return_mean": jnp.array(0.0, dtype=jnp.float32),
            "episode_count": jnp.array(0.0, dtype=jnp.float32),
            "support_mining_gold_mean": jnp.array(0.0, dtype=jnp.float32),
            "query_mining_gold_mean": jnp.array(0.0, dtype=jnp.float32),
            "meta_loss": jnp.array(0.0, dtype=jnp.float32),
            "support_total_loss": jnp.array(0.0, dtype=jnp.float32),
            "query_total_loss": jnp.array(0.0, dtype=jnp.float32),
            "query_value_loss": jnp.array(0.0, dtype=jnp.float32),
            "query_policy_loss": jnp.array(0.0, dtype=jnp.float32),
            "query_entropy": jnp.array(0.0, dtype=jnp.float32),
            "query_approx_kl": jnp.array(0.0, dtype=jnp.float32),
            "outer_value_loss": jnp.array(0.0, dtype=jnp.float32),
            "outer_policy_loss": jnp.array(0.0, dtype=jnp.float32),
            "outer_entropy": jnp.array(0.0, dtype=jnp.float32),
            "inner_grad_norm": jnp.array(0.0, dtype=jnp.float32),
            "outer_grad_norm": jnp.array(0.0, dtype=jnp.float32),
            "inner_lr": jnp.array(inner_lr, dtype=jnp.float32),
            "outer_lr": jnp.array(outer_lr, dtype=jnp.float32),
            "inner_support_loss": jnp.array(0.0, dtype=jnp.float32),
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
    steps_per_update = int(
        config.get(
            "TRAIN_STEPS_PER_UPDATE",
            int(config["ENV_KWARGS"]["num_inner_steps"]) * int(config["TRIAL_EPISODES"]),
        )
    )
    total_updates = int(config["TOTAL_TIMESTEPS"]) // steps_per_update // int(config["NUM_ENVS"])
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
                "eval_env_step": float(total_updates * steps_per_update * int(config["NUM_ENVS"])),
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
                "eval_env_step": float(updates_done * steps_per_update * int(config["NUM_ENVS"])),
            },
        )

    return runner_state[0]


def single_run(config):
    config = OmegaConf.to_container(config, resolve=True)
    config["ENV_KWARGS"]["num_outer_steps"] = int(config["TRIAL_EPISODES"])

    wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=["MAML", "FF", "META"],
        config=config,
        mode=config["WANDB_MODE"],
        name="maml_cnn_coop_mining",
    )
    train_state = train_and_evaluate(config)

    print("** Saving Results **")
    filename = f"{config['ENV_NAME']}_maml_seed{config['SEED']}"
    save_path = f"./checkpoints/maml/{filename}.pkl"
    save_params(train_state.params, save_path)
    print(f"Saved params to {save_path}")


def tune(default_config):
    """
    WandB sweep entry for MAML.
    Default behavior tunes outer/inner learning rates.
    """
    import copy

    default_config = OmegaConf.to_container(default_config, resolve=True)
    default_config["ENV_KWARGS"]["num_outer_steps"] = int(default_config["TRIAL_EPISODES"])

    sweep_config = {
        "name": "maml_coop_mining_lr_sweep",
        "method": "grid",
        "metric": {
            "name": "eval_episode_return_mean",
            "goal": "maximize",
        },
        "parameters": {
            "OUTER_LR": {"values": [3e-4]},
            "INNER_LR": {"values": [1e-3, 1e-2, 5e-1]},
            # "SEED": {"values": [30, 42, 52]},
        },
    }

    def wrapped_make_train():
        wandb.init(
            entity=default_config["ENTITY"],
            project=default_config["PROJECT"],
            mode=default_config["WANDB_MODE"],
        )

        config = copy.deepcopy(default_config)
        for k, v in dict(wandb.config).items():
            if "." in k:
                parent, child = k.split(".", 1)
                config[parent][child] = v
            else:
                config[k] = v

        # Keep rollout shape fixed for fair LR comparison.
        config["NUM_ENVS"] = 512
        config["NUM_STEPS"] = 384

        run_name = (
            f"maml_outer{config['OUTER_LR']}_inner{config['INNER_LR']}_seed{config['SEED']}"
        )
        wandb.run.name = run_name
        print("Running sweep:", run_name)
        train_and_evaluate(config)

    wandb.login()
    sweep_id = wandb.sweep(
        sweep_config, entity=default_config["ENTITY"], project=default_config["PROJECT"]
    )
    wandb.agent(sweep_id, wrapped_make_train, count=1000)


@hydra.main(version_base=None, config_path="config", config_name="maml_cnn_coop_mining")
def main(config):
    if config.get("TUNE", False):
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
