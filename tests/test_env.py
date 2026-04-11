import socialmeta
import jax
import jax.numpy as jnp

print(f"JAX version: {jax.__version__}")
print(f"JAX devices: {jax.devices()}")

print("\nTesting environment...")
env = socialmeta.make("coin_game")
print(f"Environment: {env.name}")
print(f"Num agents: {env.num_agents}")
print(f"Observation space: {env.observation_space()[0].shape}")
print(f"Action space: {env.action_space().n}")

key = jax.random.PRNGKey(0)
obs, state = env.reset(key)
print(f"\nReset successful!")
print(f"Obs shape: {obs[0].shape}")

# Actions should be array, not dict
actions = jnp.array([0, 1])  # [agent_0_action, agent_1_action]
key, subkey = jax.random.split(key)
obs_next, state_next, reward, done, info = env.step(subkey, state, actions)
print(f"Step successful!")
print(f"Reward: {reward}")
print(f"Done: {done}")

print("\n✅ All environment tests passed!")
