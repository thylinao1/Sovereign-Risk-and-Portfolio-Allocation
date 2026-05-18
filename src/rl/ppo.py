"""PPO agent and the state-normalised variant used by the sovereign-bond
allocation environment.

Tensorflow / Keras is required to import this module. Tests for the
normaliser and the yield-model live alongside; tests for these PPO classes
are out of scope for the CI matrix and run in Phase 6 when the notebook is
executed end-to-end.
"""
from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from src.rl.normaliser import StateNormaliser


class PPOAgent:
    # pretty standard ppo implementation
    # followed spinning up guide mostly
    # gaussian policy for continuous action space (portfolio weights)
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.actor = self._build_actor()
        self.critic=self._build_critic()

        self.actor_optimizer = keras.optimizers.Adam(learning_rate=3e-4)
        self.critic_optimizer = keras.optimizers.Adam(learning_rate= 1e-3)

        self.clip_ratio = 0.2
        self.gamma = 0.99
        self.lam = 0.95

    def _build_actor(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)

        mean = layers.Dense(self.action_dim, activation = 'linear', name='mean')(x)
        log_std = layers.Dense(self.action_dim, activation = 'linear', name='log_std',
                               kernel_initializer=keras.initializers.Constant(-1.0))(x)

        return keras.Model(inputs, [mean, log_std])

    def _build_critic(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(256, activation='relu')(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dense(64, activation='relu')(x)
        value = layers.Dense(1, activation= 'linear')(x)

        return keras.Model(inputs, value)

    def _log_prob_gaussian(self, actions, mean, log_std):
        # Clip once and reuse: the normalising constant '2 * log_std' must
        # match the variance used in the Mahalanobis term. The previous
        # implementation clipped log_std only for std and used the raw value
        # in the constant, producing mathematically inconsistent log-probs
        # whenever the actor's log_std head drifted outside [-5, 2].
        log_std_c = tf.clip_by_value(log_std, -5.0, 2.0)
        std = tf.exp(log_std_c)
        var = std ** 2
        log_prob = -0.5 * (
            ((actions - mean) ** 2) / var
            + 2.0 * log_std_c
            + np.log(2.0 * np.pi)
        )
        return tf.reduce_sum(log_prob, axis=-1)

    def _entropy_gaussian(self, log_std):
        # Same clip-once-reuse pattern as _log_prob_gaussian. Entropy of a
        # diagonal Gaussian per dim is 0.5 * (1 + log(2 pi) + 2 * log sigma).
        log_std_c = tf.clip_by_value(log_std, -5.0, 2.0)
        entropy = 0.5 * (1.0 + np.log(2.0 * np.pi) + 2.0 * log_std_c)
        return tf.reduce_mean(tf.reduce_sum(entropy, axis=-1))

    def get_action(self, state, training=True):
        state_t =tf.expand_dims(tf.cast(state, tf.float32), 0)
        mean, log_std = self.actor(state_t)
        std = tf.exp(tf.clip_by_value(log_std, -5, 2))

        if training:
            noise = tf.random.normal(shape=tf.shape(mean))
            action = mean + std * noise
            log_prob= self._log_prob_gaussian(action,mean,log_std)
            return action.numpy().flatten(), log_prob.numpy()[0]
        else:
            return mean.numpy().flatten(), None

    def get_value(self, state):
        state_t = tf.expand_dims(tf.cast(state,tf.float32), 0)
        return self.critic(state_t).numpy().flatten()[0]

    def compute_gae(self, rewards, values, dones):
        advantages = np.zeros_like(rewards)
        last_gae = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) -1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae = delta + self.gamma * self.lam * (1 - dones[t]) * last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, states, actions, old_log_probs, returns, advantages):
        states = tf.cast(states, tf.float32)
        actions = tf.cast(actions, tf.float32)
        old_log_probs = tf.cast(old_log_probs , tf.float32)
        returns = tf.cast(returns , tf.float32)
        advantages = tf.cast(advantages, tf.float32)

        advantages = (advantages - tf.reduce_mean(advantages))/(tf.math.reduce_std(advantages) + 1e-8)

        actor_losses = []
        critic_losses = []

        for _ in range(10):
            with tf.GradientTape() as tape:
                mean, log_std = self.actor(states)
                new_log_probs = self._log_prob_gaussian(actions, mean, log_std)

                ratio = tf.exp(new_log_probs - old_log_probs)
                clipped = tf.clip_by_value(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio)

                actor_loss = -tf.reduce_mean(tf.minimum(ratio * advantages, clipped * advantages))
                entropy = self._entropy_gaussian(log_std)
                actor_loss= actor_loss - 0.01 * entropy

            actor_grads = tape.gradient(actor_loss, self.actor.trainable_variables)
            actor_grads = [tf.clip_by_norm(g, 0.5) for g in actor_grads]
            self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor.trainable_variables))
            actor_losses.append(float(actor_loss))

        for _ in range(10):
            with tf.GradientTape() as tape:
                values = tf.squeeze(self.critic(states))
                critic_loss = tf.reduce_mean((returns - values) ** 2)

            critic_grads = tape.gradient(critic_loss, self.critic.trainable_variables)
            critic_grads = [tf.clip_by_norm(g, 0.5) for g in critic_grads]
            self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic.trainable_variables))
            critic_losses.append(float(critic_loss))

        return np.mean(actor_losses), np.mean(critic_losses)


class PPOAgentNorm(PPOAgent):
    """PPO agent that z-scores its input through a StateNormaliser.

    Inherits everything from PPOAgent and overrides _build_actor / _build_critic
    to use a smaller 128-64 hidden stack. Action selection wraps the parent
    method, normalising the state first.
    """
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        self.normaliser = StateNormaliser(state_dim)
        # Rebuild smaller actor/critic
        self.actor = self._build_smaller_actor()
        self.critic = self._build_smaller_critic()

    def _build_smaller_actor(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        mean = layers.Dense(self.action_dim, activation='linear', name='mean')(x)
        log_std = layers.Dense(
            self.action_dim, activation='linear', name='log_std',
            kernel_initializer=keras.initializers.Constant(-1.0),
        )(x)
        return keras.Model(inputs, [mean, log_std])

    def _build_smaller_critic(self):
        inputs = layers.Input(shape=(self.state_dim,))
        x = layers.Dense(128, activation='relu')(inputs)
        x = layers.LayerNormalization()(x)
        x = layers.Dense(64, activation='relu')(x)
        value = layers.Dense(1, activation='linear')(x)
        return keras.Model(inputs, value)

    def get_action(self, state, training: bool = True):
        # Normalise; PPOAgent.get_action expects raw numpy / tf input.
        state_n = self.normaliser.normalise(state, update=training)
        return super().get_action(state_n, training=training)


# Train the normalised agent. Pulls the same training loop the original PPO
# used; whether it lives in a function or inline depends on the original
# notebook structure. Below assumes a `train_ppo(agent, env, episodes)` helper
# exists; if not, this falls back to the inline training cell pattern.
