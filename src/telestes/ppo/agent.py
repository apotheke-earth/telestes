import os
import sys

import torch
import numpy as np

from tqdm import tqdm

from ..networks import PolicyNetwork, ValueNetwork

class Agent:
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        gamma: float = 0.99,
        epochs: int = 10,
        gae_lambda: float =0.95,
        policy_clip: float = 0.1,
        entropy: float = 0.001,
        batch_size: int = 64,
        actor_ridge: float = 0,
        actor_coefficient: float = 1.,
        actor_clip_grad_norm: float = None,
        critic_ridge: float = 0,
        critic_coefficient: float = .5,
        critic_clip_grad_norm: float = None,
        name: str = 'telestis',
        verbose: bool = False,
        load: bool = False,
        chkpt_dir: str = 'chkpts',
        device: str = 'cpu',
        **kwargs
    ):
        actor_hparams = {
            **kwargs.get('actor_hparams', {})
        }
        critic_hparams = {
            **kwargs.get('critic_hparams', {})
        }

        self._gamma = gamma
        self._epochs = epochs
        self._batch_size = batch_size
        self._gae_lambda = gae_lambda
        self._policy_clip = policy_clip

        self._actor_coef = actor_coefficient
        self._actor_ridge_lambda = actor_ridge

        self._critic_coef = critic_coefficient
        self._critic_ridge_lambda = critic_ridge

        self._entropy_coef = entropy

        self._name = name
        self._verbose = verbose

        script_path = os.path.dirname(os.path.abspath(sys.modules['__main__'].__file__))
        self.path = os.path.join(
            script_path,
            chkpt_dir,
            'ppo'
        )
 
        self.device = device

        self.actor = PolicyNetwork(
            input_dims=input_dims,
            output_dims=output_dims,
            device=device,
            **actor_hparams
        )
        self._actor_clip_grad_norm = actor_clip_grad_norm

        self.critic = ValueNetwork(
            input_dims=input_dims,
            device=device,
            **critic_hparams
        )
        self._critic_clip_grad_norm = critic_clip_grad_norm

        self._memory = self._set_memory()


        if self._verbose:
            actor_params = sum(p.numel() for p in self.actor.parameters())
            critic_params = sum(p.numel() for p in self.critic.parameters())
            print(f"actor ready on {self.device}. total params: {actor_params}")
            print(f"critic ready on {self.device}. total params: {critic_params}")
            print(f"{self._name} initialised with {actor_params+critic_params} params.")

        if load:
            self._load()

    def choose_action(
        self,
        state: list[any]|torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        state = self._convert_to_tensor(state).to(self.device)
        dist, value = self._sample_networks(state)

        action = dist.sample()
        probs = dist.log_prob(action)
        
        return action, probs, value

    def store(
        self,
        state: any,
        action: torch.Tensor,
        probs: torch.Tensor,
        value: torch.Tensor,
        reward: int|float,
        truncated: bool,
        terminated: bool
    ) -> None:
        """
        Append a snapshot of the current state/action pair to memory.  
        """
        self._memory['states'].append(state)
        self._memory['actions'].append(action)
        self._memory['probs'].append(probs)
        self._memory['values'].append(value)
        self._memory['rewards'].append(reward)
        self._memory['truncated'].append(truncated)
        self._memory['terminated'].append(terminated)

    def learn(self) -> dict[str, list[float]]:
        """
        Main learning function for PPO agents.
        For details look up the paper in arxiv  
        """
        num_states = len(self._memory['states'])
        loss_history = dict(
            actor = [],
            critic = [],
            entropy = [],
            total = []
        )
        for _ in tqdm(range(self._epochs), disable=not self._verbose, desc="Learning Epoch", leave=False):
            advantage = self._calculate_advantage(num_states)

            for batch in self._batch_memory(num_states):
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()

                states = [self._memory['states'][i] for i in batch]
                states = self._convert_to_tensor(states)

                losses = self._train_networks(states, advantage, batch)
                for key, tensor in zip(loss_history.keys(), losses):
                    loss_history[key].append(tensor.item())
            torch.cuda.empty_cache()
        self._memory = self._set_memory()
        torch.cuda.empty_cache()
        return loss_history

    def save(self) -> None:
        os.makedirs(self.path, exist_ok=True)

        path = os.path.join(
            self.path,
            f"{self._name}.pt"
        )

        to_save = dict(
            actor = dict(
                network = self.actor.state_dict(),
                optimizer = self.actor.optimizer.state_dict()
            ),
            critic = dict(
                network = self.critic.state_dict(),
                optimizer = self.critic.optimizer.state_dict()
            )
        )
        torch.save(
            to_save,
            path
        )
        if self._verbose:
            print(f"Agent saved at {path}")

    def _load(self) -> None:
        path = os.path.join(
            self.path,
            f"{self._name}.pt"
        )
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint {path} does not exist.")

        chkpt = torch.load(
            os.path.join(
                self.path,
                f"{self._name}.pt"
            ),
            map_location=self.device,
            weights_only=True
        )
        self.actor.load_state_dict(
            chkpt['actor']['network']
        )
        self.actor.optimizer.load_state_dict(
            chkpt['actor']['optimizer']
        )
        self.critic.load_state_dict(
            chkpt['critic']['network']
        )
        self.critic.optimizer.load_state_dict(
            chkpt['critic']['optimizer']
        )
        if self._verbose:
            print(f"Agent successfully loaded from {path}")

    def _train_networks(
        self,
        states: torch.Tensor,
        advantage: list[float],
        batch: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Training steps for the agent
        """
        dist, value = self._sample_networks(states)
        del states

        entropy = dist.entropy().mean()

        actions = [self._memory['actions'][i] for i in batch]
        actions = self._convert_to_tensor(actions)
        new_probs = dist.log_prob(actions)
        del dist, actions

        old_probs = [self._memory['probs'][i] for i in batch]
        old_probs = self._convert_to_tensor(old_probs)
        prob_ratio = (new_probs-old_probs).exp()
        del new_probs, old_probs

        batch_advantage = [advantage[i] for i in batch]
        batch_advantage = self._convert_to_tensor(batch_advantage)

        weighted_probs = batch_advantage*prob_ratio
        weighted_clipped_probs = torch.clamp(
            prob_ratio,
            min=1-self._policy_clip,
            max=1+self._policy_clip
        )*batch_advantage
        del prob_ratio

        actor_loss = torch.min(
            weighted_probs,
            weighted_clipped_probs
        ).mean()
        del weighted_probs, weighted_clipped_probs

        actor_ridge = sum(p.pow(2).sum() for p in self.actor.parameters())
        actor_loss += self._actor_ridge_lambda*actor_ridge
        del actor_ridge

        batch_values = torch.tensor(
            [self._memory['values'][i] for i in batch]
        ).to(self.device)
        returns = batch_advantage+batch_values
        del batch_advantage, batch_values

        critic_loss = ((value-returns)**2).mean()
        critic_ridge = sum(p.pow(2).sum() for p in self.critic.parameters())
        critic_loss += self._critic_ridge_lambda*critic_ridge
        del returns, critic_ridge

        total_loss = self._calculate_total_loss(
            actor_loss,
            critic_loss,
            entropy
        )
        total_loss.backward()

        if self._actor_clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(),
                max_norm=self._actor_clip_grad_norm
            )
        if self._critic_clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.critic.parameters(),
                max_norm=self._critic_clip_grad_norm
            )

        self.actor.optimizer.step()
        self.critic.optimizer.step()
        return actor_loss, critic_loss, entropy, total_loss

    def _batch_memory(self, num_states) -> list[list[int]]:
        """
        Generate batches of indices to batch different objects stored in memory
        """
        batch_start = np.arange(0, num_states, self._batch_size)
        indices = np.arange(num_states)
        np.random.shuffle(indices)
        batches = [
            indices[i:i+self._batch_size]
            for i in batch_start
        ]
        return batches

    def _calculate_advantage(self, num_states: int) -> list[float]:
        """
        Calculates GAE
        """
        rewards = self._memory['rewards']
        values = self._memory['values']
        dones = [
            int(trunc or term) for trunc, term in zip(
                self._memory['truncated'], self._memory['terminated']
            )
        ]

        gae = np.zeros(num_states)
        for t in range(len(gae)):
            discount = 1
            a_t = 0
            for k in range(t, len(gae)-1):
                a_t += discount*(rewards[k]+self._gamma*values[k+1]*(1-dones[k]))
                discount *=self._gamma*self._gae_lambda
            gae[t] = a_t
        return gae

    def _convert_to_tensor(
        self,
        state: list[any]|torch.Tensor
    ) -> torch.Tensor:
        """
        Makes sure that the state passed is a torch.Tensor
        """
        if isinstance(state, torch.Tensor):
            return state.to(self.device)
        else:
            return torch.tensor(state).to(self.device)

    def _sample_networks(
        self,
        state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Feed the actor and critic the state to get their output
        """
        dist = self.actor(state)
        value = self.critic(state)

        return dist, value

    def _set_memory(self) -> dict[str, list[any]]:
        """
        Generates a dict that will be used to store episode states
        """
        return dict(
            states=[],
            actions=[],
            probs=[],
            values=[],
            rewards=[],
            truncated=[],
            terminated=[]
        )

    def _calculate_total_loss(self, actor_loss, critic_loss, entropy):
        """
        Simple weighted sum to get the total loss
        """
        actor_contrib = self._actor_coef*actor_loss
        critic_contrib = self._critic_coef*critic_loss
        entropy_contrib = self._entropy_coef*entropy
        loss = actor_contrib + critic_contrib - entropy_contrib
        return loss
