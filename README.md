# telestes

`telestes` is a library providing transformer-based rl agents, that is extremely customizable, but also comes with sane defaults.

> source code: [https://erga.apotheke.earth/apotheke/alectors](https://erga.apotheke.earth/apotheke/telestes)  
> license: [GPLv3 or later](https://gnu.org/licenses/gpl-3.0.html)  

## Description

`telestes` serves as the rl base for `alectors`, our natural language reinforcement learning (nlrl) agents.

We decided to seperate the rl part into its own module, so that we may explore non-nlp rl tasks that might benefit from a transformer architecture.

At the same time, we are interested in expanding the different available networks; so far we support Transformers and Linear models[^1].

## Supported Architectures

The currently supported architecture is PPO. Plans exist to include SAC, GRPO, and maybe DDQN.

[^1]: we might split our network architectures into a seperate module also down the road, in order to use it on non rl research as well.

