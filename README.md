# Learn to Run a Power Network with Opponent Modeling in Deep Reinforcement Learning

L2RNP competition: https://l2rpn.chalearn.org/

Robustness Track: https://competitions.codalab.org/competitions/25426

This repository contains the source code for my MEng Computer Science Final Year Individual Project (COMP0138) at University College London (UCL).

The robustness track in the NeurIPS 2020 L2RPN challenge aims to improve the robustness of power grids. To do so, this track has an opponent that can dynamically attack the grid under a specific budget by disconnecting a power line. This approach is used to model real-life adversarial actors that can attack the power grid. These attacks are not necessary at random; they can be targeted. AI agents developed will need to overcome such unexpected events and safely operate the power grid. 

This project explores if opponent modeling could be applied in such a challenge by shaping the environment as a multi-agent setting where two agents compete. Moreover, it investigates various opponent modeling techniques and evaluates if such an approach improves the agent's performance and robustness.

The Dueling Double DQN agent is revised from the L2RPN baseline code. 

