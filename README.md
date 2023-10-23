**Supplementary material** can be found here https://github.com/sacktock/AMBS/blob/main/supplementary-material.pdf. Also, feel free to visit the full paper paper here https://arxiv.org/abs/2308.00707

*AMBS is now compatible with* [Safety Gym](https://github.com/openai/safety-gym)

# AMBS

GitHub repository for "Approximate Model-Based Shielding for Safe Reinforcement Learning"

# Main Idea

By leveraging world models for look-ahead shielding we obatin a general purpose algorithm for verifying the safety of learned RL policies. Specifically, we use DreamerV3 to simulate possible futures (traces) in the latent space of a learned dynamics model, we check each of these traces and estimate the probability of comitting a safety violation in the near future. If this probability is not sufficiently low then we override the learned policy with a safe policy trained to minimise constraint violations.

# Agents

In our experiments detailed in "Approximate Model-Based Shielding for Safe Reinforcement Learning" all agents are implemented with [JAX](https://github.com/google/jax#pip-installation-gpu-cuda), although the dependencies mostly overlap they may differ slightly depending on whether you are running a DreamerV3 based agent or a dopamine based agent.

# Dependencies

We recommend creating separate conda environments for the DreamerV3 based agents and the dopamine based agents. 
```
conda create -n jax --clone base
conda create -n jax_dopamine --clone base
```

We refer to the [DreamerV3](https://github.com/danijar/dreamerv3) repository for the list of dependencies associated with the DreamerV3 based agents. And we refer to [google/dopamine](https://github.com/google/dopamine) for the list of dependencies associated with the dopamine based agents.
```
conda activate jax
pip install //DreamerV3 dependencies
```

```
conda activate jax_dopamine
pip install //dopamine dependencies
```
Alternatively use our requirements files, although we stress that the specific [JAX](https://github.com/google/jax#pip-installation-gpu-cuda) installation required is hardware depedent.

```
conda activate jax
pip install -r requirements_jax.txt
```

```
conda activate jax_dopamine
pip install -r requirements_jax_dopamine.txt
```
# Running Experiments
For DreamerV3 based agents, navigate to the relevant subdirectory and run ```train.py```. The following command will run DreamerV3 with AMBS on Seaquest for 40M frames. The ```--env.atari.labels``` flag is used to specify the safety labels ```death```, ```early-surface```, ```out-of-oxygen```,  and the ```xlarge``` option determines the model size (```xlarge``` is the default for atari games).
```
cd dreamerV3-shield
python train.py --logdir ./logdir/seaquest/shield --configs atari xlarge --task atari_seaquest --env.atari.labels death early-surface out-of-oxygen --run.steps 10000000
```
*Random seed can be set with the ```--seed``` flag (default 0).*

For dopamine based agents, navigate to the dopamine subdirectory an run the desired agent.
```
cd dopamine
python -um dopamine.discrete_domains.train --base_dir ./logdir/seaquest/rainbow --gin_files ./dopamine/jax/agents/full_rainbow/configs/full_rainbow_seaquest.gin
```
*Random seed can be set by modifying the corresponding .gin file (e.g. JaxFullRainbowAgent.seed=0)*

# Plotting

For plotting runs we use tensorboard. Navigate to the relevant subdirectory and start tensorboard.
```
cd dreamerV3-shield
tensorboard --logdir ./logdir/seaquest
```

# Safety Gym and Penalty Techniques

Safety Gym is built on MuJoCo. For installation details of MuJoCo we refer you to [here](https://github.com/openai/mujoco-py). Additional requirements for Safety Gym include the following:
```
gym>=0.15.3
joblib>=0.14.0
mujoco_py>=2.0.2.7
numpy>=1.17.4
xmltodict>=0.12.0
``` 
*If you have followed the earlier setup and installation instructions, these dependencies may already be satisfied.*

To get AMBS to work on [Safety Gym](https://github.com/openai/safety-gym) we need to penalise the (unshielded) task policy. We provide the following three penalty techniques that help AMBS converge to a policy that maximises reward and satisfies the safety constraints of the Safety Gym environments:

- Penalty critic: a simple modification to the policy gradient that penalises unsafe states by providing introducing a weighted cost to the policy gradient. Example:
```
python train.py ./logdir/safetygym/PointGoal1/shield_penl --configs safetygym_vision large --task safetygym_Safexp-PointGoal1-v0 --penalty_coeff 1.0 --normalise_ret False --penl_critic_type vfunction --run.steps 500000
```  
- *Probabilistic Logic Policy Gradient* (PLPG) (based on this [paper](https://arxiv.org/abs/2303.03226)): renormalisation of the probabilities of the actions of the task policy, based on the level of safety. We recommend using the penalty critic in addition to PLPG. Example:
```
python train.py --logdir ./logdir/safetygym/PointGoal1/shield_plpg --configs safetygym_vision large --task safetygym_Safexp-PointGoal1-v0 --penalty_coeff 0.8 --plpg True --normalise_ret False --penl_critic_type vfunction --run.steps 500000
```
- *Counter Example Guided Policy Optimisation* (COPT): scaling the gradient of actions to zero when they lead to a safety-violation in the near future. We recommend using the penalty critic in addition to PLPG. Example:
```
python train.py --logdir ./logdir/safetygym/PointGoal1/shield_copt --configs safetygym_vision large --task safetygym_Safexp-PointGoal1-v0 --penalty_coeff 1.0 --copt True --normalise_ret False --penl_critic_type vfunction  --run.steps 500000
```

We will share our research paper outlining these techniques and our results on Safety Gym after the peer review process is complete. In the meantime please feel free to contact me directly through my personal email if you have any questions.

# Acks

We refer to the following repositories from which our code is developed:

- [DreamerV3](https://github.com/danijar/dreamerv3)
- [Bounded Prescience](https://github.com/HjalmarWijk/bounded-prescience) (safety labels for Atari games)
- [LAMBDA](https://github.com/yardenas/la-mbda) (lagrangian CPO implementation for DreamerV2)
- [Dopamine](https://github.com/google/dopamine) (IQN and Rainbow implementations)


