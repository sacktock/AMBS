# AMBS

GitHub repository for "Approximate Model-Based Shielding for Safe Reinforcement Learning"

# Main Idea

By leveraging world models for look-ahead shielding we obatin a general purpose algorithm for verifying the safety of learned RL policies. Specifically, we use DreamerV3 to simulate possible futures (traces) in the latent space of a learned dynamics model, we check each of these traces and estimate the probability of comitting a safety violation in the near future. If this probability is not sufficiently low then we override the learned policy with a safe policy trained to minimise constraint violations.

# Agents

In our experiments details in "Approximate Model-Based Shielding for Safe Reinforcement Learning" all agents are implemented with [JAX](https://github.com/google/jax#pip-installation-gpu-cuda), although the dependencies mostly overlap they may differ slightly depending on whether you are running a DreamerV3 based agent or a dopamine based agent.

# Dependencies

We recommend creating separate conda environments for the DreamerV3 based agents and the dopamine based agents. 
```
conda create -n jax --clone base
conda create -n jax_dopamine --clone base
```

We refer to the [DreamerV3 repository](https://github.com/danijar/dreamerv3) for the list of dependencies associated with the DreamerV3 based agents. And we refer to [dopamine](https://github.com/google/dopamine) for the list of dependencies associated with the dopamine based agents.
```
conda activate jax
pip install //DreamerV3 dependencies
```

```
conda activate jax_dopamine
pip install //dopamine dependencies
```

# Running Experiments
For DreamerV3 based agents navigate to the relevant subdirectory and run ```train.py```. The following command will run DreamerV3 with AMBS on Seaquest with safety labels ```death```, ```early-surface```, ```out-of-oxygen``` for 40M frames with ```xlarge``` model size (default for atari games).
```
cd dreamerV3-shield
python train.py --logdir ./logdir/seaquest/shield --configs atari xlarge --task atari_seaquest --env.atari.labels death early-surface out-of-oxygen --run.steps 10000000
```
For dopamine based agents navugate to the dopamine subdirectory an run the desired agent.
```
cd dopamine
python -um dopamine.discrete_domains.train --base_dir ./logdir/seaquest/rainbow --gin_files ./dopamine/jax/agents/full_rainbow/configs/full_rainbow_seaquest.gin
```

# Plotting

For plotting runs we use tensorboard. Navigate to the relevant subdirectory and start tensorboard
```
cd dreamerV3-shield
tensorboard --logdir ./logdir/seaquest
```

# Acks

We refer to the following repositories from which our code is developed:

- [DreamerV3](https://github.com/danijar/dreamerv3)
- [Bounded Prescience](https://github.com/HjalmarWijk/bounded-prescience) (safety labels for Atari games)
- [LAMBDA](https://github.com/yardenas/la-mbda) (lagrangian CPO implementation for DreamerV2)
- [Dopamine](https://github.com/google/dopamine) (IQN and Rainbow implementations)


