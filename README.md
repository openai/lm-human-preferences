**Status:** Archive (code is provided as-is, no updates expected)

**Status:** All references to `gs://lm-human-preferences/` should be updated to `https://openaipublic.blob.core.windows.net/lm-human-preferences/labels`.  The code provided as is no longer works.  Pull requests welcome

# lm-human-preferences

This repository contains code for the paper [Fine-Tuning Language Models from Human Preferences](https://arxiv.org/abs/1909.08593).  See also our [blog post](https://openai.com/blog/fine-tuning-gpt-2/).

We provide code for:
- Training reward models from human labels
- Fine-tuning language models using those reward models

It does not contain code for generating labels.  However, we have released human labels collected for our experiments, at `gs://lm-human-preferences/labels`.
For those interested, the question and label schemas are simple and documented in [`label_types.py`](./lm_human_preferences/label_types.py).

The code has only been tested using the smallest GPT-2 model (124M parameters).

## Instructions

This code has only been tested using Python 3.7.3.  Training has been tested on GCE machines with 8 V100s, running Ubuntu 16.04, but development also works on Mac OS X.

### Installation

- Install [pipenv](https://github.com/pypa/pipenv#installation).

- Install [tensorflow](https://www.tensorflow.org/install/gpu):  Install CUDA 10.0 and cuDNN 7.6.2, then `pipenv install tensorflow-gpu==1.13.1`.  The code may technically run with tensorflow on CPU but will be very slow.

- Install [`gsutil`](https://cloud.google.com/storage/docs/gsutil_install)

- Clone this repo.  Then:
  ```
  pipenv install
  ```

- (Recommended) Install [`horovod`](https://github.com/horovod/horovod#install) to speed up the code, or otherwise substitute some fast implementation in the `mpi_allreduce_sum` function of [`core.py`](./lm_human_preferences/utils/core.py).  Make sure to use pipenv for the install, e.g. `pipenv install horovod==0.18.1`.

### Running

The following examples assume we are aiming to train a model to continue text in a physically descriptive way.
You can read [`launch.py`](./launch.py) to see how the `descriptiveness` experiments and others are defined.

Note that we provide pre-trained models, so you can skip directly to RL fine-tuning or even to sampling from a trained policy, if desired.

#### Training a reward model

To train a reward model, use a command such as
```
experiment=descriptiveness
reward_experiment_name=testdesc-$(date +%y%m%d%H%M)
pipenv run ./launch.py train_reward $experiment $reward_experiment_name
```

This will save outputs (and tensorboard event files) to the directory `/tmp/save/train_reward/$reward_experiment_name`.  The directory can be changed via the `--save_dir` flag.

#### Finetuning a language model

Once you have trained a reward model, you can finetune against it.

First, set
```
trained_reward_model=/tmp/save/train_reward/$reward_experiment_name
```
or if using our pretrained model,
```
trained_reward_model=gs://lm-human-preferences/runs/descriptiveness/reward_model
```

Then,
```
experiment=descriptiveness
policy_experiment_name=testdesc-$(date +%y%m%d%H%M)
pipenv run ./launch.py train_policy $experiment $policy_experiment_name --rewards.trained_model $trained_reward_model --rewards.train_new_model 'off'
```

This will save outputs (and tensorboard event files) to the directory `/tmp/save/train_policy/$policy_experiment_name`.  The directory can be changed via the `--save_dir` flag.

#### Both steps at once

You can run a single command to train a reward model and then finetune against it
```
experiment=descriptiveness
experiment_name=testdesc-$(date +%y%m%d%H%M)
pipenv run ./launch.py train_policy $experiment $experiment_name
```

In this case, outputs are in the directory `/tmp/save/train_policy/$policy_experiment_name`, and the reward model is saved to a subdirectory `reward_model`.  The directory can be changed via the `--save_dir` flag.

#### Sampling from a trained policy

Specify the policy to load:
```
save_dir=/tmp/save/train_policy/$policy_experiment_name
```
or if using our pretrained model,
```
save_dir=gs://lm-human-preferences/runs/descriptiveness
```

Then run:
```
pipenv run ./sample.py sample --save_dir $save_dir --savescope policy
```

Note that this script can run on less than 8 GPUs.  You can pass the flag `--mpi 1`, for exapmle, if you only have one GPU.

## LICENSE

[MIT](./LICENSE)

## Citation

Please cite the paper with the following bibtex entry:
```
@article{ziegler2019finetuning,
  title={Fine-Tuning Language Models from Human Preferences},
  author={Ziegler, Daniel M. and Stiennon, Nisan and Wu, Jeffrey and Brown, Tom B. and Radford, Alec and Amodei, Dario and Christiano, Paul and Irving, Geoffrey},
  journal={arXiv preprint arXiv:1909.08593},
  url={https://arxiv.org/abs/1909.08593},
  year={2019}
}
```
