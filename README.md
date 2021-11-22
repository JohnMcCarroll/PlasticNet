# PlasticNet

An exploration of experimental Convolutional Neural Network (CNN)
architecture driven by my own curiosity.

## Motivation

For any problem that haunts a modern engineer, evolution has had millennia to solve.
For this reason, I'm excited about "biomimicry", or solutions that take inspiration
from complex biological life. The domain of Machine Learning is no exception, in fact recent
successes in the field often bare resemblance or were directly inspired by the manner in which
the brain learns and performs. Fully connected layers (MLPs) and stochastic gradient descent (SGD) are loosely based on a network of
neurons, CNN's filters are inspired by specialized cells in the visual cortex, 
and Reinforcement Learning (RL) training paradigms mimic the closed learning loop between
intelligent organisms, their environment, and the interactions between them.

At the time I wrote this code, I was interested in leveraging neuroplasticity (the manner
in which biological neurons strengthen some connections while pruning others) to
inform Neural Architecture Search online during a training loop. The idea was that such
a network would not only explore the highly multidimensional parameter space, using 
SGD, but would also explore the architectural space. The hope is that this could lead
to smaller networks, faster training, and perhaps better performance.

Such an approach has already proven successful for simple MLPs. See the paper
"Rigging the Lottery", a play off of the "Lottery Ticket Hypothesis" paper that came out of 
MIT a few years prior. In the latter paper, two researchers noted that the majority of
the significant information processing occurred between a small subset of neurons and connections.
That is to say that roughly 80% of a given MLP could be pruned, without losing significant performance.
"Rigging the Lottery" uses this insight to "walk" the architecture space during 
training. By starting with a sparsely connected MLP and removing the lowest weighted connections
(the least impactful on outputs) during training, only to replace them with new random connections, 
the authors were able to achieve better training times and performance from smaller networks.

Curious if this phenomenon was replicated in high dimensional networks, and a little upset that someone had beat me to 
the research, I decided to write some code and run some experiments. I came across research that showed only a few paths
of connected layers in a ResNet were significantly influential on the output. Following the same logic, I was curious
if dynamically adjusting the paths available to a Residual Neural Network could lead to higher performance and faster training,
just as had been the case in "Rigging the Lottery."


## Setup

Clone the repo to your local machine:
```buildoutcfg
$ git clone https://github.com/JohnMcCarroll/PlasticNet.git
```
Navigate to the root directory of the repository. Then, set up a virtual environment:
```buildoutcfg
$ python -m venv .
```
Activate your new virtual environment. This command is platform dependent, but for Linux it reads:
```buildoutcfg
$ source bin/activate 
```
Install dependencies:
```buildoutcfg
$ pip install -r requirements.txt
```

## Experimenting

Run training with a plastic ResNet:
```buildoutcfg
$ python research/shuffle/Gymnasium.py
```

## Results

After a few iterations on the idea, which are encapsulated in the "path", "sequential", and "shuffle" directories
and several experiment assays, it became clear that training time and performance were not significantly improved
compared to the state of the art alternative, ResNet, of similar depth. All tests involved training and inference on the CIFAR-10.
Although I feel there might be more work to do to exhaust this topic, it is my hypothesis that perhaps when operating in such a high dimensional space as the paths and parameters of a ResNet, there is no need to manually manage connections
between blocks. When there's always room for further improvement (always a direction down the gradient), given high 
dimensionality, one can reach similar performance by training longer as opposed to manually pruning and adding new paths.
Essentially, the scale of the network mitigated the benefits of neuroplastic architecture search.
