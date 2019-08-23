# Learning a Generative Transition Model for Robotic Grasping

Planning within robotic tasks usually involves a transition model of the environment's state, which are often in a high-dimensional visual space. We explicitly learn a transition model using the Bicycle GAN architecture, capturing the stochastic process of the real world and thus allowing for uncertainty estimation. Then, we apply this approach to the real-world task of bin picking, where a robot should empty a bin by grasping and shifting objects as fast as possible. The model is trained with around \num{30000} pairs of real-world images before and after a given object manipulation. Such a model allows for two important skills: First, for applications with flange-mounted cameras, the picks per hours can be increased by skipping the recording of images. Second, we use the model for planning ahead while emptying the bin, minimizing either the number of actions or maximizing the estimated reward over the next $N$ actions. We show both advantages with real-robot experiments and set a new state-of-the-art result in the YCB block test.


## YCB Block Test Benchmark

2min for grasping and placing as many objects into another bin. The gripper needs to be above the other bin for dropping.

|            | #  | Objects  | Failed Grasps  | Picks Per Hour  |
|------------|----|---------:|---------------:|----------------:|
| **Random** |  1 |      4  |             14  |          230    |
|            |  2 |      1  |             15  |          230    |
|            |  3 |      3  |             12  |          230    |
|            |  4 |      3  |             15  |          230    |
|            |  5 |      0  |             16  |          230    |
| **Single** |  1 |      12  |             0  |          230    |
|            |  2 |      14  |             0  |          230    |
|            |  3 |      12  |             1  |          230    |
|            |  4 |      13  |             0  |          230    |
|            |  5 |      13  |             1  |          230    |
| **Multiple** |  1 |      17  |             1  |          230    |
|            |  2 |      19  |             0  |          230    |
|            |  3 |      22  |             0  |          230    |
|            |  4 |      23  |             0  |          230    |
|            |  5 |      21  |             2  |          230    |
| **Single + Prediction** |  1 |      16  |             1  |          230    |
|            |  2 |      15  |             2  |          230    |
|            |  3 |      16  |             1  |          230    |
|            |  4 |      18  |             0  |          230    |
|            |  5 |      17  |             1  |          230    |
| **Multiple + Prediction** |  1 |      23  |             2  |          230    |
|            |  2 |      25  |             1  |          230    |
|            |  3 |      24  |             0  |          230    |
|            |  4 |      23  |             1  |          230    |
|            |  5 |      22  |             3  |          230    |


| Method    | Objects     | Grasp Rate   | Picks Per Hour | Video   |
|-----------|-------------|-------------:|---------------:|--------:|
| Random    |  2.2 pm 0.7 |  13.0 pm 3.8 |       66 pm 20 |         |
| Single    | 12.8 pm 0.3 |  97.0 pm 1.6 |      384 pm 10 |         |
| Multiple  | 20.4 pm 1.0 |  97.1 pm 1.6 |      612 pm 29 |         |
| Single + Prediction | 12.8 pm 0.3 |  97.0 pm 1.6 |      384 pm 10 |         |
| Multiple + Prediction | 12.8 pm 0.3 |  97.0 pm 1.6 |      384 pm 10 |         |


## Planning Ahead

### Optimize for fewest Actions

![alt text](https://raw.githubusercontent.com/pantor/learning-transition-model-for-manipulation/master/docs/assets/plan-step-fast/result-0.png)

### Optimize for Sum of Estimated Reward
