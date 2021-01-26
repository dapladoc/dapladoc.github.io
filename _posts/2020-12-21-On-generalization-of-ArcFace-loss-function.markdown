---
layout: post
title:  "On generalization of ArcFace loss function"
date:   2020-12-21 19:27:00 +0300
categories: face-recognition
katex: True
tags:
  - face recognition
  - loss
  - ArcFace
  - AirFace
  - CosFace
  - SphereFace
  - AdaCos
  - Li-ArcFace
---

[ArcFace](https://arxiv.org/abs/1801.07698){:target="_blank"} suggested generalization of loss function used by [SphereFace](https://arxiv.org/abs/1704.08063){:target="_blank"}, [CosFace](https://arxiv.org/abs/1801.09414){:target="_blank"} and some other approaches for face recognition. The loss function of ArcFace looks like


$$
\mathcal{L} = -\sum\limits_{i = 1}^{N}\log{\left(\frac{e^{s \cdot \left(\cos{\left(m_1\cdot\theta_{i, y_i} + m_2\right)} - m_3\right)}}{e^{s \cdot \left(\cos{\left(m_1\cdot\theta_{i, y_i} + m_2\right)} - m_3\right)} + \sum\limits_{\substack{j = 1 \\ j\neq y_i}}^{K}e^{s \cdot \cos{\left(\theta_{i, j}\right)}}}\right)}
$$

where $$N$$ is the number of samples in a batch, $$K$$ is the number of classes, $$\theta_{i,y_i}$$ is the angle between $$i$$-th samlpe in a batch and its class center (i.e. $$y_i$$ is the ground truth label for $$i$$-th sample), and $$s$$, $$m_1$$, $$m_2$$, $$m_3$$ are hyperparameters.

A big question is how to choose $$s$$, $$m_1$$, $$m_2$$, $$m_3$$ for a particular task. An attempt to answer this question was made in the paper [AdaCos](https://arxiv.org/abs/1905.00292){:target="_blank"} in which the approach for the loss modelling and ways to choose hyperparameters of the loss were suggested.

Following AdaCos let's consider just one summand of the loss function, and let's consider not the cross-entropy itself, but only the probability that an embedding would be classified correctly as a function of angle between this embedding and its class center:

$$
P = \frac{e^{s \cdot \left(\cos{\left(m_1\cdot\theta + m_2\right)} - m_3\right)}}{e^{s \cdot \left(\cos{\left(m_1\cdot\theta + m_2\right)} - m_3\right)} + B}
$$

where

$$
B = \sum\limits_{j = 1}^{K - 1}e^{s \cdot \cos{\left(\theta_{j}\right)}}
$$

and

$$
\mathcal{L} = -\log{\left(P\right)}.
$$

The authors of AdaCos mention that during the training the angles $$\theta_{j}$$ become close to $$\frac{\pi}{2}$$ in the very first training steps. My experience shows that it is not the whole truth and we will discuss it later. For now we assume $$\theta_{j} = \frac{\pi}{2}$$.

The problem with this function is that it is not monotonically decreasing on $$\theta$$ for some $$m_1$$ and $$m_2$$. The minimal value for this function is when $$\theta = \frac{\pi - m_2}{m_1}.$$ Let's look at the plot of the function for a particular choice of loss parameters values:

![ArcFace loss and probability](/assets/2020-12-21-On-generalization-of-ArcFace-loss-function/cos_s_60.00_m1_1.00_m2_0.90_m3_0.00.png)

Even though it could not be seen on the probabilities plot, we can see on the loss plot that probabilities that the embeddings would be classified correctly start growing for $$\theta\geq\frac{3\pi}{4}$$. It means that the loss will push the weights of the network in the direction where the angles between embeddings and their class centers are close to $$\pi$$.

So the further $$\arg\min_{\theta}{P}$$ is from $$\pi$$ the bigger probability that some of training samples will lay in the 'wrong' part of the curve, and thus loss function will push the network parameters in the wrong direction, trying to make the angle between these samples and their class center close to $$\pi$$.

Some researchers fix it brutally. For instance, [arcface-pytorch](https://github.com/ronghuaiyang/arcface-pytorch){:target="_blank"} suggests the following fix ($$m$$ here is $$m_2$$ in my notation):

{% highlight python %}
        ...
        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    ...
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)
{% endhighlight %}

But the plots look quite strange for me:

![arcface-pytorch loss fix](/assets/2020-12-21-On-generalization-of-ArcFace-loss-function/cos_arc-face-pytorch_s_60.00_m1_1.00_m2_0.90_m3_0.00.png)

As in the example above, we can see no difference on the probability plots, but loss looks odd.


Before continue, let's denote $$f(x) = \cos{(x)}$$ and rewrite probability function one more time as

$$
P = \frac{e^{s \cdot \left(f\left(m_1\cdot\theta + m_2\right) - m_3\right)}}{e^{s \cdot \left(f\left(m_1\cdot\theta + m_2\right) - m_3\right)} + B_i}
$$

$$
B_i = \sum\limits_{j = 1}^{K - 1}e^{s \cdot f\left(\theta_{i,j}\right)}.
$$


In papers of [ArcGrad](https://ieeexplore.ieee.org/document/9207251){:target="_blank"} and [AirFace](https://arxiv.org/abs/1907.12256){:target="_blank"} the authors suggested how to avoid the problem with nonmonotonicall behavior of cosine by replacing it with a linear function in a little bit different way. The loss function in AirFace is called Li-ArcFace, and its a little bit generalized version uses $$f(x) = \frac{\pi - 2\theta_{i, j}}{\pi}.$$ All these $$\pi$$ are needed only to make the function be somehow like cosine, i.e $$f(0) = 1$$ and $$f(\pi) = -1$$ which looks great.

And now we see, that actually $$f(x)$$ could be any monotonicall function. For instance, one can use the following functions:

$$f(x) = \arctan{(x)}$$

$$f(x) = \tanh{(x)}$$

$$f(x) = e^{x}$$

$$f(x) = e^{-x}$$

Lets rescale it in a such way that $$f(0) = 1$$ and $$f(\pi) = -1$$ in order to compare these functions on the same scale:

$$
f^{(rescaled)}(x) = \frac{2}{f(0) - f(\pi)}\cdot f(x) + 1 - \frac{2\cdot f(0)}{f(0) - f(\pi)}
$$

Obviously, you can turn this rescaling back by varying $$s$$ and $$m_3$$.

Plots for some of these functions are in the image below

![](/assets/2020-12-21-On-generalization-of-ArcFace-loss-function/cos_exp_linear_tanh_arctan_s_60.00_m1_1.00_m2_0.50_m3_0.00.png)

As for the image above, dashed lines are the probabilities, and continuous lines are the losses.

Of course all these functions have different properties, advantages and disadvantages. For instance, lets look on their derivatives

![](/assets/2020-12-21-On-generalization-of-ArcFace-loss-function/diff_cos_exp_linear_tanh_arctan_s_60.00_m1_1.00_m2_0.50_m3_0.00.png)

Here dashed lines are the losses, and continuous lines are their derivatives.

As we can see, for this particular choice of the hyperparameters of the loss function exponential and linear functions could make training faster in the beginning. $$\tanh$$ and $$\arctan$$ will be converging for a quite a long time (the same is true for simple MLPs). But it is harder to stop exponential function when angle is already small enough (sometimes, we do want to stop the loss pushing the network's weights when the angle is less than some threshold). And as in the example above, if $$\theta$$ is greater than $$\frac{27\pi}{32}$$ for $$\cos$$ the loss will push the network to a direction where $$\theta$$ is closer to $$\pi$$.

The last thing we need to discuss is the assumption that $$\theta_{i, j} = \frac{\pi}{2}$$ ($$j\neq y_i$$). Actually, it is true that $$\mathbb{E}\left(\theta_{i, j}\right)$$ is close to $$\frac{\pi}{2}$$, but the std for the angles is quite big. For instance, for a model from the well known repository [insightface](https://github.com/deepinsight/insightface){:target="_blank"} and for the LFW dataset mean angle for negative pairs of images (i.e. images from different classes) is about $$1.565$$ (and $$\frac{\pi}{2}\approx 1.57$$) and std is about $$0.0728$$ ($$\approx \frac{\pi}{42}$$). According to [the 3 sigma rule](https://en.wikipedia.org/wiki/68%E2%80%9395%E2%80%9399.7_rule){:target="_blank"} it means that negative angles (lets call it like that) are in the range $$[\frac{\pi}{2} - \frac{\pi}{14}, \frac{\pi}{2} + \frac{\pi}{14}]$$ (see the picture below).

![](/assets/2020-12-21-On-generalization-of-ArcFace-loss-function/lfw_scores_histogram.png)

So, assuming that negative angles varies in the range $$[\frac{\pi}{2} - \frac{\pi}{14}, \frac{\pi}{2} + \frac{\pi}{14}]$$ we will have the following picture of distributions for probabilities and loss functions

![](/assets/2020-12-21-On-generalization-of-ArcFace-loss-function/stds_cos_exp_linear_s_60.00_m1_1.00_m2_0.50_m3_0.00.png)

If everything looks approximately the same for big enough $$\theta$$, for small $$\theta$$ the situation varies in a very wide range. For instance, for the loss at level $$\mathcal{L} = 0.5$$ the cos and linear based loss functions have $$\theta$$ range about $$\frac{\pi}{10}$$ (the width in the horizontal direction of the band on the plot). At the same time, the exponent based loss function has $$\theta$$ range about $$\frac{\pi}{5}$$ which may be a problem, if you want to stop weights update for a certain $$\theta$$ threshold.