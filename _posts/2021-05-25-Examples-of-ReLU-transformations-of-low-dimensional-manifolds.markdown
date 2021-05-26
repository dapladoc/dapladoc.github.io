---
layout: post
title:  "Examples of ReLU transformations of low-dimensional manifolds"
date:   2021-05-25 12:27:00 +0300
categories: fun curious
katex: True
tags:
  - activation function
  - transformation
  - non-linearity
  - ReLU
---

Following the original paper "[MobileNetV2: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)" let's inspect how low-dimensional manifold could be transformed with a transformation like $$ReLU(A\cdot x)\cdot B$$, where $$x\in\mathbb{R}^n$$, $$A\in\mathbb{R}^{m\times n}$$ and $$B\in\mathbb{R}^{m\times n}$$.

The plan is the following:
1. Generate a 1d manifold in $$\mathbb{R}^2$$.
2. Embed the manifold in $$\mathbb{R}^n$$ by a function $$(x_1, x_2)\rightarrow \underbrace{(x_1, x_2, 0, \ldots, 0)}_n$$.
3. Apply transformation $$y = ReLU(A\cdot x)\cdot B$$. In order to visualize the result, we have to set $$m = n$$ and $$B = A^{-1}$$, and we will generate $$A$$ randomly.
4. Project $$y$$ to back $$\mathbb{R}^2$$ by function $$\underbrace{(x_1, x_2, 0, \ldots, 0)}_n\rightarrow (x_1, x_2)$$.
5. Plot projection of $$y$$.
6. For $$n = 3$$ plot $$ReLU(A\cdot x)$$.

First, let's create the manifold. It will be 1 dimensional manifold in $$\mathbb{R}^2$$. Of course we need to import `numpy` and `matplotlib` beforehand. I do this stuff in `jupyterlab`, so I import also things related to notebook widgets.

{% highlight python %}
%matplotlib widget
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from ipywidgets import Button, interact
{% endhighlight %}

As in the paper, we will use a spiral as an example of the low-dimensional manifold. The easiest way is to use an equation in polar coordinates like $$\rho = \phi$$, and then converting polar coordinates to cartesian by equations

$$
\begin{cases}
x &=& \rho\cos{\phi}\\
y &=& \rho\sin{\phi}
\end{cases}
$$

So we have

{% highlight python %}
n_points = 5000
phi = np.linspace(0, 4 * 2 * np.pi, n_points)
x = phi * np.cos(phi)
y = phi * np.sin(phi)
points = np.c_[x, y]
{% endhighlight %}

Now we have `points` a `numpy.ndarray` of shape `[n_points, 2]`. In order to visualize it, let's assign a color for each point of the manifold. We will use matplotlib built-in colormaps.

{% highlight python %}
norm = mpl.colors.Normalize(vmin=phi.min(), vmax=phi.max())
colors_mapper = cm.ScalarMappable(norm=norm, cmap=cm.viridis)
points_colors = colors_mapper.to_rgba(phi)
{% endhighlight %}

And the visualization itself
{% highlight python %}
fig, ax = plt.subplots()
ax.scatter(x, y, c=points_colors)
{% endhighlight %}

{% include image.html
    img="assets/2021-05-25-Examples-of-ReLU-transformations-of-low-dimensional-manifolds/low_dimensional_manifold.png"
    title="1d manifold in 2d"
    caption="1d manifold in 2d." %}

Next, let's define $$ReLU$$ and the whole transformation:
{% highlight python %}
def relu(x):
    """Apply ReLU to the input."""
    return np.maximum(0, x)

def transform(x):
    """Generate linear transform A randomly by using normal distribution."""
    dims = x.shape[1]
    A = np.random.randn(dims, dims)
    embedded_points = relu(x @ A)
    projected_points = embedded_points @ np.linalg.inv(A)
    return embedded_points, projected_points
{% endhighlight %}


Right now our data are in 2d space, and first we have to embed them into $$n$$-dimensional subspace by just adding $$n - 2$$ columns filled with $$0$$.
{% highlight python %}
n = 10
extended_points = np.concatenate((points, np.zeros((n_points, n - 2))), axis=1)
{% endhighlight %}

So we have `extended_points` a `numpy.ndarray` of shape `[n_points, n]`. But we still can think of it as an array of 2d points. And now we are ready to actually embed the points to $$\mathbb{R}^n$$ and then project them back to $$\mathbb{R}^2$$ with artificially added zero components, but ignore those zeroes.

{% highlight python %}
embedded_points, projected_points = transform(extended_points)

fig, ax = plt.subplots()
scatter_plot = ax.scatter(projected_points[:, 0], projected_points[:, 1], c=points_colors)
{% endhighlight %}

{% include image.html
    img="assets/2021-05-25-Examples-of-ReLU-transformations-of-low-dimensional-manifolds/transformed_points.png"
    caption="Points after transformation." %}

We can make a grid of possible plots like the following
{% highlight python %}
fig, ax = plt.subplots(3, 3)
for i in range(len(ax)):
    for j in range(len(ax[i])):
        embedded_points, projected_points = transform(extended_points)
        ax[i, j].scatter(projected_points[:, 0], projected_points[:, 1], c=points_colors, s=1)
{% endhighlight %}

{% include image.html
    img="assets/2021-05-25-Examples-of-ReLU-transformations-of-low-dimensional-manifolds/transformed_points_grid.png"
    caption="Points after transformation." %}

I use the following code to make jupyter widget, that allows to refresh the plot by pressing a button.

{% highlight python %}
_, projected_points = transform(extended_points)
fig, ax = plt.subplots()
scatter_plot = ax.scatter(projected_points[:, 0], projected_points[:, 1], c=points_colors, s=1)


def update():
    _, projected_points = transform(extended_points)
    scatter_plot.set_offsets(projected_points[:, :2])
    ax.set_xlim(projected_points[:, 0].min(), projected_points[:, 0].max())
    ax.set_ylim(projected_points[:, 1].min(), projected_points[:, 1].max())
    fig.canvas.draw()
    
button = Button(description="Refresh")
display(button)

button.on_click(update)
{% endhighlight %}

When `n = 3` we can visualize both `projected_points` and correspondent `embedded_points` with the following code

{% highlight python %}
n = 3
extended_points = np.concatenate((points, np.zeros((n_points, n - 2))), axis=1)

embedded_points, projected_points = transform(extended_points)
fig = plt.figure()
ax = fig.add_subplot(1, 2, 1)
ax3 = fig.add_subplot(1, 2, 2, projection='3d')
scatter_plot = ax.scatter(projected_points[:, 0], projected_points[:, 1], c=points_colors, s=1)
scatter_plot3 = ax3.scatter(embedded_points[:, 0], embedded_points[:, 1], embedded_points[:, 2], c=points_colors)

def update():
    embedded_points, projected_points = transform(extended_points)
    
    scatter_plot.set_offsets(projected_points[:, :2])
    ax.set_xlim(projected_points[:, 0].min(), projected_points[:, 0].max())
    ax.set_ylim(projected_points[:, 1].min(), projected_points[:, 1].max())
    
    scatter_plot3._offsets3d = (embedded_points[:, 0], embedded_points[:, 1], embedded_points[:, 2])
    ax3.set_xlim(embedded_points[:, 0].min(), embedded_points[:, 0].max())
    ax3.set_ylim(embedded_points[:, 1].min(), embedded_points[:, 1].max())
    ax3.set_zlim(embedded_points[:, 2].min(), embedded_points[:, 2].max())
    fig.canvas.draw()
    
button = Button(description="Refresh")
display(button)

button.on_click(update)
{% endhighlight %}

{% include image.html
    img="assets/2021-05-25-Examples-of-ReLU-transformations-of-low-dimensional-manifolds/transformed_points_3d.png"
    caption="Points after transformation and points embeddings in 3d." %}