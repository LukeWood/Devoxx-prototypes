---
title:
- Have you ever seen an astronaut riding a horse?
author:
- Luke Wood
theme:
- Copenhagen
date:
- October 1, 2022

---

# Generative Image Models

Two stories: (2 minutes)

(back story & introduction)

- story of my GitHub avatar + Dr. Larson + Ian
- despite working full time at Google for awhile I've kept it as a tribute to the little group
- story of recruiting Ian to KerasCV
- funny story of the little GitHub cats

Today, I'm going to teach you how to do all of this and more, 
all while understanding what is actually going on in your code

Sandy in Malibu:

(back story & introduction)

## Introduction

### Slide (1 min)

Hello everyone, my name is Luke Wood.
I'm a Keras team member, and for the last year or so my efforts have been focused on creating KerasCV: a set of computer vision extensions to Keras.

While my primary efforts have been on image classification and object detection,
today I'm here to talk to you about an area I am deeply passionate about: generative image models.

TODO(lukewood): personal history with generative image models

### The Code, Slides, Demos

- all available on GitHub, etc.

# So what the hell is an image generation model? (1 min)

Instead of telling you what a generative image model is, I will show you.

To start out, lets cover the basics and build.

Just a heads up, as part of this talk I will be coming back to explain the mechanics of mosts of these model types, but for now lets just give an overview of what they can do.

## Types of Models

model zoo slide

### Random Sample Models (1 min)

MNIST, Shoes, etc

Generative modeling for the last 10 years.  
I've been building random sample models since I was in my undergrad.

### Flash forward to today (1 min)

(yes of course other models existed, they just were not popular or commercially viable due to low quality)

DallE-2

### Text to Image Generation Models (5 min)

- 3 or 4 slides of prompts

### Image In-Painting  (2 min)

- show the original picture
- show another with more people filled in

### Textual Inversion (5 minutes)

(back story)

- story of my GitHub avatar + Dr. Larson + Ian
- despite working full time at Google for awhile I've kept it as a tribute to the little group
- story of recruiting Ian to KerasCV
- funny story of the little GitHub cats

### Image+Text to Image Generation Models

## Overview of Implementations
So next, lets start talking about how you can accomplish some of these.
We'll start simple, and we'll build up to the more complicated stuff.

### Variational AutoEncoders

- code demo
- imaging scaling this concept

### Diffusion Models

- significance & implementation

# Modern Models

So last spring OpenAI released Dall-E 2, which has turned out to be the first commercially successful image generation model.

Stable Diffusion is a powerful, open-source text-to-image generation model.

### Dall-E2

### ImageGen

# ... and now
# Stable Diffusion

- why everyone is excited about SD
- we're going to literally only look at this model for the rest of the talk

# Open Source

Where it came from, who made it, etc.

# Architecture Overview

Unlike what you might expect at this point, Stable Diffusion doesn't actually run on magic.
It's a kind of "latent diffusion model". Let's dig into what that means.

You may be familiar with the idea of _super-resolution_:
it's possible to train a deep learning model to _denoise_ an input image -- and thereby turn it into a higher-resolution
version. The deep learning model doesn't do this by magically recovering the information that's missing from the noisy, low-resolution
input -- rather, the model uses its training data distribution to hallucinate the visual details that would be most likely
given the input. To learn more about super-resolution, you can check out the following Keras.io tutorials:

- [Image Super-Resolution using an Efficient Sub-Pixel CNN](https://keras.io/examples/vision/super_resolution_sub_pixel/)
- [Enhanced Deep Residual Networks for single-image super-resolution](https://keras.io/examples/vision/edsr/)

![Super-resolution](https://i.imgur.com/M0XdqOo.png)

When you push this idea to the limit, you may start asking -- what if we just run such a model on pure noise?
The model would then "denoise the noise" and start hallucinating a brand new image. By repeating the process multiple
times, you can get turn a small patch of noise into an increasingly clear and high-resolution artificial picture.

This is the key idea of latent diffusion, proposed in
[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) in 2020.
To understand diffusion in depth, you can check the Keras.io tutorial
[Denoising Diffusion Implicit Models](https://keras.io/examples/generative/ddim/).

![Denoising diffusion](https://i.imgur.com/FSCKtZq.gif)

Now, to go from latent diffusion to a text-to-image system,
you still need to add one key feature: the ability to control the generated visual contents via prompt keywords.
This is done via "conditioning", a classic deep learning technique which consists of concatenating to the
noise patch a vector that represents a bit of text, then training the model on a dataset of {image: caption} pairs.

This gives rise to the Stable Diffusion architecture. Stable Diffusion consists of three parts:

- A text encoder, which turns your prompt into a latent vector.
- A diffusion model, which repeatedly "denoises" a 64x64 latent image patch.
- A decoder, which turns the final 64x64 latent patch into a higher-resolution 512x512 image.

First, your text prompt gets projected into a latent vector space by the text encoder,
which is simply a pretrained, frozen language model. Then that prompt vector is concatenate
to a randomly generated noise patch, which is repeatedly "denoised" by the decoder over a series
of "steps" (the more steps you run the clearer and nicer your image will be -- the default value is 50 steps).

Finally, the 64x64 latent image is sent through the decoder to properly render it in high resolution.

![The Stable Diffusion architecture](https://i.imgur.com/2uC8rYJ.png)

All-in-all, it's a pretty simple system -- the Keras implementation
fits in four files that represent less than 500 lines of code in total:

# Training Overview

So how the hell do I train this thing?  Seriously, how do I train it

## Introduce the VAE

- A callback to an earlier episode...

## Introduce CLIP

## Introduce the Diffusion Model

## Introduce LAOIN-5B

- introduce mr. hedge fund

# So what the hell can I do with this model?

## Text to Image

## Textual Inversion

## Image to Image

## Image in-painting

# Demo Time!

- text to image
