---
language:
- en
license: mit
size_categories:
- 1K<n<10K
task_categories:
- question-answering
- video-text-to-text
- robotics
tags:
- vlm
- navigation
- vqa
---

# SocialNav-SUB: Benchmarking VLMs for Scene Understanding in Social Robot Navigation

This is the accompying dataset for the Social Navigation Scene Understanding Benchmark (SocialNav-SUB) which is a Visual Question Answering (VQA) dataset and benchmark designed to evaluate Vision-Language Models (VLMs) for scene understanding in real-world social robot navigation scenarios. SocialNav-SUB provides a unified framework for evaluating VLMs against human and rule-based baselines across VQA tasks requiring spatial, spatiotemporal, and social reasoning in social robot navigation. It aims to identify critical gaps in the social scene understanding capabilities of current VLMs, setting the stage for further research in foundation models for social robot navigation.

This VQA dataset provides:
1) A set of VQA prompts for scene understanding of social navigation scenarios
2) Additional data for prompts including odometry information and 3D human tracking estimations.
3) A set of multiple human labels for each VQA prompt (multiple human answers per prompt)

For more information, please see the following:

- Paper: [SocialNav-SUB: Benchmarking VLMs for Scene Understanding in Social Robot Navigation](https://arxiv.org/abs/2509.08757)
- Project Page: [https://larg.github.io/socialnav-sub](https://larg.github.io/socialnav-sub)
- Code: [https://github.com/michaelmunje/SocialNavSUB](https://github.com/LARG/SocialNavSUB)
