# A5 Model Architecture

A5WCEFW is the DS-Fall-RD deep temporal multitask reference.

## Input

- Shape: 50 x 12.
- Feature set: tilt12.

## Structure

- Dual accelerometer/gyroscope stream encoder.
- Depthwise-separable temporal convolution blocks.
- Shared temporal encoder.
- Task-specific attention pooling.
- Fall head: non-fall/fall.
- Direction head: forward/backward/lateral.
- Weighted cross entropy for fall and direction in the reference setup.

## Current Role

A5 remains the main deep temporal baseline and the safest direction expert.
Its fall head is retained as an auxiliary/reference result when the final
hybrid pipeline uses a separate ML fall expert.
