# Detector

A simple OpenCV-based object detector library.

## Usage

The following steps illustrate how to use the library:

1. Call `create()`, which returns a pointer to the created detector instance.
2. Call `detect()` as manay times as required.
3. Call `release()` to free resources.

The `create()` function takes a path to the Darknet configuration file (.cfg) along with a path to the trained weights (.weights) file.

The `detect()` function populates the array of detections passed. The number of objects detected is returned.

## Building

Requires OpenCV 4.4.0 or later to be installed. Also requires CUDA and cuDNN to be installed. See the [cuDNN documentation](https://docs.nvidia.com/deeplearning/sdk/cudnn-install/index.html#installwindows) for details.

## Pre-built Binaries

You can find a pre-built DLL for Windows x64 in `Builds\x64\Release`. This has been built for CUDA versions 6.1, 7.0 and 7.5. You will therefore require a Pascal or Turning card.