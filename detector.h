#pragma once

#include <vector>
#include <string>

#include <opencv2/dnn.hpp>

#define EXPORT extern "C" __declspec(dllexport)

struct detector
{
	cv::dnn::Net net;
	cv::Mat blob;
	std::vector<cv::Mat> detections;
	std::vector<cv::String> output_layer_names;
	std::vector<int> classes;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;
	std::vector<int> indices;
};

struct bbox
{
	int class_id;
	int x;
	int y;
	int width;
	int height;
	float confidence;
};

EXPORT detector* create(const char* configuration_path, const char* weights_path);
EXPORT void release(detector* instance);
EXPORT int detect(detector* instance, float* source, int width, int height, bool swap, bbox* boxes);