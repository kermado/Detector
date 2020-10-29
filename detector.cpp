
#include "detector.h"

detector* create(const char* configuration_path, const char* weights_path)
{
	detector* instance = new detector();

	instance->net = cv::dnn::readNet(weights_path, configuration_path);
	instance->net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
	instance->net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
	instance->output_layer_names = instance->net.getUnconnectedOutLayersNames();

	return instance;
}

void release(detector* instance)
{
	instance->blob.release();
	delete instance;
}

int detect(detector* instance, float* source, int width, int height, bbox* boxes)
{
	cv::Mat& blob = instance->blob;

	if (blob.rows != height || blob.cols != width)
	{
		int sizes[] = { 1, 3, height, width };
		blob.create(4, sizes, CV_32F);
	}

	const int channels = blob.channels();

	float* destination = (float*)blob.data;
	for (int c = 0; c < 3; ++c)
	{
		const int destination_channel_offset = width * height * c;
		const int source_channel_offset = width * height * (2 - c);
		for (int y = 0; y < height; ++y)
		{
			const int destination_row_offset = destination_channel_offset + width * y;
			const int source_row_offset = source_channel_offset + width * y;
			for (int x = 0; x < width; ++x)
			{
				destination[destination_row_offset + x] = source[source_row_offset + x];
			}
		}
	}

	cv::dnn::Net& net = instance->net;

	instance->detections.clear();
	net.setInput(blob);
	net.forward(instance->detections, instance->output_layer_names);

	// Network produces output blob with a shape NxC where N is a number of
	// detected objects and C is a number of classes + 4 where the first 4
	// numbers are [center_x, center_y, width, height]
	instance->classes.clear();
	instance->confidences.clear();
	instance->boxes.clear();
	for (const cv::Mat& detection : instance->detections)
	{
		const int box_count = detection.rows;
		for (int i = 0; i < box_count; ++i)
		{
			float bcx = detection.at<float>(i, 0) * width;
			float bcy = detection.at<float>(i, 1) * height;
			float bw = detection.at<float>(i, 2) * width;
			float bh = detection.at<float>(i, 3) * height;

			float xmin = bcx - bw * 0.5F;
			float xmax = bcx + bw * 0.5F;
			float ymin = bcy - bh * 0.5F;
			float ymax = bcy + bh * 0.5F;

			int x = (int)xmin;
			int y = (int)ymin;
			int w = (int)xmax - (int)xmin;
			int h = (int)ymax - (int)ymin;

			if (x < 0) { x = 0; }
			if (y < 0) { y = 0; }
			if (x >= width) { x = width - 1; }
			if (y >= height) { y = height - 1; }

			instance->boxes.push_back(cv::Rect(x, y, w, h));

			cv::Mat scores = detection.row(i).colRange(5, detection.cols);
			cv::Point class_point;
			double confidence;
			minMaxLoc(scores, 0, &confidence, 0, &class_point);

			instance->classes.push_back(class_point.x);
			instance->confidences.push_back((float)confidence);
		}
	}

	instance->indices.clear();
	cv::dnn::NMSBoxes(instance->boxes, instance->confidences, 0.01, 0.4, instance->indices);

	const int count = instance->indices.size();
	for (int i = 0; i < count; ++i)
	{
		int idx = instance->indices[i];
		boxes[i].class_id = instance->classes[idx];
		boxes[i].x = instance->boxes[idx].x;
		boxes[i].y = instance->boxes[idx].y;
		boxes[i].width = instance->boxes[idx].width;
		boxes[i].height = instance->boxes[idx].height;
		boxes[i].confidence = instance->confidences[idx];
	}

	return count;
}