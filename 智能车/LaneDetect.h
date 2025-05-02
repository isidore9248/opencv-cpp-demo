#ifndef LANEDETECT_H
#define LANEDETECT_H

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <atomic>
#include "CamCap.h"

class LaneDetect {
public:
	LaneDetect(CamCap& camCap);
	~LaneDetect();

	void start();
	void stop();
	cv::Mat getLatestFrame();

private:
	void detectLoop();
	cv::Mat processFrame(const cv::Mat& frame);
	void detectLaneLines(const cv::Mat& binary, cv::Mat& output);

	CamCap& camCap_;
	std::atomic<bool> stopFlag_;
	std::thread detectThread_;
	cv::Mat latestOutput_;
	std::mutex outputMutex_;
};

#endif // LANEDETECT_H
