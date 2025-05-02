#ifndef CAMCAP_H
#define CAMCAP_H

#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

class CamCap {
public:
	CamCap(int cameraIndex = 0, size_t maxQueueSize = 16);
	~CamCap();

	void start();
	void stop();

	bool popFrame(cv::Mat& frame);

	std::condition_variable& getCond();

private:
	void captureLoop();

	std::queue<cv::Mat> frameQueue_;
	std::mutex queueMutex_;
	std::condition_variable queueCond_;
	int cameraIndex_;
	size_t maxQueueSize_;
	std::atomic<bool> stopFlag_;
	std::thread captureThread_;

	friend class LaneDetect;
};

#endif // CAMCAP_H
