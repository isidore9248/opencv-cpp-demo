#include "CamCap.h"

CamCap::CamCap(int cameraIndex, size_t maxQueueSize)
	: cameraIndex_(cameraIndex), maxQueueSize_(maxQueueSize), stopFlag_(false) {
}

CamCap::~CamCap() {
	stop();
}

void CamCap::start() {
	captureThread_ = std::thread(&CamCap::captureLoop, this);
}

void CamCap::stop() {
	stopFlag_.store(true);
	queueCond_.notify_all();
	if (captureThread_.joinable()) {
		captureThread_.join();
	}
}

void CamCap::captureLoop() {
	cv::VideoCapture cap(cameraIndex_);
	if (!cap.isOpened()) {
		std::cerr << "Error: Unable to open camera." << std::endl;
		return;
	}

	while (!stopFlag_.load()) {
		cv::Mat frame;
		if (!cap.read(frame)) continue;

		std::unique_lock<std::mutex> lock(queueMutex_);
		queueCond_.wait(lock, [this] {
			return frameQueue_.size() < maxQueueSize_ || stopFlag_.load();
			});

		if (stopFlag_.load()) break;

		frameQueue_.push(frame.clone());
		lock.unlock();
		queueCond_.notify_all();
	}

	cap.release();
}

bool CamCap::popFrame(cv::Mat& frame) {
	std::unique_lock<std::mutex> lock(queueMutex_);
	if (frameQueue_.empty()) return false;
	frame = frameQueue_.front();
	frameQueue_.pop();
	lock.unlock();
	queueCond_.notify_all();
	return true;
}

std::condition_variable& CamCap::getCond() {
	return queueCond_;
}