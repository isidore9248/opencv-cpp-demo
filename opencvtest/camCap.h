/*
 * @Description:
 * @Version: v1.0.0
 * @Author: isidore-chen
 * @Date: 2025-04-27 21:59:33
 * @Copyright: Copyright (c) 2025 CAUC
 */
#pragma once
#include <opencv2/opencv.hpp>
#include <queue>
#include <mutex>
#include <atomic>
#include <condition_variable>

#include "laneDetect.h"

class camCap
{
public:
	camCap() : running(true) {}
	~camCap() { stop(); }

	//最终外部调用部分
public:
	void displayLoop(const std::string& windowName = "camPic");
	void captureThread(int camera_index);
	void LaneDetectThread();

private:
	void stop();
	bool isRunning() const;

	bool getFrameFromQueue(cv::Mat& frame); // 从队列获取帧

	static laneDetect& GetLaneDetectInstance()
	{
		static laneDetect instance;
		return instance;
	}

private:
	std::queue<cv::Mat> frame_queue; //帧队列
	std::mutex frame_mutex; //帧队列互斥锁

	std::condition_variable frame_cv;
	std::atomic<bool> running; // 线程运行标志

	//车道检测使用的私有变量
	cv::Point left_line[2];  // 定义左侧直线端点
	cv::Point right_line[2]; // 定义右侧直线端点
};
