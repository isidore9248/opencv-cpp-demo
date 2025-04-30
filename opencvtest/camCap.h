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

class camCap
{
public:
	camCap() : running(true) {}
	~camCap() { stop(); }
	void stop();
	bool isRunning() const;

public:
	void captureThread(int camera_index);
	void displayLoop(const std::string& windowName = "camPic");
	void LaneDetectThread();

private:
	//显示帧队列私有函数
	bool getFrame(cv::Mat& frame);
	bool getProcessedFrame(cv::Mat& frame);

private:
	// Helper functions for LaneDetectThread
	bool getFrameFromQueue(cv::Mat& frame); // 从队列获取帧
	void preprocessFrame(const cv::Mat& input, cv::Mat& gray, cv::Mat& binary); // 预处理帧（灰度+Canny）
	void findAndFilterContours(const cv::Mat& binary, const cv::Size& imageSize,
		std::vector<std::vector<cv::Point>>& filteredContours); // 查找并过滤轮廓
	void drawFilteredContours(const cv::Size& imageSize, const cv::Scalar& color,
		const std::vector<std::vector<cv::Point>>& contours, cv::Mat& outputImage); // 绘制过滤后的轮廓
	void combineAndStoreFrame(const cv::Mat& originalFrame, const cv::Mat& lineOverlay); // 合并图像并存储
	cv::Mat fitLines(cv::Mat& image, cv::Point* left_line, cv::Point* right_line);

private:
	cv::Mat processed_frame_;	//显示的最终图像
	std::mutex processed_frame_mutex_;	//显示图像互斥锁
	std::queue<cv::Mat> frame_queue; //帧队列
	std::mutex frame_mutex; //帧队列互斥锁

	std::condition_variable frame_cv;
	std::atomic<bool> running; // 线程运行标志

	cv::Point left_line[2];  // 定义左侧直线端点
	cv::Point right_line[2]; // 定义右侧直线端点
};
