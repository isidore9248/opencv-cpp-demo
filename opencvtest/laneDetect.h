#pragma once
class laneDetect
{
	//friend class camCap;
public:
	laneDetect() = default; // 默认构造函数
	~laneDetect() = default; // 默认析构函数

	//private:
public:
	void preprocessFrame(const cv::Mat& input, cv::Mat& gray, cv::Mat& binary); // 预处理帧（灰度+Canny）
	void findAndFilterContours(const cv::Mat& binary, const cv::Size& imageSize,
		std::vector<std::vector<cv::Point>>& filteredContours); // 查找并过滤轮廓
	void drawFilteredContours(const cv::Size& imageSize, const cv::Scalar& color,
		const std::vector<std::vector<cv::Point>>& contours, cv::Mat& outputImage); // 绘制过滤后的轮廓
	void combineAndStoreFrame(const cv::Mat& originalFrame, const cv::Mat& lineOverlay); // 合并图像并存储
	cv::Mat fitLines(cv::Mat& image, cv::Point* left_line, cv::Point* right_line);
	bool getProcessedFrame(cv::Mat& frame);
private:
	cv::Mat processed_frame_;	//显示的叠加车道显示的最终图像
	std::mutex processed_frame_mutex_;	//显示图像互斥锁
};
