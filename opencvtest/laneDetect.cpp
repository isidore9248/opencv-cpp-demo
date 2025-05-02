#include "camCap.h"
#include "laneDetect.h"

#include <opencv2/opencv.hpp>

/// <summary>
/// 对图像进行预处理，包括灰度转换和 Canny 边缘检测。
/// </summary>
/// <param name="input">输入的原始图像。</param>
/// <param name="gray">输出的灰度图像。</param>
/// <param name="binary">输出的二值图像。</param>
void laneDetect::preprocessFrame(const cv::Mat &input, cv::Mat &gray, cv::Mat &binary)
{
	cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
	// Canny 边缘检测
	cv::Canny(gray, binary, 150, 300); //*Canny 检测的是亮度变化剧烈的地方，而不是特定的颜色。
}

/// <summary>
/// 查找轮廓并根据条件进行过滤。
/// </summary>
/// <param name="binary">输入的二值图像。</param>
/// <param name="imageSize">图像尺寸。</param>
/// <param name="filteredContours">输出的过滤后的轮廓集合。</param>
void laneDetect::findAndFilterContours(const cv::Mat &binary, const cv::Size &imageSize, std::vector<std::vector<cv::Point>> &filteredContours)
{
	std::vector<std::vector<cv::Point>> contours;									// 存储找到的所有轮廓
	cv::findContours(binary, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE); // 查找外部轮廓

	filteredContours.clear(); // 清除上一次遍历的结果

	for (const auto &contour : contours)
	{
		double length = cv::arcLength(contour, true); // 计算周长
		double area = cv::contourArea(contour);		  // 计算面积

		// 周长和面积过小
		if (length < 5.0 || area < 10.0)
		{
			continue;
		}

		cv::Rect rect = cv::boundingRect(contour); // 计算边界矩形
		// 位置偏下
		if (rect.y > imageSize.height - 50)
		{
			continue;
		}

		cv::RotatedRect rrt = cv::minAreaRect(contour); // 计算最小外接旋转矩形
		double angle = std::abs(rrt.angle);				// 获取角度
		// 角度不合适
		if (angle < 20.0 || angle > 84.0)
		{
			continue;
		}

		// 椭圆拟合角度不合适
		if (contour.size() > 5) // 确保点数足够
		{
			cv::RotatedRect errt = cv::fitEllipse(contour);
			if (((errt.angle < 5.0) || (errt.angle > 160.0)) && (80.0 < errt.angle && errt.angle < 100.0))
			{
				continue;
			}
		}

		// 成功遍历
		filteredContours.push_back(contour);
	}
}

/// <summary>
/// 在空白图像上绘制过滤后的轮廓。
/// </summary>
/// <param name="imageSize">图像尺寸。</param>
/// <param name="color">轮廓颜色。</param>
/// <param name="contours">过滤后的轮廓集合。</param>
/// <param name="outputImage">输出的图像。</param>
void laneDetect::drawFilteredContours(const cv::Size &imageSize, const cv::Scalar &color, const std::vector<std::vector<cv::Point>> &contours, cv::Mat &outputImage)
{
	outputImage = cv::Mat::zeros(imageSize, CV_8UC1); // 创建与原图大小相同的单通道黑色图像
	for (size_t i = 0; i < contours.size(); ++i)
	{
		cv::drawContours(outputImage, contours, static_cast<int>(i), color, 2, 8); // 绘制轮廓
	}
}

/// <summary>
/// 将原始帧与车道线叠加，并存储结果。
/// </summary>
/// <param name="originalFrame">原始帧。</param>
/// <param name="lineOverlay">车道线叠加图像。</param>
void laneDetect::combineAndStoreFrame(const cv::Mat &originalFrame, const cv::Mat &lineOverlay)
{
	cv::Mat dst; // 最终叠加结果
	// 按权重叠加原始帧和车道线图像
	cv::addWeighted(originalFrame, 0.8, lineOverlay, 0.5, 0, dst);

	{															  // 块作用域，用于自动管理锁
		std::lock_guard<std::mutex> lock(processed_frame_mutex_); // 加锁
		processed_frame_ = dst.clone();							  // 存储处理后的帧
	} // 锁在此处自动释放
}

/// <summary>
/// 拟合车道线并返回叠加结果图像。
/// </summary>
/// <param name="image">输入的轮廓图像。</param>
/// <param name="left_line">左侧车道线点。</param>
/// <param name="right_line">右侧车道线点。</param>
/// <returns>叠加了车道线的图像。</returns>
cv::Mat laneDetect::fitLines(cv::Mat &image, cv::Point *left_line, cv::Point *right_line) // 车道线拟合函数（保持不变）
{
	int height = image.rows; // 获取图像高度
	int width = image.cols;	 // 获取图像宽度

	int cx = width / 2;	 // 图像中心的 x 坐标
	int cy = height / 2; // 图像中心的 y 坐标

	std::vector<cv::Point> left_pts;  // 存储左侧车道线的点
	std::vector<cv::Point> right_pts; // 存储右侧车道线的点
	cv::Vec4f left;					  // 用于存储左侧车道线拟合结果

	// 遍历图像左半部分，收集左侧车道线的点
	for (int i = 100; i < (cx - 10); i++)
	{
		for (int j = cy; j < height; j++)
		{
			int pv = image.at<uchar>(j, i); // 获取像素值
			if (pv == 255)					// 如果像素值为白色（车道线）
			{
				left_pts.push_back(cv::Point(i, j)); // 将点加入左侧点集合
			}
		}
	}

	// 遍历图像右半部分，收集右侧车道线的点
	for (int i = cx; i < (width - 20); i++)
	{
		for (int j = cy; j < height; j++)
		{
			int pv = image.at<uchar>(j, i); // 获取像素值
			if (pv == 255)					// 如果像素值为白色（车道线）
			{
				right_pts.push_back(cv::Point(i, j)); // 将点加入右侧点集合
			}
		}
	}

	// 初始化为全黑色三通道
	cv::Mat out = cv::Mat::zeros(image.size(), CV_8UC3);

	// 如果左侧点集合足够多，进行拟合
	if (left_pts.size() > 2)
	{
		// 最小二乘法拟合
		cv::fitLine(left_pts, left, cv::DIST_L1, 0, 0.01, 0.01);

		double k1 = left[1] / left[0];		  // 计算斜率
		double step = left[3] - k1 * left[2]; // 计算截距

		int x1 = int((height - step) / k1);	 // 计算直线底部的 x 坐标
		int y2 = int((cx - 25) * k1 + step); // 计算直线顶部的 y 坐标

		cv::Point left_spot_1 = cv::Point(x1, height);		// 左侧直线底部点
		cv::Point left_spot_end = cv::Point((cx - 25), y2); // 左侧直线顶部点

		cv::line(out, left_spot_1, left_spot_end, cv::Scalar(0, 0, 255), 8, 8, 0); // 在输出图像上绘制左侧车道线
		left_line[0] = left_spot_1;												   // 更新左侧车道线的起点
		left_line[1] = left_spot_end;											   // 更新左侧车道线的终点
	}
	else
	{
		cv::line(out, left_line[0], left_line[1], cv::Scalar(0, 0, 255), 8, 8, 0); // 如果点不足，使用之前的车道线绘制
	}

	// 如果右侧点集合足够多，进行拟合
	if (right_pts.size() > 2)
	{
		cv::Point spot_1 = right_pts[0];					  // 右侧直线底部点
		cv::Point spot_end = right_pts[right_pts.size() - 1]; // 右侧直线顶部点

		int x1 = spot_1.x;
		int y1 = spot_1.y;

		int x2 = spot_end.x;
		int y2 = spot_end.y;

		cv::line(out, spot_1, spot_end, cv::Scalar(0, 0, 255), 8, 8, 0); // 在输出图像上绘制右侧车道线
		right_line[0] = spot_1;											 // 更新右侧车道线的起点
		right_line[1] = spot_end;										 // 更新右侧车道线的终点
	}
	else
	{
		// 如果点不足，保留上次的车道线
		cv::line(out, right_line[0], right_line[1], cv::Scalar(0, 0, 255), 8, 8, 0);
	}

	return out;
}

/// <summary>
/// 获取处理后的图像帧。
/// </summary>
/// <param name="frame">输出的处理后图像帧。</param>
/// <returns>是否成功获取帧。</returns>
bool laneDetect::getProcessedFrame(cv::Mat &frame)
{
	std::lock_guard<std::mutex> lock(processed_frame_mutex_);
	if (!processed_frame_.empty())
	{
		frame = processed_frame_.clone();
		return true;
	}
	return false;
}