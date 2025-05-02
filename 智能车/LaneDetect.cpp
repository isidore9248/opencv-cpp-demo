#include "LaneDetect.h"

LaneDetect::LaneDetect(CamCap& camCap)
	: camCap_(camCap), stopFlag_(false) {
}

LaneDetect::~LaneDetect() {
	stop();
}

void LaneDetect::start() {
	detectThread_ = std::thread(&LaneDetect::detectLoop, this);
}

void LaneDetect::stop() {
	stopFlag_.store(true);
	camCap_.getCond().notify_all();
	if (detectThread_.joinable()) {
		detectThread_.join();
	}
}

cv::Mat LaneDetect::getLatestFrame() {
	std::lock_guard<std::mutex> lock(outputMutex_);
	return latestOutput_.clone();
}

void LaneDetect::detectLoop() {
	while (!stopFlag_.load()) {
		cv::Mat frame;
		{
			std::unique_lock<std::mutex> lock(camCap_.queueMutex_);
			camCap_.getCond().wait(lock, [this] {
				return !camCap_.frameQueue_.empty() || stopFlag_.load();
				});

			if (stopFlag_.load()) break;

			frame = camCap_.frameQueue_.front();
			camCap_.frameQueue_.pop();
		}
		camCap_.getCond().notify_all();

		cv::Mat result = processFrame(frame);
		{
			std::lock_guard<std::mutex> lock(outputMutex_);
			latestOutput_ = result;
		}
	}
}

cv::Mat LaneDetect::processFrame(const cv::Mat& inputFrame) {
	cv::Mat gray, blurred, thresh;
	cv::cvtColor(inputFrame, gray, cv::COLOR_BGR2GRAY);
	cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 0);
	cv::threshold(blurred, thresh, 0, 255, cv::THRESH_BINARY | cv::THRESH_OTSU);

	cv::Mat output = inputFrame.clone();
	detectLaneLines(thresh, output);
	return output;
}

void LaneDetect::detectLaneLines(const cv::Mat& binaryImg, cv::Mat& outputImg)
{
	// Constants (similar to original macros)
	const double bottomRatio = 0.73; // proportion of image height to scan from bottom
	const int minPointDist = 3;      // max distance between consecutive points on lane
	const int diffThreshold = 60;    // difference threshold for zebra-like detection

	if (binaryImg.empty() || outputImg.empty()) {
		return;
	}

	int rows = binaryImg.rows;
	int cols = binaryImg.cols;
	int scanStartY = static_cast<int>(rows * bottomRatio);
	scanStartY = std::min(scanStartY, rows - 1);

	// Count white pixels in each column from bottom up to scanStartY
	std::vector<int> whiteCount(cols, 0);
	for (int x = 0; x < cols; ++x) {
		for (int y = scanStartY; y >= 0; --y) {
			if (binaryImg.at<uchar>(y, x) != 0) {
				whiteCount[x]++;
			}
			else {
				break;
			}
		}
	}

	// Zebra-like detection (difference between adjacent columns)
	int diffCount = 0;
	for (int x = 1; x < cols; ++x) {
		if (std::abs(whiteCount[x] - whiteCount[x - 1]) >= diffThreshold) {
			diffCount++;
		}
	}
	bool zebraDetected = (diffCount >= 6);

	// Find the column with maximum white count from left and right
	int maxLeftCount = 0, maxLeftCol = 0;
	int maxRightCount = 0, maxRightCol = cols - 1;
	for (int x = 0; x < cols; ++x) {
		if (whiteCount[x] > maxLeftCount) {
			maxLeftCount = whiteCount[x];
			maxLeftCol = x;
		}
	}
	for (int x = cols - 1; x >= 0; --x) {
		if (whiteCount[x] > maxRightCount) {
			maxRightCount = whiteCount[x];
			maxRightCol = x;
		}
	}

	// Data structure to hold lane edge points
	struct PointInfo { cv::Point point; int flag; };

	std::vector<PointInfo> leftLane;
	std::vector<PointInfo> rightLane;
	std::vector<PointInfo> middleLane;

	int scanHeight = std::min(maxLeftCount, maxRightCount);
	scanHeight = std::min(scanHeight, scanStartY);

	// Scan from bottom to top for each lane
	for (int y = scanStartY; y >= scanStartY - scanHeight; --y) {
		// Left lane: find transition from white to black (edge)
		for (int x = maxLeftCol; x >= 0; --x) {
			bool isEdge = (binaryImg.at<uchar>(y, x) != 0) &&
				(x == 0 || binaryImg.at<uchar>(y, x - 1) == 0);
			if (isEdge) {
				if (leftLane.empty() ||
					(std::abs(x - leftLane.back().point.x) <= minPointDist &&
						std::abs(y - leftLane.back().point.y) <= minPointDist)) {
					leftLane.push_back({ cv::Point(x, y), 0 });
				}
				maxLeftCol = x;
				break;
			}
		}
		// Right lane: find transition from white to black (edge)
		for (int x = maxRightCol; x < cols; ++x) {
			bool isEdge = (binaryImg.at<uchar>(y, x) != 0) &&
				(x == cols - 1 || binaryImg.at<uchar>(y, x + 1) == 0);
			if (isEdge) {
				if (rightLane.empty() ||
					(std::abs(x - rightLane.back().point.x) <= minPointDist &&
						std::abs(y - rightLane.back().point.y) <= minPointDist)) {
					rightLane.push_back({ cv::Point(x, y), 0 });
				}
				maxRightCol = x;
				break;
			}
		}
		// If both lane edges found for this row, compute middle point
		if (!leftLane.empty() && !rightLane.empty()) {
			cv::Point leftPt = leftLane.back().point;
			cv::Point rightPt = rightLane.back().point;
			// Midpoint between left and right lane edges
			cv::Point midPt((leftPt.x + rightPt.x) / 2, (leftPt.y + rightPt.y) / 2);
			if (middleLane.empty() ||
				(std::abs(midPt.x - middleLane.back().point.x) <= minPointDist &&
					std::abs(midPt.y - middleLane.back().point.y) <= minPointDist)) {
				middleLane.push_back({ midPt, 0 });
			}
		}
	}

	// Draw left lane points in blue
	for (const auto& pt : leftLane) {
		cv::circle(outputImg, pt.point, 2, cv::Scalar(255, 0, 0), cv::FILLED);
	}
	// Draw right lane points in red
	for (const auto& pt : rightLane) {
		cv::circle(outputImg, pt.point, 2, cv::Scalar(0, 0, 255), cv::FILLED);
	}
	// Draw middle lane if available (green lines)
	for (size_t i = 0; i + 1 < middleLane.size(); ++i) {
		cv::line(outputImg, middleLane[i].point, middleLane[i + 1].point, cv::Scalar(0, 255, 0), 2);
	}
}