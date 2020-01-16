/*
 * General.h
 *
 *  Created on: Nov 13, 2013
 *      Author: coert
 */

#ifndef GENERAL_H_
#define GENERAL_H_

#include <opencv2/core/core.hpp>
#include <opencv2/core/operations.hpp>
#include <string>

#define PATH_SEP "/"

namespace nl_uu_science_gmt
{

// Version and Main OpenCV window name
const static std::string VERSION = "2.5";
const static std::string VIDEO_WINDOW = "Video";
const static std::string SCENE_WINDOW = "OpenGL 3D scene";

// Some OpenCV colors
const static cv::Vec3b Color_BLUE = cv::Vec3b(255, 0, 0);
const static cv::Vec3b Color_GREEN = cv::Vec3b(0, 255, 0);
const static cv::Vec3b Color_RED = cv::Vec3b(0, 0, 255);
const static cv::Vec3b Color_YELLOW = cv::Vec3b(255, 255, 0);
const static cv::Vec3b Color_MAGENTA = cv::Vec3b(255, 0, 255);
const static cv::Vec3b Color_CYAN = cv::Vec3b(0, 255, 255);
const static cv::Vec3b Color_WHITE = cv::Vec3b(255, 255, 255);
const static cv::Vec3b Color_BLACK = cv::Vec3b(0, 0, 0);

class General
{
public:
	static const std::string CBConfigFile;
	static const std::string IntrinsicsFile;
	static const std::string CalibrationVideo;
	static const std::string CheckerboadVideo;
	static const std::string CheckerboadCorners;
	static const std::string VideoFile;
	static const std::string BackgroundImageFile;
	static const std::string ConfigFile;

	static bool fexists(const std::string &);
};

} /* namespace nl_uu_science_gmt */

#endif /* GENERAL_H_ */
