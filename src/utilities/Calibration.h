#ifndef CALIBRATION_H_
#define CALIBRATION_H_

#include <opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

#include <vector>
#include <math.h>
#include <iostream>

using namespace cv;
using namespace std;

namespace nl_uu_science_gmt
{
struct CalibParameters
{
	Mat intrinsics = Mat_<double>(3, 3);
	Mat distCoeffs;
	vector<Mat> rvecs;
	vector<Mat> tvecs;
};

class Calibration
{
public:
	static bool saveIntrinsicsXML(string folder, CalibParameters calib);
	static bool loadIntrinsicsXML(string folder, CalibParameters& calib);
	static void calibrate(vector<String> paths, Size size, int cameraNum);
};

} /* namespace nl_uu_science_gmt */

#endif