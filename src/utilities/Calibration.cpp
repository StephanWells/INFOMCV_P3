#include "Calibration.h"

using namespace cv;
using namespace std;

namespace nl_uu_science_gmt
{
// Saves the camera calibration to a calibration.xml file, given a folder path.
bool Calibration::saveIntrinsicsXML(string folder, CalibParameters calib)
{
	bool success = true;
	FileStorage calibFile = FileStorage(folder + "intrinsics.xml", FileStorage::WRITE);

	if (calibFile.isOpened())
	{
		calibFile << "CameraMatrix" << calib.intrinsics;
		calibFile << "DistortionCoeffs" << calib.distCoeffs;
		calibFile.release();
	}
	else
	{
		success = false;
	}

	return success;
}

// Loads the camera calibration from a calibration.xml file into the calib variable.
bool Calibration::loadIntrinsicsXML(string folder, CalibParameters& calib)
{
	bool success = true;
	FileStorage calibFile = FileStorage(folder + "calibration.xml", FileStorage::READ);

	if (calibFile.isOpened())
	{
		calibFile["CameraMatrix"] >> calib.intrinsics;
		calibFile["DistortionCoeffs"] >> calib.distCoeffs;
		calibFile.release();
	}
	else
	{
		success = false;
	}

	return success;
}

// Calibrates the camera, saving the calibration to a file.
void Calibration::calibrate(vector<String> paths, Size size, int cameraNum)
{
	vector<vector<Point3f>> objPoints;
	vector<vector<Point2f>> imgPoints;
	vector<Point3f> obj;

	for (int j = 0; j < size.width * size.height; j++)
	{
		obj.push_back(Point3f((j / size.width), (j % size.width), 0.0f));
	}

	vector<Point2f> corners;
	Mat imgGrey, imgChess;

	//VideoCapture capture = VideoCapture(0);

	for (int i = 0; i < paths.size(); i++)
	{
		//capture >> imgChess;
		imgChess = imread(paths[i]);
		cvtColor(imgChess, imgGrey, COLOR_RGB2GRAY);

		bool found = findChessboardCorners(imgGrey, size, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FILTER_QUADS);

		if (found)
		{
			cornerSubPix(imgGrey, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(imgChess, size, corners, found);

			imgPoints.push_back(corners);
			objPoints.push_back(obj);
		}

		imshow("Calibration Image " + to_string(i + 1), imgChess);

		waitKey(0);
	}

	CalibParameters calib;

	Mat intrinsics = Mat(3, 3, CV_64F);
	calib.intrinsics.ptr<double>(0)[0] = 1;
	calib.intrinsics.ptr<double>(1)[1] = 1;
	calibrateCamera(objPoints, imgPoints, imgChess.size(), calib.intrinsics, calib.distCoeffs, calib.rvecs, calib.tvecs);

	saveIntrinsicsXML("data/cam" + to_string(cameraNum) + "/", calib);
 	destroyAllWindows();
}

} /* namespace nl_uu_science_gmt */