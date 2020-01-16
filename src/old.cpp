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

#define ROW_CORNERS 9
#define COL_CORNERS 6
#define SQUARE_LENGTH 25

// Stores the camera calibration parameters.
struct Calibration
{
	Mat intrinsics = Mat_<double>(3, 3);
	Mat distCoeffs;
	vector<Mat> rvecs;
	vector<Mat> tvecs;
};

// Saves the camera calibration to a calibration.xml file, given a folder path.
bool saveCalibration(string folder, Calibration calib)
{
	bool success = true;
	FileStorage calibFile = FileStorage(folder + "calibration.xml", FileStorage::WRITE);

	if (calibFile.isOpened())
	{
		calibFile << "intrinsicsMat" << calib.intrinsics;
		calibFile << "distCoeffsMat" << calib.distCoeffs;
		calibFile << "rvecsMat" << calib.rvecs;
		calibFile << "tvecsMat" << calib.tvecs;
		calibFile.release();
	}
	else
	{
		success = false;
	}

	return success;
}

// Loads the camera calibration from a calibration.xml file into the calib variable.
bool loadCalibration(string folder, Calibration& calib)
{
	bool success = true;
	FileStorage calibFile = FileStorage(folder + "calibration.xml", FileStorage::READ);

	if (calibFile.isOpened())
	{
		calibFile["intrinsicsMat"] >> calib.intrinsics;
		calibFile["distCoeffsMat"] >> calib.distCoeffs;
		calibFile["rvecsMat"] >> calib.rvecs;
		calibFile["tvecsMat"] >> calib.tvecs;
		calibFile.release();
	}
	else
	{
		success = false;
	}

	return success;
}

// Calibrates the camera, saving the calibration to a file.
void calibrate(vector<String> paths, Size size)
{
	vector<vector<Point3f>> objPoints;
	vector<vector<Point2f>> imgPoints;
	vector<Point3f> obj;
	
	for (int j = 0; j < ROW_CORNERS * COL_CORNERS; j++)
	{
		obj.push_back(Point3f((j / COL_CORNERS) * SQUARE_LENGTH, (j % COL_CORNERS) * SQUARE_LENGTH, 0.0f));
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
		}

		imshow("Calibration Image " + to_string(i + 1), imgChess);

		imgPoints.push_back(corners);
		objPoints.push_back(obj);

		waitKey(0);
	}

	Calibration calib;

	Mat intrinsics = Mat(3, 3, CV_64F);
	calib.intrinsics.ptr<double>(0)[0] = 1;
	calib.intrinsics.ptr<double>(1)[1] = 1;
	calibrateCamera(objPoints, imgPoints, imgChess.size(), calib.intrinsics, calib.distCoeffs, calib.rvecs, calib.tvecs);

	saveCalibration("data/calibration/", calib);
	destroyAllWindows();
}

// Returns the [R|t] extrinsics matrix.
Mat getRTMatrix(Mat rvec, Mat tvec)
{
	Mat r = Mat(3, 3, CV_64F);
	Rodrigues(rvec, r); // Converting from a rotation vector rvec to a rotation matrix r.
	//cout << r << endl << endl;
	//cout << tvec << endl << endl;
	Mat rt = Mat(3, 4, CV_64F);

	// Concatenating the rotation matrix with the translation matrix.
	rt.ptr<double>(0)[0] = r.ptr<double>(0)[0];
	rt.ptr<double>(0)[1] = r.ptr<double>(0)[1];
	rt.ptr<double>(0)[2] = r.ptr<double>(0)[2];
	rt.ptr<double>(0)[3] = tvec.ptr<double>(0)[0],

	rt.ptr<double>(1)[0] = r.ptr<double>(1)[0];
	rt.ptr<double>(1)[1] = r.ptr<double>(1)[1];
	rt.ptr<double>(1)[2] = r.ptr<double>(1)[2];
	rt.ptr<double>(1)[3] = tvec.ptr<double>(1)[0];

	rt.ptr<double>(2)[0] = r.ptr<double>(2)[0];
	rt.ptr<double>(2)[1] = r.ptr<double>(2)[1];
	rt.ptr<double>(2)[2] = r.ptr<double>(2)[2];
	rt.ptr<double>(2)[3] = tvec.ptr<double>(2)[0];

	return rt;
}

// Performs explicit calculation of K[R|t].
Mat getCameraMatrix(Mat intrinsics, Mat rvec, Mat tvec)
{
	Mat k = intrinsics;
	Mat rt = getRTMatrix(rvec, tvec);
	
	return k * rt; // Final camera matrix.
}

// Scales an [x, y, z] vector to [x/z, y/z, 1].
Mat scaleToHomogeneous(Mat vec)
{
	vec.ptr<double>(0)[0] /= vec.ptr<double>(2)[0];
	vec.ptr<double>(0)[1] /= vec.ptr<double>(2)[0];

	return vec;
}

// Takes the x and y values of a 3D vector and makes a 2D point out of them.
Point2f convertTo2D(Mat vec)
{
	return Point2f(vec.ptr<double>(0)[0], vec.ptr<double>(1)[0]);
}

// Projects from world coordinates to image coordinates.
Point2f project(Mat vec, Mat p)
{
	return convertTo2D(scaleToHomogeneous(p * vec));
}

// Draws 3D axes.
void drawAxesAt(Mat origin, Mat& img, Mat p, float lineLength)
{
	double axisLength = lineLength * SQUARE_LENGTH;
	origin = origin * SQUARE_LENGTH;
	origin.ptr<double>(3)[0] = 1.0;

	// World coordinates of the axes.
	Mat axisX = (Mat_<double>(4, 1) << axisLength, 0.0, 0.0, 0) + origin;
	Mat axisY = (Mat_<double>(4, 1) << 0.0, axisLength, 0.0, 0) + origin;
	Mat axisZ = (Mat_<double>(4, 1) << 0.0, 0.0, axisLength, 0) + origin;

	// Converting to 2-dimensional points.
	Point2f imgOrigin2D = project(origin, p);
	Point2f imgAxisX2D = project(axisX, p);
	Point2f imgAxisY2D = project(axisY, p);
	Point2f imgAxisZ2D = project(axisZ, p);

	// Drawing the axis lines.
	line(img, imgOrigin2D, imgAxisX2D, CV_RGB(255, 0, 0), 4, 8, 0);
	line(img, imgOrigin2D, imgAxisY2D, CV_RGB(0, 255, 0), 4, 8, 0);
	line(img, imgOrigin2D, imgAxisZ2D, CV_RGB(0, 0, 255), 4, 8, 0);
}

// Draws cube.
void drawCubeAt(Mat origin, Mat& img, Mat p, float lineLength, Scalar colour)
{
	double cubeLength = lineLength * SQUARE_LENGTH;
	origin = origin * SQUARE_LENGTH;
	origin.ptr<double>(3)[0] = 1.0;

	// World coordinates of the cube.
	Mat cubeX = (Mat_<double>(4, 1) << cubeLength, 0.0, 0.0, 0) + origin;
	Mat cubeY = (Mat_<double>(4, 1) << 0.0, cubeLength, 0.0, 0) + origin;
	Mat cubeZ = (Mat_<double>(4, 1) << 0.0, 0.0, cubeLength, 0) + origin;
	Mat cubeXCorner = (Mat_<double>(4, 1) << cubeLength, 0.0, cubeLength, 0) + origin;
	Mat cubeYCorner = (Mat_<double>(4, 1) << 0.0, cubeLength, cubeLength, 0) + origin;
	Mat cubeCornerTop = (Mat_<double>(4, 1) << cubeLength, cubeLength, cubeLength, 0) + origin;
	Mat cubeCornerBottom = (Mat_<double>(4, 1) << cubeLength, cubeLength, 0.0, 0) + origin;

	// Converting to 2-dimensional points.
	Point2f imgOrigin2D = project(origin, p);
	Point2f imgCubeX2D = project(cubeX, p);
	Point2f imgCubeY2D = project(cubeY, p);
	Point2f imgCubeZ2D = project(cubeZ, p);
	Point2f imgCubeXCorner2D = project(cubeXCorner, p);
	Point2f imgCubeYCorner2D = project(cubeYCorner, p);
	Point2f imgCubeCornerTop2D = project(cubeCornerTop, p);
	Point2f imgCubeCornerBottom2D = project(cubeCornerBottom, p);

	// Drawing the cube.
	line(img, imgOrigin2D, imgCubeX2D, colour, 2, 8, 0);
	line(img, imgOrigin2D, imgCubeY2D, colour, 2, 8, 0);
	line(img, imgOrigin2D, imgCubeZ2D, colour, 2, 8, 0);
	line(img, imgCubeX2D, imgCubeXCorner2D, colour, 2, 8, 0);
	line(img, imgCubeX2D, imgCubeCornerBottom2D, colour, 2, 8, 0);
	line(img, imgCubeY2D, imgCubeYCorner2D, colour, 2, 8, 0);
	line(img, imgCubeY2D, imgCubeCornerBottom2D, colour, 2, 8, 0);
	line(img, imgCubeZ2D, imgCubeXCorner2D, colour, 2, 8, 0);
	line(img, imgCubeZ2D, imgCubeYCorner2D, colour, 2, 8, 0);
	line(img, imgCubeCornerBottom2D, imgCubeCornerTop2D, colour, 2, 8, 0);
	line(img, imgCubeXCorner2D, imgCubeCornerTop2D, colour, 2, 8, 0);
	line(img, imgCubeYCorner2D, imgCubeCornerTop2D, colour, 2, 8, 0);
}

// Draws a 3D axis and cube onto each image in the paths parameter.
void draw(vector<String> paths, Calibration calib)
{
	// Iterating through every image.
	for (int i = 0; i < paths.size(); i++)
	{
		Mat p = getCameraMatrix(calib.intrinsics, calib.rvecs[i], calib.tvecs[i]);
		Mat img = imread(paths[i]);
		Mat tmp = img.clone();
		undistort(tmp, img, calib.intrinsics, calib.distCoeffs);

		Mat origin = (Mat_<double>(4, 1) << 0.0, 0.0, 0.0, 1.0);

		drawAxesAt(origin, img, p, 2.0);
		drawCubeAt(origin + (Mat_<double>(4, 1) << 1.0, 1.0, 0.0, 1.0), img, p, 1.0, CV_RGB(174, 28, 40));
		drawCubeAt(origin + (Mat_<double>(4, 1) << 3.5, 2.0, 1.0, 1.0), img, p, 1.0, CV_RGB(25, 33, 123));
		drawCubeAt(origin + (Mat_<double>(4, 1) << 5.0, 0.0, 0.0, 1.0), img, p, 1.0, CV_RGB(77, 194, 60));

		// Showing the image.
		imshow("Output Image " + to_string(i + 1), img);
		waitKey(0);
	}
	
	destroyAllWindows();
}

// Draws a 3D axis and cube on a live webcam feed.
void drawLive(Calibration calib)
{
	vector<Point3f> obj;
	for (int j = 0; j < ROW_CORNERS * COL_CORNERS; j++)
	{
		obj.push_back(Point3f((j / COL_CORNERS) * SQUARE_LENGTH, (j % COL_CORNERS) * SQUARE_LENGTH, 0.0f));
	}

	VideoCapture cap;
	cap.open(0);
	if (!cap.isOpened()) { cerr << "ERROR! Unable to open camera" << endl; return; }

	Mat frame;
	for (;;)
	{
		cap >> frame;
		flip(frame, frame, 1);

		Mat imgGrey; vector<Point2f> corners;
		Size boardSize = cvSize(COL_CORNERS, ROW_CORNERS);
		cvtColor(frame, imgGrey, COLOR_RGB2GRAY);
		bool found = findChessboardCorners(imgGrey, boardSize, corners, CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_FILTER_QUADS);

		if (found)
		{
			cornerSubPix(imgGrey, corners, Size(11, 11), Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS | CV_TERMCRIT_ITER, 30, 0.1));
			drawChessboardCorners(frame, boardSize, corners, found);

			Mat tmp = frame.clone();
			undistort(tmp, frame, calib.intrinsics, calib.distCoeffs);

			Mat rvec = calib.rvecs[0];
			Mat tvec = calib.tvecs[0];
			solvePnP(obj, corners, calib.intrinsics, calib.distCoeffs, rvec, tvec, true, SOLVEPNP_ITERATIVE);

			Mat origin = (Mat_<double>(4, 1) << 0.0, 0.0, 0.0, 1.0);
			Mat p = getCameraMatrix(calib.intrinsics, rvec, tvec);

			drawAxesAt(origin, frame, p, 2.0);
			drawCubeAt(origin + (Mat_<double>(4, 1) << 1.0, 1.0, 0.0, 1.0), frame, p, 1.0, CV_RGB(174, 28, 40));
			drawCubeAt(origin + (Mat_<double>(4, 1) << 3.5, 2.0, 1.0, 1.0), frame, p, 1.0, CV_RGB(25, 33, 123));
			drawCubeAt(origin + (Mat_<double>(4, 1) << 5.0, 0.0, 0.0, 1.0), frame, p, 1.0, CV_RGB(77, 194, 60));
		}

		imshow("Live Feed", frame);
		if (waitKey(5) >= 0) break;
	}
}

// Main function.
int main_old()
{
	vector<String> paths;

	paths.push_back("data/chess/chess_photo1.png");
	paths.push_back("data/chess/chess_photo2.png");
	paths.push_back("data/chess/chess_photo3.png");
	paths.push_back("data/chess/chess_photo4.png");
	paths.push_back("data/chess/chess_photo5.png");

	Size boardSize = cvSize(COL_CORNERS, ROW_CORNERS);
	Calibration calib;

	//calibrate(paths, boardSize);
	loadCalibration("data/calibration/", calib);

	draw(paths, calib); // Use this to draw over the images.
	//drawLive(calib); // Use this to draw over the webcam feed.

	return 0;
}