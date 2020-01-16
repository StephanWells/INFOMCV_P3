/*
 * Reconstructor.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Reconstructor.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/types_c.h>
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <cassert>
#include <stack>
#include <iostream>
#include <ppl.h>

#include "../utilities/General.h"

//#define USE_SQUARES
#define USE_OCCLUSION_WEIGHTS

using namespace std;
using namespace cv;
using namespace concurrency;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Voxel reconstruction class
 */
Reconstructor::Reconstructor(
		const vector<Camera*> &cs) :
				m_cameras(cs),
				m_height(2560),
				m_step(32),
				m_sqr_size(13)
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_plane_size.area() > 0)
			assert(m_plane_size.width == m_cameras[c]->getSize().width && m_plane_size.height == m_cameras[c]->getSize().height);
		else
			m_plane_size = m_cameras[c]->getSize();
	}

	const size_t edge = 2 * m_height;
	m_voxels_amount = (edge / m_step) * (edge / m_step) * (m_height / m_step);

	initialize();
}

/**
 * Deconstructor
 * Free the memory of the pointer vectors
 */
Reconstructor::~Reconstructor()
{
	for (size_t c = 0; c < m_corners.size(); ++c)
		delete m_corners.at(c);
	for (size_t v = 0; v < m_voxels.size(); ++v)
		delete m_voxels.at(v);
}

/**
 * Create some Look Up Tables
 * 	- LUT for the scene's box corners
 * 	- LUT with a map of the entire voxelspace: point-on-cam to voxels
 * 	- LUT with a map of the entire voxelspace: voxel to cam points-on-cam
 */
void Reconstructor::initialize()
{
	// Cube dimensions from [(-m_height, m_height), (-m_height, m_height), (0, m_height)]
	const int xL = -m_height;
	const int xR = m_height;
	const int yL = -m_height;
	const int yR = m_height;
	const int zL = 0;
	const int zR = m_height;
	const int plane_y = (yR - yL) / m_step;
	const int plane_x = (xR - xL) / m_step;
	const int plane = plane_y * plane_x;

	// Save the 8 volume corners
	// bottom
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zL));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zL));

	// top
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zR));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zR));

	// Initialise cluster colours.
	default_colours.push_back(Color_RED);
	default_colours.push_back(Color_GREEN);
	default_colours.push_back(Color_BLUE);
	default_colours.push_back(Color_MAGENTA);
	cluster_colours = default_colours;

	// Acquire some memory for efficiency
	cout << "Initializing " << m_voxels_amount << " voxels ";
	m_voxels.resize(m_voxels_amount);

	int z;
	int pdone = 0;
#pragma omp parallel for schedule(auto) private(z) shared(pdone)
	for (z = zL; z < zR; z += m_step)
	{
		const int zp = (z - zL) / m_step;
		int done = cvRound((zp * plane / (double) m_voxels_amount) * 100.0);

#pragma omp critical
		if (done > pdone)
		{
			pdone = done;
			cout << done << "%..." << flush;
		}

		int y, x;
		for (y = yL; y < yR; y += m_step)
		{
			const int yp = (y - yL) / m_step;

			for (x = xL; x < xR; x += m_step)
			{
				const int xp = (x - xL) / m_step;

				// Create all voxels
				Voxel* voxel = new Voxel;
				voxel->x = x;
				voxel->y = y;
				voxel->z = z;
				voxel->camera_projection = vector<Point>(m_cameras.size());
				voxel->valid_camera_projection = vector<int>(m_cameras.size(), 0);

				const int p = zp * plane + yp * plane_x + xp;  // The voxel's index

				parallel_for(size_t(0), m_cameras.size(), [&](size_t c)
				{
					Point point = m_cameras[c]->projectOnView(Point3f((float)x, (float)y, (float)z));

					// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
					voxel->camera_projection[(int)c] = point;

					// If it's within the camera's FoV, flag the projection
					if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height)
						voxel->valid_camera_projection[(int)c] = 1;
				});

				/*for (size_t c = 0; c < m_cameras.size(); ++c)
				{
					Point point = m_cameras[c]->projectOnView(Point3f((float) x, (float) y, (float) z));

					// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
					voxel->camera_projection[(int) c] = point;

					// If it's within the camera's FoV, flag the projection
					if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height)
						voxel->valid_camera_projection[(int) c] = 1;
				}*/

				//Writing voxel 'p' is not critical as it's unique (thread safe)
				m_voxels[p] = voxel;
			}
		}
	}

	cout << "done!" << endl;
}

/**
 * Count the amount of camera's each voxel in the space appears on,
 * if that amount equals the amount of cameras, add that voxel to the
 * visible_voxels vector
 */
void Reconstructor::update()
{
	m_visible_voxels.clear();
	std::vector<Voxel*> visible_voxels;

	int v;
#pragma omp parallel for schedule(auto) private(v) shared(visible_voxels)
	for (v = 0; v < (int) m_voxels_amount; ++v)
	{
		int camera_counter = 0;
		Voxel* voxel = m_voxels[v];

		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			if (voxel->valid_camera_projection[c])
			{
				const Point point = voxel->camera_projection[c];

				//If there's a white pixel on the foreground image at the projection point, add the camera
				if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 255) ++camera_counter;
			}
		}

		// If the voxel is present on all cameras
		if (camera_counter == m_cameras.size())
		{
#pragma omp critical //push_back is critical
			visible_voxels.push_back(voxel);
		}
	}

	m_visible_voxels.insert(m_visible_voxels.end(), visible_voxels.begin(), visible_voxels.end());
}

// Performs initial cluster, and saves offline cluster models for a single camera.
void Reconstructor::initialiseCluster(int cluster_num, int camera_num)
{
	Mat labels = cluster(cluster_num, true); // Labels assigned after clustering.
	vector<Mat> colourModel = clusterLabelsToModels(labels, cluster_num, camera_num, true); // Colour models for each cluster for a camera.

	cluster_models.push_back(colourModel);
	recolourVoxels(labels, cluster_num);
}

// Updates the cluster models and colours.
void Reconstructor::updateClusters(int cluster_num)
{
	Mat labels = cluster(cluster_num);
	vector<vector<Mat>> colourModels;
	vector<float> weights(m_cameras.size(), 1.0f);
	
#ifdef USE_OCCLUSION_WEIGHTS
	vector<int> occlusions = getCurrentFrameOcclusions(labels);
	weights = setWeights(occlusions); // Weight each camera's contribution based on how much occlusion is present in the camera.
#endif

	for (int i = 0; i < m_cameras.size(); i++)
	{
		colourModels.push_back(clusterLabelsToModels(labels, cluster_num, i)); // Colour models for each cluster for each camera.
	}

	updateClusterColours(colourModels, weights);
	recolourVoxels(labels, cluster_num);
}

// Method that, given a list containing the number of occlusions present in every camera, returns the camera index with the fewest occlusions.
int Reconstructor::getBestCamera(vector<int> occlusions)
{
	int bestIndex = 0;
	
	for (int i = 0; i < occlusions.size(); i++)
	{
		if (occlusions[i] < occlusions[bestIndex])
		{
			bestIndex = i;
		}
	}

	return bestIndex;
}

// Returns the number of occlusions per camera for the current frame.
vector<int> Reconstructor::getCurrentFrameOcclusions(Mat labels)
{
	vector<int> occlusions(m_cameras.size(), 1);
	vector<Mat> occlusionMats;
	
	Mat temp = Mat::zeros(m_plane_size, CV_8UC4); // 4 values corresponding to 4 clusters.

	for (int i = 0; i < m_cameras.size(); i++)
	{
		occlusionMats.push_back(temp.clone());
	}

	for (int i = 0; i < m_visible_voxels.size(); i++)
	{
		for (int j = 0; j < m_cameras.size(); j++)
		{
			Point currentPoint = m_visible_voxels[i]->camera_projection[j];
			int currentLabel = labels.at<int>(i);
			Vec4b occlusionVal = (occlusionMats[j].at<Vec4b>(currentPoint.y, currentPoint.x));

			if (!(occlusionVal[currentLabel] == 255))
			{
				for (int i = 0; i < 4; i++)
				{
					if (occlusionVal[i] == 255)
					{
						occlusions[j]++;
					}
				}

				occlusionMats[j].at<Vec4b>(currentPoint.y, currentPoint.x)[currentLabel] = 255;
			}
		}
	}

	for (int i = 0; i < occlusions.size(); i++)
	{
		cout << "Camera " << i << ": " << to_string(occlusions[i]) << endl;
	}

	return occlusions;
}

// Given a list of occlusion counts, returns a set of weights to multiply the final histogram distance calculations with.
vector<float> Reconstructor::setWeights(vector<int> occlusions)
{
	vector<float> weights(m_cameras.size(), 1.0f);

	for (int i = 0; i < weights.size(); i++)
	{
		weights[i] /= (float)occlusions[i];
	}

	return weights;
}

// Takes a set of cluster labels and returns the cluster models for a given camera.
vector<Mat> Reconstructor::clusterLabelsToModels(Mat labels, int cluster_num, int camera_num, bool init)
{
	vector<Mat> clusterColours = getColours(labels, cluster_num, camera_num, init);
	vector<Mat> modelsSingleCamera;

	for (int i = 0; i < clusterColours.size(); i++)
	{
		Mat clusterModel = getHistogram(clusterColours.at(i));
		modelsSingleCamera.push_back(clusterModel);
	}

	return modelsSingleCamera;
}

// Gets the colours for all of the clusters of a given camera.
vector<Mat> Reconstructor::getColours(Mat labels, int cluster_num, int camera_num, bool init)
{
	vector<vector<Point>> clusterImages; // Vector of points for each cluster.

	for (int i = 0; i < cluster_num; i++)
	{
		vector<Point> image;
		clusterImages.push_back(image);
	}

	for (int i = 0; i < m_visible_voxels.size(); i++)
	{
		Voxel* tmp = m_visible_voxels.at(i);

		if (!isOccluded(tmp, camera_num))
		{
			if (!init) clusterImages.at(labels.at<int>(i)).push_back(tmp->camera_projection.at(camera_num));
			else clusterImages.at(init_order[camera_num][labels.at<int>(i)]).push_back(tmp->camera_projection.at(camera_num));
		}
	}

	vector<Mat> coloursVec;

	for (int i = 0; i < cluster_num; i++)
	{
		vector<Vec3b> colours = concatenateColours(camera_num, clusterImages.at(i), 5);
		Mat coloursMat = colourVecToMat(colours);
		coloursVec.push_back(coloursMat);
	}

	return coloursVec;
}

// Checks if the pixel projection of a given voxel is occluded for a given camera.
bool Reconstructor::isOccluded(Voxel* vox, int cameraNum)
{
	return false; // To be continued...
}

// Returns a list of colours from a given list of points on a given camera, considering only those with a V value above the threshold.
vector<Vec3b> Reconstructor::concatenateColours(int camera_num, vector<Point> points, double pixelsThresh)
{
	vector<Vec3b> concatenatedVec;

	for (int i = 0; i < points.size(); i++)
	{
		int x = points.at(i).x;
		int y = points.at(i).y;
		Mat currentFrame = m_cameras.at(camera_num)->getFrame();
		Vec3b bgrColour = currentFrame.at<Vec3b>(y, x);
		Vec3b hsvColour = Vec3bBGR2HSV(bgrColour[0], bgrColour[1], bgrColour[2]);

		if (hsvColour[2] >= pixelsThresh)
		{
			concatenatedVec.push_back(hsvColour);
		}
	}

	return concatenatedVec;
}

// Converts a single colour from BGR to HSV.
Vec3b Reconstructor::Vec3bBGR2HSV(uchar B, uchar G, uchar R)
{
	Mat hsv;
	Mat bgr(1, 1, CV_8UC3, Vec3b(B, G, R));
	cvtColor(bgr, hsv, CV_BGR2HSV);
	return Vec3b(hsv.data[0], hsv.data[1], hsv.data[2]);
}

// Converts a vector of colours to a Mat to pass into OpenCV functions.
Mat Reconstructor::colourVecToMat(vector<Vec3b> vec)
{
	Mat colours = Mat(1, vec.size(), CV_8UC3, Vec3b(0, 0, 0));

	for (int i = 0; i < vec.size(); i++)
	{
		colours.at<Vec3b>(0, i) = vec[i];
	}

	return colours;
}

// Converts a vector of points to a Mat to pass into OpenCV functions.
Mat Reconstructor::pointVecToMat(vector<Point2f> vec)
{
	Mat points = Mat::zeros(m_height / 16, m_height / 16, CV_8UC1);

	for (int i = 0; i < vec.size(); i++)
	{
		points.at<uchar>((vec[i].x / 32) + 80, (vec[i].y / 32) + 80) = 255;
	}

	return points;
}

// Returns the histogram from a matrix of colours representing a single cluster.
Mat Reconstructor::getHistogram(Mat inputColours)
{
	// Histogram parameters.
	int hBins = 90, sBins = 32;
	int histSize[] = { hBins, sBins };
	float hRange[] = { 0, 180 };
	float sRange[] = { 0, 256 };
	const float* histRange[] = { hRange, sRange };
	const int channels[] = { 0, 1 };

	/*int histSize = 15;
	float range[] = { 0, 180 };
	const float* histRange = { range };
	vector<Mat> hsvSplit;
	split(inputColours, hsvSplit);*/

	Mat hist;
	calcHist(&inputColours, 1, channels, Mat(), hist, 2, histSize, histRange, true, false);
	//calcHist(&hsvSplit[0], 1, 0, Mat(), hist, 1, &histSize, &histRange, true, false);
	normalize(hist, hist, 1.0, 0, NORM_L1); // Normalise the histograms for comparison.

	return hist;
}

// Takes the models for each cluster for each camera and compares them to the offline cluster_models to update the cluster colours for the frame.
void Reconstructor::updateClusterColours(vector<vector<Mat>> models, vector<float> weights)
{
	vector<vector<Mat>> tmpModels = cluster_models; // Vector used to store the models for comparison.
	vector<vector<double>> distMat; // Matrix used to store the distances between offline models (rows) and online models (columns).
	vector<vector<Point2i>> indicesMat; // Matrix used to store the indices of the distance matrix.

	for (int i = 0; i < models.at(0).size(); i++)
	{
		distMat.push_back(vector<double>(models.at(0).size()));

		vector<Point2i> temp;

		for (int j = 0; j < models.at(0).size(); j++)
		{
			temp.push_back(Point2i(i, j));
		}

		indicesMat.push_back(temp);
	}

	for (int i = 0; i < models.at(0).size(); i++) // For each input model.
	{
		// Finding the distances to each offline model.
		for (int j = 0; j < tmpModels.at(0).size(); j++) // For each offline model.
		{
			double tempDist = 0;

			// Find the total distance between the models for each camera.
			for (int k = 0; k < models.size(); k++)
			{
				tempDist += weights[k] * distanceBetweenHists(models.at(k).at(i), tmpModels.at(k).at(j));
			}

			distMat[i][j] = tempDist;
		}
	}

	while (!distMat.empty())
	{
		Point2i minimum = minMat(distMat);

		cluster_colours.at(indicesMat[minimum.x][minimum.y].x) = default_colours.at(indicesMat[minimum.x][minimum.y].y);
		distMat.erase(distMat.begin() + minimum.x);
		indicesMat.erase(indicesMat.begin() + minimum.x);

		for (int j = 0; j < distMat.size(); j++)
		{
			distMat[j].erase(distMat[j].begin() + minimum.y);
			indicesMat[j].erase(indicesMat[j].begin() + minimum.y);
		}
	}
}

// Returns the minimum value of a given matrix.
Point2i Reconstructor::minMat(vector<vector<double>> matrix)
{
	double minDistance = DBL_MAX;
	Point2i minIndex = Point2i(0, 0);

	for (int i = 0; i < matrix.size(); i++)
	{
		for (int j = 0; j < matrix[i].size(); j++)
		{
			if (matrix[i][j] < minDistance)
			{
				minIndex = Point2i(i, j);
				minDistance = matrix[i][j];
			}
		}
	}

	return minIndex;
}

// Returns the distance between two histograms.
double Reconstructor::distanceBetweenHists(Mat hist1, Mat hist2)
{
	return compareHist(hist1, hist2, CV_COMP_CHISQR);
}

// Cluster the voxels.
Mat Reconstructor::cluster(int cluster_num, bool init)
{
	std::vector<cv::Point2f> data;
	
	for (int i = 0; i < m_visible_voxels.size(); i++)
	{
		bool to_push = true;

#ifdef USE_SQUARES
		if (prev_centers.rows == cluster_num)
		{
			int xposvox = m_visible_voxels[i]->x / 32;
			int yposvox = m_visible_voxels[i]->y / 32;

			for (int j = 0; j < cluster_num; j++)
			{
				int xposcen = (int)prev_centers.at<float>(j, 0) / 32;
				int yposcen = (int)prev_centers.at<float>(j, 1) / 32;

				int xdist = abs(xposvox - xposcen);
				int ydist = abs(yposvox - yposcen);

				if (xdist <= m_sqr_size && ydist <= m_sqr_size) break;

				if (j == cluster_num - 1)
				{
					to_push = false;
					m_visible_voxels.erase(m_visible_voxels.begin() + i);
					i--;
					break;
				}
				else continue;
			}
		}
#endif

		if (to_push) data.push_back(cv::Point2f(m_visible_voxels[i]->x, m_visible_voxels[i]->y));
	}

	Mat testimg(m_height / 16, m_height / 16, CV_8UC3, Vec3b(0, 0, 0));

	cv::imwrite("data/centers.png", testimg);

	Mat labels, centers;
	cv::kmeans(data, cluster_num, labels, TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 4, 0.1), 10, KMEANS_PP_CENTERS, centers);
	if (!init) prev_centers = centers.clone();

	return labels;
}

// Colour the voxels to their assigned cluster colour.
void Reconstructor::recolourVoxels(Mat labels, int cluster_num)
{
	for (int i = 0; i < labels.rows; i++)
	{
		cv::Vec3b color_space;
		color_space = cluster_colours.at(labels.at<int>(i));
		m_visible_voxels[i]->color = color_space;
	}

	if (prev_centers.rows == cluster_num)
	{
		for (int i = 0; i < cluster_num; i++)
		{
			Voxel* trackpoint = new Voxel();
			trackpoint->x = (int)prev_centers.at<float>(i, 0);
			trackpoint->y = (int)prev_centers.at<float>(i, 1);
			trackpoint->z = 0;

			trackpoint->color = cluster_colours.at(i);

			if (prev_tracking_voxels.size() > 0)
			{
				Voxel* prevtrackpoint = new Voxel();
				for (int j = 0; j < prev_tracking_voxels.size(); j++)
					if (prev_tracking_voxels[j]->color == trackpoint->color) prevtrackpoint = prev_tracking_voxels[j];

				int prevx = prevtrackpoint->x / 32;
				int currx = trackpoint->x / 32;
				int prevy = prevtrackpoint->y / 32;
				int curry = trackpoint->y / 32;

				float sqdist = (prevx - currx) * (prevx - currx) + (prevy - curry) * (prevy - curry);

				if (sqdist > 10) m_flicker_amount++;
			}

			m_tracking_voxels.push_back(trackpoint);
		}

		prev_tracking_voxels.clear();
		prev_tracking_voxels.push_back(m_tracking_voxels[m_tracking_voxels.size() - 4]);
		prev_tracking_voxels.push_back(m_tracking_voxels[m_tracking_voxels.size() - 3]);
		prev_tracking_voxels.push_back(m_tracking_voxels[m_tracking_voxels.size() - 2]);
		prev_tracking_voxels.push_back(m_tracking_voxels[m_tracking_voxels.size() - 1]);		
	}
}

} /* namespace nl_uu_science_gmt */