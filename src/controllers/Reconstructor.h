/*
 * Reconstructor.h
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#ifndef RECONSTRUCTOR_H_
#define RECONSTRUCTOR_H_

#include <opencv2/core/core.hpp>
#include <stddef.h>
#include <vector>
#include <iostream>

#include "Camera.h"

namespace nl_uu_science_gmt
{

class Reconstructor
{
public:
	/*
	 * Voxel structure
	 * Represents a 3D pixel in the half space
	 */
	struct Voxel
	{
		int x, y, z;                               // Coordinates
		cv::Vec3b color;                          // Color
		std::vector<cv::Point> camera_projection;  // Projection location for camera[c]'s FoV (2D)
		std::vector<int> valid_camera_projection;  // Flag if camera projection is in camera[c]'s FoV
	};

private:
	const std::vector<std::vector<int>> init_order = { { 2,3,1,0 },{ 2,0,3,1 },{ 0,3,1,2 },{ 1,2,0,3 } };

	const std::vector<Camera*> &m_cameras;					// vector of pointers to cameras
	const int m_height;										// Cube half-space height from floor to ceiling
	const int m_step;										// Step size (space between voxels)
	const int m_sqr_size;

	std::vector<cv::Point3f*> m_corners;					// Cube half-space corner locations

	size_t m_voxels_amount;									// Voxel count
	int m_flicker_amount = 0;
	cv::Size m_plane_size;									// Camera FoV plane WxH

	std::vector<Voxel*> m_voxels;							// Pointer vector to all voxels in the half-space
	std::vector<Voxel*> m_visible_voxels;					// Pointer vector to all visible voxels
	std::vector<Voxel*> m_tracking_voxels;
	std::vector<Voxel*> prev_tracking_voxels;
	std::vector<cv::Vec3b> default_colours;				// Vector of default colours for each cluster.
	std::vector<cv::Vec3b> cluster_colours;				// Vector defining which cluster label gets which colour.
	std::vector<std::vector<cv::Mat>> cluster_models;		// Vector of colour models used for each cluster for each camera.
	cv::Mat prev_centers;

	void initialize();

public:
	Reconstructor(
			const std::vector<Camera*> &);
	virtual ~Reconstructor();

	void update();
	cv::Mat cluster(int cluster_num, bool init = false);
	void recolourVoxels(cv::Mat labels, int cluster_num);
	std::vector<cv::Mat> getColours(cv::Mat labels, int cluster_num, int camera_num, bool init = false);
	bool isOccluded(Voxel* vox, int cameraNum);
	std::vector<cv::Vec3b> concatenateColours(int camera_num, std::vector<cv::Point> points, double pixelsThresh = 0);
	cv::Vec3b Vec3bBGR2HSV(uchar B, uchar G, uchar R);
	cv::Mat colourVecToMat(std::vector<cv::Vec3b> vec);
	cv::Mat pointVecToMat(std::vector<cv::Point2f> vec);
	cv::Mat getHistogram(cv::Mat inputColours);
	void updateClusterColours(std::vector<std::vector<cv::Mat>> models, std::vector<float> weights);
	double distanceBetweenHists(cv::Mat hist1, cv::Mat hist2);
	void initialiseCluster(int cluster_num, int camera_num);
	void updateClusters(int cluster_num);
	std::vector<cv::Mat> clusterLabelsToModels(cv::Mat labels, int cluster_num, int camera_num, bool init = false);
	cv::Point2i minMat(std::vector<std::vector<double>> matrix);
	std::vector<int> getCurrentFrameOcclusions(cv::Mat labels);
	int getBestCamera(std::vector<int> occlusions);
	std::vector<float> setWeights(std::vector<int> occlusions);

	const void printFlickerCount() const
	{
		std::cout << m_flicker_amount << std::endl;
	}

	const void printTrackingVoxels() const
	{
		cv::Mat trackimg(m_height / (m_step / 2), m_height / (m_step / 2), CV_8UC3, cv::Vec3b(0, 0, 0));

		for (int i = 0; i < m_tracking_voxels.size(); i++)
		{
			int xpos = (m_tracking_voxels[i]->x / (m_step / 2)) + (m_height / m_step);
			int ypos = (m_tracking_voxels[i]->y / (m_step / 2)) + (m_height / m_step);

			trackimg.at<cv::Vec3b>(xpos, ypos) = m_tracking_voxels[i]->color;
		}

		cv::imwrite("data/tracking.png", trackimg);
	}

	const std::vector<Voxel*>& getVisibleVoxels() const
	{
		return m_visible_voxels;
	}

	const std::vector<Voxel*>& getTrackingVoxels() const
	{
		return m_tracking_voxels;
	}

	const std::vector<Voxel*>& getVoxels() const
	{
		return m_voxels;
	}

	void setVisibleVoxels(
			const std::vector<Voxel*>& visibleVoxels)
	{
		m_visible_voxels = visibleVoxels;
	}

	void setVoxels(
			const std::vector<Voxel*>& voxels)
	{
		m_voxels = voxels;
	}

	void clearTrackingVoxels()
	{
		m_tracking_voxels.clear();
	}

	const std::vector<cv::Point3f*>& getCorners() const
	{
		return m_corners;
	}

	int getSize() const
	{
		return m_height;
	}

	const cv::Size& getPlaneSize() const
	{
		return m_plane_size;
	}
};

} /* namespace nl_uu_science_gmt */

#endif /* RECONSTRUCTOR_H_ */
