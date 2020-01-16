#include <cstdlib>
#include <string>
#include <vector>

#include "utilities/General.h"
#include "utilities/Calibration.h"
#include "VoxelReconstruction.h"

using namespace nl_uu_science_gmt;

void generateIntrinsics()
{
	for (int i = 0; i < 4; i++)
	{
		vector<String> paths;

		for (int j = 0; j < 10; j++) paths.push_back("data/cam" + to_string(i + 1) + "/frames_intrinsics/scene" + to_string(j) + ".png");

		int cb_width = 0, cb_height = 0;
		int cb_square_size = 0;

		// Read the checkerboard properties (XML)
		FileStorage fs;
		fs.open("data" + string(PATH_SEP) + General::CBConfigFile, FileStorage::READ);
		if (fs.isOpened())
		{
			fs["CheckerBoardWidth"] >> cb_width;
			fs["CheckerBoardHeight"] >> cb_height;
			fs["CheckerBoardSquareSize"] >> cb_square_size;
		}
		fs.release();

		Size boardSize = cvSize(cb_width, cb_height);
		Calibration::calibrate(paths, boardSize, i + 1);
	}
}

int main(int argc, char** argv)
{
	//generateIntrinsics();

	VoxelReconstruction::showKeys();
	VoxelReconstruction vr("data" + std::string(PATH_SEP), 4);
	vr.run(argc, argv);

	return EXIT_SUCCESS;
}