#pragma once

#define BLUETEAM 0
#define REDTEAM 1

class ArmorPlate
{
public:
	cv::Mat pre_image;
	cv::Mat armor_image;
	cv::VideoCapture capture_plate;

	int myteam;


	ArmorPlate();
	bool CameraInit(int device);
	void AutoShoot();

private:
	void ImgPreprosses(const cv::Mat& src);


};


class ArmorRect
{
public:
	cv::RotatedRect armors;
};