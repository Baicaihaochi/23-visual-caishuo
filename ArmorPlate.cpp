#include <opencv2/opencv.hpp>
#include<opencv2/imgproc/types_c.h>
#include <opencv2/highgui/highgui_c.h>
#include "ArmorPlate.h"

using namespace std;
using namespace cv;


ArmorPlate::ArmorPlate()
{
	myteam = REDTEAM;
}

bool ArmorPlate::CameraInit(int device)
{
	capture_plate.open(device);
	if (!capture_plate.isOpened())
	{
		cout << "The capture has something wrong!";
		return false;
	}
	else return true;
}

cv::RotatedRect& adjustRec(cv::RotatedRect& rec)
{
	using std::swap;

	float& width = rec.size.width;
	float& height = rec.size.height;
	float& angle = rec.angle;


	while (angle >= 90.0) angle -= 180.0;
	while (angle < -90.0) angle += 180.0;

	
	
		if (angle >= 45.0)
		{
			swap(width, height);
			angle -= 90.0;
		}
		else if (angle < -45.0)
		{
			swap(width, height);
			angle += 90.0;
		}
	

	return rec;
}

void ArmorPlate::AutoShoot()
{
/*
	armor_image = imread("1920.png");
	ImgPreprosses(armor_image, pre_image);
	//imshow("预处理图", pre_image);
	waitKey(0);
*/
	VideoCapture cap("1.avi");
	while (true)
	{
		Mat frame;
		cap >> frame;
		if(frame.empty())
		{
			break;
		}
		ImgPreprosses(frame);
		namedWindow("video",WINDOW_AUTOSIZE);
		imshow("video",frame);
		waitKey(30);
		destroyAllWindows;
	}
	

}



void drawall(vector<RotatedRect> rec,Mat img)
{
	for (int i = 0; i < rec.size(); i++)
	{
		Point2f p[4];
		rec[i].points(p);
		line(img, p[0], p[1], Scalar(0, 0, 255), 1, 8, 0);
		line(img, p[1], p[2], Scalar(0, 0, 255), 1, 8, 0);
		line(img, p[2], p[3], Scalar(0, 0, 255), 1, 8, 0);
		line(img, p[3], p[0], Scalar(0, 0, 255), 1, 8, 0);
	}
}



void ArmorPlate::ImgPreprosses(const Mat& src)
{
	Mat grayImg;
	Mat binBrightImg;
	vector<RotatedRect> lightInfos;
	double MaxValue;
	vector<Mat> channels;

	Mat HSVImg;
	Mat image;
	cvtColor(src,HSVImg,COLOR_BGR2HSV); 
	split(HSVImg, channels);
	minMaxLoc(channels[2], 0, &MaxValue, 0, 0); 
	threshold(channels[2], channels[2], MaxValue*0.98, 255, THRESH_BINARY);
	Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
	medianBlur(channels[2], channels[2], 3);
	morphologyEx(channels[2], channels[2], MORPH_DILATE, element, Point(-1, -1), 1);

	binBrightImg = channels[2];
	vector<vector<Point>> lightContours;
	cv::findContours(binBrightImg.clone(), lightContours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);


	lightInfos.clear();

	for (const auto& contour : lightContours)
	{
		float lightContourArea = contourArea(contour);
		if (contour.size() <= 5 ||lightContourArea <10 ) continue;
		
		RotatedRect lightRec = fitEllipse(contour);
		RotatedRect minAreaRec = minAreaRect(contour);
		adjustRec(lightRec);
		if(lightRec.size.width>src.rows || lightRec.size.width>src.cols||lightRec.size.height>src.rows || lightRec.size.height>src.cols) continue;
		if(!(0 <= lightRec.center.x && 0 <= lightRec.size.width && lightRec.center.x + lightRec.size.width <= src.cols && 0 <= lightRec.center.y && 0 <= lightRec.size.height && lightRec.center.y + lightRec.size.height <= src.rows)) continue;
		if ((lightRec.size.width / lightRec.size.height) >0.8)
			continue;
		int x = lightRec.center.x - lightRec.size.width;
		if (x < 0)
			continue;
		int y = lightRec.center.y - lightRec.size.height;
		if (y < 0)
			continue;

		if (lightRec.size.width / lightRec.size.height > 1.0 ||
			lightContourArea / lightRec.size.area() < 0.5)
			continue;
		lightRec.size.width *= 0.9;
		lightRec.size.height *= 0.9;

		Rect boundRect = lightRec.boundingRect();
		Mat lightImg = src(boundRect);

		if((lightRec.size.height>10&& (lightRec.size.height < 150)&&(lightRec.angle<45||lightRec.angle>135)))
			lightInfos.push_back(lightRec);


	}

	vector<RotatedRect> armors;
	vector<ArmorRect> armorRects;
	ArmorRect armorRect;

	armors.clear();
	armorRects.clear();

	if (lightInfos.size()<=1)
	{
		cout << "There's no light contours in quality." << endl;
	}
	
	sort(lightInfos.begin(), lightInfos.end(), [](const RotatedRect& ld1, const RotatedRect& ld2)
	{
		return ld1.center.x < ld2.center.x;
	});

	for (int i = 0; i < lightInfos.size(); i++)
	{
		for (int j = i + 1; j < lightInfos.size(); j++)
		{
			const RotatedRect& left = lightInfos[i];
			const RotatedRect& right = lightInfos[j];

			double heightDiff = abs(left.size.height - right.size.height);
			double widthDiff = abs(left.size.width - right.size.width);
			double angleDiff = abs(left.angle - right.angle);
			double yDiff = abs(left.center.y - right.center.y);
			double xDiff = abs(left.center.x - right.center.x);
			double meanheight = (left.size.height + right.size.height)/2;
			double yDiffRatio = yDiff / meanheight;
			double xDiffRatio = xDiff / meanheight;
			double dis= sqrt((left.center.x - right.center.x)*(left.center.x - right.center.x) + (left.center.y - right.center.y)*(left.center.y - right.center.y));
			double ratio = dis / meanheight;
			float heightDiff_ratio = heightDiff / max(left.size.height, right.size.height);

			if (angleDiff > 10 || xDiffRatio < 0.5 || yDiffRatio>0.7||ratio>3||ratio<1)
				continue;

			armorRect.armors.center.x = (left.center.x + right.center.x) / 2;
			armorRect.armors.center.y = (left.center.y + right.center.y) / 2;
			armorRect.armors.angle= (left.angle + right.angle) / 2;
			//cout << left.angle << endl;
			//armorRect.armors.angle = 0;
			if (180 - angleDiff < 3)
				armorRect.armors.angle += 90;
			armorRect.armors.size.height= (left.size.height + right.size.height) / 2;
			armorRect.armors.size.width = sqrt((left.center.x - right.center.x)*(left.center.x - right.center.x) + (left.center.y - right.center.y)*(left.center.y - right.center.y));

			double nL = armorRect.armors.size.height;
			double nW = armorRect.armors.size.width;

			//change
			if (nL < nW)
			{
				armorRect.armors.size.height = nL;
				armorRect.armors.size.width = nW;
			}
			else
			{
				armorRect.armors.size.height = nW;
				armorRect.armors.size.width = nL;
			}

			armorRects.emplace_back(armorRect);
			armors.push_back(armorRect.armors);
		}
	}
	if (armorRects.empty())
		cout << "There is no armor in quality!" << endl;
	
	drawall(armors, src);
//	imshow("", src);
}