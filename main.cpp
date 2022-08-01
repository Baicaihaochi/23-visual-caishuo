#include <iostream>
#include <opencv2/opencv.hpp>
#include "ArmorPlate.h"

using namespace std;
using namespace cv;

bool CameraRead(ArmorPlate& armor_param);

int main()
{
	ArmorPlate armor;

	armor.CameraInit(0);

	//while (1)
	//{
		
		//if (!CameraRead(armor))
			//continue;
	
		armor.AutoShoot();

	//}
	return 0;
}

bool CameraRead(ArmorPlate& armor_param)
{
	armor_param.capture_plate.read(armor_param.armor_image);
	if (!armor_param.armor_image.data)
	{
		cout << "The camera has not read a image!" << endl;
		armor_param.CameraInit(0);
		return false;
	}
	else
		return true;
}
