#include "../../stdafx.h"
#include "../../DccsCanny.h"
#include "../../XhDccsBase.h"
#include "../../xhGradsGMM.h"
#include "../../xhGrayGMM.h"
#include "opencv2/opencv.hpp"
#include <stdlib.h>
#include <string>

using namespace cv;
using namespace std;

int main()
{
	init_platform(0);
	string path(MAX_PATH, '\0');
	path.resize(GetModuleFileNameA(NULL, &path[0], MAX_PATH)+1);
	path.erase(path.rfind('\\')+1);
	path.append("tulips.jpg");
	Mat img = imread(path);
	Mat gray = imread(path, IMREAD_GRAYSCALE);
	clock t;
	t.start();
	xhGradsGMM gmm;
	gmm.SetVideoFrameInfo(768, 576);
	string imgsec = "C:\\opencv3\\opencv\\sources\\samples\\data\\768x576\\1.jpg";
	double sum = 0;
	int framenum = 1;
	for(;;)
	{
		img = imread(imgsec);
		if(img.empty()) break;
		string numstr = imgsec.substr(imgsec.rfind('\\')+1, imgsec.rfind('.')-imgsec.rfind('\\')-1);
		istringstream istr(numstr);
		istr >> framenum;
		double t = getTickCount();
		gmm.SetVideoFrame(img.data, NULL, framenum);
		BYTE* bw;
		gmm.GetGmmBw(&bw);
		double fps = 1.0/((getTickCount()-t)/getTickFrequency());
		printf("fps: %.1ffps\n", fps);
		if(framenum > 1)
			sum += fps;
		imshow("", Mat(img.size(), CV_8UC1, bw));
		if(waitKey(25) == 27)
			break;
		ostringstream ostr;
		ostr << framenum+1;
		numstr = ostr.str();
		imgsec.erase(imgsec.rfind('\\')+1);
		imgsec.append(numstr+".jpg");
	}
	t.stop();
	printf("sum: %.1f, num: %d\n", sum, framenum);
	printf("%.ffps\n", sum/(framenum-1));
// 	imshow("", img);
// 	waitKey(0);
	release_platform();
	return 0;
}
