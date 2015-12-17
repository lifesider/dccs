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
	for(;;)
	{
		img = imread(imgsec);
		if(img.empty()) break;
		int framenum = 1;
		string numstr = imgsec.substr(imgsec.rfind('\\')+1, imgsec.rfind('.')-imgsec.rfind('\\')-1);
		istringstream istr(numstr);
		istr >> framenum;
		double t = getTickCount();
		gmm.SetVideoFrame(img.data, NULL, framenum);
		printf("SetVideoFrame: %.2fms\n", (getTickCount()-t)*1000/getTickFrequency());
		BYTE* bw;
		gmm.GetGmmBw(&bw);
		imshow("", Mat(img.size(), CV_8UC1, bw));
		if(waitKey(30) == 27)
			break;
		ostringstream ostr;
		ostr << framenum+1;
		numstr = ostr.str();
		imgsec.erase(imgsec.rfind('\\')+1);
		imgsec.append(numstr+".jpg");
	}
	t.stop();
	printf("%.2fms\n", t.gettimems());
// 	imshow("", img);
// 	waitKey(0);
	release_platform();
	return 0;
}
