#include "StdAfx.h"
#include "ObjectnessTrain.h"
#include "CmShow.h"


void ObjectnessTrain::trainObjectnessModel()
{
	generateTrainData();

}

void ObjectnessTrain::generateTrainData()
{
	const int trainNum = _dataSet.trainNum;
	const int NUM_NEG_BOX = 100; // nubmer of negative windows sampled from each image
	xTrainP.reserve(1000000);
	xTrainN.reserve(1000000);

	for (int i = 0; i < trainNum; i++)
	{
		const int NUM_GT_BOX = _dataSet.imgFr[i].size();
		Mat im3u = imread(_dataSet.imgPath + "\\" + _dataSet.imgPathName[i]);

		// positive samples
		for (int k =0; k < NUM_GT_BOX; k++)
		{
			const Vec4i &bbgt = _dataSet.imgFr[i][k];
			Vec4i bbs(bbgt[0], bbgt[1], min(bbgt[2], im3u.cols - 1), min(bbgt[3], im3u.rows - 1));
			Mat mag1f, magF1f;
			mag1f = getFeature(im3u, bbs);
			// flip the train image
			flip(mag1f, magF1f, CV_FLIP_HORIZONTAL);
			xTrainP.push_back(mag1f);
			xTrainP.push_back(magF1f);
		}

		// negative samples
		for (int k = 0; k < NUM_NEG_BOX; k++)
		{
			int x1 = rand() % im3u.cols, x2 = rand() % im3u.cols;
			int y1 = rand() % im3u.rows, y2 = rand() % im3u.rows;

			if (x1 == x2 || y1 == y2)
				continue;
			Vec4i bb(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2));
			if (maxIntUnion(bb, _dataSet.imgFr[i]) < 0.5)
				xTrainN.push_back(getFeature(im3u, bb));
		}
	}
}



Mat ObjectnessTrain::getFeature(CMat &img3u, const Vec4i &bb)
{
	int x = bb[0], y = bb[1];
	Rect reg(x, y, bb[2] -  x, bb[3] - y);
	Mat subImg3u, mag1f, mag1u;
	resize(img3u(reg), subImg3u, Size(_W, _W));
	gradientMag(subImg3u, mag1u);
	mag1u.convertTo(mag1f, CV_32F);
	return mag1f;
}

void ObjectnessTrain::gradientMag(CMat &imgBGR3u, Mat &mag1u)
{
	gradientRGB(imgBGR3u, mag1u);
}

void ObjectnessTrain::gradientRGB(CMat &bgr3u, Mat &mag1u)
{
	const int H = bgr3u.rows, W = bgr3u.cols;
	Mat Ix(H, W, CV_32S), Iy(H, W, CV_32S);

	// Left/right most column Ix
	for (int y = 0; y < H; y++){
		Ix.at<int>(y, 0) = bgrMaxDist(bgr3u.at<Vec3b>(y, 1), bgr3u.at<Vec3b>(y, 0))*2;
		Ix.at<int>(y, W-1) = bgrMaxDist(bgr3u.at<Vec3b>(y, W-1), bgr3u.at<Vec3b>(y, W-2))*2;
	}

	// Top/bottom most column Iy
	for (int x = 0; x < W; x++)	{
		Iy.at<int>(0, x) = bgrMaxDist(bgr3u.at<Vec3b>(1, x), bgr3u.at<Vec3b>(0, x))*2;
		Iy.at<int>(H-1, x) = bgrMaxDist(bgr3u.at<Vec3b>(H-1, x), bgr3u.at<Vec3b>(H-2, x))*2; 
	}

	// Find the gradient for inner regions
	for (int y = 0; y < H; y++){
		const Vec3b *dataP = bgr3u.ptr<Vec3b>(y);
		for (int x = 2; x < W; x++)
			Ix.at<int>(y, x-1) = bgrMaxDist(dataP[x-2], dataP[x]); //  bgr3u.at<Vec3b>(y, x+1), bgr3u.at<Vec3b>(y, x-1));
	}
	for (int y = 1; y < H-1; y++){
		const Vec3b *tP = bgr3u.ptr<Vec3b>(y-1);
		const Vec3b *bP = bgr3u.ptr<Vec3b>(y+1);
		for (int x = 0; x < W; x++)
			Iy.at<int>(y, x) = bgrMaxDist(tP[x], bP[x]);
	}
	gradientXY(Ix, Iy, mag1u);
}

/*void ObjectnessTest::gradientGray(CMat &bgr3u, Mat &mag1u)
{
	Mat g1u;
	cvtColor(bgr3u, g1u, CV_BGR2GRAY); 
	const int H = g1u.rows, W = g1u.cols;
	Mat Ix(H, W, CV_32S), Iy(H, W, CV_32S);

	// Left/right most column Ix
	for (int y = 0; y < H; y++){
		Ix.at<int>(y, 0) = abs(g1u.at<byte>(y, 1) - g1u.at<byte>(y, 0)) * 2;
		Ix.at<int>(y, W-1) = abs(g1u.at<byte>(y, W-1) - g1u.at<byte>(y, W-2)) * 2;
	}

	// Top/bottom most column Iy
	for (int x = 0; x < W; x++)	{
		Iy.at<int>(0, x) = abs(g1u.at<byte>(1, x) - g1u.at<byte>(0, x)) * 2;
		Iy.at<int>(H-1, x) = abs(g1u.at<byte>(H-1, x) - g1u.at<byte>(H-2, x)) * 2; 
	}

	// Find the gradient for inner regions
	for (int y = 0; y < H; y++)
		for (int x = 1; x < W-1; x++)
			Ix.at<int>(y, x) = abs(g1u.at<byte>(y, x+1) - g1u.at<byte>(y, x-1));
	for (int y = 1; y < H-1; y++)
		for (int x = 0; x < W; x++)
			Iy.at<int>(y, x) = abs(g1u.at<byte>(y+1, x) - g1u.at<byte>(y-1, x));

	gradientXY(Ix, Iy, mag1u);
}


void ObjectnessTest::gradientHSV(CMat &bgr3u, Mat &mag1u)
{
	Mat hsv3u;
	cvtColor(bgr3u, hsv3u, CV_BGR2HSV);
	const int H = hsv3u.rows, W = hsv3u.cols;
	Mat Ix(H, W, CV_32S), Iy(H, W, CV_32S);

	// Left/right most column Ix
	for (int y = 0; y < H; y++){
		Ix.at<int>(y, 0) = vecDist3b(hsv3u.at<Vec3b>(y, 1), hsv3u.at<Vec3b>(y, 0));
		Ix.at<int>(y, W-1) = vecDist3b(hsv3u.at<Vec3b>(y, W-1), hsv3u.at<Vec3b>(y, W-2));
	}

	// Top/bottom most column Iy
	for (int x = 0; x < W; x++)	{
		Iy.at<int>(0, x) = vecDist3b(hsv3u.at<Vec3b>(1, x), hsv3u.at<Vec3b>(0, x));
		Iy.at<int>(H-1, x) = vecDist3b(hsv3u.at<Vec3b>(H-1, x), hsv3u.at<Vec3b>(H-2, x)); 
	}

	// Find the gradient for inner regions
	for (int y = 0; y < H; y++)
		for (int x = 1; x < W-1; x++)
			Ix.at<int>(y, x) = vecDist3b(hsv3u.at<Vec3b>(y, x+1), hsv3u.at<Vec3b>(y, x-1))/2;
	for (int y = 1; y < H-1; y++)
		for (int x = 0; x < W; x++)
			Iy.at<int>(y, x) = vecDist3b(hsv3u.at<Vec3b>(y+1, x), hsv3u.at<Vec3b>(y-1, x))/2;

	gradientXY(Ix, Iy, mag1u);
}*/

void ObjectnessTrain::gradientXY(CMat &x1i, CMat &y1i, Mat &mag1u)
{
	const int H = x1i.rows, W = x1i.cols;
	mag1u.create(H, W, CV_8U);
	for (int r = 0; r < H; r++){
		const int *x = x1i.ptr<int>(r), *y = y1i.ptr<int>(r);
		byte* m = mag1u.ptr<byte>(r);
		for (int c = 0; c < W; c++)
			m[c] = min(x[c] + y[c], 255);   //((int)sqrt(sqr(x[c]) + sqr(y[c])), 255);
	}
}