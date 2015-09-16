#include "StdAfx.h"
#include "ObjectnessTest.h"
#include "CmShow.h"

int ObjectnessTest::loadTrainedModel(string modelPath)
{
	if (modelPath.size() == 0)
		modelPath = _modelPath;
	
	string s1 = modelPath + "\\" + "ObjNessB2W8MAXBGR.wS1";
	string s2 = modelPath + "\\" + "ObjNessB2W8MAXBGR.wS2";
	Mat filters1f, reW1f;
	if (!matRead(s1, filters1f))
	{
		printf("Cannot load model\n");
		return false;
	}

	CV_Assert(filters1f.type() == CV_32F);
	_svmFilter = filters1f;

	return true;
}

// for dataset
void ObjectnessTest::getFaceProposalsForImgsFast(vector<vector<Vec4i>> &_frsImgs, int numDetPerSize)
{
	const int testNum = _dataSet.testNum;
	vector<vector<Vec4i>> boxesTests;
	boxesTests.resize(testNum);
	
	for (int i = 0; i < testNum; i++)
		boxesTests[i].reserve(10000);

	// predict
	printf("Start predicting\n");
	for (int i = 0; i < testNum; i++)
	{
		printf("Process %d images..\n", i);
		Mat img = imread(_dataSet.imgPath + "\\" + _dataSet.imgPathName[i]);
		getFaceProposaksForPerImgFast(img, boxesTests[i], numDetPerSize);
	}

	// save
	_frsImgs.resize(testNum);
	for (int i = 0; i < testNum; i++)
	{
		_frsImgs[i].resize(boxesTests[i].size());
		for (int j = 0; j < boxesTests[i].size(); j++)
			_frsImgs[i][j] = boxesTests[i][j];
	}

	evaluatePerImgRecall(_frsImgs, 10000);
}

// for single image
void ObjectnessTest::getFaceProposaksForPerImgFast(Mat &img3u, vector<Vec4i> &frsPerImg, int numDetPerSize)
{
	// predict stage I
	const int imgW = img3u.cols, imgH = img3u.rows;
	const int maxFace = min(imgW, imgH);
	const int minFace = 12;
	const double maxScale = 8.0 / minFace, minScale = 8.0 / maxFace;
	for (double scale = maxScale; scale >= minScale; scale = scale * 0.9)
	{
		int height = cvRound(imgH * scale), width = cvRound(imgW * scale);
		Mat im3u, matchCost1f, mag1u;

		// NG feature extract and process
		resize(img3u, im3u, Size(cvRound(_W / (scale * maxFace / imgW)), cvRound(_W / (scale * maxFace / imgH))));
		gradientMag(im3u, mag1u);
		matchCost1f = _bingF.matchTemplate(mag1u);
		ValStructVec<float, Point> matchCost;
		nonMaxSup(matchCost1f, matchCost, _NSS, numDetPerSize, 1);

		// Find true locations 
		double ratioX = min(height, width)/_W, ratioY = min(height, width)/_W;
		int iMax = min(matchCost.size(), numDetPerSize);
		for (int i = 0; i < iMax; i++)
		{
			float mVal = matchCost(i);
			Point pnt = matchCost[i];
			Vec4i box(cvRound(pnt.x * ratioX), cvRound(pnt.y*ratioY));
			box[2] = cvRound(min(box[0] + min(height, width), imgW));
			box[3] = cvRound(min(box[1] + min(height, width), imgH));
			box[0] ++;
			box[1] ++;
			frsPerImg.push_back(box); 
		}
	}
	//NOTE: predict stage II  is not used for our task
}

// Read matrix from binary file
bool ObjectnessTest::matRead(const string& filename, Mat& _M)
{
	FILE* f = fopen(_S(filename), "rb");
	if (f == NULL)
		return false;
	char buf[8];
	int pre = fread(buf,sizeof(char), 5, f);
	if (strncmp(buf, "CmMat", 5) != 0)	{
		printf("Invalidate CvMat data file %s\n", _S(filename));
		return false;
	}
	int headData[3]; // Width, height, type
	fread(headData, sizeof(int), 3, f);
	Mat M(headData[1], headData[0], headData[2]);
	fread(M.data, sizeof(char), M.step * M.rows, f);
	fclose(f);
	M.copyTo(_M);
	return true;
}

void ObjectnessTest::gradientMag(CMat &imgBGR3u, Mat &mag1u)
{
	gradientRGB(imgBGR3u, mag1u);
}

void ObjectnessTest::gradientRGB(CMat &bgr3u, Mat &mag1u)
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

void ObjectnessTest::gradientXY(CMat &x1i, CMat &y1i, Mat &mag1u)
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

void ObjectnessTest::nonMaxSup(CMat &matchCost1f, ValStructVec<float, Point> &matchCost, int NSS, int maxPoint, bool fast)
{
	const int _h = matchCost1f.rows, _w = matchCost1f.cols;
	Mat isMax1u = Mat::ones(_h, _w, CV_8U), costSmooth1f;
	ValStructVec<float, Point> valPnt;
	matchCost.reserve(_h * _w);
	valPnt.reserve(_h * _w);
	if (fast){
		blur(matchCost1f, costSmooth1f, Size(3, 3));
		for (int r = 0; r < _h; r++){
			const float* d = matchCost1f.ptr<float>(r);
			const float* ds = costSmooth1f.ptr<float>(r);
			for (int c = 0; c < _w; c++)
				if (d[c] >= ds[c])
					valPnt.pushBack(d[c], Point(c, r));
		}
	}
	else{
		for (int r = 0; r < _h; r++){
			const float* d = matchCost1f.ptr<float>(r);
			for (int c = 0; c < _w; c++)
				valPnt.pushBack(d[c], Point(c, r));
		}
	}

	valPnt.sort();
	for (int i = 0; i < valPnt.size(); i++){
		Point &pnt = valPnt[i];
		if (isMax1u.at<byte>(pnt)){
			matchCost.pushBack(valPnt(i), pnt);
			for (int dy = -NSS; dy <= NSS; dy++) for (int dx = -NSS; dx <= NSS; dx++){
				Point neighbor = pnt + Point(dx, dy);
				if (!CHK_IND(neighbor))
					continue;
				isMax1u.at<byte>(neighbor) = false;
			}
		}
		if (matchCost.size() >= maxPoint)
			return;
	}
}

void ObjectnessTest::evaluatePerImgRecall(const vector<vector<Vec4i>> &boxesTests, const int numDet)
{
	vecD recalls(numDet);
	vecD avgScore(numDet);
	const int TEST_NUM = _dataSet.testNum;
	for (int i = 0; i < TEST_NUM; i++){
		const vector<Vec4i> &boxesGT = _dataSet.imgFr[i];
		const vector<Vec4i> &boxes = boxesTests[i];
		const int gtNumCrnt = boxesGT.size();
		vecI detected(gtNumCrnt);
		vecD score(gtNumCrnt);
		double sumDetected = 0, abo = 0;
		for (int j = 0; j < numDet; j++){
			if (j >= (int)boxes.size()){
				recalls[j] += sumDetected/gtNumCrnt;
				avgScore[j] += abo/gtNumCrnt;
				continue;
			}

			for (int k = 0; k < gtNumCrnt; k++)	{
				double s = DataSet::interUnio(boxes[j], boxesGT[k]);
				score[k] = max(score[k], s);
				detected[k] = score[k] >= 0.5 ? 1 : 0;
			}
			sumDetected = 0, abo = 0;
			for (int k = 0; k < gtNumCrnt; k++)	
				sumDetected += detected[k], abo += score[k];
			recalls[j] += sumDetected/gtNumCrnt;
			avgScore[j] += abo/gtNumCrnt;
		}
	}

	for (int i = 0; i < numDet; i++){
		recalls[i] /=  TEST_NUM;
		avgScore[i] /= TEST_NUM;
	}

	int idx[8] = {1, 10, 100, 1000, 2000, 3000, 4000, 5000};
	for (int i = 0; i < 8; i++){
		if (idx[i] > numDet)
			continue;
		printf("%d:%.3g,%.3g\t", idx[i], recalls[idx[i] - 1], avgScore[idx[i] - 1]);
	}
	printf("\n");
}

void ObjectnessTest::illuTestReults(string &imgPath, string &savePath, const vector<Vec4i> &gtBoxesTest, const vector<Vec4i> &boxesTests)
{
	const int gtNumCrnt = gtBoxesTest.size();
	Mat img = imread(imgPath);
	Mat bboxMatchImg = Mat::zeros(img.size(), CV_32F);

	vecD score(gtNumCrnt);
	vector<Vec4i> bboxMatch(gtNumCrnt);
	for (int j = 0; j < boxesTests.size(); j++)
	{
		const Vec4i &bb = boxesTests[j];
		for (int k = 0; k < gtNumCrnt; k++)	{
			double mVal = DataSet::interUnio(boxesTests[j], gtBoxesTest[k]);
			if (mVal < score[k])
				continue;
			score[k] = mVal;
			bboxMatch[k] = boxesTests[j];
		}
	}

	for (int k = 0; k < gtNumCrnt; k++)
	{
		const Vec4i &bb = bboxMatch[k];
		rectangle(img, Point(bb[0], bb[1]), Point(bb[2], bb[3]), Scalar(0), 3);
		rectangle(img, Point(bb[0], bb[1]), Point(bb[2], bb[3]), Scalar(255, 255, 255), 2);
		rectangle(img, Point(bb[0], bb[1]), Point(bb[2], bb[3]), Scalar(0, 0, 255), 1);
	}

	imwrite(savePath, img);
}
