#include "StdAfx.h"
#include "ObjectnessTrain.h"
#include "CmShow.h"


void ObjectnessTrain::trainObjectnessModel()
{
	cout << "Generating training data..." << endl;
	generateTrainData();
	cout <<"Training model.." <<endl;
	trainStageI();
	cout << "Training complete.." <<endl;
}

void ObjectnessTrain::generateTrainData()
{
	const int trainNum = _dataSet.trainNum;
	const int NUM_NEG_BOX = 100; // nubmer of negative windows sampled from each image
	//vector<Mat> xTrainP, xTrainN;
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

	int numP = xTrainP.size();
	int numN = xTrainN.size();
	int iP = 0, iN = 0;
	Mat xP1f(numP, _W*_W, CV_32F), xN1f(numN, _W*_W, CV_32F);
	for (int i = 0; i < numP; i++)
	{
		memcpy(xP1f.ptr(iP++), xTrainP[i].data, _W*_W*sizeof(float));
	}
	for (int i = 0; i < numN; i++)
	{
		memcpy(xN1f.ptr(iN++), xTrainN[i].data, _W*_W*sizeof(float));
	}

	matWrite(_modelPath + "\\" + "data" + ".xP", xP1f);
	matWrite(_modelPath + "\\" + "data" + ".xN", xN1f);

}

void ObjectnessTrain::trainStageI()
{
	vector<Mat> pX, nX;
	pX.reserve(200000), nX.reserve(200000);
	Mat xP1f, xN1f;
	CV_Assert(matRead(_modelPath + "\\" + "data" + ".xP", xP1f) && matRead(_modelPath + "\\" + "data" + ".xN", xN1f));
	for (int r = 0; r < xP1f.rows; r++)
		pX.push_back(xP1f.row(r));
	for (int r = 0; r < xN1f.rows; r++)
		nX.push_back(xN1f.row(r));

	Mat crntW = trainSVM(pX, nX, L1R_L2LOSS_SVC, 10, 1);
	crntW = crntW.colRange(0, crntW.cols - 1).reshape(1, _W);
	CV_Assert(crntW.size() == Size(_W, _W));
	matWrite(_modelPath + "\\" + "ObjNessB2W8MAXBGR" + ".wS1", crntW);
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

Mat ObjectnessTrain::trainSVM(const vector<Mat> &pX1f, const vector<Mat> &nX1f, int sT, double C, double bias, double eps, int maxTrainNum)
{
	vecI ind(nX1f.size());
	for (size_t i = 0; i < ind.size(); i++)
		ind[i] = i;
	int numP = pX1f.size(), feaDim = pX1f[0].cols;
	int totalSample = numP + nX1f.size();
	if (totalSample > maxTrainNum)
		random_shuffle(ind.begin(), ind.end());
	totalSample = min(totalSample, maxTrainNum);
	Mat X1f(totalSample, feaDim, CV_32F);
	vecI Y(totalSample);
	for(int i = 0; i < numP; i++){
		pX1f[i].copyTo(X1f.row(i));
		Y[i] = 1;
	}
	for (int i = numP; i < totalSample; i++){
		nX1f[ind[i - numP]].copyTo(X1f.row(i));
		Y[i] = -1;
	}
	return trainSVM(X1f, Y, sT, C, bias, eps);
}

// Training SVM with feature vector X and label Y. 
// Each row of X is a feature vector, with corresponding label in Y.
// Return a CV_32F weight Mat
Mat ObjectnessTrain::trainSVM(CMat &X1f, const vecI &Y, int sT, double C, double bias, double eps)
{
	// Set SVM parameters
	parameter param; {
		param.solver_type = sT; // L2R_L2LOSS_SVC_DUAL;
		param.C = C;
		param.eps = eps; // see setting below
		param.p = 0.1;
		param.nr_weight = 0;
		param.weight_label = NULL;
		param.weight = NULL;
		CV_Assert(X1f.rows == Y.size() && X1f.type() == CV_32F);
	}

	// Initialize a problem
	feature_node *x_space = NULL;
	problem prob;{
		prob.l = X1f.rows;
		prob.bias = bias;
		prob.y = Malloc(double, prob.l);
		prob.x = Malloc(feature_node*, prob.l);
		const int DIM_FEA = X1f.cols;
		prob.n = DIM_FEA + (bias >= 0 ? 1 : 0);
		x_space = Malloc(feature_node, (prob.n + 1) * prob.l);
		int j = 0;
		for (int i = 0; i < prob.l; i++){
			prob.y[i] = Y[i];
			prob.x[i] = &x_space[j];
			const float* xData = X1f.ptr<float>(i);
			for (int k = 0; k < DIM_FEA; k++){
				x_space[j].index = k + 1;
				x_space[j++].value = xData[k];
			}
			if (bias >= 0){
				x_space[j].index = prob.n;
				x_space[j++].value = bias;
			}
			x_space[j++].index = -1;
		}
		CV_Assert(j == (prob.n + 1) * prob.l);
	}

	// Training SVM for current problem
	const char*  error_msg = check_parameter(&prob, &param);
	if(error_msg){
		fprintf(stderr,"ERROR: %s\n",error_msg);
		exit(1);
	}
	model *svmModel = train(&prob, &param);
	Mat wMat(1, prob.n, CV_64F, svmModel->w);
	wMat.convertTo(wMat, CV_32F);
	free_and_destroy_model(&svmModel);
	destroy_param(&param);
	free(prob.y);
	free(prob.x);
	free(x_space);
	return wMat;
}

// Write matrix to binary file
bool ObjectnessTrain::matWrite(CStr& filename, CMat& _M)
{
	Mat M;
	_M.copyTo(M);
	FILE* file = fopen(_S(filename), "wb");
	if (file == NULL || M.empty())
		return false;
	fwrite("CmMat", sizeof(char), 5, file);
	int headData[3] = {M.cols, M.rows, M.type()};
	fwrite(headData, sizeof(int), 3, file);
	fwrite(M.data, sizeof(char), M.step * M.rows, file);
	fclose(file);
	return true;
}

// Read matrix from binary file
bool ObjectnessTrain::matRead(const string& filename, Mat& _M)
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