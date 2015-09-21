#include "CnnFace.h"
#include "ValStructVec.h"


int CnnFace::loadTrainedModel()
{
	_cnn = new Net();
	if(_cnn->LoadFromFile(_modelPath.c_str()))
	{
		cout << "Load cnn model error!" <<endl;
		return false;
	}
	return true;
}


void CnnFace::getFaceDetectionPerImg(Mat &img3u, vector<Vec4i> &boxProposal, ValStructVec<float, Vec4i> &valBoxes, float thr)
{
	Mat img1u;
	float *data = (float *)malloc(_netSize*_netSize*sizeof(float));
	cvtColor(img3u, img1u, CV_BGR2GRAY);
	valBoxes.reserve(2000);

	for (int i = 0; i < boxProposal.size(); i++)
	{
		// data preprocess
		int xtl = boxProposal[i][0] - 1, ytl = boxProposal[i][1] - 1;
		int xbr = boxProposal[i][2] - 1, ybr = boxProposal[i][3] - 1;
		Rect reg(xtl, ytl, xbr - xtl + 1, ybr - ytl +1);
		Mat patch;
		resize(img1u(reg), patch, Size(_netSize, _netSize));
		int height = patch.rows, width = patch.cols;
		for (int j = 0; j < height*width; j++)
			data[j] = static_cast<float>(patch.data[j]) / 255.0;

		// cnn predict
		_cnn->TakeInput(data, _netSize, _netSize, 1);
		_cnn->Forward();
		
		// results
		float *prob = _cnn->get_blob(_netProbLayer)->data +1;
		if (*prob > thr)
		{
			valBoxes.pushBack(*prob, boxProposal[i]);
#ifdef _DEBUG
			char *temp_name = (char *)malloc(sizeof(char)*1000);
			sprintf(temp_name, "test_%d_%d_%d_%d.jpg", xtl, ytl, xbr, ybr);
			imwrite(temp_name, img1u(reg));
#endif
		}
	}
	free(data);
	data = NULL;
}