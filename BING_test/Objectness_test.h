#include "ValStructVec.h"
#include "FilterBING.h"
#include "Dataset.h"

class Objectness
{
public:
	Objectness(DataSet &dataSet, int W = 8, int NSS = 2);
	~Objectness(void){};

	//int loadTrainedModel(string modelName);

private:
	const int _W;
	const int _NSS;
	
	DataSet _dataSet;
	string _modelName;
	Mat _svmFilter; //Filters learned at stage I
	FilterBING _bingF;   // BING Filter


};
