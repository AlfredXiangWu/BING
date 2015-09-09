#include "ValStructVec.h"
#include "FilterBING.h"
#include "Dataset.h"

class Objectness
{
public:
	Objectness(DataSet &dataSet, int W = 8, int NSS = 2);
	~Objectness(void);

	int loadTrainedModel(string modelName);



};
