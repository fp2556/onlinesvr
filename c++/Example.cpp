/******************************************************************************
*                       ONLINE SUPPORT VECTOR REGRESSION                      *
*                      Copyright 2006 - Francesco Parrella                    *
*                                                                             *
*This program is distributed under the terms of the GNU General Public License*
******************************************************************************/



#include "OnlineSVR.h"
#include <math.h>

using namespace onlinesvr; 

int main ()  
{ 

	// Make a new OnlineSVR
	OnlineSVR* SVR = new OnlineSVR();

	// Set parameters
	SVR->SetC(1);
	SVR->SetEpsilon(0.01);
	SVR->SetKernelType(OnlineSVR::KERNEL_RBF);
	SVR->SetKernelParam(30);
	SVR->SetVerbosity(OnlineSVR::VERBOSITY_NORMAL);	

	// Build training set
	Matrix<double>* TrainingSetX = Matrix<double>::RandMatrix(20,1);
	Vector<double>* TrainingSetY = new Vector<double>();
	for (int i=0; i<TrainingSetX->GetLengthRows(); i++)
		TrainingSetY->Add(sin(TrainingSetX->GetValue(i,0)));
	
	// Train OnlineSVR
	SVR->Train(TrainingSetX,TrainingSetY);

	// Show OnlineSVR info
	SVR->ShowInfo();

	// Predict some new values
	Matrix<double>* TestSetX = new Matrix<double>();
	Vector<double>* X1 = new Vector<double>();
	Vector<double>* X2 = new Vector<double>();
	X1->Add(0);
	X2->Add(1);
	TestSetX->AddRowRef(X1);
	TestSetX->AddRowRef(X2);
	Vector<double>* PredictedY = SVR->Predict(TestSetX);
	cout << "f(0) = " << PredictedY->GetValue(0) << endl;
	cout << "f(1) = " << PredictedY->GetValue(1) << endl;	

	// Forget some samples
	Vector<int>* RemainingSamples = SVR->GetRemainingSetIndexes()->Clone();
	SVR->Forget(RemainingSamples);

	// Save OnlineSVR
	SVR->SaveOnlineSVR("Sin.svr");

	// Delete	
	delete SVR;
	delete TrainingSetX;
	delete TrainingSetY;	
	delete TestSetX;
	delete PredictedY;
	delete RemainingSamples;	
	
	return 0;
}
