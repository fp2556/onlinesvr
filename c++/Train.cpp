/******************************************************************************
*                       ONLINE SUPPORT VECTOR REGRESSION                      *
*                      Copyright 2006 - Francesco Parrella                    *
*                                                                             *
*This program is distributed under the terms of the GNU General Public License*
******************************************************************************/


#ifndef TRAIN_CPP
#define TRAIN_CPP

#include <iostream>
#include "time.h"
#include "OnlineSVR.h"


namespace onlinesvr
{

	// Learning Operations
	int OnlineSVR::Train (Matrix<double>* X, Vector<double>* Y)
	{
		// Initialization
		time_t StartTime = time(NULL);
		int Flops = 0;
		this->ShowMessage("Starting Training...\n",1);
		
		// Learning
		for (int i=0; i<X->GetLengthRows(); i++) {
			
			// Element already trained
			int Index = this->X->IndexOf(X->GetRowRef(i));
			if (Index>-1 && Y->Values[i] == this->Y->Values[Index]) {				
				continue;
			}

			// Show Informations
			this->ShowMessage(" ",2);
			this->ShowMessage(" ",3);
			char Line[80];
			sprintf(Line,"Training %d/%d",i+1,X->GetLengthRows());			
			this->ShowMessage(Line,1);
			// Training
			Flops += this->Learn(X->GetRowRef(i),Y->GetValue(i));
		}

		// Stabilize the results
		if (this->StabilizedLearning) {
			int StabilizationNumber = 0;
			while (!this->VerifyKKTConditions()) {
				Flops += this->Stabilize();
				StabilizationNumber ++;
				if (StabilizationNumber>this->GetSamplesTrainedNumber()) {
					this->ShowMessage("Error: it's impossible to stabilize the OnlineSVR. Please add or remove some samples.", VERBOSITY_NORMAL);
					break;
				}
			}
		}

		if (this->Verbosity>=3)
			this->ShowDetails();

		// Show Execution Time
		time_t EndTime = time(NULL);
		long LearningTime = static_cast<long>(EndTime-StartTime);
		this->ShowMessage(" ",2);
		this->ShowMessage(" ",3);
		char Line[80];
		char* TimeElapsed = this->TimeToString(LearningTime);
		sprintf(Line, "\nTrained %d elements correctly in %s.\n", X->GetLengthRows(), TimeElapsed);
		delete TimeElapsed;
		this->ShowMessage(Line,1);

		return Flops;
	}

	// Learning Operations
	int OnlineSVR::Train (Matrix<double>* X, Vector<double>* Y, Matrix<double>* TestSetX, Vector<double>* TestSetY)
	{
		// Initialization
		time_t StartTime = time(NULL);
		int Flops = 0;
		this->ShowMessage("Starting Training...\n",1);
		Vector<double>* MeanErrors = new Vector<double>();
		Vector<double>* Variances = new Vector<double>();
		Vector<double>* Predictions = new Vector<double>();
		
		// Learning
		for (int i=0; i<X->GetLengthRows(); i++) {
			// Show Informations
			this->ShowMessage(" ",2);
			this->ShowMessage(" ",3);
			char Line[80];
			sprintf(Line,"Training %d/%d",i+1,X->GetLengthRows());			
			this->ShowMessage(Line,1);
			// Training
			Predictions->Add(this->Predict(X->GetRowRef(i)));
			Flops += this->Learn(X->GetRowRef(i),Y->GetValue(i));
			Vector<double>* Errors = this->Margin(TestSetX, TestSetY);
			MeanErrors->Add(Errors->MeanAbs());
			Variances->Add(Errors->Variance());
			delete Errors;
		}

		// Stabilize the results
		if (this->StabilizedLearning) {
			int StabilizationNumber = 0;
			while (!this->VerifyKKTConditions()) {
				Flops += this->Stabilize();
				StabilizationNumber ++;
				if (StabilizationNumber>this->GetSamplesTrainedNumber()) {
					this->ShowMessage("Error: it's impossible to stabilize the OnlineSVR. Please add or remove some samples.", VERBOSITY_NORMAL);
					break;
				}
			}
		}

		if (this->Verbosity>=3)
			this->ShowDetails();

		// Show Execution Time
		time_t EndTime = time(NULL);
		long LearningTime = static_cast<long>(EndTime-StartTime);
		this->ShowMessage(" ",2);
		this->ShowMessage(" ",3);
		char Line[80];
		char* TimeElapsed = this->TimeToString(LearningTime);
		sprintf(Line, "\nTrained %d elements correctly in %s.\n", X->GetLengthRows(), TimeElapsed);	
		delete TimeElapsed;
		this->ShowMessage(Line,1);

		// Save the files
		MeanErrors->Save("MeanErrors.txt");
		Variances->Save("Variances.txt");
		Predictions->Save("Predictions.txt");
		delete MeanErrors;
		delete Variances;
		delete Predictions;
		return Flops;
	}

	// Learning Operations
	int OnlineSVR::Train (Matrix<double>* X, Vector<double>* Y, int TrainingSize, int TestSize)
	{
		// Initialization
		time_t StartTime = time(NULL);
		int Flops = 0;
		this->ShowMessage("Starting Training...\n",1);
		Vector<double>* TestErrors = new Vector<double>();

		// Learning
		for (int i=0; i<X->GetLengthRows()-TrainingSize-TestSize+1; i++) {
			// Show Informations
			this->ShowMessage(" ",2);
			this->ShowMessage(" ",3);
			char Line[80];
			sprintf(Line,"Training %d/%d",i+1,X->GetLengthRows());			
			this->ShowMessage(Line,1);
			// Learning
			Matrix<double>* TrainingSetX = X->ExtractRows(i, i+TrainingSize-1);
			Vector<double>* TrainingSetY = Y->Extract(i, i+TrainingSize-1);
			Matrix<double>* TestSetX = X->ExtractRows(i+TrainingSize, i+TrainingSize+TestSize-1);
			Vector<double>* TestSetY = Y->Extract(i+TrainingSize, i+TrainingSize+TestSize-1);
			this->Clear();
			this->Train(TrainingSetX, TrainingSetY);
			Vector<double>* Margins = this->Margin(TestSetX, TestSetY);
			TestErrors->Add(Margins->MeanAbs());
			delete TrainingSetX;
			delete TrainingSetY;
			delete TestSetX;
			delete TestSetY;
			delete Margins;

		}

		if (this->Verbosity>=3)
			this->ShowDetails();

		// Show Execution Time
		time_t EndTime = time(NULL);
		long LearningTime = static_cast<long>(EndTime-StartTime);
		this->ShowMessage(" ",2);
		this->ShowMessage(" ",3);
		char Line[80];
		char* TimeElapsed = this->TimeToString(LearningTime);
		sprintf(Line, "\nTrained %d elements correctly in %s.\n", X->GetLengthRows(), TimeElapsed);
		delete TimeElapsed;
		this->ShowMessage(Line,1);

		// Save the files
		TestErrors->Save("TestErrors.txt");
		delete TestErrors;
		return Flops;
	}

	int OnlineSVR::Train (double**X, double *Y, int ElementsNumber, int ElementsSize)
	{	
		Matrix<double>* NewX = new Matrix<double>(X, ElementsNumber, ElementsSize);
		Vector<double>* NewY = new Vector<double>(Y, ElementsNumber);
		int Flops = Train(NewX,NewY);
		delete NewX;
		delete NewY;
		return Flops;
	}

	int OnlineSVR::Train (Vector<double>* X, double Y)
	{
		int Flops;
		Matrix<double>* X1 = new Matrix<double>();
		Vector<double>* Y1 = new Vector<double>();
		X1->AddRowCopy(X);
		Y1->Add(Y);
		Flops = this->Train(X1,Y1);
		delete X1;
		delete Y1;
		return Flops;
	}

	int OnlineSVR::Learn (Vector<double>* X, double Y)
	{
		// Inizializations
		this->X->AddRowCopy(X);
		this->Y->Add(Y);
		this->Weights->Add(0);
		this->SamplesTrainedNumber ++;
		if (this->SaveKernelMatrix) {
			this->AddSampleToKernelMatrix(X);
		}	
		int Flops = 0;
		double Epsilon = this->Epsilon;
		bool NewSampleAdded = false;
		int SampleIndex = this->SamplesTrainedNumber-1;

		// CASE 0: Right classified sample
		if (ABS(this->Margin(X,Y))<=Epsilon) {
			this->AddSampleToRemainingSet(SampleIndex);
			NewSampleAdded = true;
			Flops ++;
			return Flops;
		}

		// Find the Margin
		Vector<double>* H = this->Margin(this->X,this->Y);	

		// Main Loop
		while (!NewSampleAdded) {

			// Check Iterations Number
			Flops ++;
			if (Flops > (this->GetSamplesTrainedNumber()+1)*100) {
				cerr << endl << "Learning Error. Infinite Loop." << endl;
				exit(1);
			}
			
			// KKT CONDITION CHECKING - TODO
			//if (!this->VerifyKKTConditions(H)) {
			//	this->ShowDetails(H,SampleIndex);
			//	int x = 0;
			//}

			// Find Beta and Gamma
			Vector<double>* Beta = this->FindBeta(SampleIndex);
			Vector<double>* Gamma = this->FindGamma(Beta,SampleIndex);
					
			// Find Min Variation
			double MinVariation = 0;
			int Flag = -1;
			int MinIndex = -1;		
			FindLearningMinVariation (H, Beta, Gamma, SampleIndex, &MinVariation, &MinIndex, &Flag);

			// Update Weights and Bias		
			this->UpdateWeightsAndBias (&H, Beta, Gamma, SampleIndex, MinVariation);

			// Move the Sample with Min Variaton to the New Set
			switch (Flag) {
				
				// CASE 1: Add the sample to the support set
				case 1:
					this->AddSampleToSupportSet (&H, Beta, Gamma, SampleIndex, MinVariation);
					NewSampleAdded = true;
					break;			
				
				// CASE 2: Add the sample to the error set
				case 2:
					this->AddSampleToErrorSet (SampleIndex, MinVariation);
					NewSampleAdded = true;
					break;			

				// CASE 3: Move Sample from SupportSet to ErrorSet/RemainingSet
				case 3:
					this->MoveSampleFromSupportSetToErrorRemainingSet (MinIndex, MinVariation);
					break;

				// CASE 4: Move Sample from ErrorSet to SupportSet
				case 4:
					this->MoveSampleFromErrorSetToSupportSet (&H, Beta, Gamma, MinIndex, MinVariation);
					break;

				// CASE 5: Move Sample from RemainingSet to SupportSet
				case 5:
					this->MoveSampleFromRemainingSetToSupportSet (&H, Beta, Gamma, MinIndex, MinVariation);
					break;
			}

			// Clear
			delete Beta;
			delete Gamma;
		}

		// Clear
		delete H;

		return Flops;
	}

}
	
#endif
