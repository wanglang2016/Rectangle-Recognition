//
//  TrainCore.cpp
//  opencv_test
//
//  Created by 王朗 on 2017/6/3.
//  Copyright © 2017年 王朗. All rights reserved.
//

#include "TrainCore.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//extern QTextCodec *textCodec;
#include <vector>
#include <iterator>
using namespace std;
using namespace cv;

class MySVM: public ml::SVM {
public:
    double get_svm_rho() {
        return this->getDecisionFunction(0, svm_alpha, svm_svidx);
    }
    vector<float> svm_alpha;
    vector<float> svm_svidx;
    float svm_rho;
};

void TrainSVM(list <string> positiveSampleList, list<string> negativeSampleList, SVMInfo svmInfo, string resultFileName)
{
//    int numPositiveSample = (int)positiveSampleList.size();
    vector<vector<float> > featureVector;
    vector<int> resultVector;
    
    //cv::HOGDescriptor hog;
    list<string>::iterator it;
    for(it = positiveSampleList.begin(); it != positiveSampleList.end(); it++)
    {
        cv::Mat img;
        //qDebug() << positiveSampleList.at(i).filePath();
        img = cv::imread(it->data());
        std::vector<float> descriptors;
        
        cv::HOGDescriptor hog( cv::Size(svmInfo.windowWidth,svmInfo.windowHeight), cv::Size(svmInfo.blockWidth, svmInfo.blockHeight),
                              cv::Size(svmInfo.overlapStrideWidth, svmInfo.overlapStrideHeight), cv::Size(svmInfo.cellWidth, svmInfo.cellHeight), svmInfo.nbins);
        hog.compute(img, descriptors, cv::Size(svmInfo.overlapStrideWidth, svmInfo.overlapStrideHeight));
        featureVector.push_back(descriptors);
        resultVector.push_back(1);
    }
    
    int numNegativeSample = 0;
    for(it = negativeSampleList.begin(); it != negativeSampleList.end(); it++)
    {
        cv::Mat img;
        //qDebug() << negativeSampleList.at(i);
        img = cv::imread(it->data());
        std::vector<float> descriptors;
        if(img.cols > svmInfo.windowWidth && img.rows > svmInfo.windowHeight)
        {
            cv::HOGDescriptor hog( cv::Size(svmInfo.windowWidth,svmInfo.windowHeight), cv::Size(svmInfo.blockWidth, svmInfo.blockHeight),
                                  cv::Size(svmInfo.overlapStrideWidth, svmInfo.overlapStrideHeight), cv::Size(svmInfo.cellWidth, svmInfo.cellHeight), svmInfo.nbins);
            std::vector<cv::Point> location;
            location.push_back(cv::Point(abs(svmInfo.windowWidth - img.cols)/2,abs(svmInfo.windowHeight- img.rows)/2));
            hog.compute(img, descriptors, cv::Size(svmInfo.overlapStrideWidth, svmInfo.overlapStrideHeight), cv::Size(), location);
        }
        else
        {
            cv::HOGDescriptor hog( cv::Size(svmInfo.windowWidth,svmInfo.windowHeight), cv::Size(svmInfo.blockWidth, svmInfo.blockHeight),
                                  cv::Size(svmInfo.overlapStrideWidth, svmInfo.overlapStrideHeight), cv::Size(svmInfo.cellWidth, svmInfo.cellHeight), svmInfo.nbins);
            hog.compute(img, descriptors, cv::Size(svmInfo.overlapStrideWidth, svmInfo.overlapStrideHeight));
        }
        //qDebug () << textCodec->fromUnicode(negativeSampleList.at(i)).data() ;
        
        //int feature_vec_length = descriptors.size();
        featureVector.push_back(descriptors);
        resultVector.push_back(0);
        numNegativeSample++;
    }
    
    cv::TermCriteria criteria;
    
    criteria = cvTermCriteria(CV_TERMCRIT_EPS,1000,FLT_EPSILON);

    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();
    svm->setType(cv::ml::SVM::Types::C_SVC);
    svm->setKernel(cv::ml::SVM::KernelTypes::LINEAR);
    svm->setTermCriteria(criteria);
    
//    Mat trainData( (int)featureVector.size(), (int)featureVector.at(0).size(), CV_32FC1 );
//    Mat resData( (int)featureVector.size(), (int)featureVector.at(0).size(), CV_32FC1 );
    CvMat *trainData, *resData;
    trainData =  cvCreateMat((int)featureVector.size(), (int)featureVector.at(0).size(), CV_32FC1);
    resData = cvCreateMat((int)featureVector.size(), 1, CV_32SC1);
    
    for(int i = 0; i < (int)featureVector.size(); ++i)
    {
        std::vector<float> descriptors = featureVector.at(i);
        int feature_vec_length = (int)descriptors.size();
        for(int j = 0; j < feature_vec_length; ++j)
        {
            CV_MAT_ELEM(*trainData, float, i, j) = descriptors[j];
        }
        CV_MAT_ELEM(*resData, float, i, 0) = resultVector.at(i);
    }
    Mat td(trainData->rows, trainData->cols, CV_32FC1, trainData->data.fl);
    Mat rd(resData->rows, resData->cols, CV_32SC1, resData->data.fl);
    svm->train(td, cv::ml::SampleTypes::ROW_SAMPLE, rd);
//    svm->save("test.xml");
    
    
    Mat sv = svm->getSupportVectors();
    vector<float> svm_alpha;
    vector<float> svm_svidx;
    
    float rho = svm->getDecisionFunction(0, svm_alpha, svm_svidx);
    vector<float> myDetector;
    for(int i = 0; i < sv.cols; ++i){
        double temp = 0.;
        for(int j = 0; j < svm_alpha.size(); ++j){
            temp += svm_alpha.at(j) * sv.at<float> (j,i);
        }
        myDetector.push_back(-temp);
    }
    myDetector.push_back(rho);
    

//    //covert svm xml to hog xml
//#ifdef	DEBUG_INFO
//    printf("Convert SVM xml to HOG xml\n");
//#endif
    cv::HOGDescriptor hogT( cv::Size(svmInfo.windowWidth,svmInfo.windowHeight), cv::Size(svmInfo.blockWidth, svmInfo.blockHeight),
                           cv::Size(svmInfo.overlapStrideWidth, svmInfo.overlapStrideHeight), cv::Size(svmInfo.cellWidth, svmInfo.cellHeight), svmInfo.nbins);
    hogT.setSVMDetector(myDetector);
//    

////////    qDebug () << hogDetectorFileName;
    hogT.save( "hogDetector.xml" );
////
    
    cvReleaseMat(&trainData);
    cvReleaseMat(&resData);
    return ;
}


