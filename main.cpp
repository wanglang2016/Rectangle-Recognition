////
////  main.cpp
////  opencv_test
////
////  Created by 王朗 on 2017/4/20.
////  Copyright © 2017年 王朗. All rights reserved.
////
//


#include <opencv2/ml/ml.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <opencv2/video/video.hpp>
//#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/features2d/features2d.hpp>
//#include<opencv2/legacy/legacy.hpp>
#include <stdio.h>
#include <string.h>
#include <ctype.h>
#include <string>
#include <fstream>
#include <list>
#include <vector>
#include "TrainCore.hpp"

using namespace cv;
using namespace std;

typedef pair<double, double> P;

int main(int argc, char** argv)
{

//    Mat img = imread("/Users/wanglang/Desktop/Test.png");
//    cout << img.size() << endl;
//    HOGDescriptor hog;
//    hog.load("/Users/wanglang/Documents/Language C++/opencv_test/DerivedData/Build/Products/Debug/detector.xml");
////    hog.load("/Users/wanglang/Downloads/detector2.xml");
////    cout << hog.blockSize.width << endl;
//    vector<Rect> location;
//    hog.detectMultiScale(img, location);
//    for( int i = 0; i < location.size(); i++ ) {
//        rectangle(img, location.at(i), cv::Scalar(0, 0, 255));
//    }
//    resize(img, img, cv::Size(600,600));
//    namedWindow("res", 1);
//    imshow("res", img);
//    waitKey();
////////----------------*----------------*--------------*----------*------------------------
    SVMInfo svmInfo(30, 48, 12, 12, 6, 6, 6, 6, 9);
    list<string>positiveList;
    list<string>negativeList;
//    list<int> resultVector;
    positiveList.clear();
    negativeList.clear();
    
    ifstream fin("/Users/wanglang/Desktop/samples/negative/negative.txt", ios::in);
    char s[100];
    while( !fin.eof() ) {
        fin.getline(s, 100);
        negativeList.push_back("/Users/wanglang/Desktop/samples/negative/"+(string)s);
    }
    fin.close();
    ifstream ffin("/Users/wanglang/Desktop/samples/positive/positive.txt", ios::in);
    while( !ffin.eof() ) {
        ffin.getline(s, 100);
        positiveList.push_back("/Users/wanglang/Desktop/samples/positive/"+(string)s);
    }
    fin.close();

    TrainSVM(positiveList, negativeList, svmInfo, string());
    
    return 0;
}

