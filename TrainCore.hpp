//
//  TrainCore.hpp
//  opencv_test
//
//  Created by 王朗 on 2017/6/3.
//  Copyright © 2017年 王朗. All rights reserved.
//

#ifndef TRAINCORE_H
#define TRAINCORE_H

//#include <QtWidgets>
//#include <QtCore>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <string>
#include <list>
using namespace std;

struct SVMInfo
{
    int windowWidth;
    int windowHeight;
    int blockWidth;
    int blockHeight;
    int cellWidth;
    int cellHeight;
    int overlapStrideWidth;
    int overlapStrideHeight;
    int nbins;
    SVMInfo(int a, int b, int c, int d, int e, int f, int g, int h, int i) {
        windowWidth = a;
        windowHeight = b;
        blockWidth = c;
        blockHeight = d;
        cellWidth = e;
        cellHeight = f;
        overlapStrideWidth = g;
        overlapStrideHeight = h;
        nbins = i;
    }
};

void TrainSVM(list <string> postiveSampleList, list<string> negativeSampleList, SVMInfo svmInfo, string resultFileName);




#endif /* TrainCore_hpp */
