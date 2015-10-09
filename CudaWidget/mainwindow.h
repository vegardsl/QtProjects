#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTCore>
#include <QDebug>
#include<QFileDialog>
#include<QtCore>
#include <QMetaType>
#include <QtMath>

#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudaimgproc.hpp>

#include "opencv2/cudaoptflow.hpp"
#include "opencv2/cudaarithm.hpp"

#include "stdio.h"
#include "stdlib.h"
#include <cstdint>
#include <iomanip>
#include <iostream>

#include "videostreamthread.h"


using namespace cv;
using namespace cv::cuda;
using namespace std;

#define MAX_NUM_LINES 125

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();


private slots:
    void on_pushButton_clicked();

public slots:

    void prepAndDisplayNewFrame(cv::Mat mat);

private:
    Ui::MainWindow *ui;

    QImage MainWindow::convertOpenCVMatToQtQImage(Mat mat);
    bool MainWindow::storeLine(Point p1, Point p2, int lineNumber);
    void MainWindow::printStoredLines(int numLines);
    int MainWindow::calculateLineIntersect(int numLines);

    VideoStreamThread mVideoStreamThread;

    //vector<double> lines;
    double **lines;
    double **lineIntersections;

    /* ----  Vanishing point calculations  ---- */
    double imageHeight = 480.0;
    double imageWidth = 640.0;
    //double a[2], b[2]; // Size of kernel.
    //double a, b;
    double x_padding = 4;
    double lambda = 25.0; // Incresing lambda reduce the "Fuzzyness" of kernel.
    double sigma = 0.1; // sigma = [0,1)

    int numStoredPoints;
    int sizeAllocated;

    double vanishingPoint[2];
    int *ptr_vanishingPoint;
    int **ptr_pointsInEllipse;

    void splitEllipse(double semiMajorAxis,
                      double semiMinorAxis,
                      Point center,
                      double angle,
                      vector<Vec4i> lines);

    bool pointInsideEllipse(double semiMajorAxis,
                            double semiMinorAxis,
                            Point center,
                            double angle,
                            Point point);

    int storeLineIntersection(int x, int y, double weight, int numStoredPoints, int sizeAllocated);
    int expandLineIntersectionStorage(int sizeAllocated);
    void findVanishingPoint(int numStoredPoints);
    bool pointFitsInCell(int x, int y, int i, int j);
    double calcPointWeightContribution(int x,
                                       int y,
                                       int weight,
                                       int rowNum,
                                       int colNum);
    double sigmoidKernel_1D(int x,
                            int rowNum,
                            char selectedAxis);
};

#endif // MAINWINDOW_H
