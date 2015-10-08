#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QTCore>
#include <QDebug>
#include<QFileDialog>
#include<QtCore>

#include <opencv2/opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudafilters.hpp>
#include <opencv2/cudaimgproc.hpp>

#include <opencv2/core/utility.hpp>

using namespace cv;
using namespace cv::cuda;
using namespace std;

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    ~MainWindow();

private:
    Ui::MainWindow *ui;

    void MainWindow::runStereoApp();
    QImage MainWindow::convertOpenCVMatToQtQImage(Mat mat);

    Mat leftSource, rightSource, leftGray, rightGray;
    GpuMat d_leftGray, d_rightGray, leftSobel, rightSobel;

    Mat disp, disp8;
    GpuMat d_disp;

    Ptr<cuda::StereoBM> gpu_bm = createStereoBM(16,9);

    void runGpuStereoAlg();

    /* ---- CPU Stereo Algorithm declarations -------------------------------*/
    enum { STEREO_BM=0,
           STEREO_SGBM=1,
           STEREO_HH=2,
           STEREO_VAR=3,
           STEREO_3WAY=4
         };

    int alg = STEREO_BM;
    int SADWindowSize = 0, numberOfDisparities = 0;
    bool no_display = false;
    float scale = 1.f;

    Ptr<cv::StereoBM> bm = cv::StereoBM::create(16,9);
    Ptr<StereoSGBM> sgbm = StereoSGBM::create(0,16,3);

    void runCpuStereoAlg();
};

#endif // MAINWINDOW_H
