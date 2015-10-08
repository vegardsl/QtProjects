#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    runStereoApp();
}

MainWindow::~MainWindow()
{
    delete ui;
}

void morphOperations(Mat &thresh)
{
    //create structuring element that will be used to "dilate" and "erode" image.
    //the element chosen here is a 3px by 3px rectangle

    Mat erodeElement = getStructuringElement( MORPH_RECT,Size(6,6));
    //dilate with larger element so make sure object is nicely visible
    Mat dilateElement = getStructuringElement( MORPH_ELLIPSE,Size(11,11));

    erode(thresh,thresh,erodeElement);
    erode(thresh,thresh,erodeElement);


    dilate(thresh,thresh,dilateElement);
    dilate(thresh,thresh,dilateElement);
}

void MainWindow::runCpuStereoAlg()
{
    Size img_size = leftSource.size();

    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;

    //bm->setROI1(roi1);
    //bm->setROI2(roi2);
    bm->setPreFilterCap(31);
    bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
    bm->setMinDisparity(0);
    bm->setNumDisparities(numberOfDisparities);
    bm->setTextureThreshold(10);
    bm->setUniquenessRatio(15);
    bm->setSpeckleWindowSize(100);
    bm->setSpeckleRange(32);
    bm->setDisp12MaxDiff(1);

    sgbm->setPreFilterCap(63);
    int sgbmWinSize = SADWindowSize > 0 ? SADWindowSize : 3;
    sgbm->setBlockSize(sgbmWinSize);

    int cn = leftSource.channels();

    sgbm->setP1(8*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setP2(32*cn*sgbmWinSize*sgbmWinSize);
    sgbm->setMinDisparity(0);
    sgbm->setNumDisparities(numberOfDisparities);
    sgbm->setUniquenessRatio(10);
    sgbm->setSpeckleWindowSize(100);
    sgbm->setSpeckleRange(32);
    sgbm->setDisp12MaxDiff(1);
    if(alg==STEREO_HH)
        sgbm->setMode(StereoSGBM::MODE_HH);
    else if(alg==STEREO_SGBM)
        sgbm->setMode(StereoSGBM::MODE_SGBM);
    //else if(alg==STEREO_3WAY)
      //  sgbm->setMode(StereoSGBM::MODE_SGBM_3WAY);

    Mat disp, disp8;
    //Mat img1p, img2p, dispp;
    //copyMakeBorder(img1, img1p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);
    //copyMakeBorder(img2, img2p, 0, 0, numberOfDisparities, 0, IPL_BORDER_REPLICATE);

    int64 t = getTickCount();
    if( alg == STEREO_BM )
        bm->compute(leftGray, rightGray, disp);
    else if( alg == STEREO_SGBM || alg == STEREO_HH || alg == STEREO_3WAY )
        sgbm->compute(leftSource, rightSource, disp);
    t = getTickCount() - t;
    printf("Time elapsed: %fms\n", t*1000/getTickFrequency());

    //disp = dispp.colRange(numberOfDisparities, img1p.cols);
    if( alg != STEREO_VAR )
        disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
    else
        disp.convertTo(disp8, CV_8U);
    if( !no_display )
    {
        namedWindow("left", 1);
        imshow("left", leftSource);
        namedWindow("right", 1);
        imshow("right", rightSource);
        namedWindow("disparity", 0);
        imshow("disparity", disp8);
        printf("press any key to continue...");
        fflush(stdout);
        waitKey();
        printf("\n");
    }
}

void MainWindow::runGpuStereoAlg()
{
    Size img_size = leftSource.size();
    numberOfDisparities = numberOfDisparities > 0 ? numberOfDisparities : ((img_size.width/8) + 15) & -16;

    qDebug() << "Number of disparities: "+QString::number(numberOfDisparities);
    //Ptr<cuda::StereoBeliefPropagation> bm = createStereoBeliefPropagation(64,5,5,3);
    //Ptr<cuda::StereoBeliefPropagation> bm = createStereoBeliefPropagation(64,5,5,CV_32F);

    //gpu_bm->setPreFilterType(CV_STEREO_BM_BASIC);
    gpu_bm->setPreFilterCap(31);
    gpu_bm->setBlockSize(SADWindowSize > 0 ? SADWindowSize : 9);
    gpu_bm->setMinDisparity(-32);
    gpu_bm->setNumDisparities(numberOfDisparities);
    gpu_bm->setTextureThreshold(10);
    gpu_bm->setUniquenessRatio(15);
    gpu_bm->setSpeckleWindowSize(5);
    gpu_bm->setSpeckleRange(15);
    gpu_bm->setDisp12MaxDiff(1);

    Ptr<StereoBeliefPropagation> bp = createStereoBeliefPropagation(numberOfDisparities, 8, 2, CV_16S);
    bp->setMaxDataTerm(25.0);
    bp->setDataWeight(0.1);
    bp->setMaxDiscTerm(15.0);
    bp->setDiscSingleJump(1.0);

    int64 t = getTickCount();
    gpu_bm->compute(d_leftGray, d_rightGray, d_disp);

    d_disp.download(disp);


    Mat gray_filtered;
    Mat maskMat;
    //inRange(disp, Scalar(0, 0, 0), Scalar(150, 150, 150),maskMat);// Create a mask for noise filtering.
    disp.copyTo(gray_filtered, maskMat);    // Filter out false close disparities.
    //imshow("masked disp", gray_filtered);
    //morphOperations(gray_filtered);

    t = getTickCount() - t;
    cout << "Time elapsed: " + to_string(t*1000/getTickFrequency()) << endl;
    //disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
    imshow("disp", disp);
    //imshow("filtered disp", gray_filtered);
    waitKey();
}

void MainWindow::runStereoApp()
{
    QString strFileNameLeft = QFileDialog::getOpenFileName();
    QString strFileNameRight = QFileDialog::getOpenFileName();

    leftSource = imread(strFileNameLeft.toStdString()); // open left image
    rightSource = imread(strFileNameRight.toStdString()); // open right image

    cv::cvtColor(leftSource, leftGray, COLOR_BGR2GRAY);
    cv::cvtColor(rightSource, rightGray, COLOR_BGR2GRAY);

    int type = leftGray.type();
    cout<<to_string(type)<<endl;

    Ptr<cuda::Filter> sobel = createSobelFilter(leftGray.type(),leftGray.type(),1,1,7);
    qDebug() << "filter created";
    d_leftGray.upload(leftGray);
    d_rightGray.upload(rightGray);

    //sobel->apply(d_leftGray,leftSobel);
    //sobel->apply(d_rightGray,rightSobel);

    qDebug() << "filter applied";

    Mat imSobel;
    //leftSobel.download(imSobel);
    //imshow("sobel",imSobel);


    runCpuStereoAlg();
    //runGpuStereoAlg();
    /*
    GpuMat dst, dst1;

    Ptr<cuda::Filter> erodeFilter = cuda::createMorphologyFilter(MORPH_ERODE, d_disp.type(), element);
    erodeFilter->apply(d_disp, dst);

    Ptr<cuda::Filter> dilateFilter = cuda::createMorphologyFilter(MORPH_DILATE, d_disp.type(), element);
    dilateFilter->apply(dst, dst1);
*/
    QImage qimgDisparity = convertOpenCVMatToQtQImage(disp);

    ui->labelDisparity->setPixmap(QPixmap::fromImage(qimgDisparity));

    waitKey();
}

QImage MainWindow::convertOpenCVMatToQtQImage(Mat mat) {
    if(mat.channels() == 1) {                   // if grayscale image
        return QImage((uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_Indexed8);     // declare and return a QImage
    } else if(mat.channels() == 3) {            // if 3 channel color image
        cv::cvtColor(mat, mat, CV_BGR2RGB);     // invert BGR to RGB
        return QImage((uchar*)mat.data, mat.cols, mat.rows, mat.step, QImage::Format_RGB888);       // declare and return a QImage
    } else {
        qDebug() << "in convertOpenCVMatToQtQImage, image was not 1 channel or 3 channel, should never get here";
    }
    return QImage();        // return a blank QImage if the above did not work
}
