#-------------------------------------------------
#
# Project created by QtCreator 2015-09-23T20:02:53
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = CudaWidget
TEMPLATE = app


SOURCES += main.cpp\
        mainwindow.cpp \
    videostreamthread.cpp

HEADERS  += mainwindow.h \
    videostreamthread.h

FORMS    += mainwindow.ui

INCLUDEPATH +=  D:/Utvikling/opencv/build/include \
                D:/Utvikling/opencv/sources/modules/core/include \
                D:/Utvikling/opencv/sources/modules/cudaarithm/include \
                D:/Utvikling/opencv/sources/modules/cudabgsegm/include \
                D:/Utvikling/opencv/sources/modules/cudacodec/include \
                D:/Utvikling/opencv/sources/modules/cudafeatures2d/include \
                D:/Utvikling/opencv/sources/modules/modules/cudafilters/include \
                D:/Utvikling/opencv/sources/modules/cudaimgproc/include \
                D:/Utvikling/opencv/sources/modules/cudalegacy/include \
                D:/Utvikling/opencv/sources/modules/cudaobjdetect/include \
                D:/Utvikling/opencv/sources/modules/cudaoptflow/include \
                D:/Utvikling/opencv/sources/modules/cudastereo/include \
                D:/Utvikling/opencv/sources/modules/cudawarping/include \
                D:/Utvikling/opencv/sources/modules/cudev/include

LIBS += -LD:\\Utvikling\\opencv\\mybuild\\lib\\Debug \
    -lopencv_calib3d300d \
    -lopencv_core300d \
    -lopencv_cudaarithm300d \
    -lopencv_cudabgsegm300d \
    -lopencv_cudacodec300d \
    -lopencv_cudafeatures2d300d \
    -lopencv_cudafilters300d \
    -lopencv_cudaimgproc300d \
    -lopencv_cudalegacy300d \
    -lopencv_cudaobjdetect300d \
    -lopencv_cudaoptflow300d \
    -lopencv_cudastereo300d \
    -lopencv_cudawarping300d \
    -lopencv_cudev300d \
    -lopencv_features2d300d \
    -lopencv_flann300d \
    -lopencv_hal300d \
    -lopencv_highgui300d \
    -lopencv_imgcodecs300d \
    -lopencv_imgproc300d \
    -lopencv_ml300d \
    -lopencv_objdetect300d \
    -lopencv_photo300d \
    -lopencv_shape300d \
    -lopencv_stitching300d \
    -lopencv_superres300d \
    -lopencv_ts300d \
    -lopencv_video300d \
    -lopencv_videoio300d \
    -lopencv_videostab300d
