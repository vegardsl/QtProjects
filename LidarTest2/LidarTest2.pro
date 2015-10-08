TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt

SOURCES += main.cpp

LIBS += D:\Utvikling\urg_library-1.9.9\vs2010\cpp\Debug\urg_cpp.lib \
        D:\Utvikling\urg_library-1.9.9\vs2010\cpp\Debug\urg.lib \
        D:\Utvikling\urg_library-1.9.9\vs2010\c\Debug\urg.lib \
        -lws2_32

INCLUDEPATH +=  D:/Utvikling/urg_library-1.9.9/include/cpp \
                D:/Utvikling/urg_library-1.9.9/include/c

