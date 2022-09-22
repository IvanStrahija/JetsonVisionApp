QT -= gui
QT += core
CONFIG += c++11 console
CONFIG -= app_bundle
DEFINES += QT_DEPRECATED_WARNINGS
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target
DESTDIR = $$system(pwd)
OBJECTS_DIR = $$DESTDIR/Obj
# C++ flags
QMAKE_CXXFLAGS_RELEASE =-03
SOURCES += main.cpp
QMAKE_CFLAGS_ISYSTEM = -I
CUDA_DIR = /usr/local/cuda-10.2
INCLUDEPATH += /usr/local/include/opencv4/
INCLUDEPATH += $$CUDA_DIR/include
INCLUDEPATH += -I/usr/include
INCLUDEPATH += -I/usr/local/cuda-10.2/include
QMAKE_LIBDIR += $$CUDA_DIR/lib64
LIBS += -lcudart -lcuda
LIBS += -L$$PWD/../../../../usr/lib/ -lueye_api
LIBS += -L/usr/local/lib -lopencv_stitching -lopencv_calib3d -
lopencv_core -lopencv_dnn -lopencv_features2d -lopencv_flann -
lopencv_gapi -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc -
lopencv_ml -lopencv_objdetect -lopencv_photo -lopencv_video -
lopencv_videoio -lopencv_cudaimgproc -lopencv_cudafilters -
lopencv_cudaarithm
CUDA_ARCH = sm_72
