#include <iostream>

#include <stdlib.h>

#include <urg_sensor.h>
#include <urg_utils.h>


using namespace std;

int main()
{
    urg_t urg;
    int ret;
    long *length_data;
    int length_data_size;

    const char connect_device[] = "COM11";
    const long connect_baudrate = 115200;

    // センサに対して接続を行う。
    ret = urg_open(&urg, URG_SERIAL, connect_device, connect_baudrate);
    if (ret < 0)
    {
        // todo: check error code
        printf("Unable to open urg device.");
        getchar();
        return 1;
    }

    // データ受信のための領域を確保する
    length_data = (long *)malloc(sizeof(long) * urg_max_data_size(&urg));
    // todo: check length_data is not NULL

    // センサから距離データを取得する。
    ret = urg_start_measurement(&urg, URG_DISTANCE, 1, 0);
    if (ret < 0)
    {
        // todo: check error code
        return 1;
    }
    length_data_size = urg_get_distance(&urg, length_data, NULL);
    if (length_data_size <= 0)
    {
        // todo: check error code
        return 1;
    }
    // todo: process length_data array
    printf("Size of data: %d  \n", length_data_size);
    for (int i = 0; i < length_data_size; i++)
    {
        printf("Length measurement: %ld \n", length_data[i]);
    }

    // センサとの接続を閉じる。
    urg_close(&urg);


    printf("Yo");

    getchar();
    return 0;
}
