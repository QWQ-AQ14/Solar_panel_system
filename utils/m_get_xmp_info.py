import pyexiv2
import PIL.Image
from PIL.ExifTags import TAGS, GPSTAGS
import os
import exifread
# 该文件获取图片XMP信息包括：经纬度、高度、偏航角、俯仰角、翻滚角、焦距、图像尺寸大小

#将经纬度信息转换为小数点形式
def decimal_coords(coords, ref):
 decimal_degrees = float(coords[0]) + float(coords[1] / 60) + float(coords[2] / 3600)
 if ref == "S" or ref == "W":
     decimal_degrees = -decimal_degrees
 return decimal_degrees

# 将经纬度转换为小数形式
def convert_to_decimal(info):
    gps = [x.replace(' ', '') for x in info[1:-1].split(',')]
    # 度
    if '/' in gps[0]:
        deg = gps[0].split('/')
        if deg[0] == '0' or deg[1] == '0':
            gps_d = 0
        else:
            gps_d = float(deg[0]) / float(deg[1])
    else:
        gps_d = float(gps[0])
    # 分
    if '/' in gps[1]:
        minu = gps[1].split('/')
        if minu[0] == '0' or minu[1] == '0':
            gps_m = 0
        else:
            gps_m = (float(minu[0]) / float(minu[1])) / 60
    else:
        gps_m = float(gps[1]) / 60
    # 秒
    if '/' in gps[2]:
        sec = gps[2].split('/')
        if sec[0] == '0' or sec[1] == '0':
            gps_s = 0
        else:
            gps_s = (float(sec[0]) / float(sec[1])) / 3600
    else:
        gps_s = float(gps[2]) / 3600

    decimal_gps = gps_d + gps_m + gps_s
    # # 如果是南半球或是西半球
    # if gps[3] == 'W' or gps[3] == 'S' or gps[3] == "83" or gps[3] == "87":
    #     return str(decimal_gps * -1)
    # else:
    #     return str(decimal_gps)
    return decimal_gps

def to_float(str):
    # 去掉前面的符号 并转换为浮点数
    str = str[1:len(str)]
    str = float(str)
    return str

def get_xmp_info(img_path):

    # with pyexiv2.Image(img_path) as img:
    #     data = img.read_exif()
    try:
        # DJI XH2
        img = pyexiv2.Image(img_path)
        data = img.read_xmp()
        exif = img.read_exif()
        longitude = data['Xmp.drone-dji.GpsLongitude']
        latitude = data['Xmp.drone-dji.GpsLatitude']

        RelativeAltitude = data['Xmp.drone-dji.RelativeAltitude']
        roll = data['Xmp.drone-dji.FlightRollDegree']
        yaw = data['Xmp.drone-dji.FlightYawDegree']
        pitch = data['Xmp.drone-dji.FlightPitchDegree']

        img_widh = exif['Exif.Photo.PixelXDimension']
        img_height = exif['Exif.Photo.PixelYDimension']

        focal_length = exif['Exif.Photo.FocalLength'].split('/')
        focal_length = float(focal_length[0]) / float(focal_length[1])
        dist = {'Longitude': to_float(longitude), 'Latitude': to_float(latitude), 'Altitude': to_float(RelativeAltitude),
                        'Flight-Roll': float(roll), 'Flight-Pitch': float(pitch), 'Flight-Yaw': float(yaw),
                        'FocalLength': focal_length,'ImageWidth': float(img_widh),'ImageHeight': float(img_height)
                      }
    except:
        # DJI XT2
        try:
            f = open(img_path, 'rb')
            exif_dict = exifread.process_file(f)
        except:
            return
        Altitude = exif_dict['GPS GPSAltitude'].printable[0:-1].replace(" ", "").replace("/", ",").split(",")
        Altitude = float(Altitude[0]) / (float(Altitude[1]) * 10)
        # 获取图像焦距
        focal = exif_dict['EXIF FocalLength'].printable[:].replace("'", "")
        img_wid = float(str(exif_dict['EXIF ExifImageWidth']))
        img_height = float(str(exif_dict['EXIF ExifImageLength']))
        longitude = convert_to_decimal(str(exif_dict['GPS GPSLongitude']))
        latitude = convert_to_decimal(str(exif_dict['GPS GPSLatitude']))
        dist = {'Longitude': longitude, 'Latitude': latitude,
                'Altitude': Altitude,
                'FocalLength': float(focal), 'ImageWidth': img_wid, 'ImageHeight': img_height
                }


    return dist