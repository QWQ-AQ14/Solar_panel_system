import pyexiv2
#将经纬度信息转换为小数点形式
def decimal_coords(coords, ref):
 decimal_degrees = float(coords[0]) + float(coords[1] / 60) + float(coords[2] / 3600)
 if ref == "S" or ref == "W":
     decimal_degrees = -decimal_degrees
 return decimal_degrees

def to_float(str):
    # 去掉前面的符号 并转换为浮点数
    str = str[1:len(str)]
    str = float(str)
    return str

def get_xmp_info(img_path):
    with pyexiv2.Image(img_path) as img:
        data = img.read_exif()
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
    return dist