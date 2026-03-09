# In[] Convert TIFF format to NetCDF format
#  original LAI4g can be downloaded throygh https://zenodo.org/records/8281930

from osgeo import gdal
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import calendar
import pandas as pd
import geopandas as gpd
import salem


def tiff2nc(path):
    data = gdal.Open(path)
    im_width = data.RasterXSize  
    im_height = data.RasterYSize  
    im_bands = data.RasterCount  
    """
    GeoTransform 的含义：
        影像左上角横坐标：im_geotrans[0]，对应经度
        影像左上角纵坐标：im_geotrans[3]，对应纬度

        遥感图像的水平空间分辨率(纬度间隔)：im_geotrans[5]
        遥感图像的垂直空间分辨率(经度间隔)：im_geotrans[1]
        通常水平和垂直分辨率相等

        如果遥感影像方向没有发生旋转，即上北下南，则 im_geotrans[2] 与 im_geotrans[4] 为 0

    计算图像地理坐标：
        若图像中某一点的行数和列数分别为 row 和 column，则该点的地理坐标为：
            经度：xGeo = im_geotrans[0] + col * im_geotrans[1] + row * im_geotrans[2]
            纬度：yGeo = im_geotrans[3] + col * im_geotrans[4] + row * im_geotrans[5]
    """
    im_geotrans = data.GetGeoTransform()  # 获取仿射矩阵，含有 6 个元素的元组

    im_proj = data.GetProjection()  # 获取地理信息

    """
    GetRasterBand(bandNum)，选择要读取的波段数，bandNum 从 1 开始
    ReadAsArray(xoff, yoff, xsize, ysize)，一般就按照下面这么写，偏移量都是 0 ，返回 ndarray 数组
    """
    im_data = data.GetRasterBand(1).ReadAsArray(xoff=0, yoff=0, win_xsize=im_width, win_ysize=im_height)
    # 根据im_proj得到图像的经纬度信息
    im_lon = [im_geotrans[0] + i * im_geotrans[1] for i in range(im_width)]
    im_lat = [im_geotrans[3] + i * im_geotrans[5] for i in range(im_height)]

    im_nc = xr.DataArray(im_data, coords=[im_lat, im_lon], dims=['lat', 'lon'])
    return im_nc


#  1982~1990
filepath='G:/GIMMS_LAI4g/all/GIMMS_LAI4g_AVHRR_MODIS_consolidated_1982_1990/' #文件路径
for y in range(1982, 1991):  # 遍历年
    for m in range(1, 13): 
        for a in range(1,3):
            day_num = calendar.monthrange(y, m)[1]  # 根据年月，获取当月日数
            filename = 'GIMMS_LAI4g_V1.2_'+str(y) + str(m).zfill(2) +str(a).zfill(2)+'.tif'  # 文件名
    #         print(filename)GIMMS_LAI4g(AVHRR)_V1.2_20110101
            day_nc = tiff2nc(filepath+filename)
            
            time = pd.Timestamp(str(y) + str(m).zfill(2) +str(a).zfill(2))
            day_nc = day_nc.expand_dims(time=[time])
            day_nc = day_nc.where(day_nc != 65535, np.nan)
            day_nc = day_nc/1000
            day_nc = day_nc.astype('float32')
            if filename=='GIMMS_LAI4g_V1.2_19820101.tif':
                merged=day_nc
            if filename!='GIMMS_LAI4g_V1.2_19820101.tif':
                merged=xr.concat([merged, day_nc], dim='time')
                

dataset = merged.to_dataset(name="LAI")
dataset.to_netcdf('G:/GIMMS_LAI4g/merged/'+str('dataset1982_1990')+'.nc') 

#  1991~2000

filepath='G:/GIMMS_LAI4g/all/GIMMS_LAI4g_AVHRR_MODIS_consolidated_1991_2000/' #文件路径
for y in range(1991, 2001):  # 遍历年
    for m in range(1, 13): 
        for a in range(1,3):
            day_num = calendar.monthrange(y, m)[1]  # 根据年月，获取当月日数
            filename = 'GIMMS_LAI4g_V1.2_'+str(y) + str(m).zfill(2) +str(a).zfill(2)+'.tif'  # 文件名
    #         print(filename)GIMMS_LAI4g(AVHRR)_V1.2_20110101
            day_nc = tiff2nc(filepath+filename)
            
            time = pd.Timestamp(str(y) + str(m).zfill(2) +str(a).zfill(2))
            day_nc = day_nc.expand_dims(time=[time])
            day_nc = day_nc.where(day_nc != 65535, np.nan)
            day_nc = day_nc/1000
            day_nc = day_nc.astype('float32')
            if filename=='GIMMS_LAI4g_V1.2_19910101.tif':
                merged=day_nc
            if filename!='GIMMS_LAI4g_V1.2_19910101.tif':
                merged=xr.concat([merged, day_nc], dim='time')

dataset = merged.to_dataset(name="LAI")
dataset.to_netcdf('G:/GIMMS_LAI4g/merged/'+str('dataset1991_2000')+'.nc') 

#  2001~2010

filepath='G:/GIMMS_LAI4g/all/GIMMS_LAI4g_AVHRR_MODIS_consolidated_2001_2010/' #文件路径
for y in range(2001, 2011):  # 遍历年
    for m in range(1, 13): 
        for a in range(1,3):
            day_num = calendar.monthrange(y, m)[1]  # 根据年月，获取当月日数
            filename = 'GIMMS_LAI4g_V1.2_'+str(y) + str(m).zfill(2) +str(a).zfill(2)+'.tif'  # 文件名
    #         print(filename)GIMMS_LAI4g(AVHRR)_V1.2_20110101
            day_nc = tiff2nc(filepath+filename)
            
            time = pd.Timestamp(str(y) + str(m).zfill(2) +str(a).zfill(2))
            day_nc = day_nc.expand_dims(time=[time])
            day_nc = day_nc.where(day_nc != 65535, np.nan)
            day_nc = day_nc/1000
            day_nc = day_nc.astype('float32')
            if filename=='GIMMS_LAI4g_V1.2_20010101.tif':
                merged=day_nc
            if filename!='GIMMS_LAI4g_V1.2_20010101.tif':
                merged=xr.concat([merged, day_nc], dim='time')
                
dataset = merged.to_dataset(name="LAI")
dataset.to_netcdf('G:/GIMMS_LAI4g/merged/'+str('dataset2001_2010')+'.nc') 

#   2011~2020

filepath='G:/GIMMS_LAI4g/all/GIMMS_LAI4g_AVHRR_MODIS_consolidated_2011_2020/' #文件路径
for y in range(2011, 2021):  # 遍历年
    for m in range(1, 13): 
        for a in range(1,3):
            day_num = calendar.monthrange(y, m)[1]  # 根据年月，获取当月日数
            filename = 'GIMMS_LAI4g_V1.2_'+str(y) + str(m).zfill(2) +str(a).zfill(2)+'.tif'  # 文件名
    #         print(filename)GIMMS_LAI4g(AVHRR)_V1.2_20110101
            day_nc = tiff2nc(filepath+filename)
            
            time = pd.Timestamp(str(y) + str(m).zfill(2) +str(a).zfill(2))
            day_nc = day_nc.expand_dims(time=[time])
            day_nc = day_nc.where(day_nc != 65535, np.nan)
            day_nc = day_nc/1000
            day_nc = day_nc.astype('float32')
            if filename=='GIMMS_LAI4g_V1.2_20110101.tif':
                merged=day_nc
            if filename!='GIMMS_LAI4g_V1.2_20110101.tif':
                merged=xr.concat([merged, day_nc], dim='time')
                
dataset = merged.to_dataset(name="LAI")
dataset.to_netcdf('G:/GIMMS_LAI4g/merged/'+str('dataset2011_2020')+'.nc') 




# In[] nearest interpolation

import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import calendar
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter 

#  station info (latitude, longitude)
data = pd.read_csv('F:/global_wind/1982-2020merge2/merged_month.csv')
data=data[['STATION', 'LATITUDE', 'LONGITUDE']].drop_duplicates()
sta = data.STATION.values
lat = data.LATITUDE.values
lon = data.LONGITUDE.values

f =  xr.open_dataset('F:/GIMMS_LAI4g/merged/dataset1982_1990.nc')
LAI1 = f.LAI.resample(time='1YE').mean()
lat_grid = f.lat
lon_grid = f.lon
f2 =  xr.open_dataset('F:/GIMMS_LAI4g/merged/dataset1991_2000.nc')
f3 =  xr.open_dataset('F:/GIMMS_LAI4g/merged/dataset2001_2010.nc')
f4 =  xr.open_dataset('F:/GIMMS_LAI4g/merged/dataset2011_2020.nc')
LAI2 = f2.LAI.resample(time='1YE').mean()
LAI3 = f3.LAI.resample(time='1YE').mean()
LAI4 = f4.LAI.resample(time='1YE').mean()

LAI_merged = xr.concat([LAI1, LAI2], dim='time')
LAI_merged = xr.concat([LAI_merged, LAI3], dim='time')
LAI_merged = xr.concat([LAI_merged, LAI4], dim='time')

lat_da = xr.DataArray(lat, coords={'sta': sta}, dims=['sta'])
lon_da = xr.DataArray(lon, coords={'sta': sta}, dims=['sta'])

# nearest interpolation
LAI_interp = LAI_merged.interp(lat=lat_da, lon=lon_da, method='nearest')

# To DataFrame
df = LAI_interp.to_dataframe(name='LAI').reset_index()

df.to_csv('F:/dir.csv',  index=False)
