
# In[]   Fig 1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pwlf
import xarray as xr


df = pd.read_csv('F:/merged_filtered_common_stations.csv')
df = df.rename(columns={'STATION': 'sta', 'LATITUDE': 'lat', 'LONGITUDE': 'lon'})
data = pd.read_csv('F:/LAI.csv')
data = data.groupby('sta').filter(lambda x: x['LAI'].notna().any())
df_avg = data.groupby(['sta', 'lat', 'lon'])['LAI'].mean().reset_index()
data['DATE'] = pd.to_datetime(data['time'])
data['YEAR'] = data['DATE'].dt.year
df = pd.merge(df, df_avg[['sta', 'lat', 'lon']], on=['sta', 'lat', 'lon'], how='inner')
df['WDSP']= df['WDSP'] * 0.514444
df1 = (
    df
    .groupby(['sta', 'YEAR'], as_index=False)
    .agg({
        'WDSP': 'mean',
        'lat': 'first',
        'lon': 'first'
    })
    .rename(columns={'WDSP': 'WDSP'})
)
stations=df1['sta'].unique()
station_coords = df1[['sta', 'lat', 'lon']].drop_duplicates()
years = np.arange(1982, 2021, 1)
c=[]
chucunqian=[]
chucunhou=[]
pTP=[]
for i in station_coords['sta']:
    b=df1.loc[df1['sta'].isin([i])]
    my_pwlf = pwlf.PiecewiseLinFit(years, b['WDSP'])
    a=[]
    for m in range(1986,2017):
        x0 = np.array([1982, m, 2020])
        ssr=my_pwlf.fit_with_breaks(x0)
        a.append(ssr)
    a=a.index(min(a))+1986
    c.append(a)
    x0 = np.array([1982, a, 2020])
    ssr=my_pwlf.fit_with_breaks(x0)
    slopes = my_pwlf.calc_slopes()
    chucunqian.append(slopes[0])
    chucunhou.append(slopes[1])
    p = my_pwlf.p_values(method='linear')
    pTP.append(p[2])
station_coords['TP'] = c
station_coords['TPqian'] = chucunqian
station_coords['TPhou'] = chucunhou
station_coords['TP_pvalue'] = pTP
station_coords = station_coords[station_coords['TP_pvalue'] < 0.05].reset_index(drop=True)

ds = xr.open_dataset('F:/global_wind/landcover/ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992-v2.0.7cds.nc') 
land1992= ds.lccs_class


ds1 = xr.open_dataset('F:/global_wind/landcover/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc') 
land2020= ds1.lccs_class


station_coords['land2020'] = pd.NA
station_coords['land1992'] = pd.NA


for idx, row in station_coords.iterrows():

    v2020 = land2020.sel(
        lat = row['lat'],
        lon = row['lon'],
        method = 'nearest'
    ).values.item()  
    
    v1992 = land1992.sel(
        lat = row['lat'],
        lon = row['lon'],
        method = 'nearest'
    ).values.item()
    
    station_coords.at[idx, 'land2020'] = v2020
    station_coords.at[idx, 'land1992'] = v1992

station_coords = station_coords[station_coords['land1992'] == station_coords['land2020']]

non_veg = [ 
    0,    # no_data
    190,  # urban
    200,  # bare_areas
    201,  # bare_areas_consolidated
    202,  # bare_areas_unconsolidated
    210,  # water
    220,  # snow_and_ice
]


station_coords = station_coords[~station_coords['land2020'].isin(non_veg)].copy()

mask = station_coords[['TPqian', 'TPhou']].abs().le(0.5).all(axis=1)
station_coords = station_coords[mask]

valid_stas = station_coords['sta'].tolist()

df1 = df1[df1['sta'].isin(valid_stas)]          # :contentReference[oaicite:0]{index=0}

data = data[data['sta'].isin(valid_stas)] 





lai_ann = data.groupby('YEAR')['LAI'].agg(['mean','count','std']).rename(columns={'mean':'LAI_mean','count':'n_LAI','std':'LAI_std'})
wind_ann = df1.groupby('YEAR')['WDSP'].agg(['mean','count','std']).rename(columns={'mean':'WDSP_mean','count':'n_WDSP','std':'WDSP_std'})


z = 1.96
lai_ann['LAI_ci95'] = z * lai_ann['LAI_std'] / np.sqrt(lai_ann['n_LAI'])
wind_ann['WDSP_ci95'] = z * wind_ann['WDSP_std'] / np.sqrt(wind_ann['n_WDSP'])


years = np.arange(1982, 2021, 1)
my_pwlf = pwlf.PiecewiseLinFit(years, wind_ann['WDSP_mean'])
a=[]
for m in range(1986,2017):
    x0 = np.array([1982, m, 2020])
    ssr=my_pwlf.fit_with_breaks(x0)
    a.append(ssr)
a=a.index(min(a))+1986
ssr=my_pwlf.fit_with_breaks(np.array([1982, a, 2020]))
# breaks = my_pwlf.fit(2)
slopes = my_pwlf.calc_slopes()
rsq = my_pwlf.r_squared()
# se = my_pwlf.standard_errors()
xHat = np.linspace(min(years), max(years), num=10000)
yHat = my_pwlf.predict(xHat)
p = my_pwlf.p_values(method='linear')
beta = my_pwlf.beta
# t = my_pwlf.beta / my_pwlf.se



my_pwlf1 = pwlf.PiecewiseLinFit(years, lai_ann['LAI_mean'])

ssr=my_pwlf1.fit_with_breaks(np.array([1982, a, 2020]))
p1 = my_pwlf1.p_values(method='linear')
slopes1 = my_pwlf1.calc_slopes()
beta1 = my_pwlf1.beta
rsq1 = my_pwlf1.r_squared()

xHat1 = np.linspace(min(years), max(years), num=10000)
yHat1 = my_pwlf1.predict(xHat1)



fig1 = plt.figure(figsize=(15, 6))
ax = fig1.add_axes([0.1, 0.2, 0.6, 0.7])
ax.set_ylabel('Wind Speed (m/s)')

ax.plot(years, wind_ann['WDSP_mean'], label='Wind Speed', linestyle='--',
        color='blue', linewidth=2)
ax.plot(xHat, yHat, color='blue', linewidth=2)
ax.fill_between(years,
                wind_ann['WDSP_mean'] - wind_ann['WDSP_ci95'] ,
                wind_ann['WDSP_mean']  + wind_ann['WDSP_ci95'] ,
                color='#c6dbef', alpha=0.3)


ax2 = ax.twinx()
ax2.set_ylabel('LAI')
# ax2.set_ylim(1.1, 1.5)
# ax2.set_yticks(np.arange(1.1, 1.51, 0.05))


ax2.plot(years, lai_ann['LAI_mean'] , label='LAI', linestyle='--',
         color='red', linewidth=2)
ax2.plot(xHat1, yHat1,  color='red', linewidth=2)
ax2.fill_between(years,
                 lai_ann['LAI_mean']  - lai_ann['LAI_ci95'],
                 lai_ann['LAI_mean']  + lai_ann['LAI_ci95'],
                 color='#fcbba1', alpha=0.3)


lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2,
          labels1 + labels2,
          loc='upper center',
          bbox_to_anchor=(0.5, -0.15),
          fancybox=True,
          shadow=True,
          ncol=2)

# plt.title('wind speed and LAI time series')
plt.grid(linestyle='--', linewidth=0.5, alpha=0.7)
ax.text(0.65, 0.81, "Turning point=2009", 
        fontsize=12, color='black', ha='center', va='center', transform=ax.transAxes)
ax.text(0.65, 0.74, "R²=0.85, P<0.001", 
        fontsize=12, color='blue', ha='center', va='center', transform=ax.transAxes)
ax.text(0.65, 0.68, "R²=0.54, P<0.001", 
        fontsize=12, color='red', ha='center', va='center', transform=ax.transAxes)


ax.axvline(x=2009, color='black', linewidth=1, linestyle='--', alpha=0.5)

f2_ax2 = fig1.add_axes([0.177, 0.25, 0.145, 0.08])


x = np.arange(2)
width = 0.35


f2_ax2.bar(x - width/2, slopes*10, width, label='Wind Trend', color='blue')

f2_ax2.bar(x + width/2, slopes1*10, width, label='LAI Trend', color='red')
f2_ax2.set_xticks(x)
f2_ax2.set_xticklabels(['Before TP', 'After TP'])
f2_ax2.spines['top'].set_visible(False)
f2_ax2.spines['right'].set_visible(False)
f2_ax2.tick_params(axis='y', labelsize=8)

f2_ax2.axhline(y=0, color='black', linewidth=1)
f2_ax2.text( 0- width/2, slopes[0]*10 + 0.2, 'P < 0.01', ha='center', va='bottom', color='blue', fontsize=8.5)
f2_ax2.text( 1- width/2, slopes[1]*10 + 0.05, 'P < 0.01', ha='center', va='bottom', color='blue', fontsize=8.5)
f2_ax2.text(0 + width/2, slopes1[0]*10 + 0.1, 'ns', ha='center', va='bottom', color='red', fontsize=8.5)
f2_ax2.text(1 + width/2+0.1, slopes1[1]*10 + 0.05, 'P < 0.01', ha='center', va='bottom', color='red', fontsize=8.5)



# In[]  Extended fig 1a, 
import xarray as xr
import pwlf
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

df = pd.read_csv('F:/merged_filtered_common_stations.csv')
df = df.rename(columns={'STATION': 'sta', 'LATITUDE': 'lat', 'LONGITUDE': 'lon'})


data = pd.read_csv('F:/LAI.csv')
data = data.groupby('sta').filter(lambda x: x['LAI'].notna().any())

df_avg = data.groupby(['sta', 'lat', 'lon'])['LAI'].mean().reset_index()

data['DATE'] = pd.to_datetime(data['time'])
data['YEAR'] = data['DATE'].dt.year

df = pd.merge(df, df_avg[['sta', 'lat', 'lon']], on=['sta', 'lat', 'lon'], how='inner')

df['WDSP']= df['WDSP'] * 0.514444
df1 = (
    df
    .groupby(['sta', 'YEAR'], as_index=False)
    .agg({
        'WDSP': 'mean',
        'lat': 'first',
        'lon': 'first'
    })
    .rename(columns={'WDSP': 'WDSP'})
)

stations=df1['sta'].unique()
station_coords = df1[['sta', 'lat', 'lon']].drop_duplicates()


years = np.arange(1982, 2021, 1)
c=[]
chucunqian=[]
chucunhou=[]
pTP=[]
for i in station_coords['sta']:
    b=df1.loc[df1['sta'].isin([i])]
    my_pwlf = pwlf.PiecewiseLinFit(years, b['WDSP'])
    a=[]
    for m in range(1986,2017):
        x0 = np.array([1982, m, 2020])
        ssr=my_pwlf.fit_with_breaks(x0)
        a.append(ssr)
    a=a.index(min(a))+1986
    c.append(a)
    x0 = np.array([1982, a, 2020])
    ssr=my_pwlf.fit_with_breaks(x0)
    slopes = my_pwlf.calc_slopes()
    chucunqian.append(slopes[0])
    chucunhou.append(slopes[1])
    p = my_pwlf.p_values(method='linear')
    pTP.append(p[2])


station_coords['TP'] = c
station_coords['TPqian'] = chucunqian
station_coords['TPhou'] = chucunhou
station_coords['TP_pvalue'] = pTP
station_coords = station_coords[station_coords['TP_pvalue'] < 0.05].reset_index(drop=True)


ds = xr.open_dataset('F:/global_wind/landcover/ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992-v2.0.7cds.nc') 
land1992= ds.lccs_class


ds1 = xr.open_dataset('F:/global_wind/landcover/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc') 
land2020= ds1.lccs_class


station_coords['land2020'] = pd.NA
station_coords['land1992'] = pd.NA


for idx, row in station_coords.iterrows():
    
    v2020 = land2020.sel(
        lat = row['lat'],
        lon = row['lon'],
        method = 'nearest'
    ).values.item()  
    
    v1992 = land1992.sel(
        lat = row['lat'],
        lon = row['lon'],
        method = 'nearest'
    ).values.item()
    
    station_coords.at[idx, 'land2020'] = v2020
    station_coords.at[idx, 'land1992'] = v1992

station_coords = station_coords[station_coords['land1992'] == station_coords['land2020']]
non_veg = [ 
    0,    # no_data
    190,  # urban
    200,  # bare_areas
    201,  # bare_areas_consolidated
    202,  # bare_areas_unconsolidated
    210,  # water
    220,  # snow_and_ice
]


station_coords = station_coords[~station_coords['land2020'].isin(non_veg)].copy()
mask = station_coords[['TPqian', 'TPhou']].abs().le(0.5).all(axis=1)
station_coords = station_coords[mask]

valid_stas = station_coords['sta'].tolist()

df1 = df1[df1['sta'].isin(valid_stas)]        

data = data[data['sta'].isin(valid_stas)] 

station_coords['lat_bin'] = station_coords['lat'].round(0)  


grouped = (
    station_coords
    .groupby('lat_bin')['TP']
    .agg(['mean', 'std', 'count'])
    .reset_index()
)




grouped['sem'] = grouped['std'] / np.sqrt(grouped['count']) 
grouped['ci95'] = 1.96 * grouped['sem']





norm = mpl.colors.Normalize(vmin=1986,vmax=2016)


fig, ax = plt.subplots(figsize=(12,6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
ax.add_feature(cfeature.OCEAN, facecolor='white')

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False; gl.right_labels = False
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
gl.ylocator = mticker.FixedLocator(np.arange(-90,91,30))
gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':16}; gl.ylabel_style = {'size':16}


sc = ax.scatter(station_coords['lon'], station_coords['lat'], c=station_coords['TP'],
                norm=norm, cmap='Purples', alpha=0.7,s=30,
                transform=ccrs.PlateCarree())

cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.1, fraction=0.06)
cbar.ax.text(1.02, 0.5, 'Wind Speed Turning Point',
             transform=cbar.ax.transAxes, va='center', ha='left', fontsize=16)
cbar.ax.tick_params(labelsize=14)


divider = make_axes_locatable(ax)
ax2 = divider.append_axes(
    'right', 
    size='18%', 
    pad=0.5, 
    sharey=ax,
    axes_class=plt.Axes      
)

ax2.plot(grouped['mean'], grouped['lat_bin'],
         color='black', linewidth=3, label='Mean wind speed')
ax2.fill_betweenx(
    grouped['lat_bin'],
    grouped['mean'] - grouped['ci95'],
    grouped['mean'] + grouped['ci95'],
    color='gray',    
    alpha=0.5,
    label='95% CI'
)

ax2.set_ylim(ax.get_ylim())

ax2.set_xticks([ 1980, 2000, 2020])
lat_ticks = np.array([0,20,40,60,80,-20,-40,-60])
lat_labels = [f"{abs(lat)}°{'N' if lat>0 else 'S'}" if lat!=0 else '0°'
              for lat in lat_ticks]
ax2.set_yticks(lat_ticks); ax2.set_yticklabels(lat_labels)


ax2.tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelleft=False)  

plt.tight_layout()

# In[]    Extended data fig 1b, first running 1a
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

grouped = (
    station_coords
    .groupby('lat_bin')['TPqian']
    .agg(['mean', 'std', 'count'])
    .reset_index()
)




grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])  
grouped['ci95'] = 1.96 * grouped['sem']


kde1 = gaussian_kde(station_coords['TPqian'])
x_min = min(station_coords['TPqian'])
x_max = max(station_coords['TPqian'])
x = np.linspace(x_min, x_max, 1000)
kde_values = kde1(x)



norm = TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2)


fig, ax = plt.subplots(figsize=(12,6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
ax.add_feature(cfeature.OCEAN, facecolor='white')

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False; gl.right_labels = False
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
gl.ylocator = mticker.FixedLocator(np.arange(-90,91,30))
gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':16}; gl.ylabel_style = {'size':16}


sc = ax.scatter(station_coords['lon'], station_coords['lat'], c=station_coords['TPqian'],
                norm=norm, cmap="RdYlBu_r", alpha=0.7,s=30,
                transform=ccrs.PlateCarree())

cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.1, fraction=0.06)
cbar.ax.text(1.02, 0.5, 'Wind Speed Trend',
             transform=cbar.ax.transAxes, va='center', ha='left', fontsize=16)
cbar.ax.tick_params(labelsize=16)


divider = make_axes_locatable(ax)
ax2 = divider.append_axes(
    'right', 
    size='18%', 
    pad=0.6, 
    sharey=ax,
    axes_class=plt.Axes     
)

ax2.plot(grouped['mean'], grouped['lat_bin'],
         color='black', linewidth=3, label='Mean wind speed')
ax2.fill_betweenx(
    grouped['lat_bin'],
    grouped['mean'] - grouped['ci95'],
    grouped['mean'] + grouped['ci95'],
    color='gray',    # 设置填充为灰色
    alpha=0.5,
    label='95% CI'
)

ax2.set_ylim(ax.get_ylim())

ax2.set_xticks([ -0.25, 0, 0.25])
lat_ticks = np.array([0,20,40,60,80,-20,-40,-60])
lat_labels = [f"{abs(lat)}°{'N' if lat>0 else 'S'}" if lat!=0 else '0°'
              for lat in lat_ticks]
ax2.set_yticks(lat_ticks); ax2.set_yticklabels(lat_labels)

import matplotlib.ticker as mticker
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

ax2.tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelleft=False)  
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)


ax_ins = inset_axes(
    ax,
    width="100%",    
    height="95%",
    bbox_to_anchor=(0.0, 0.08, 0.2, 0.22), 
    bbox_transform=ax.transAxes,          
    loc='lower left',                     
    borderpad=2.5                            
)



ax_ins.set_title("Trend density", fontsize=16)
ax_ins.plot(x, kde_values, label='Trend density', color='black', linewidth=1.5)

ax_ins.axvline(x=0, color='black', linewidth=0.5, linestyle='--', alpha=0.7)





# In[]    Extended data fig 1c, first running 1a
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


station_coords = station_coords[station_coords['TPhou'] >= -1].reset_index(drop=True)
grouped = (
    station_coords
    .groupby('lat_bin')['TPhou']
    .agg(['mean', 'std', 'count'])
    .reset_index()
)




grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])  
grouped['ci95'] = 1.96 * grouped['sem']


kde1 = gaussian_kde(station_coords['TPhou'])
x_min = min(station_coords['TPhou'])
x_max = max(station_coords['TPhou'])
x = np.linspace(x_min, x_max, 1000)
kde_values = kde1(x)



norm = TwoSlopeNorm(vmin=-0.2, vcenter=0, vmax=0.2)


fig, ax = plt.subplots(figsize=(12,6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
ax.add_feature(cfeature.OCEAN, facecolor='white')

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False; gl.right_labels = False
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
gl.ylocator = mticker.FixedLocator(np.arange(-90,91,30))
gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':16}; gl.ylabel_style = {'size':16}


sc = ax.scatter(station_coords['lon'], station_coords['lat'], c=station_coords['TPhou'],
                norm=norm, cmap="RdYlBu_r", alpha=0.7,s=30,
                transform=ccrs.PlateCarree())


cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.1, fraction=0.06)
cbar.ax.text(1.02, 0.5, 'Wind Speed Trend',
             transform=cbar.ax.transAxes, va='center', ha='left', fontsize=16)
cbar.ax.tick_params(labelsize=16)


divider = make_axes_locatable(ax)
ax2 = divider.append_axes(
    'right', 
    size='18%', 
    pad=0.6, 
    sharey=ax,
    axes_class=plt.Axes     
)

ax2.plot(grouped['mean'], grouped['lat_bin'],
         color='black', linewidth=3, label='Mean wind speed')
ax2.fill_betweenx(
    grouped['lat_bin'],
    grouped['mean'] - grouped['ci95'],
    grouped['mean'] + grouped['ci95'],
    color='gray',   
    alpha=0.5,
    label='95% CI'
)

ax2.set_ylim(ax.get_ylim())

ax2.set_xticks([ -0.25, 0, 0.25])
lat_ticks = np.array([0,20,40,60,80,-20,-40,-60])
lat_labels = [f"{abs(lat)}°{'N' if lat>0 else 'S'}" if lat!=0 else '0°'
              for lat in lat_ticks]
ax2.set_yticks(lat_ticks); ax2.set_yticklabels(lat_labels)
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)
import matplotlib.ticker as mticker
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

ax2.tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelleft=False) 



ax_ins = inset_axes(
    ax,
    width="100%",    
    height="95%",
    bbox_to_anchor=(0.0, 0.08, 0.2, 0.22),  
    bbox_transform=ax.transAxes,           
    loc='lower left',                      
    borderpad=2.5                           
)



ax_ins.set_title("Trend density", fontsize=16)
ax_ins.plot(x, kde_values, label='Trend density', color='black', linewidth=1.5)

ax_ins.axvline(x=0, color='black', linewidth=0.5, linestyle='--', alpha=0.7)



# In[]    Extended data fig 1d 
import xarray as xr
import pwlf
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# 读取数据
df = pd.read_csv('F:/merged_filtered_common_stations.csv')
df = df.rename(columns={'STATION': 'sta', 'LATITUDE': 'lat', 'LONGITUDE': 'lon'})


data = pd.read_csv('F:/LAI.csv')
data = data.groupby('sta').filter(lambda x: x['LAI'].notna().any())

df_avg = data.groupby(['sta', 'lat', 'lon'])['LAI'].mean().reset_index()

data['DATE'] = pd.to_datetime(data['time'])
data['YEAR'] = data['DATE'].dt.year

df = pd.merge(df, df_avg[['sta', 'lat', 'lon']], on=['sta', 'lat', 'lon'], how='inner')

df['WDSP']= df['WDSP'] * 0.514444
df1 = (
    df
    .groupby(['sta', 'YEAR'], as_index=False)
    .agg({
        'WDSP': 'mean',
        'lat': 'first',
        'lon': 'first'
    })
    .rename(columns={'WDSP': 'WDSP'})
)

stations=df1['sta'].unique()
station_coords = df1[['sta', 'lat', 'lon']].drop_duplicates()


years = np.arange(1982, 2021, 1)
c=[]
chucunqian=[]
chucunhou=[]
pTP=[]
for i in station_coords['sta']:
    b=df1.loc[df1['sta'].isin([i])]
    my_pwlf = pwlf.PiecewiseLinFit(years, b['WDSP'])
    a=[]
    for m in range(1986,2017):
        x0 = np.array([1982, m, 2020])
        ssr=my_pwlf.fit_with_breaks(x0)
        a.append(ssr)
    a=a.index(min(a))+1986
    c.append(a)
    x0 = np.array([1982, a, 2020])
    ssr=my_pwlf.fit_with_breaks(x0)
    slopes = my_pwlf.calc_slopes()
    chucunqian.append(slopes[0])
    chucunhou.append(slopes[1])
    p = my_pwlf.p_values(method='linear')
    pTP.append(p[2])


station_coords['TP'] = c
station_coords['TPqian'] = chucunqian
station_coords['TPhou'] = chucunhou
station_coords['TP_pvalue'] = pTP
station_coords = station_coords[station_coords['TP_pvalue'] < 0.05].reset_index(drop=True)
ds = xr.open_dataset('F:/global_wind/landcover/ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992-v2.0.7cds.nc') 
land1992= ds.lccs_class


ds1 = xr.open_dataset('F:/global_wind/landcover/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc') 
land2020= ds1.lccs_class


station_coords['land2020'] = pd.NA
station_coords['land1992'] = pd.NA


for idx, row in station_coords.iterrows():

    v2020 = land2020.sel(
        lat = row['lat'],
        lon = row['lon'],
        method = 'nearest'
    ).values.item()  
    
    v1992 = land1992.sel(
        lat = row['lat'],
        lon = row['lon'],
        method = 'nearest'
    ).values.item()
    
    station_coords.at[idx, 'land2020'] = v2020
    station_coords.at[idx, 'land1992'] = v1992

station_coords = station_coords[station_coords['land1992'] == station_coords['land2020']]
non_veg = [ 
    0,    # no_data
    190,  # urban
    200,  # bare_areas
    201,  # bare_areas_consolidated
    202,  # bare_areas_unconsolidated
    210,  # water
    220,  # snow_and_ice
]


station_coords = station_coords[~station_coords['land2020'].isin(non_veg)].copy()
mask = station_coords[['TPqian', 'TPhou']].abs().le(0.5).all(axis=1)
station_coords = station_coords[mask]

valid_stas = station_coords['sta'].tolist()

df1 = df1[df1['sta'].isin(valid_stas)]          # :contentReference[oaicite:0]{index=0}

data = data[data['sta'].isin(valid_stas)] 



chucunqian1=[]
chucunhou1=[]
pTP_LAI=[]

for i in station_coords['sta']:
    d=data.loc[data['sta'].isin([i])]
    TP=station_coords.loc[station_coords['sta'].isin([i])]['TP'].values
    TP = TP[0] 
    my_pwlf1 = pwlf.PiecewiseLinFit(years, d['LAI'])
    x0 = np.array([1982, TP, 2020])
    ssr=my_pwlf1.fit_with_breaks(x0)
    slopes = my_pwlf1.calc_slopes()
    chucunqian1.append(slopes[0])
    chucunhou1.append(slopes[1])
    p = my_pwlf1.p_values(method='linear')
    pTP_LAI.append(p[2])

station_coords['TPqian_LAI'] = chucunqian1
station_coords['TPhou_LAI'] = chucunhou1
station_coords['TP_pvalue_LAI'] = pTP_LAI


# ————————————————————————————————————————extende data table 2_______________

mask = (station_coords['TPhou'] > -0.05) & (station_coords['TPhou'] <= -0.02)
mask = (station_coords['TPhou'] <-0.05) 
count_in_range = mask.sum()
total_count = 772
proportion = count_in_range / total_count
print(proportion)
print(count_in_range)


mask = (station_coords['TPhou_LAI'] > -0.05) & (station_coords['TPhou_LAI'] <= -0.02)
mask = (station_coords['TPhou_LAI'] <=-0.05) 
count_in_range = mask.sum()
total_count = 772
proportion = count_in_range / total_count
print(proportion)
print(count_in_range)








station_coords['lat_bin'] = station_coords['lat'].round(0)  


grouped = (
    station_coords
    .groupby('lat_bin')['TPqian_LAI']
    .agg(['mean', 'std', 'count'])
    .reset_index()
)



grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])  
grouped['ci95'] = 1.96 * grouped['sem']




kde1 = gaussian_kde(station_coords['TPqian_LAI'])
x_min = min(station_coords['TPqian_LAI'])
x_max = max(station_coords['TPqian_LAI'])
x = np.linspace(x_min, x_max, 1000)
kde_values = kde1(x)



norm = TwoSlopeNorm(vmin=-0.06, vcenter=0, vmax=0.06)


fig, ax = plt.subplots(figsize=(12,6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
ax.add_feature(cfeature.OCEAN, facecolor='white')

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False; gl.right_labels = False
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
gl.ylocator = mticker.FixedLocator(np.arange(-90,91,30))
gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':16}; gl.ylabel_style = {'size':16}


sc = ax.scatter(station_coords['lon'], station_coords['lat'], c=station_coords['TPqian_LAI'],
                norm=norm, cmap="BrBG", alpha=0.7,s=30,
                transform=ccrs.PlateCarree())

cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.1, fraction=0.06)
cbar.ax.text(1.02, 0.5, 'LAI Trend ',
             transform=cbar.ax.transAxes, va='center', ha='left', fontsize=16)
cbar.ax.tick_params(labelsize=16)


divider = make_axes_locatable(ax)
ax2 = divider.append_axes(
    'right', 
    size='18%', 
    pad=0.6, 
    sharey=ax,
    axes_class=plt.Axes      
)

ax2.plot(grouped['mean'], grouped['lat_bin'],
         color='black', linewidth=3, label='')
ax2.fill_betweenx(
    grouped['lat_bin'],
    grouped['mean'] - grouped['ci95'],
    grouped['mean'] + grouped['ci95'],
    color='gray',    
    alpha=0.5,
    label='95% CI'
)

ax2.set_ylim(ax.get_ylim())
# ax2.set_xlim(0, 7)
ax2.set_xticks([ -0.1, 0, 0.06])
lat_ticks = np.array([0,20,40,60,80,-20,-40,-60])
lat_labels = [f"{abs(lat)}°{'N' if lat>0 else 'S'}" if lat!=0 else '0°'
              for lat in lat_ticks]
ax2.set_yticks(lat_ticks); ax2.set_yticklabels(lat_labels)

import matplotlib.ticker as mticker
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

ax2.tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelleft=False)  
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)


ax_ins = inset_axes(
    ax,
    width="100%",    
    height="95%",
    bbox_to_anchor=(0.0, 0.08, 0.2, 0.22),  
    bbox_transform=ax.transAxes,           
    loc='lower left',                     
    borderpad=2.5                            
)


ax_ins.set_title("Trend density", fontsize=16)
ax_ins.plot(x, kde_values, label='Trend density', color='black', linewidth=1.5)
# ax_ins.set_xlim(-0.2, 0.2)
ax_ins.axvline(x=0, color='black', linewidth=0.5, linestyle='--', alpha=0.7)






# In[]    Extended data fig 1e ,first running 1d

grouped = (
    station_coords
    .groupby('lat_bin')['TPhou_LAI']
    .agg(['mean', 'std', 'count'])
    .reset_index()
)



grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])  
grouped['ci95'] = 1.96 * grouped['sem']




kde1 = gaussian_kde(station_coords['TPhou_LAI'])
x_min = min(station_coords['TPhou_LAI'])
x_max = max(station_coords['TPhou_LAI'])
x = np.linspace(x_min, x_max, 1000)
kde_values = kde1(x)



norm = TwoSlopeNorm(vmin=-0.06, vcenter=0, vmax=0.06)


fig, ax = plt.subplots(figsize=(12,6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
ax.add_feature(cfeature.OCEAN, facecolor='white')

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False; gl.right_labels = False
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
gl.ylocator = mticker.FixedLocator(np.arange(-90,91,30))
gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':16}; gl.ylabel_style = {'size':16}


sc = ax.scatter(station_coords['lon'], station_coords['lat'], c=station_coords['TPhou_LAI'],
                norm=norm, cmap="BrBG", alpha=0.7,s=30,
                transform=ccrs.PlateCarree())

cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.1, fraction=0.06)
cbar.ax.text(1.02, 0.5, 'LAI Trend',
             transform=cbar.ax.transAxes, va='center', ha='left', fontsize=16)
cbar.ax.tick_params(labelsize=16)


divider = make_axes_locatable(ax)
ax2 = divider.append_axes(
    'right', 
    size='18%', 
    pad=0.6, 
    sharey=ax,
    axes_class=plt.Axes      
)

ax2.plot(grouped['mean'], grouped['lat_bin'],
         color='black', linewidth=3, label='')
ax2.fill_betweenx(
    grouped['lat_bin'],
    grouped['mean'] - grouped['ci95'],
    grouped['mean'] + grouped['ci95'],
    color='gray',   
    alpha=0.5,
    label='95% CI'
)

ax2.set_ylim(ax.get_ylim())

ax2.set_xticks([ -0.1, 0, 0.06])
lat_ticks = np.array([0,20,40,60,80,-20,-40,-60])
lat_labels = [f"{abs(lat)}°{'N' if lat>0 else 'S'}" if lat!=0 else '0°'
              for lat in lat_ticks]
ax2.set_yticks(lat_ticks); ax2.set_yticklabels(lat_labels)

import matplotlib.ticker as mticker
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

ax2.tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelleft=False)  
ax2.axvline(x=0, color='black', linestyle='--', linewidth=1)


ax_ins = inset_axes(
    ax,
    width="100%",    
    height="95%",
    bbox_to_anchor=(0.0, 0.08, 0.2, 0.22),  
    bbox_transform=ax.transAxes,          
    loc='lower left',                     
    borderpad=2.5                          
)



ax_ins.set_title("Trend density", fontsize=16)
ax_ins.plot(x, kde_values, label='Trend density', color='black', linewidth=1.5)
# ax_ins.set_xlim(-0.2, 0.2)
ax_ins.axvline(x=0, color='black', linewidth=0.5, linestyle='--', alpha=0.7)



# In[] Fig 2

import xarray as xr
import pwlf
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio
# from matplotlib.patches import Patch 



df = pd.read_csv('F:/merged_filtered_common_stations.csv')
df = df.rename(columns={'STATION': 'sta', 'LATITUDE': 'lat', 'LONGITUDE': 'lon'})


data = pd.read_csv('F:/LAI.csv')
data = data.groupby('sta').filter(lambda x: x['LAI'].notna().any())

df_avg = data.groupby(['sta', 'lat', 'lon'])['LAI'].mean().reset_index()

data['DATE'] = pd.to_datetime(data['time'])
data['YEAR'] = data['DATE'].dt.year

df = pd.merge(df, df_avg[['sta', 'lat', 'lon']], on=['sta', 'lat', 'lon'], how='inner')

df['WDSP']= df['WDSP'] * 0.514444
df1 = (
    df
    .groupby(['sta', 'YEAR'], as_index=False)
    .agg({
        'WDSP': 'mean',
        'lat': 'first',
        'lon': 'first'
    })
    .rename(columns={'WDSP': 'WDSP'})
)

stations=df1['sta'].unique()
station_coords = df1[['sta', 'lat', 'lon']].drop_duplicates()


years = np.arange(1982, 2021, 1)
c=[]
chucunqian=[]
chucunhou=[]
pTP=[]
for i in station_coords['sta']:
    b=df1.loc[df1['sta'].isin([i])]
    my_pwlf = pwlf.PiecewiseLinFit(years, b['WDSP'])
    a=[]
    for m in range(1986,2017):
        x0 = np.array([1982, m, 2020])
        ssr=my_pwlf.fit_with_breaks(x0)
        a.append(ssr)
    a=a.index(min(a))+1986
    c.append(a)
    x0 = np.array([1982, a, 2020])
    ssr=my_pwlf.fit_with_breaks(x0)
    slopes = my_pwlf.calc_slopes()
    chucunqian.append(slopes[0])
    chucunhou.append(slopes[1])
    p = my_pwlf.p_values(method='linear')
    pTP.append(p[2])


station_coords['TP'] = c
station_coords['TPqian'] = chucunqian
station_coords['TPhou'] = chucunhou
station_coords['TP_pvalue'] = pTP
station_coords = station_coords[station_coords['TP_pvalue'] < 0.05].reset_index(drop=True)
ds = xr.open_dataset('F:/global_wind/landcover/ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992-v2.0.7cds.nc') 
land1992= ds.lccs_class


ds1 = xr.open_dataset('F:/global_wind/landcover/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc') 
land2020= ds1.lccs_class


station_coords['land2020'] = pd.NA
station_coords['land1992'] = pd.NA


for idx, row in station_coords.iterrows():

    v2020 = land2020.sel(
        lat = row['lat'],
        lon = row['lon'],
        method = 'nearest'
    ).values.item()  
    
    v1992 = land1992.sel(
        lat = row['lat'],
        lon = row['lon'],
        method = 'nearest'
    ).values.item()
    
    station_coords.at[idx, 'land2020'] = v2020
    station_coords.at[idx, 'land1992'] = v1992

station_coords = station_coords[station_coords['land1992'] == station_coords['land2020']]
non_veg = [ 
    0,    # no_data
    190,  # urban
    200,  # bare_areas
    201,  # bare_areas_consolidated
    202,  # bare_areas_unconsolidated
    210,  # water
    220,  # snow_and_ice
]


station_coords = station_coords[~station_coords['land2020'].isin(non_veg)].copy()

mask = station_coords[['TPqian', 'TPhou']].abs().le(0.5).all(axis=1)
station_coords = station_coords[mask]

valid_stas = station_coords['sta'].tolist()

df1 = df1[df1['sta'].isin(valid_stas)]         

data = data[data['sta'].isin(valid_stas)] 


chucunqian1=[]
chucunhou1=[]
pTP_LAI=[]

for i in station_coords['sta']:
    d=data.loc[data['sta'].isin([i])]
    TP=station_coords.loc[station_coords['sta'].isin([i])]['TP'].values
    TP = TP[0] 
    my_pwlf1 = pwlf.PiecewiseLinFit(years, d['LAI'])
    x0 = np.array([1982, TP, 2020])
    ssr=my_pwlf1.fit_with_breaks(x0)
    slopes = my_pwlf1.calc_slopes()
    chucunqian1.append(slopes[0])
    chucunhou1.append(slopes[1])
    p = my_pwlf1.p_values(method='linear')
    pTP_LAI.append(p[2])

station_coords['TPqian_LAI'] = chucunqian1
station_coords['TPhou_LAI'] = chucunhou1
station_coords['TP_pvalue_LAI'] = pTP_LAI


station_coords['lat_bin'] = station_coords['lat'].round(0)  


with rasterio.open("F:/global_wind/climate classification/1991_2020/koppen_geiger_0p1.tif") as src:

    coords = [(x, y) for x, y in zip(station_coords['lon'], station_coords['lat'])]

    samples = list(src.sample(coords))
   
    values = [s[0] for s in samples]
    

station_coords['climate_zone'] = values




bins = [0, 3, 7, 16, 28, 30]
zone_labels = ['0-3', '4-7', '8-16', '17-28', '29-30']
zone_names  = ['Tropical', 'Arid', 'Temperate', 'Cold', 'Polar']
cmap = plt.get_cmap('tab10')
colors = {label: cmap(i) for i, label in enumerate(zone_labels)}






# In[] first running fig 2
# ——————————————————————————————————————————————————————————————————————————————

station_coords['zone_class'] = pd.cut(
    station_coords['climate_zone'],
    bins=bins,
    labels=zone_labels,
    right=True,
    include_lowest=True
)
import seaborn as sns

mask = station_coords[['TPqian', 'TPhou']].abs().le(0.5).all(axis=1)
station_coords = station_coords[mask]

df_long = station_coords.melt(
    id_vars=['zone_class'],
    value_vars=['TPqian', 'TPhou'],
    var_name='Variable',
    value_name='Value'
)

zone_names = ['Tropical', 'Arid', 'Temperate', 'Cold', 'Polar']

fig, ax = plt.subplots()


ax = sns.boxplot(
    data=df_long,
    x='zone_class',
    y='Value',
    hue='Variable',
    # notch=True
)
from statannotations.Annotator import Annotator

# 1. Build a list of pairwise comparisons for each zone
zones = df_long['zone_class'].unique()
pairs = [((zone, 'TPqian'), (zone, 'TPhou')) for zone in zones]

# 2. Initialize the annotator with your Axes, data, and pairing info
annotator = Annotator(
    ax,
    pairs,
    data=df_long,
    x='zone_class',
    y='Value',
    hue='Variable'
)

# 3. Choose a test and formatting
annotator.configure(
    test='Wilcoxon',              # independent t-test (Welch’s)
    text_format='star',             # show *, **, *** instead of raw p-values
    loc='inside',                   # place brackets inside the plot
    comparisons_correction='bonferroni'  # adjust for multiple zones
)

# 4. Compute and draw the annotations
annotator.apply_and_annotate()

ticks = range(len(zone_names))

ax.set_xticks(ticks)
ax.set_xticklabels(zone_names, rotation=0, ha='center')
leg = ax.get_legend()
leg.set_title('')  
leg.get_frame().set_linewidth(0)  
handles, labels = ax.get_legend_handles_labels()
new_labels = []
for lab in labels:
    if lab == 'TPqian':
        new_labels.append('pre-TP')
    elif lab == 'TPhou':
        new_labels.append('post-TP')
    else:
        new_labels.append(lab)

ax.legend(handles, new_labels, title='', frameon=False, loc='best')



ax.set_xlabel('Climate Zone')
ax.set_ylabel('Wind Trend')

# LAI 
# ——————————————————————————————————————————————————————————————————————————————


import seaborn as sns
station_coords['zone_class'] = pd.cut(
    station_coords['climate_zone'],
    bins=bins,
    labels=zone_labels,
    right=True,
    include_lowest=True
)

mask = station_coords[['TPqian', 'TPhou']].abs().le(0.5).all(axis=1)
station_coords = station_coords[mask]

df_long = station_coords.melt(
    id_vars=['zone_class'],
    value_vars=['TPqian_LAI', 'TPhou_LAI'],
    var_name='Variable',
    value_name='Value'
)

zone_names = ['Tropical', 'Arid', 'Temperate', 'Cold', 'Polar']

fig, ax = plt.subplots()


ax = sns.boxplot(
    data=df_long,
    x='zone_class',
    y='Value',
    hue='Variable',
    # notch=True
)
from statannotations.Annotator import Annotator

# 1. Build a list of pairwise comparisons for each zone
zones = df_long['zone_class'].unique()
pairs = [((zone, 'TPqian_LAI'), (zone, 'TPhou_LAI')) for zone in zones]

# 2. Initialize the annotator with your Axes, data, and pairing info
annotator = Annotator(
    ax,
    pairs,
    data=df_long,
    x='zone_class',
    y='Value',
    hue='Variable'
)

# 3. Choose a test and formatting
annotator.configure(
    test='Wilcoxon',              
    text_format='star',             # show *, **, *** instead of raw p-values
    loc='inside',                   # place brackets inside the plot
    comparisons_correction='bonferroni'  # adjust for multiple zones
)

# 4. Compute and draw the annotations
annotator.apply_and_annotate()

ticks = range(len(zone_names))

ax.set_xticks(ticks)
ax.set_xticklabels(zone_names, rotation=0, ha='center')
leg = ax.get_legend()
leg.set_title('')  
leg.get_frame().set_linewidth(0)  
handles, labels = ax.get_legend_handles_labels()
new_labels = []
for lab in labels:
    if lab == 'TPqian_LAI':
        new_labels.append('pre-TP')
    elif lab == 'TPhou_LAI':
        new_labels.append('post-TP')
    else:
        new_labels.append(lab)

ax.legend(handles, new_labels, title='', frameon=False, loc='best')



ax.set_xlabel('Climate Zone')
ax.set_ylabel('LAI Trend')
# ax.set_title('Comparison of TPqian_LAI vs. TPhou_LAI Across Zones')
plt.tight_layout()
fig.savefig("F:/global_wind/picture/five LAI.png", dpi=600, bbox_inches='tight')  




counts = station_coords['zone_class'].value_counts()
print(counts)


# Extended data fig 4
# ——————————————————————————————————————————————————————————————————————————————
station_coords['zone_class'] = pd.cut(
    station_coords['climate_zone'],
    bins=bins,
    labels=zone_labels,
    right=True,
    include_lowest=True
)
from matplotlib.colors import TwoSlopeNorm
mask = station_coords[['TPqian', 'TPhou']].abs().le(0.5).all(axis=1)
station_coords = station_coords[mask]

forest_df = station_coords[station_coords['zone_class'] == '8-16']

grouped = forest_df.groupby('lat_bin')['TPqian_LAI'].agg(['mean', 'std', 'count']).reset_index()


grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])  
grouped['ci95'] = 1.96 * grouped['sem']


norm = TwoSlopeNorm(vmin=-0.05, vcenter=0, vmax=0.05)


fig, ax = plt.subplots(figsize=(12,6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
ax.add_feature(cfeature.OCEAN, facecolor='white')

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False; gl.right_labels = False
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
gl.ylocator = mticker.FixedLocator(np.arange(-90,91,30))
gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':16}; gl.ylabel_style = {'size':16}


sc = ax.scatter(forest_df['lon'], forest_df['lat'], c=forest_df['TPqian_LAI'],
                norm=norm, cmap="BrBG", alpha=0.9,s=30,
                transform=ccrs.PlateCarree())

cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.1, fraction=0.06)

cbar.ax.tick_params(labelsize=16)


divider = make_axes_locatable(ax)
ax2 = divider.append_axes(
    'right', 
    size='18%', 
    pad=0.6, 
    sharey=ax,
    axes_class=plt.Axes     
)

ax2.plot(grouped['mean'], grouped['lat_bin'],
         color='black', linewidth=3, label='Mean wind speed')
grouped['lat_bin'] = grouped['lat_bin'].astype(float)
grouped['mean']    = grouped['mean'].astype(float)
grouped['ci95']    = grouped['ci95'].astype(float)
ax2.fill_betweenx(
    grouped['lat_bin'],
    grouped['mean'] - grouped['ci95'],
    grouped['mean'] + grouped['ci95'],
    color='gray',    
    alpha=0.5,
    label='95% CI'
)

ax2.set_ylim(ax.get_ylim())
# ax2.set_xlim(0, 7)
ax2.set_xticks([ -0.1, 0, 0.1])
lat_ticks = np.array([0,20,40,60,80,-20,-40,-60])
lat_labels = [f"{abs(lat)}°{'N' if lat>0 else 'S'}" if lat!=0 else '0°'
              for lat in lat_ticks]
ax2.set_yticks(lat_ticks); ax2.set_yticklabels(lat_labels)

import matplotlib.ticker as mticker
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

# ax2.set_xlabel('Wind Speed (m s$^{-1}$)', fontsize=12)
ax2.tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelleft=False) 


fig.savefig("F:/global_wind/picture/772LAI-Temperate.png", dpi=600, bbox_inches='tight')


# ——————————————————————————————————————————————————————————————————————————————
station_coords['zone_class'] = pd.cut(
    station_coords['climate_zone'],
    bins=bins,
    labels=zone_labels,
    right=True,
    include_lowest=True
)
from matplotlib.colors import TwoSlopeNorm
mask = station_coords[['TPqian', 'TPhou']].abs().le(0.5).all(axis=1)
station_coords = station_coords[mask]

forest_df = station_coords[station_coords['zone_class'] == '8-16']

grouped = forest_df.groupby('lat_bin')['TPhou_LAI'].agg(['mean', 'std', 'count']).reset_index()


grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])  
grouped['ci95'] = 1.96 * grouped['sem']


norm = TwoSlopeNorm(vmin=-0.05, vcenter=0, vmax=0.05)

fig, ax = plt.subplots(figsize=(12,6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
ax.add_feature(cfeature.OCEAN, facecolor='white')

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False; gl.right_labels = False
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
gl.ylocator = mticker.FixedLocator(np.arange(-90,91,30))
gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':16}; gl.ylabel_style = {'size':16}


sc = ax.scatter(forest_df['lon'], forest_df['lat'], c=forest_df['TPhou_LAI'],
                norm=norm, cmap="BrBG", alpha=0.9,s=30,
                transform=ccrs.PlateCarree())

cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.1, fraction=0.06)

cbar.ax.tick_params(labelsize=16)


divider = make_axes_locatable(ax)
ax2 = divider.append_axes(
    'right', 
    size='18%', 
    pad=0.6, 
    sharey=ax,
    axes_class=plt.Axes     
)

ax2.plot(grouped['mean'], grouped['lat_bin'],
         color='black', linewidth=3, label='Mean wind speed')
grouped['lat_bin'] = grouped['lat_bin'].astype(float)
grouped['mean']    = grouped['mean'].astype(float)
grouped['ci95']    = grouped['ci95'].astype(float)
ax2.fill_betweenx(
    grouped['lat_bin'],
    grouped['mean'] - grouped['ci95'],
    grouped['mean'] + grouped['ci95'],
    color='gray',    # 设置填充为灰色
    alpha=0.5,
    label='95% CI'
)

ax2.set_ylim(ax.get_ylim())

lat_ticks = np.array([0,20,40,60,80,-20,-40,-60])
lat_labels = [f"{abs(lat)}°{'N' if lat>0 else 'S'}" if lat!=0 else '0°'
              for lat in lat_ticks]
ax2.set_yticks(lat_ticks); ax2.set_yticklabels(lat_labels)

import matplotlib.ticker as mticker
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

# ax2.set_xlabel('Wind Speed (m s$^{-1}$)', fontsize=12)
ax2.tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelleft=False)  


fig.savefig("F:/global_wind/picture/772LAI-Temperate.png", dpi=600, bbox_inches='tight')


# Draw the global distribution of cold among the five types and the LAI trend before the turning point
# ——————————————————————————————————————————————————————————————————————————————
station_coords['zone_class'] = pd.cut(
    station_coords['climate_zone'],
    bins=bins,
    labels=zone_labels,
    right=True,
    include_lowest=True
)
from matplotlib.colors import TwoSlopeNorm
mask = station_coords[['TPqian', 'TPhou']].abs().le(0.5).all(axis=1)
station_coords = station_coords[mask]

forest_df = station_coords[station_coords['zone_class'] == '17-28']

grouped = forest_df.groupby('lat_bin')['TPqian_LAI'].agg(['mean', 'std', 'count']).reset_index()


grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])  
grouped['ci95'] = 1.96 * grouped['sem']


norm = TwoSlopeNorm(vmin=-0.05, vcenter=0, vmax=0.05)


fig, ax = plt.subplots(figsize=(12,6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
ax.add_feature(cfeature.OCEAN, facecolor='white')

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False; gl.right_labels = False
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
gl.ylocator = mticker.FixedLocator(np.arange(-90,91,30))
gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':16}; gl.ylabel_style = {'size':16}


sc = ax.scatter(forest_df['lon'], forest_df['lat'], c=forest_df['TPqian_LAI'],
                norm=norm, cmap="BrBG", alpha=0.9,s=30,
                transform=ccrs.PlateCarree())

cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.1, fraction=0.06)

cbar.ax.tick_params(labelsize=16)


divider = make_axes_locatable(ax)
ax2 = divider.append_axes(
    'right', 
    size='18%', 
    pad=0.6, 
    sharey=ax,
    axes_class=plt.Axes      
)

ax2.plot(grouped['mean'], grouped['lat_bin'],
         color='black', linewidth=3, label='Mean wind speed')
grouped['lat_bin'] = grouped['lat_bin'].astype(float)
grouped['mean']    = grouped['mean'].astype(float)
grouped['ci95']    = grouped['ci95'].astype(float)
ax2.fill_betweenx(
    grouped['lat_bin'],
    grouped['mean'] - grouped['ci95'],
    grouped['mean'] + grouped['ci95'],
    color='gray',    
    alpha=0.5,
    label='95% CI'
)

ax2.set_ylim(ax.get_ylim())
# ax2.set_xlim(0, 7)
ax2.set_xticks([ -0.05, 0, 0.05])
lat_ticks = np.array([0,20,40,60,80,-20,-40,-60])
lat_labels = [f"{abs(lat)}°{'N' if lat>0 else 'S'}" if lat!=0 else '0°'
              for lat in lat_ticks]
ax2.set_yticks(lat_ticks); ax2.set_yticklabels(lat_labels)

import matplotlib.ticker as mticker
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

# ax2.set_xlabel('Wind Speed (m s$^{-1}$)', fontsize=12)
ax2.tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelleft=False)  



#  Draw the global distribution of cold among the five types and the trend of LAI after the turning point
# ——————————————————————————————————————————————————————————————————————————————
station_coords['zone_class'] = pd.cut(
    station_coords['climate_zone'],
    bins=bins,
    labels=zone_labels,
    right=True,
    include_lowest=True
)
from matplotlib.colors import TwoSlopeNorm
mask = station_coords[['TPqian', 'TPhou']].abs().le(0.5).all(axis=1)
station_coords = station_coords[mask]

forest_df = station_coords[station_coords['zone_class'] == '17-28']

grouped = forest_df.groupby('lat_bin')['TPhou_LAI'].agg(['mean', 'std', 'count']).reset_index()


grouped['sem'] = grouped['std'] / np.sqrt(grouped['count'])  
grouped['ci95'] = 1.96 * grouped['sem']


norm = TwoSlopeNorm(vmin=-0.05, vcenter=0, vmax=0.05)


fig, ax = plt.subplots(figsize=(12,6), subplot_kw={'projection': ccrs.PlateCarree()})
ax.set_global()
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
ax.add_feature(cfeature.OCEAN, facecolor='white')

gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False; gl.right_labels = False
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
gl.ylocator = mticker.FixedLocator(np.arange(-90,91,30))
gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':16}; gl.ylabel_style = {'size':16}


sc = ax.scatter(forest_df['lon'], forest_df['lat'], c=forest_df['TPhou_LAI'],
                norm=norm, cmap="BrBG", alpha=0.9,s=30,
                transform=ccrs.PlateCarree())

cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.1, fraction=0.06)

cbar.ax.tick_params(labelsize=16)


divider = make_axes_locatable(ax)
ax2 = divider.append_axes(
    'right', 
    size='18%', 
    pad=0.6, 
    sharey=ax,
    axes_class=plt.Axes      
)

ax2.plot(grouped['mean'], grouped['lat_bin'],
         color='black', linewidth=3, label='Mean wind speed')
grouped['lat_bin'] = grouped['lat_bin'].astype(float)
grouped['mean']    = grouped['mean'].astype(float)
grouped['ci95']    = grouped['ci95'].astype(float)
ax2.fill_betweenx(
    grouped['lat_bin'],
    grouped['mean'] - grouped['ci95'],
    grouped['mean'] + grouped['ci95'],
    color='gray',    
    alpha=0.5,
    label='95% CI'
)

ax2.set_ylim(ax.get_ylim())
# ax2.set_xlim(0, 7)
ax2.set_xticks([ -0.05, 0, 0.05])
lat_ticks = np.array([0,20,40,60,80,-20,-40,-60])
lat_labels = [f"{abs(lat)}°{'N' if lat>0 else 'S'}" if lat!=0 else '0°'
              for lat in lat_ticks]
ax2.set_yticks(lat_ticks); ax2.set_yticklabels(lat_labels)

import matplotlib.ticker as mticker
ax2.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

ax2.tick_params(labelsize=16)
ax2.yaxis.set_tick_params(labelleft=False)  



# In[]  Fig 3


import pwlf
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio
# from scipy.stats import linregress

import xarray as xr

df = pd.read_csv('F:/merged_filtered_common_stations.csv')
df = df.rename(columns={'STATION': 'sta', 'LATITUDE': 'lat', 'LONGITUDE': 'lon'})


data = pd.read_csv('F:/LAI.csv')
data = data.groupby('sta').filter(lambda x: x['LAI'].notna().any())

df_avg = data.groupby(['sta', 'lat', 'lon'])['LAI'].mean().reset_index()

data['DATE'] = pd.to_datetime(data['time'])
data['YEAR'] = data['DATE'].dt.year

df = pd.merge(df, df_avg[['sta', 'lat', 'lon']], on=['sta', 'lat', 'lon'], how='inner')

df['WDSP']= df['WDSP'] * 0.514444
df1 = (
    df
    .groupby(['sta', 'YEAR'], as_index=False)
    .agg({
        'WDSP': 'mean',
        'lat': 'first',
        'lon': 'first'
    })
    .rename(columns={'WDSP': 'WDSP'})
)

stations=df1['sta'].unique()
station_coords = df1[['sta', 'lat', 'lon']].drop_duplicates()


years = np.arange(1982, 2021, 1)
c=[]
chucunqian=[]
chucunhou=[]
pTP=[]
for i in station_coords['sta']:
    b=df1.loc[df1['sta'].isin([i])]
    my_pwlf = pwlf.PiecewiseLinFit(years, b['WDSP'])
    a=[]
    for m in range(1986,2017):
        x0 = np.array([1982, m, 2020])
        ssr=my_pwlf.fit_with_breaks(x0)
        a.append(ssr)
    a=a.index(min(a))+1986
    c.append(a)
    x0 = np.array([1982, a, 2020])
    ssr=my_pwlf.fit_with_breaks(x0)
    slopes = my_pwlf.calc_slopes()
    chucunqian.append(slopes[0])
    chucunhou.append(slopes[1])
    p = my_pwlf.p_values(method='linear')
    pTP.append(p[2])


station_coords['TP'] = c
station_coords['TPqian'] = chucunqian
station_coords['TPhou'] = chucunhou
station_coords['TP_pvalue'] = pTP
station_coords = station_coords[station_coords['TP_pvalue'] < 0.05].reset_index(drop=True)


chucunqian1=[]
chucunhou1=[]
pTP_LAI=[]

for i in station_coords['sta']:
    d=data.loc[data['sta'].isin([i])]
    TP=station_coords.loc[station_coords['sta'].isin([i])]['TP'].values
    TP = TP[0] 
    my_pwlf1 = pwlf.PiecewiseLinFit(years, d['LAI'])
    x0 = np.array([1982, TP, 2020])
    ssr=my_pwlf1.fit_with_breaks(x0)
    slopes = my_pwlf1.calc_slopes()
    chucunqian1.append(slopes[0])
    chucunhou1.append(slopes[1])
    p = my_pwlf1.p_values(method='linear')
    pTP_LAI.append(p[2])

station_coords['TPqian_LAI'] = chucunqian1
station_coords['TPhou_LAI'] = chucunhou1
station_coords['TP_pvalue_LAI'] = pTP_LAI

station_coords['lat_bin'] = station_coords['lat'].round(0)  



ds = xr.open_dataset('F:/global_wind/landcover/ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992-v2.0.7cds.nc') 
land1992= ds.lccs_class


ds1 = xr.open_dataset('F:/global_wind/landcover/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc') 
land2020= ds1.lccs_class


station_coords['land2020'] = pd.NA
station_coords['land1992'] = pd.NA


for idx, row in station_coords.iterrows():

    v2020 = land2020.sel(
        lat = row['lat'],
        lon = row['lon'],
        method = 'nearest'
    ).values.item()  
    
    v1992 = land1992.sel(
        lat = row['lat'],
        lon = row['lon'],
        method = 'nearest'
    ).values.item()
    
    station_coords.at[idx, 'land2020'] = v2020
    station_coords.at[idx, 'land1992'] = v1992

station_coords = station_coords[station_coords['land1992'] == station_coords['land2020']]

non_veg = [ 
    0,    # no_data
    190,  # urban
    200,  # bare_areas
    201,  # bare_areas_consolidated
    202,  # bare_areas_unconsolidated
    210,  # water
    220,  # snow_and_ice
]


station_coords = station_coords[~station_coords['land2020'].isin(non_veg)].copy()

mask = station_coords[['TPqian', 'TPhou']].abs().le(0.5).all(axis=1)
station_coords = station_coords[mask]


attrs = land2020.attrs

values   = list(attrs['flag_values'])               # e.g. [0, 10, 11, 12, ...]
colors   = attrs['flag_colors'].split()              # e.g. ['#ffff64', '#ffff64', '#ffff00', ...]
meanings = attrs['flag_meanings'].split()            # e.g. ['no_data', 'cropland_rainfed', ...]

mapping = {
    v: {'color': c, 'meaning': m}
    for v, c, m in zip(values, colors, meanings)
}


station_coords['color'] = station_coords['land2020'].map(lambda v: mapping.get(v, {}).get('color'))
station_coords['meaning'] = station_coords['land2020'].map(lambda v: mapping.get(v, {}).get('meaning'))


# ——————————————————————————————————————————————————————————————————————————————
broad_map = {
    # No Data
    'no_data': 'No Data',
    
    # 1. （Cropland）
    'cropland_rainfed':                     'Cropland',
    'cropland_rainfed_herbaceous_cover':    'Cropland',
    'cropland_rainfed_tree_or_shrub_cover': 'Cropland',
    'cropland_irrigated':                   'Cropland',
    
    # 2. （Mosaic）
    'mosaic_cropland':          'Mosaic',
    'mosaic_natural_vegetation': 'Mosaic',
    'mosaic_tree_and_shrub':    'Mosaic',
    'mosaic_herbaceous':        'Mosaic',
    
    # 3. （Forest）
    'tree_broadleaved_evergreen_closed_to_open': 'Forest',
    'tree_broadleaved_deciduous_closed_to_open': 'Forest',
    'tree_broadleaved_deciduous_closed':         'Forest',
    'tree_broadleaved_deciduous_open':           'Forest',
    'tree_needleleaved_evergreen_closed_to_open':'Forest',
    'tree_needleleaved_evergreen_closed':        'Forest',
    'tree_needleleaved_evergreen_open':          'Forest',
    'tree_needleleaved_deciduous_closed_to_open':'Forest',
    'tree_needleleaved_deciduous_closed':        'Forest',
    'tree_needleleaved_deciduous_open':          'Forest',
    'tree_mixed':                               'Forest',
    
    # 4.（Shrub & Herbaceous）
    'shrubland':             'Shrub & Herbaceous',
    'shrubland_evergreen':   'Shrub & Herbaceous',
    'shrubland_deciduous':   'Shrub & Herbaceous',
    'sparse_shrub':          'Shrub & Herbaceous',
    'sparse_herbaceous':     'Shrub & Herbaceous',
    
    # 5. （Grass & Moss）
    'grassland':            'Grass & Moss',
    'lichens_and_mosses':   'Grass & Moss',
    'sparse_vegetation':    'Grass & Moss',
    
    # 6. （Wetlands）
    'tree_cover_flooded_fresh_or_brakish_water': 'Wetlands',
    'tree_cover_flooded_saline_water':          'Wetlands',
    'shrub_or_herbaceous_cover_flooded':        'Wetlands',
    
    # 7. （Urban & Bare）
    'urban':                       'Urban & Bare',
    'bare_areas':                  'Urban & Bare',
    'bare_areas_consolidated':     'Urban & Bare',
    'bare_areas_unconsolidated':   'Urban & Bare',
    'sparse_tree':                 'Urban & Bare',  #
    
    # 8. （Water & Snow/Ice）
    'water':         'Water & Snow/Ice',
    'snow_and_ice':  'Water & Snow/Ice',
}
station_coords['broad_type'] = station_coords['meaning'].map(broad_map)

import seaborn as sns

mask = station_coords[['TPqian', 'TPhou']].abs().le(0.5).all(axis=1)
station_coords = station_coords[mask]

df_long = station_coords.melt(
    id_vars=['broad_type'], 
    value_vars=['TPqian_LAI', 'TPhou_LAI'],
    var_name='Period',
    value_name='LAI Trend'
)
broad_types = df_long['broad_type'].unique()

df_long['Period'] = df_long['Period'].map({
    'TPqian_LAI': 'Before TP',
    'TPhou_LAI': 'After TP'
})
from statannotations.Annotator import Annotator

fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_long, x='broad_type', y='LAI Trend', hue='Period')

pairs = [((bt, 'Before TP'), (bt, 'After TP')) for bt in broad_types]


annotator = Annotator(ax, pairs, data=df_long, x='broad_type', y='LAI Trend', hue='Period')
annotator.configure(
    test='Wilcoxon',              # independent t-test (Welch’s)
    text_format='star',             # show *, **, *** instead of raw p-values
    loc='inside',                   # place brackets inside the plot
    comparisons_correction='bonferroni'  # adjust for multiple zones
)
annotator.apply_and_annotate()

leg = ax.get_legend()  
leg.set_frame_on(False)
leg.set_title('')
handles, labels = ax.get_legend_handles_labels()
new_labels = ['pre-TP' if lbl=='Before TP' else 'post-TP' for lbl in labels]
ax.legend(handles, new_labels, frameon=False, title='')

new_labels = [lbl.get_text().replace(' ', '\n', 1) for lbl in ax.get_xticklabels()]
ax.set_xticklabels(new_labels)
ax.set_xlabel('Vegetation Zone', fontsize=12)  


# Eight types of wind speed box line diagrams before and after turns have been drawn
# ——————————————————————————————————————————————————————————————————————————————
broad_map = {
    # No Data
    'no_data': 'No Data',
    
    # 1. （Cropland）
    'cropland_rainfed':                     'Cropland',
    'cropland_rainfed_herbaceous_cover':    'Cropland',
    'cropland_rainfed_tree_or_shrub_cover': 'Cropland',
    'cropland_irrigated':                   'Cropland',
    
    # 2. （Mosaic）
    'mosaic_cropland':          'Mosaic',
    'mosaic_natural_vegetation': 'Mosaic',
    'mosaic_tree_and_shrub':    'Mosaic',
    'mosaic_herbaceous':        'Mosaic',
    
    # 3. （Forest）
    'tree_broadleaved_evergreen_closed_to_open': 'Forest',
    'tree_broadleaved_deciduous_closed_to_open': 'Forest',
    'tree_broadleaved_deciduous_closed':         'Forest',
    'tree_broadleaved_deciduous_open':           'Forest',
    'tree_needleleaved_evergreen_closed_to_open':'Forest',
    'tree_needleleaved_evergreen_closed':        'Forest',
    'tree_needleleaved_evergreen_open':          'Forest',
    'tree_needleleaved_deciduous_closed_to_open':'Forest',
    'tree_needleleaved_deciduous_closed':        'Forest',
    'tree_needleleaved_deciduous_open':          'Forest',
    'tree_mixed':                               'Forest',
    
    # 4. （Shrub & Herbaceous）
    'shrubland':             'Shrub & Herbaceous',
    'shrubland_evergreen':   'Shrub & Herbaceous',
    'shrubland_deciduous':   'Shrub & Herbaceous',
    'sparse_shrub':          'Shrub & Herbaceous',
    'sparse_herbaceous':     'Shrub & Herbaceous',
    
    # 5. （Grass & Moss）
    'grassland':            'Grass & Moss',
    'lichens_and_mosses':   'Grass & Moss',
    'sparse_vegetation':    'Grass & Moss',
    
    # 6.（Wetlands）
    'tree_cover_flooded_fresh_or_brakish_water': 'Wetlands',
    'tree_cover_flooded_saline_water':          'Wetlands',
    'shrub_or_herbaceous_cover_flooded':        'Wetlands',
    
    # 7. （Urban & Bare）
    'urban':                       'Urban & Bare',
    'bare_areas':                  'Urban & Bare',
    'bare_areas_consolidated':     'Urban & Bare',
    'bare_areas_unconsolidated':   'Urban & Bare',
    'sparse_tree':                 'Urban & Bare',  # 
    
    # 8. （Water & Snow/Ice）
    'water':         'Water & Snow/Ice',
    'snow_and_ice':  'Water & Snow/Ice',
}
station_coords['broad_type'] = station_coords['meaning'].map(broad_map)

import seaborn as sns

mask = station_coords[['TPqian', 'TPhou']].abs().le(0.5).all(axis=1)
station_coords = station_coords[mask]

df_long = station_coords.melt(
    id_vars=['broad_type'], 
    value_vars=['TPqian', 'TPhou'],
    var_name='Period',
    value_name='Wind Trend'
)
broad_types = df_long['broad_type'].unique()

df_long['Period'] = df_long['Period'].map({
    'TPqian': 'Before TP',
    'TPhou': 'After TP'
})
from statannotations.Annotator import Annotator
# 画图
fig, ax = plt.subplots(figsize=(12, 6))
sns.boxplot(data=df_long, x='broad_type', y='Wind Trend', hue='Period')

pairs = [((bt, 'Before TP'), (bt, 'After TP')) for bt in broad_types]


annotator = Annotator(ax, pairs, data=df_long, x='broad_type', y='Wind Trend', hue='Period')
annotator.configure(
    test='Wilcoxon',              # independent t-test (Welch’s)
    text_format='star',             # show *, **, *** instead of raw p-values
    loc='inside',                   # place brackets inside the plot
    comparisons_correction='bonferroni'  # adjust for multiple zones
)
annotator.apply_and_annotate()

leg = ax.get_legend()  
leg.set_frame_on(False)
leg.set_title('')
handles, labels = ax.get_legend_handles_labels()
new_labels = ['pre-TP' if lbl=='Before TP' else 'post-TP' for lbl in labels]
ax.legend(handles, new_labels, frameon=False, title='')

new_labels = [lbl.get_text().replace(' ', '\n', 1) for lbl in ax.get_xticklabels()]
ax.set_xticklabels(new_labels)
ax.set_xlabel('Vegetation Zone', fontsize=12)  

counts = station_coords['broad_type'].value_counts()
print(counts)
counts = station_coords['meaning'].value_counts()
print(counts)



# In[]     Bootstrap R² Fig 4


import pwlf
import pandas as pd
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import numpy as np
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import rasterio
# from scipy.stats import linregress
# from matplotlib.patches import Patch 
import xarray as xr



import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm


def bootstrap_eta2_with_r2_and_weights(df, formula, B=300, ci=0.95, random_state=None):
    """
    Bootstrap overall R², partial η² and factor weight proportions within R² for each term.
    Returns a summary DataFrame with mean and confidence intervals for R2, partial η², and weights.
    """
    rng = np.random.default_rng(random_state)
    records = []

    for _ in range(B):
        # 
        idx = rng.choice(df.index, size=len(df), replace=True)
        sample = df.loc[idx]

        # 
        model = smf.ols(formula, data=sample).fit()
        anova_res = anova_lm(model, typ=3)

        # 
        terms = anova_res.index.difference(['Residual', 'Intercept'])
        ss_resid = anova_res.at['Residual', 'sum_sq']
        ss_terms = anova_res.loc[terms, 'sum_sq']
        ss_model = ss_terms.sum()

        
        entry = {'R2': model.rsquared}

        # 
        for term in terms:
            ss_t = anova_res.at[term, 'sum_sq']
            entry[f'{term}_eta2'] = ss_t / (ss_t + ss_resid)
            entry[f'{term}_wt']   = ss_t / ss_model if ss_model > 0 else np.nan

        records.append(entry)

    df_res = pd.DataFrame(records)

    # 
    lower_pct = (1 - ci) / 2 * 100    # 2.5
    upper_pct = (ci + (1 - ci) / 2) * 100  # 97.5

    # 
    summary = []
    for col in df_res:
        data = df_res[col].dropna()
        summary.append({
            'term':     col,
            'mean':     data.mean(),
            f'{lower_pct:.1f}th_pct': np.percentile(data, lower_pct),
            f'{upper_pct:.1f}th_pct': np.percentile(data, upper_pct)
        })

    summary_df = pd.DataFrame(summary).set_index('term')
    return summary_df




df = pd.read_csv('F:/merged_filtered_common_stations.csv')
df = df.rename(columns={'STATION': 'sta', 'LATITUDE': 'lat', 'LONGITUDE': 'lon'})
df_tempre = pd.read_csv('F:/merged_filtered_common_stations_tempre.csv')


df_tempre = df_tempre[df_tempre['PRCP'] != 99.99]


df_tempre1=(
    df_tempre
    .groupby(['STATION', 'YEAR'], as_index=False)
    .agg({
        'PRCP': 'mean',
        'TEMP': 'mean',

    })
)


df_tempre1['TEMP']=(df_tempre1['TEMP']-32)/1.8

df_tempre1['PRCP']= df_tempre1['PRCP'] *25.4


data = pd.read_csv('F:/LAI.csv')
data = data.groupby('sta').filter(lambda x: x['LAI'].notna().any())

df_avg = data.groupby(['sta', 'lat', 'lon'])['LAI'].mean().reset_index()

data['DATE'] = pd.to_datetime(data['time'])
data['YEAR'] = data['DATE'].dt.year

df = pd.merge(df, df_avg[['sta', 'lat', 'lon']], on=['sta', 'lat', 'lon'], how='inner')

df['WDSP']= df['WDSP'] * 0.514444
df1 = (
    df
    .groupby(['sta', 'YEAR'], as_index=False)
    .agg({
        'WDSP': 'mean',
        'lat': 'first',
        'lon': 'first'
    })
    .rename(columns={'WDSP': 'WDSP'})
)

stations=df1['sta'].unique()
station_coords = df1[['sta', 'lat', 'lon']].drop_duplicates()


years = np.arange(1982, 2021, 1)
c=[]
chucunqian=[]
chucunhou=[]
pTP=[]
for i in station_coords['sta']:
    b=df1.loc[df1['sta'].isin([i])]
    my_pwlf = pwlf.PiecewiseLinFit(years, b['WDSP'])
    a=[]
    for m in range(1986,2017):
        x0 = np.array([1982, m, 2020])
        ssr=my_pwlf.fit_with_breaks(x0)
        a.append(ssr)
    a=a.index(min(a))+1986
    c.append(a)
    x0 = np.array([1982, a, 2020])
    ssr=my_pwlf.fit_with_breaks(x0)
    slopes = my_pwlf.calc_slopes()
    chucunqian.append(slopes[0])
    chucunhou.append(slopes[1])
    p = my_pwlf.p_values(method='linear')
    pTP.append(p[2])


station_coords['TP'] = c
station_coords['TPqian'] = chucunqian
station_coords['TPhou'] = chucunhou
station_coords['TP_pvalue'] = pTP
station_coords = station_coords[station_coords['TP_pvalue'] < 0.05].reset_index(drop=True)

ds = xr.open_dataset('F:/global_wind/landcover/ESACCI-LC-L4-LCCS-Map-300m-P1Y-1992-v2.0.7cds.nc') 
land1992= ds.lccs_class


ds1 = xr.open_dataset('F:/global_wind/landcover/C3S-LC-L4-LCCS-Map-300m-P1Y-2020-v2.1.1.nc') 
land2020= ds1.lccs_class


station_coords['land2020'] = pd.NA
station_coords['land1992'] = pd.NA


for idx, row in station_coords.iterrows():

    v2020 = land2020.sel(
        lat = row['lat'],
        lon = row['lon'],
        method = 'nearest'
    ).values.item()  
    
    v1992 = land1992.sel(
        lat = row['lat'],
        lon = row['lon'],
        method = 'nearest'
    ).values.item()
    
    station_coords.at[idx, 'land2020'] = v2020
    station_coords.at[idx, 'land1992'] = v1992

station_coords = station_coords[station_coords['land1992'] == station_coords['land2020']]
non_veg = [ 
    0,    # no_data
    190,  # urban
    200,  # bare_areas
    201,  # bare_areas_consolidated
    202,  # bare_areas_unconsolidated
    210,  # water
    220,  # snow_and_ice
]


station_coords = station_coords[~station_coords['land2020'].isin(non_veg)].copy()

mask = station_coords[['TPqian', 'TPhou']].abs().le(0.5).all(axis=1)
station_coords = station_coords[mask]


valid_stas = station_coords['sta'].tolist()

df1 = df1[df1['sta'].isin(valid_stas)]          # :contentReference[oaicite:0]{index=0}

data = data[data['sta'].isin(valid_stas)] 

df_tempre1 = df_tempre1.rename(columns={'STATION': 'sta'})

df_merged = pd.merge(
    df_tempre1,
    df1,
    on=['sta', 'YEAR'],
    how='inner'  
)



sta_list = station_coords['sta'].unique()



df_filtered = df_merged[df_merged['sta'].isin(sta_list)].copy()
dataLAI = data[data['sta'].isin(sta_list)].copy()
dataLAI['DATE'] = pd.to_datetime(dataLAI['time'])
dataLAI['YEAR'] = dataLAI['DATE'].dt.year


df_filtered = (
    df_filtered
    .merge(
        dataLAI[['sta','YEAR','LAI']],
        on=['sta','YEAR'],
        how='inner'   
    )
)

chucun = df_filtered[['sta', 'lat', 'lon']].drop_duplicates()


# ————————————————————————————————————————bootstrap——————————————————————————————————————

tem=[]
pre=[]
win=[]
r2=[]
r22=[]



for i in sta_list:
    
    data = pd.DataFrame({
        'y': df_filtered.loc[df_filtered['sta'] == i, 'LAI'].values,
        'x1': df_filtered.loc[df_filtered['sta'] == i, 'TEMP'].values,
        'x2': df_filtered.loc[df_filtered['sta'] == i, 'PRCP'].values,
        'x3': df_filtered.loc[df_filtered['sta'] == i, 'WDSP'].values,
    })
    

    if data['x2'].eq(0).all():
        print(f"Skipping station {i}: PRCP (x2) is all zeros")
        tem.append(np.nan)
        pre.append(np.nan)
        win.append(np.nan)
        r2.append(np.nan)
        r22.append(np.nan)
        continue
    result = bootstrap_eta2_with_r2_and_weights(data,formula='y ~ x1 + x2 + x3', B=300,ci=0.95,random_state=42)
    
    
    
    data1 = pd.DataFrame({
        'y': df_filtered.loc[df_filtered['sta'] == i, 'LAI'].values,
        'x1': df_filtered.loc[df_filtered['sta'] == i, 'TEMP'].values,
        'x2': df_filtered.loc[df_filtered['sta'] == i, 'PRCP'].values,
    })
    result1 = bootstrap_eta2_with_r2_and_weights(data1,formula='y ~ x1 + x2', B=300,ci=0.95,random_state=42)
    

    # lm_model = stats.lm('y ~ x1 + x2 + x3', data=r_dataframe)
    # importance = relaimpo.calc_relimp(lm_model, type='lmg', rela=True)
    # lmg_values = np.array(importance.do_slot('lmg'))
    # r2 = importance.do_slot('R2')
    mean=result['mean']
    mean1=result1['mean']

    tem.append(mean.iloc[2])
    pre.append(mean.iloc[4])
    win.append(mean.iloc[6])
    r2.append(mean.iloc[0])
    r22.append(mean1.iloc[0])

chucun['tem']=tem
chucun['pre']=pre
chucun['win']=win
chucun['r2']=r2
chucun['r22']=r22
chucun = chucun[chucun['r2'].notna()]


# __________________________global distribution map—————————————————————————————
import matplotlib.colors as mcolors
from matplotlib.colors import BoundaryNorm
import matplotlib.cm as cm

col_map = {'tem': 1, 'pre': 2, 'win': 3}
chucun['max_type'] = (
    chucun[['tem', 'pre', 'win']]
    .idxmax(axis=1)
    .map(col_map)
)

color_map = {1: 'red', 2: 'blue', 3: 'yellow'}
chucun['plot_color'] = chucun['max_type'].map(color_map)


fig = plt.figure(figsize=(8,6))
ax = plt.axes(projection=ccrs.PlateCarree())
ax.coastlines()

sc = ax.scatter(
    chucun['lon'], 
    chucun['lat'], 
    c=chucun['plot_color'], 
    alpha=0.7, 
    transform=ccrs.PlateCarree(),
    s=10,  
    # edgecolor='k',  
)
ax.set_global()
ax.add_feature(cfeature.COASTLINE, linewidth=1)
ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='black')
ax.add_feature(cfeature.OCEAN, facecolor='white')
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=0.5, color='gray', alpha=0.5, linestyle='--')
gl.top_labels = False; gl.right_labels = False
gl.xlocator = mticker.FixedLocator(np.arange(-180,181,30))
gl.ylocator = mticker.FixedLocator(np.arange(-90,91,30))
gl.xformatter = LONGITUDE_FORMATTER; gl.yformatter = LATITUDE_FORMATTER
gl.xlabel_style = {'size':14}; gl.ylabel_style = {'size':14}
cmap = mcolors.ListedColormap(['red', 'blue', 'yellow'])
bounds = [0.5, 1.5, 2.5, 3.5]
norm = BoundaryNorm(bounds, cmap.N)
cbar_ax = fig.add_axes([0.15, 0.06, 0.4, 0.015])
cb = fig.colorbar(
    cm.ScalarMappable(norm=norm, cmap=cmap),  
    cax=cbar_ax,
    orientation='horizontal',
    boundaries=bounds,
    ticks=[1, 2, 3]
)
cb.set_ticklabels(['tm', 'pre', 'win'])
cb.ax.tick_params(labelsize=16, pad=4)


# ————————————————————————————————————latitudinal distribution————————————————————————————————————
chucun['lat_bin'] = chucun['lat'].round(0)
grouped = chucun.groupby('lat_bin')[['tem','pre','win']].mean().reset_index()
divider = make_axes_locatable(ax)
ax2 = divider.append_axes(
    'right',
    size='25%',    
    pad=0.3,       #
    sharey=ax,
    axes_class=plt.Axes  #
)

colors = [ 'red', 'blue','yellow']
series = [grouped['tem'], grouped['pre'],grouped['win']]
bottom = np.zeros_like(grouped['lat_bin'], dtype=float)

for vals, c in zip(series, colors):
 
    ax2.fill_betweenx(
        grouped['lat_bin'],
        bottom,
        bottom + vals,
        color=c,
        alpha=0.8
    )
    bottom += vals


ax2.set_xlabel('Weight')

ax2.yaxis.set_visible(False)

