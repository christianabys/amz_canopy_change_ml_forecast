#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import packages
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio.features
from rasterio import open as r_open
import rasterio.warp
from rasterio.plot import show as r_show 
from subprocess import Popen
from rasterstats import zonal_stats
#import packages
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import (ExtraTreesRegressor,
                             GradientBoostingRegressor,
                             RandomForestRegressor)
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from shapely.geometry import Polygon, Point


# In[100]:


pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows',200)


# In[ ]:


#seperate out 2005
from osgeo import gdal

driver = gdal.GetDriverByName('GTiff')
file = gdal.Open('/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/Data_2/tif_coarse/merge_4326_coarse_210.tif')
band = file.GetRasterBand(1)
lista = band.ReadAsArray()

# reclassification
for j in  range(file.RasterXSize):
    for i in  range(file.RasterYSize):
        if lista[i,j] > 0 and lista[i,j] < 5:
            if lista[i,j] > 0:
                lista[i,j] = 1
        else:
            lista[i,j] = 0

# create new file
file2 = driver.Create( '/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/Data_2/tif_coarse/merge_4326_reclass_2005.tif', file.RasterXSize , file.RasterYSize , 1)
file2.GetRasterBand(1).WriteArray(lista)

# spatial ref system
proj = file.GetProjection()
georef = file.GetGeoTransform()
file2.SetProjection(proj)
file2.SetGeoTransform(georef)
file2.FlushCache()


# In[3]:


cu = gpd.read_file("/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/Data_2/CT10_P/CT10_P.shp")
cu.index = pd.Index(range(1, len(cu) + 1), name='px_id')


# In[4]:


pas = pd.read_csv("/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/Data/amazon_pas.csv",encoding = 'latin1')


# In[5]:


px_p = gpd.read_file("/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/Data_2/amazon_px.csv")
px_p = gpd.GeoDataFrame(px_p, geometry=gpd.points_from_xy(px_p.x, px_p.y))
px_p = px_p.set_geometry(px_p['geometry'])
px_p.crs = {'init': 'epsg:4326', 'no_defs': True}
px_p = px_p.to_crs("EPSG:4326")


# In[6]:


cu_px = gpd.sjoin(px_p , cu,how = "right" ,op='intersects')
cu_px = cu_px.dropna(subset=['vcf00','vcf05','distedg00','distedg05','distpa','traveltime','slope','elev','floodable'])
convert_dict = {'distedg05':'float','distedg00':'float','distpa':'float','traveltime':'float','slope':'float','elev':'float','floodable':'float','wdpaid':'int64','vcf05':'float','vcf00':'float'}
cu_px = cu_px.astype(convert_dict)
cu_px_g = cu_px.groupby('px_id')['distedg00','distedg05','distpa','traveltime','slope','elev','floodable','CU_ID1'].mean()
cu_px = cu_px.reset_index().drop_duplicates(subset = 'px_id').set_index('px_id')
cu_px = cu_px.drop(columns = ['distedg00','distedg05','distpa','traveltime','slope','elev','floodable','CU_ID1'] )
cu_px = cu_px.merge(cu_px_g,on ='px_id')
cu_px = cu_px.drop(['CD_GEOCODB','NM_BAIRRO','NM_SUBDIST',
                   'POP'],axis=1)
cu_px.index = pd.Index(range(1, len(cu_px) + 1), name='px_id')
#get area column
cu_px = cu_px.to_crs({'init':'epsg:3857'})
cu_px[['area']] = cu_px['geometry'].area/10**6
cu_px = cu_px.to_crs({'init':'epsg:4326'})
cu_px = pd.merge(cu_px, pas,on='wdpaid',how = 'left').fillna('NP')
cu_px.index = pd.Index(range(1, len(cu_px) + 1), name='px_id')


# In[7]:


#ZS_2000
zs_2002 = pd.DataFrame(zonal_stats(cu_px, 
                        '/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/Data_2/tif_coarse/merge_4326_reclass_2002.tif', 
                        stats = ['mean'], all_touched = False, nodata = -9999 ))


# In[8]:


#ZS 2019
zs_2019 = pd.DataFrame(zonal_stats(cu_px, 
                        '/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/Data_2/tif_coarse/merge_4326_reclass_2019.tif', 
                        stats = ['mean'], all_touched = False, nodata = -9999 ))


# In[9]:


#ZS HDR
zs_hdr = pd.DataFrame(zonal_stats(cu_px, 
                        '/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/Data_2/hdr/hdr_clipped.tif', 
                        stats = ['mean'], all_touched = False, nodata = -9999 ))


# In[10]:


#ZS_POP_2002
zs_pop_2002 = pd.DataFrame(zonal_stats(cu_px, 
                        '/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/Data_2/POP/bra_ppp_2002_1km_Aggregated.tif', 
                        stats = ['sum'], all_touched = False, nodata = -9999 ))


# In[11]:


#ZS_POP_2019
zs_pop_2019 = pd.DataFrame(zonal_stats(cu_px, 
                        '/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/Data_2/POP/bra_ppp_2019_1km_Aggregated.tif', 
                        stats = ['sum'], all_touched = False, nodata = -9999 ))


# In[12]:


zs_2002_t = pd.DataFrame(zs_2002['mean'] * 100)
zs_2002_t.index = pd.Index(range(1, len(zs_2002_t) + 1), name='px_id')
zs_2019_t = pd.DataFrame(zs_2019['mean'] * 100)
zs_2019_t.index = pd.Index(range(1, len(zs_2019_t) + 1), name='px_id')
zs_hdr.index = pd.Index(range(1, len(zs_hdr) + 1), name='px_id')
zs_pop_2002 = pd.DataFrame(zs_pop_2002['sum'])
zs_pop_2002.index =  pd.Index(range(1, len(zs_pop_2002) + 1), name='px_id')
zs_pop_2019 = pd.DataFrame(zs_pop_2002['sum'])
zs_pop_2019.index =  pd.Index(range(1, len(zs_pop_2019) + 1), name='px_id')


# In[13]:


cu_px = cu_px.join(zs_pop_2002['sum'].abs())
cu_px = cu_px.rename(columns = {'sum':'pop_2002'})
cu_px = cu_px.join(zs_pop_2019['sum'].abs())
cu_px = cu_px.rename(columns = {'sum':'pop_2019'})


# In[14]:


#cast joins and rename columns
cu_px = cu_px.join(zs_2019_t['mean'])
cu_px = cu_px.rename(columns = {'mean':'mean_2019'})
cu_px = cu_px.join(zs_2002_t['mean'])
cu_px = cu_px.rename(columns = {'mean':'mean_2002'})
cu_px = cu_px.join(zs_hdr['mean'])
cu_px = cu_px.rename(columns = {'mean':'hdr'})
cu_px.loc[cu_px.hdr > 100,'hdr'] = 100
cu_px = cu_px.dropna(subset = ['mean_2019','mean_2002','hdr'])
#create p_f_change class 
cu_px[['p_f_change']] = cu_px['mean_2019'] - cu_px['mean_2002']
#hdr_2019
cu_px[['hdr_2019']] = cu_px['hdr'] + cu_px['p_f_change']
cu_px.loc[cu_px.hdr_2019 > 100,'hdr_2019'] = 100


# In[32]:


cu_px[['distedg19']] = (cu_px['distedg05']) * 0.55333


# In[15]:


#cost function
cu_px[['cost']] = np.asarray((cu_px ['area'] * 247.105) * 3953.54)


# In[137]:


#rename protected areas for legend
cu_px = cu_px.replace(to_replace = ['IT'],value = 'Indigenous_Territory')
cu_px = cu_px.replace(to_replace = ['SP'],value = 'Strict_Protection')
cu_px = cu_px.replace(to_replace = ['SU'],value = 'Sustainable_Use')
cu_px = cu_px.replace(to_replace = ['IT'],value = 'Indigenous_Territory')
cu_px = cu_px.replace(to_replace = ['NP'],value = 'Not_Protected')


# In[ ]:


Distance to major road operation
roads = gpd.read_file("/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/Data_2/roads/raods_sa.shp")
roads = roads[['OBJECTID','LENGTH_KM',
               'Shape_Leng','geometry']]
roads = roads.to_crs("EPSG:3857")
roads = gpd.GeoSeries(roads.geometry)
cu_px_c = gpd.GeoDataFrame(
    cu_px, geometry=gpd.points_from_xy(cu_px.x, cu_px.y))
cu_px_c = cu_px_c.sample(20)
cu_px_c = cu_px_c.set_crs(roads.crs)
def min_distance(polygons, roads):
    return cu_px_c =
cu_px_c['dist_to_roads'] = cu_px_c.geometry.apply(lambda x: min_distance(x, roads))
cu_px_c = gpd.GeoDataFrame(cu_px_c)
cu_px = cu_px.merge(cu_px_c,on = 'px_id',how='inner')


# In[140]:


#visualize percent forest cover change from 2002 to 2019
#looks pretty good
fig,ax = plt.subplots()
cu_px.plot('p_f_change', alpha = 0.5, cmap = 'RdYlGn_r', legend = True,ax = ax)
cu_px.plot(ax = ax, color = 'none', edgecolor = 'black',linewidth = 0.05)
plt.title("Percent Forest Change %: 2002 to 2019")
fig.set_size_inches(17,10)
plt.savefig("/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/figures/p_f_change.png",bbox_inches='tight',dpi=150)


# In[141]:


fig,ax = plt.subplots()
cu_px.plot('mean_2019', alpha = 0.5, cmap = 'RdYlGn_r', legend = True,ax = ax)
cu_px.plot(ax = ax, color = 'none', edgecolor = 'black',linewidth = 0.05)
plt.title("Average Deforestation %: 2019")
fig.set_size_inches(17,10)
plt.savefig("/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/figures/deforestation_2019.png",bbox_inches='tight',dpi=150)


# In[142]:


fig,ax = plt.subplots()
cu_px.plot('mean_2002', alpha = 0.5, cmap = 'RdYlGn_r', legend = True,ax = ax)
cu_px.plot(ax = ax, color = 'none', edgecolor = 'black',linewidth = 0.05)
plt.title("Average Deforestation %: 2002")
fig.set_size_inches(17,10)
plt.savefig("/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/figures/deforestation_2002.png",bbox_inches='tight',dpi=150)


# In[313]:


fig,ax = plt.subplots()
cu_px.plot('hdr', alpha = 0.5, cmap = 'RdYlGn_r', legend = True,ax = ax)
cu_px.plot(ax = ax, color = 'none', edgecolor = 'black',linewidth = 0.05)
plt.title("Human Development Index")
fig.set_size_inches(17,10)
plt.savefig("/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/figures/HDR.png",bbox_inches='tight',dpi=150)


# In[314]:


fig,ax = plt.subplots()
cu_px.plot('hdr_2019', alpha = 0.5, cmap = 'RdYlGn_r', legend = True,ax = ax)
cu_px.plot(ax = ax, color = 'none', edgecolor = 'black',linewidth = 0.05)
plt.title("Human Development Index 2019 (Adjusted with percent forest cover change)")
fig.set_size_inches(17,10)


# In[ ]:


# fig,ax = plt.subplots()
# cu_px.plot('dist_to_roads', alpha = 0.5, cmap = 'RdYlGn_r', legend = True,ax = ax)
# #cu_px.plot(ax = ax, color = 'none', edgecolor = 'black',linewidth = 0.05)
# plt.title("Distance to Major Roads")
# fig.set_size_inches(17,10)


# # ML

# In[225]:


#cu_px = cu_px.drop(columns = ['pred','pred_p_f_change','pred_2019'])
cu_px = cu_px.drop(columns = ['pred'])


# In[226]:


#create x columns 
#TRAINING DATA
XCOLS_FC = ['distpa','traveltime','slope','elev','floodable','mean_2002','hdr','pop_2002','distedg00']
#XCOLS_FC = ['distpa','traveltime','slope','elev','mean_2002','hdr','pop_2002','distedg00']
#create fc
fc = cu_px[XCOLS_FC].isnull().sum(1).eq(0)
fc &= cu_px['patype'].ne('SP')
#fc &= cu_px['patype'].ne('IT')
X_fc = cu_px.loc[fc,XCOLS_FC].copy()
y_fc = cu_px.loc[fc, 'p_f_change'].copy() 


# In[227]:


coords = cu_px[['x', 'y']]


# In[228]:


#Model
model = ExtraTreesRegressor(n_estimators = 300, bootstrap = True, oob_score = True, n_jobs=-1)
#model = RandomForestRegressor(n_estimators=250, oob_score=True)


# In[229]:


#predict forest change #fit 
model.fit(X_fc.join(coords), y_fc)
y_pred = pd.DataFrame(model.predict(X_fc.join(coords)), index=X_fc.index)


# In[230]:


X = X_fc
y = y_fc
mse_train = []
mse_test = []
for i_train, i_test in KFold(n_splits=5, shuffle=True).split(X):
    X_train, X_test = X.iloc[i_train], X.iloc[i_test]
    y_train, y_test = y.iloc[i_train], y.iloc[i_test]
    #model = ExtraTreesRegressor(n_estimators = 300, bootstrap = True, oob_score = True, n_jobs=-1)
    #model = GradientBoostingRegressor(n_estimators=100)
    model = RandomForestRegressor(n_estimators=250, oob_score=True)
    model.fit(X_train, y_train)
    mse_train += [mean_squared_error(y_train, model.predict(X_train))]
    mse_test += [mean_squared_error(y_test, model.predict(X_test))]
print('Avg MSE (within-sample):', round(np.mean(mse_train), 4))
print('Avg MSE (out-of-sample):', round(np.mean(mse_test), 4))
print('R2:',r2_score(y_fc,y_pred))


# In[231]:


#join predictions back to cu_px and rename pred column
cu_px = pd.merge(cu_px, y_pred, on  = 'px_id',how = 'left').fillna(0)
cu_px = cu_px.rename(columns = {0:'pred'})


# In[318]:


feat_imp = pd.Series(model.feature_importances_, index=X.columns)
x = feat_imp.sort_values()
fig,ax = plt.subplots()
x.plot(kind = 'barh')
plt.title("Feature Importance: Training")
plt.savefig("/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/figures/feature_importance.png",bbox_inches='tight',dpi=150)


# In[233]:


#visualize percent forest cover change from 2002 to 2019
#looks pretty good
fig,ax = plt.subplots()
cu_px.plot('p_f_change', alpha = 0.5, cmap = 'RdYlGn_r', legend = True,ax = ax)
cu_px.plot(ax = ax, color = 'none', edgecolor = 'black',linewidth = 0.05)
plt.title("Percent Forest Change %: 2002 to 2019")
fig.set_size_inches(17,10)
plt.savefig("/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/figures/p_f_change.png",bbox_inches='tight',dpi=150)


# In[234]:


fig,ax = plt.subplots()
cu_px.plot('pred', alpha = 0.5, cmap = 'RdYlGn_r', legend = True,ax = ax)
cu_px.plot(ax = ax, color = 'none', edgecolor = 'black',linewidth = 0.05)
plt.title("Predicted Forest Threat: Training Set")
fig.set_size_inches(17,10)
plt.savefig("/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/figures/predicted_deforest_rate_training.png",bbox_inches='tight',dpi=150)


# In[235]:


#predict with 2019 data
#create x columns
XCOLS_FC = ['distpa','traveltime','slope','elev','floodable','mean_2019','hdr_2019','pop_2019','distedg19']
#XCOLS_FC = ['distpa','traveltime','slope','elev','mean_2019','hdr_2019','pop_2019','distedg19']


#create fc
fc = cu_px[XCOLS_FC].isnull().sum(1).eq(0)
fc &= cu_px['patype'].ne('SP')
#fc &= cu_px['patype'].ne('IT')
X_fc = cu_px.loc[fc,XCOLS_FC].copy()
y_fc = cu_px.loc[fc,'p_f_change'].copy()


# In[236]:


y_pred_2019 = pd.DataFrame(model.predict(X_fc), index = X_fc.index)
cu_px = pd.merge(cu_px, y_pred_2019, on ='px_id',how = 'left').fillna(0)
cu_px = cu_px.rename(columns = {0:'pred_2019'})


# In[237]:


#calculate total percent change since 2002
cu_px[['pred_p_f_change']] = cu_px['pred_2019'] + cu_px['p_f_change']
cu_px.loc[cu_px.pred_p_f_change > 100,'pred_p_f_change'] = 100


# In[238]:


X = X_fc
y = y_fc
mse_train = []
mse_test = []
for i_train, i_test in KFold(n_splits=5, shuffle=True).split(X):
    X_train, X_test = X.iloc[i_train], X.iloc[i_test]
    y_train, y_test = y.iloc[i_train], y.iloc[i_test]
    model = ExtraTreesRegressor(n_estimators = 500, bootstrap = True, oob_score = True, n_jobs=-1)
    model.fit(X_train, y_train)
    mse_train += [mean_squared_error(y_train, model.predict(X_train))]
    mse_test += [mean_squared_error(y_test, model.predict(X_test))]
print('Avg MSE (within-sample):', round(np.mean(mse_train), 4))
print('Avg MSE (out-of-sample):', round(np.mean(mse_test), 4))


# In[239]:


fig,ax = plt.subplots()
cu_px.plot('pred_p_f_change', alpha = 0.5, cmap = 'RdYlGn_r', legend = True,ax = ax)
cu_px.plot(ax = ax, color = 'none', edgecolor = 'black',linewidth = 0.05)
plt.title("Predicted % Deforestation: 2002 - 2036")
fig.set_size_inches(17,10)
plt.savefig("/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/figures/predicted_2002_2036_training.png",bbox_inches='tight',dpi=150)


# In[240]:


fig,ax = plt.subplots()
cu_px.plot('pred_2019', alpha = 0.5, cmap = 'RdYlGn_r', legend = True,ax = ax)
cu_px.plot(ax = ax, color = 'none', edgecolor = 'black',linewidth = 0.05)
plt.title("Predicted % Deforestation632 2019 - 2036")
fig.set_size_inches(17,10)
plt.savefig("/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/figures/predicted_2036.png",bbox_inches='tight',dpi=150)


# In[241]:


fig,ax = plt.subplots()
cu_px.plot('patype', alpha = 0.5, cmap = 'RdYlGn_r', legend = True,ax = ax)
cu_px.plot(ax = ax, color = 'none', edgecolor = 'black',linewidth = 0.05)
plt.title("Protected Areas")
fig.set_size_inches(17,10)
plt.savefig("/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/figures/protected_areas].png",bbox_inches='tight',dpi=150)


# In[245]:


cu_px['area'].sum()


# In[243]:


cu_px['p_f_change'].mean()


# In[242]:


cu_px['pred_2019'].mean()


# Policy Decisions

# In[ ]:


#set budget
BUDGET = 75000000


# In[47]:


#print area estimates
#area forested in 2002
cu_px['deforest_2002_area'] = cu_px['area'] * (cu_px['mean_2002']/100)
cu_px[['deforest_2019_area']] = cu_px['area'] * (cu_px['mean_2019']/100)
cu_px['deforest_2036_area'] = cu_px['area'] * (cu_px['pred_p_f_change']/100)


# In[84]:


dm = cu_px.copy()


# In[48]:


dm['area'].sum()


# In[49]:


dm['deforest_2002_area'].sum() 


# In[50]:


dm['deforest_2019_area'].sum()


# In[51]:


dm['deforest_2036_area'].sum()


# In[247]:


dm['deforest_2002_area'].sum() / dm['area'].sum()


# In[52]:


dm['deforest_2019_area'].sum() / cu_px['area'].sum()


# In[53]:


dm['deforest_2036_area'].sum() / cu_px['area'].sum()


# In[257]:


i = dm['deforest_2036_area'].sort_values(ascending = False).head(20)


# In[264]:


dm = dm.sort_values('deforest_2036_area',ascending = False)


# In[ ]:





# In[265]:


dm['cost_sum'] = dm['deforest_area']


# In[309]:


i_area_loss =  dm[dm['deforest_2036_area'].gt(2000)]
i_area_loss['deforest_2002_area'].sum()


# In[302]:



fig,ax = plt.subplots()
i_area_loss.plot('deforest_2036_area', alpha = 0.5, cmap = 'RdYlGn_r', legend = True,ax = ax)
cu_px.plot(ax = ax, color = 'black', edgecolor = 'black',linewidth = 0.05, alpha = .1)
plt.title("Deforestation Loss greater than 2000 kilometers ")
fig.set_size_inches(17,10)
plt.savefig("/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/figures/Highest_Threat_area.png",bbox_inches='tight',dpi=150)


# In[311]:


i_threat = dm[dm['pred_2019'].gt(40)]
i_threat['deforest_2036_area'].sum()


# In[312]:





# In[300]:


i_threat = dm[dm['pred_2019'].gt(40)]
fig,ax = plt.subplots()
i_threat.plot('pred_2019', alpha = 0.5, cmap = 'RdYlGn_r', legend = True,ax = ax)
cu_px.plot(ax = ax, color = 'black', edgecolor = 'black',linewidth = 0.05, alpha = .1)
plt.title("Deforestation Rate Greater than 40% ")
fig.set_size_inches(17,10)
plt.savefig("/Users/christianabys/Desktop/School/Boston_University/2020/Data_Science/Project/figures/Highest_Threat_rate.png",bbox_inches='tight',dpi=150)


# In[ ]:




