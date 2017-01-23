from mpl_toolkits.basemap import Basemap
from mpl_toolkits.basemap import cm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import rgb2hex
from matplotlib.patches import Polygon


def res_visulization(rawTest_data, pred_result):
    # popdensity_ori is the base of popular of state
    popdensity_ori = {'New Jersey': 0., 'Rhode Island': 0., 'Massachusetts': 0., 'Connecticut': 0.,
                      'Maryland': 0., 'New York': 0., 'Delaware': 0., 'Florida': 0., 'Ohio': 0., 'Pennsylvania': 0.,
                      'Illinois': 0., 'California': 0., 'Hawaii': 0., 'Virginia': 0., 'Michigan': 0.,
                      'Indiana': 0., 'North Carolina': 0., 'Georgia': 0., 'Tennessee': 0., 'New Hampshire': 0.,
                      'South Carolina': 0., 'Louisiana': 0., 'Kentucky': 0., 'Wisconsin': 0., 'Washington': 0.,
                      'Alabama': 0., 'Missouri': 0., 'Texas': 0., 'West Virginia': 0., 'Vermont': 0.,
                      'Minnesota': 0., 'Mississippi': 0., 'Iowa': 0., 'Arkansas': 0., 'Oklahoma': 0.,
                      'Arizona': 0., 'Colorado': 0., 'Maine': 0., 'Oregon': 0., 'Kansas': 0., 'Utah': 0.,
                      'Nebraska': 0., 'Nevada': 0., 'Idaho': 0., 'New Mexico': 0., 'South Dakota': 0.,
                      'North Dakota': 0., 'Montana': 0., 'Wyoming': 0., 'Alaska': 0.}
    # popdensity is to store result of prediction
    popdensity_pred = {'New Jersey': 0., 'Rhode Island': 0., 'Massachusetts': 0., 'Connecticut': 0.,
                       'Maryland': 0., 'New York': 0., 'Delaware': 0., 'Florida': 0., 'Ohio': 0., 'Pennsylvania': 0.,
                       'Illinois': 0., 'California': 0., 'Hawaii': 0., 'Virginia': 0., 'Michigan': 0.,
                       'Indiana': 0., 'North Carolina': 0., 'Georgia': 0., 'Tennessee': 0., 'New Hampshire': 0.,
                       'South Carolina': 0., 'Louisiana': 0., 'Kentucky': 0., 'Wisconsin': 0., 'Washington': 0.,
                       'Alabama': 0., 'Missouri': 0., 'Texas': 0., 'West Virginia': 0., 'Vermont': 0.,
                       'Minnesota': 0., 'Mississippi': 0., 'Iowa': 0., 'Arkansas': 0., 'Oklahoma': 0.,
                       'Arizona': 0., 'Colorado': 0., 'Maine': 0., 'Oregon': 0., 'Kansas': 0., 'Utah': 0.,
                       'Nebraska': 0., 'Nevada': 0., 'Idaho': 0., 'New Mexico': 0., 'South Dakota': 0.,
                       'North Dakota': 0., 'Montana': 0., 'Wyoming': 0., 'Alaska': 0.}

    idx = 0
    for obj in rawTest_data['results']:
        user_location = obj['user_location']
        popdensity_ori[user_location] += (obj['polarity'] - 1)
        popdensity_pred[user_location] += (pred_result[idx] - 1)
        idx += 1

    print('popdensity_ori')
    print(popdensity_ori)
    print('------------------------------------------------------------')
    print('popdensity_pred')
    print(popdensity_pred)
    print('------------------------------------------------------------')

    # Lambert Conformal Projection
    fig = plt.figure(figsize=(14, 6))
    # original sentiment value
    ax1 = fig.add_axes([0.05, 0.20, 0.40, 0.75])
    # analysis result
    ax3 = fig.add_axes([0.50, 0.20, 0.40, 0.75])
    # m1 is use map
    m1 = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49, projection='lcc',
                 lat_1=33, lat_2=45, lon_0=-95, ax=ax1)
    shp_info = m1.readshapefile('./shapefile/st99_d00', 'states', drawbounds=True)
    print(shp_info)

    # cities
    cities = ['WA', 'OR', 'CA', 'NV', 'MT', 'ID', 'WY',
              'UT', 'CO', 'AZ', 'NM', 'ND', 'SD', 'NE',
              'KS', 'OK', 'TX', 'MN', 'IA', 'MO', 'AR',
              'LA', 'WI', 'IL', 'MI', 'IN', 'OH', 'KY',
              'TN', 'MS', 'AL', 'PA', 'WV', 'GA', 'ME',
              'VT', 'NY', 'VA', 'NC', 'SC', 'FL', 'AL']
    lat = [47.40, 44.57, 36.12, 38.31, 46.92, 44.24, 42.75,
           40.15, 39.06, 33.73, 34.84, 47.53, 44.30, 41.125,
           38.526, 35.565, 31.05, 45.69, 42.01, 38.46, 34.97,
           31.17, 44.27, 40.35, 43.33, 39.85, 40.39, 37.67,
           35.75, 32.74, 61.37, 40.59, 38.49, 33.04, 44.69,
           44.045, 42.165, 37.77, 35.63, 33.86, 27.77, 32.81]
    lon = [-121.49, -122.07, -119.68, -117.05, -110.45, -114.48, -107.30,
           -111.86, -105.31, -111.43, -106.25, -99.93, -99.44, -98.27,
           -96.726, -96.93, -97.56, -93.90, -93.21, -92.29, -92.37,
           -91.87, -89.62, -88.99, -84.54, -86.26, -82.76, -84.67,
           -86.70, -89.68, -152.40, -77.21, -80.95, -83.64, -69.38,
           -72.71, -74.95, -78.17, -79.81, -80.945, -81.67, -86.79]
    # colors based on population
    colors_ori = {}
    colors_pred = {}
    statenames = []
    # gradient color
    cmap = cm.GMT_polar
    inverse = [(value, key) for key, value in popdensity_ori.items()]
    vmin_ori = min(inverse)[0]
    vmax_ori = max(inverse)[0]
    inverse = [(value, key) for key, value in popdensity_pred.items()]
    vmin_pred = min(inverse)[0]
    vmax_pred = max(inverse)[0]

    print('vmax: ')
    print(vmax_ori)
    print(m1.states_info[0].keys())
    for shapedict in m1.states_info:
        statename = shapedict['NAME']
        # skip DC and Puerto Rico
        if statename not in ['District of Columbia', 'Puerto Rico']:
            pop = popdensity_ori[statename]
            pop_pred = popdensity_pred[statename]
            # calling colormap with value between 0 and 1 returns
            # rgba value.  Invert color range (hot colors are high
            # population), take sqrt root to spread out colors more
            if pop == 0:
                colors_ori[statename] = cmap(0.5)[:3]
            elif pop < 0:
                colors_ori[statename] = cmap(1.0 - np.sqrt((pop - vmin_ori) / (0 - vmin_ori)))[:3]
            else:
                colors_ori[statename] = cmap(0.5 - np.sqrt((pop - 0) / (vmax_ori - 0)))[:3]

            if pop_pred == 0:
                colors_pred[statename] = cmap(0.5)[:3]
            elif pop_pred < 0:
                colors_pred[statename] = cmap(1.0 - np.sqrt((pop_pred - vmin_pred) / (0 - vmin_pred)))[:3]
            else:
                colors_pred[statename] = cmap(0.5 - np.sqrt((pop_pred - 0) / (vmax_pred - 0)))[:3]
        statenames.append(statename)
    print('colors_ori')
    print(colors_ori)
    print('------------------------------------------------------------')
    print('colors_pred')
    print(colors_pred)
    print('------------------------------------------------------------')

    for nshape, seg in enumerate(m1.states):
        if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
            color = rgb2hex(colors_ori[statenames[nshape]])
            poly = Polygon(seg, facecolor=color, edgecolor=color)
            ax1.add_patch(poly)

    # # longitude and langtiude
    # m1.drawparallels(np.arange(25, 65, 20), labels=[1, 0, 0, 0])
    # m1.drawmeridians(np.arange(-120, -40, 20), labels=[0, 0, 0, 1])

    # abbreviation
    x, y = m1(lon, lat)
    for city, xc, yc in zip(cities, x, y):
        ax1.text(xc - 60000, yc - 50000, city)
    ax1.set_title('Twitter-based sentiment analysis about Hillary ')

    m2 = Basemap(llcrnrlon=-119, llcrnrlat=22, urcrnrlon=-64, urcrnrlat=49, projection='lcc',
                 lat_1=33, lat_2=45, lon_0=-95, ax=ax3)
    m2.readshapefile('./shapefile/st99_d00', 'states', drawbounds=True)

    for nshape, seg in enumerate(m2.states):
        if statenames[nshape] not in ['District of Columbia', 'Puerto Rico']:
            color = rgb2hex(colors_pred[statenames[nshape]])
            poly = Polygon(seg, facecolor=color, edgecolor=color)
            ax3.add_patch(poly)
    # # longitude and langtiude
    # m2.drawparallels(np.arange(25, 65, 20), labels=[1, 0, 0, 0])
    # m2.drawmeridians(np.arange(-120, -40, 20), labels=[0, 0, 0, 1])
    x, y = m2(lon, lat)
    for city, xc, yc in zip(cities, x, y):
        ax3.text(xc - 60000, yc - 50000, city)
    ax3.set_title('Random Forest Prediction about Hillary ')

    # add gradient color bar
    ax2 = fig.add_axes([0.05, 0.10, 0.9, 0.05])
    norm = mpl.colors.Normalize(vmin=-1, vmax=1)
    cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, orientation='horizontal', ticks=[-1, 0, 1])
    cb1.ax.set_xticklabels(['negative', 'natural', 'positive'])
    cb1.set_label('Sentiment')

    plt.show()
