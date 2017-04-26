# cap_functs.py

import numpy as np
import pandas as pd
import re
from sklearn.neighbors import KDTree


def rmse(y_pred, y_true):
    leng = len(y_pred)
    if leng == len(y_true):
        return np.sqrt((1/float(leng)) * np.sum((y_true-y_pred)**2))


def mae(y_pred, y_true):
    leng = len(y_pred)
    if leng == len(y_true):
        return (1/float(leng)) * np.sum(np.abs(y_true-y_pred))
 

def percent_within_x(y_pred, y_test, x):
    err_arr = 100.*(y_pred-y_test)/y_test
    tmp = []
    for num in err_arr:
        if np.abs(num) < x:
            tmp.append(num)
    return 100.*(len(tmp)/float(len(err_arr)))


def median_error(y_pred, y_test):
    err_arr = 100.*(y_pred-y_test)/y_test
    return np.median(np.abs(err_arr))


def print_percents(y_pred, y_true):
    print "Percent within 5 of price:   %0.3f"%(percent_within_x(y_pred, y_true, 5))
    print "Percent within 10 of price:  %0.3f"%(percent_within_x(y_pred, y_true, 10))
    print "Percent within 20 of price:  %0.3f"%(percent_within_x(y_pred, y_true, 20))
    print ""
    print "Median error (percent):      %0.3f"%(median_error(y_pred, y_true))
    print ""


def print_percents_log(y_pred_l, y_true_l):
    pred_tmp, true_tmp = np.expm1(y_pred_l), np.expm1(y_true_l)
    print "Percent within 5 of price:   %0.3f"%(percent_within_x(pred_tmp, true_tmp, 5))
    print "Percent within 10 of price:  %0.3f"%(percent_within_x(pred_tmp, true_tmp, 10))
    print "Percent within 20 of price:  %0.3f"%(percent_within_x(pred_tmp, true_tmp, 20))
    print ""
    print "Median error (percent):      %0.3f"%(median_error(pred_tmp, true_tmp))


def circle(lat_cent, long_cent, radius, pts):
    x = []
    y = []
    pi_conv = 180/math.pi
    for i in list(np.linspace(0,1, pts)):
        angle = i*(math.pi*2)
        x.append((long_cent + radius*math.cos(angle)))
        y.append((lat_cent + radius*math.sin(angle)))
    return x, y    


def latlong_pt_matrices(cat, cat_df):
    tmp = cat_df[cat_df.Category == cat]
    return tmp[['longitude', 'latitude']].as_matrix()


# use function below if points are given as an array of coordinates
def find_neigh_cnt(col_name, X_pts, centers_df, r=0.0045, leaf_size=2):
    tmp = centers_df.loc[:, ['longitude', 'latitude']].as_matrix()
    tree = KDTree(X_pts, leaf_size)     
    cnts_per_center = tree.query_radius(tmp, r, count_only=True)
    centers_df[col_name] = cnts_per_center
    return centers_df


# use function below if points are within a dataframe and associated with different categories
def find_cnts_per_cat(list_of_cats, cat_df, cat_col_name, centers_df, r=0.0045, leaf_size=2):
    '''
    list_of_cats = list of categories as strings
    cat_df =       dataframe that contains some type of category column
    cat_col_name = name of category column within cat_df
    centers__df =  destination df to add counts to
    r =            radius size, appropriate to search area unit
    '''
    for cat in list_of_cats:
        tmp_cat = cat_df[cat_df[cat_col_name] == cat]
        cat_pts = tmp_cat[['longitude', 'latitude']].as_matrix()
        tmp_cent = centers_df.loc[:, ['longitude', 'latitude']].as_matrix()
        tree = KDTree(cat_pts, leaf_size)     
        cnts_per_center = tree.query_radius(tmp_cent, r, count_only=True)
        centers_df[cat+'_cnt'] = cnts_per_center
    return centers_df


# use this function to strip out Lat/Long boundary points from the polygon strings
def convert_mp_vals(df, column):
    all_pts = []
    array = df[column].values
    for line in array:
        str_pts = re.findall(r'-122\.\d+ 37\.\d+', line)
        for coords in str_pts:
            tmp = coords.split(' ')
            tmp = map(float, tmp)
            all_pts.append(tmp)
    return np.array(all_pts)


def compile_model_data(keyword, df_start, cats_311, cats_crime, df_crime, df_311,
    r_crime=0.002, r_311=0.002):
    '''This is used specifially for adding points for given categories of the 
    crime and 311 datasets. 
    '''

    df_mod = pd.DataFrame(df_start)
    
    if keyword == 'crime':
        return find_cnts_per_cat(cats_crime, df_crime, 'Category',
                                 df_mod, r=r_crime)
        
    elif keyword == '311':
        if 'Graffiti' in cats_311:
            cats_311.remove('Graffiti')
            df_mod = find_cnts_per_cat(cats_311, df_311, 'Category', 
                                       df_mod, r=r_311)
            rec_graff = df_311[(df_311.Category == 'Graffiti Public Property') | 
                                 (df_311.Category == 'Graffiti Private Property')]
            rec_graff_pts = rec_graff[['longitude', 'latitude']].as_matrix()
            return find_neigh_cnt('graffiti_cnt', rec_graff_pts, df_mod, r=r_311)
        else:
            return find_cnts_per_cat(cats_311, df_311, 'Category', 
                                     df_mod, r=r_311)
    elif keyword == 'crime and 311':
        df_mod = find_cnts_per_cat(cats_crime, df_crime, 'Category', df_mod, r=r_crime)
        if 'Graffiti' in cats_311:
            cats_311.remove('Graffiti')
            df_mod = find_cnts_per_cat(cats_311, df_311, 'Category', 
                                       df_mod, r=r_311)
            rec_graff = df_311[(df_311.Category == 'Graffiti Public Property') | 
                                 (df_311.Category == 'Graffiti Private Property')]
            rec_graff_pts = rec_graff[['longitude', 'latitude']].as_matrix()
            return find_neigh_cnt('graffiti_cnt', rec_graff_pts, df_mod, r=r_311)
        else:
            return find_cnts_per_cat(cats_311, df_311, 'Category', 
                                     df_mod, r=r_311)
    else:
        return 'Parameters wrong...'

