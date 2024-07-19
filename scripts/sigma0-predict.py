#!/usr/bin/env python3
import time
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from math import ceil
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
import joblib
import argparse

model = CatBoostClassifier()
model.load_model("../models/sigma0.bin")

def get_info_for_predict(dataset_path):
    dataframe = pd.read_csv(dataset_path)
    df = df.drop(["address", "Unnamed: 0", "uid", "Unnamed: 0.1", 'Unnamed: 0.2', 'year1', 'total_floors', 'fundament'], axis=1)
    data = dataframe.T.to_dict('dict')
    return data

def predict_build_year(price_pr, floor_pr, total_floors_pr, rooms_pr, area_pr, city_pr, remont_pr, balcon_pr, view_window_pr, row): 
    if year == '0' or year == '0.0' or year == 0.0 or year == 0 or year == 'empty':
        x = pd.DataFrame(columns=['price', 'floors', 'rooms', 'kitchen', 'city', 'square', 'type_perec', 'balcon'])
        result = model.predict(x_pred)
    result = result.reshape(1, -1)
    result = scaler.inverse_transform(result)
    result = ceil(result[0][0])  
    return result

def df_save(dataset_path):
    df.to_csv(dataset_path, index=False)

if __name__ == "__main__":
    year_list = []
    parser = argparse.ArgumentParser(description="Program for predict build year")
    parser.add_argument('-f', "--file", help="Path to target CSV file")
    parser.add_argument('-o', '--output', help="Path to output CSV file")
    parser.add_argument('-v', "--verbose", action='verbose_flag', help="Print verbose info")
    parser.parse_args()
    
    start_time = time.perf_counter()
    data = get_info_for_predict(parser.f)
    for i in range(len(data)):
        j = data[i]
        price_pr = j["price"]
        floor_pr = j["floor"]
        rooms_pr = j["rooms"]
        kitchen_pr = j["kitchen"]
        city_pr = j['city']
        square_pr = j["square"]
        type_perec_pr = j["type_perec"]
        balcon_pr = j["balcon"]
        year_pr = j['year']
        year_pr = predict_build_year(price_pr, floor_pr, rooms_pr, kitchen_pr, city_pr, square_pr, type_perec_pr, balcon_pr, model)
        j['year'] = year_pr
        
        if verbose_flag == True:
            print("price", price_pr) 
            print("floor", floor_pr)
            print("floors", total_floors_pr)
            print("rooms", rooms_pr)
            print("area", area_pr)
            print("city", city_pr)
            print("remont", remont_pr)
            print("balcon", balcon_pr)
            print("view_window", view_window_pr)
            print()
            print("Year", year_pr)
            print(j)
            print("-"*20)
            input()
            
        print("Working...", str(i+1) + "/" + str(len(data)), ceil((i+1)*100/len(data)), "%")
    df_save(parser.o)
    
    program_time = time.perf_counter() - start_time
    print("The program finished in", program_time, "seconds.")    
    print(*result_list)
