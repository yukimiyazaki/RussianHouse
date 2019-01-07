import numpy as np
import pandas as pd
from keras.utils import np_utils


#header=0をすると、0行目をヘッダとして読み込んでくれる
macro_data = pd.read_csv('macro.csv', header=0)
#列名の抽出
#    macro_data.columns
'''
Index(['timestamp', 'oil_urals', 'gdp_quart', 'gdp_quart_growth', 'cpi', 'ppi',
   'gdp_deflator', 'balance_trade', 'balance_trade_growth', 'usdrub',
   'eurrub', 'brent', 'net_capital_export', 'gdp_annual',
   'gdp_annual_growth', 'average_provision_of_build_contract',
   'average_provision_of_build_contract_moscow', 'rts', 'micex',
   'micex_rgbi_tr', 'micex_cbi_tr', 'deposits_value', 'deposits_growth',
   'deposits_rate', 'mortgage_value', 'mortgage_growth', 'mortgage_rate',
   'grp', 'grp_growth', 'income_per_cap',
   'real_dispos_income_per_cap_growth', 'salary', 'salary_growth',
   'fixed_basket', 'retail_trade_turnover',
   'retail_trade_turnover_per_cap', 'retail_trade_turnover_growth',
   'labor_force', 'unemployment', 'employment',
   'invest_fixed_capital_per_cap', 'invest_fixed_assets',
   'profitable_enterpr_share', 'unprofitable_enterpr_share',
   'share_own_revenues', 'overdue_wages_per_cap', 'fin_res_per_cap',
   'marriages_per_1000_cap', 'divorce_rate', 'construction_value',
   'invest_fixed_assets_phys', 'pop_natural_increase', 'pop_migration',
   'pop_total_inc', 'childbirth', 'mortality', 'housing_fund_sqm',
   'lodging_sqm_per_cap', 'water_pipes_share', 'baths_share',
   'sewerage_share', 'gas_share', 'hot_water_share',
   'electric_stove_share', 'heating_share', 'old_house_share',
   'average_life_exp', 'infant_mortarity_per_1000_cap',
   'perinatal_mort_per_1000_cap', 'incidence_population',
   'rent_price_4+room_bus', 'rent_price_3room_bus', 'rent_price_2room_bus',
   'rent_price_1room_bus', 'rent_price_3room_eco', 'rent_price_2room_eco',
   'rent_price_1room_eco', 'load_of_teachers_preschool_per_teacher',
   'child_on_acc_pre_school', 'load_of_teachers_school_per_teacher',
   'students_state_oneshift', 'modern_education_share',
   'old_education_build_share', 'provision_doctors', 'provision_nurse',
   'load_on_doctors', 'power_clinics', 'hospital_beds_available_per_cap',
   'hospital_bed_occupancy_per_year', 'provision_retail_space_sqm',
   'provision_retail_space_modern_sqm', 'turnover_catering_per_cap',
   'theaters_viewers_per_1000_cap', 'seats_theather_rfmin_per_100000_cap',
   'museum_visitis_per_100_cap', 'bandwidth_sports',
   'population_reg_sports_share', 'students_reg_sports_share',
   'apartment_build', 'apartment_fund_sqm'],
  dtype='object')
'''
train_data = pd.read_csv('train.csv', header=0)
#train_datemp.indexta.columns
#non_categoricalな列と、categoricalな列に分ける
#NAは現段階で考慮しない
categorical=[11,12,29,33,34,35,36,37,38,39,40,106,114,118,152,]
#non_categorical=np.arange(292)
non_categorical =[0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30,31,32,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,107,108,109,110,111,112,113,115,116,117,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291]

#non_categoricalの列を取り出す
train_data[non_categorical]
#NaNを-1に置換する
#sample: df.replace('0', np.nan)
new_non_categorical = train_data[non_categorical].replace(np.nan,'-1')
temp_categorical = (train_data[categorical]).replace(np.nan,'-1')

#pd.get_dummiesでcategoricalに変換する
#categoriesを抽出する
temp_categorical_lists = train_data[categorical].columns
temp_categorical_dummy = pd.get_dummies(temp_categorical[temp_categorical_lists])
#dummyの時点でo
##new_categorical =np_utils.to_categorical(temp_categorical_dummy)

#save csv files
new_non_categorical.to_csv('train_new_non_categorical_train.csv')
temp_categorical_dummy.to_csv('train_new_categorical_dummy_train.csv')

####convert test data
test_data = pd.read_csv('test.csv', header=0)
non_categorical_test =[0,1,2,3,4,5,6,7,8,9,10,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,30,31,32,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,107,108,109,110,111,112,113,115,116,117,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,134,135,136,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,153,154,155,156,157,158,159,160,161,162,163,164,165,166,167,168,169,170,171,172,173,174,175,176,177,178,179,180,181,182,183,184,185,186,187,188,189,190,191,192,193,194,195,196,197,198,199,200,201,202,203,204,205,206,207,208,209,210,211,212,213,214,215,216,217,218,219,220,221,222,223,224,225,226,227,228,229,230,231,232,233,234,235,236,237,238,239,240,241,242,243,244,245,246,247,248,249,250,251,252,253,254,255,256,257,258,259,260,261,262,263,264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,]
new_non_categorical_test = test_data[non_categorical_test].replace(np.nan,'-1')
temp_categorical_test = (test_data[categorical]).replace(np.nan,'-1')
temp_categorical_lists_test = test_data[categorical].columns
temp_categorical_dummy_test = pd.get_dummies(temp_categorical_test[temp_categorical_lists_test])
new_non_categorical_test.to_csv('test_new_non_categorical.csv')
temp_categorical_dummy_test.to_csv('test_new_categorical_dummy.csv')