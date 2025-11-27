import pandas as pd
import numpy as np
from operation import OperationModel


def cal_solar_output(solar_radiation_list, temperature_list, ppv):
    return [ppv * 0.9 * r / 1000 * (1 - 0.0035 * (t - 25)) for r, t in zip(solar_radiation_list, temperature_list)]


def cal_wind_output(wind_speed_list, pwt):
    ret = [0 for _ in range(len(wind_speed_list))]
    for i in range(len(wind_speed_list)):
        w = wind_speed_list[i]
        if 2.5 <= w < 9:
            ret[i] = (w ** 3 - 2.5 ** 3) / (9 ** 3 - 2.5 ** 3) * pwt
        elif 9 <= w < 25:
            ret[i] = pwt
    return ret


operation_data = pd.read_csv('mergedData.csv')
operation_list = np.array(operation_data).tolist()
typical_days = dict()
typical_data = pd.read_excel('typicalDayData.xlsx')
typical_day_id = typical_data["typicalDayId"]
days_str = typical_data["days"]
for i in range(len(typical_day_id)):
    days_list = list(map(int, days_str[i].split(",")))
    typical_days[typical_day_id[i]] = days_list

ppv = 1710.86   # 光伏额定功率
pwt = 1648.98   # 风电额定功率
pgt = 2217.91   # 燃气轮机额定功率
php = 2.79      # 电热泵额定功率
pec = 5.17      # 电制冷额定功率
pac = 305.72    # 吸收式制冷额定功率
pes = 0.04      # 电储能额定功率
phs = 2351.50   # 热储能额定功率
pcs = 400.82    # 冷储能额定功率
net_ele_load = [0 for _ in range(8760)]  # 电净负荷
net_heat_load = [0 for _ in range(8760)]  # 热净负荷
net_cool_load = [0 for _ in range(8760)]  # 冷净负荷
time_step = 24
ele_price = [0.1598 for _ in range(time_step)]
gas_price = [0.0286 for _ in range(time_step)]
ele_load = [0 for _ in range(time_step)]
heat_load = [0 for _ in range(time_step)]
cool_load = [0 for _ in range(time_step)]
solar_radiation_list = [0 for _ in range(time_step)]
wind_speed_list = [0 for _ in range(time_step)]
temperature_list = [0 for _ in range(time_step)]
is_success = True
for cluster_medoid in typical_days.keys():
    time_start = (cluster_medoid - 1) * 24
    # 下层模型参数设置
    for t in range(time_start, time_start + time_step):
        ele_load[t % 24] = operation_list[t][0]
        heat_load[t % 24] = operation_list[t][1]
        cool_load[t % 24] = operation_list[t][2]
        solar_radiation_list[t % 24] = operation_list[t][3]
        wind_speed_list[t % 24] = operation_list[t][4]
        temperature_list[t % 24] = operation_list[t][5]
    pv_output = cal_solar_output(solar_radiation_list, temperature_list, ppv)
    wt_output = cal_wind_output(wind_speed_list, pwt)
    # 底层模型初始化及优化
    operation_model = OperationModel('01/01/2019', time_step, ele_price, gas_price,
                                     ele_load, heat_load, cool_load, wt_output, pv_output,
                                     pgt, php, pec, pac, pes, phs, pcs)
    try:
        # 优化并获取结果
        operation_model.optimise()
        complementary_results = operation_model.get_complementary_results()
        operation_model.result_process("electricity bus")
    except Exception:
        is_success = False
        break
    for d in typical_days[cluster_medoid]:
        start_index = (d - 1) * 24
        for i in range(24):
            if complementary_results["grid"][i] >= complementary_results["electricity overflow"][i]:
                net_ele_load[start_index + i] = complementary_results["grid"][i]
            else:
                net_ele_load[start_index + i] = 0 - complementary_results["electricity overflow"][i]
            if complementary_results["heat source"][i] >= complementary_results["heat overflow"][i]:
                net_heat_load[start_index + i] = complementary_results["heat source"][i]
            else:
                net_heat_load[start_index + i] = 0 - complementary_results["heat overflow"][i]
            if complementary_results["cool source"][i] >= complementary_results["cool overflow"][i]:
                net_cool_load[start_index + i] = complementary_results["cool source"][i]
            else:
                net_cool_load[start_index + i] = 0 - complementary_results["cool overflow"][i]