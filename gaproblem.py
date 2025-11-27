# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 15:27:15 2021

@author: Frank
"""
import geatpy as ea
import pandas as pd
import numpy as np
from operation import OperationModel
import multiprocessing as mp
from multiprocessing import Pool as ProcessPool
from multiprocessing.dummy import Pool as ThreadPool



class MyProblem(ea.Problem):  # 继承Problem父类
    def __init__(self, PoolType):
        operation_data = pd.read_csv('mergedData.csv')
        self.operation_list = np.array(operation_data).tolist()
        self.typical_days = dict()
        typical_data = pd.read_excel('typicalDayData.xlsx')
        typical_day_id = typical_data["typicalDayId"]
        days_str = typical_data["days"]
        for i in range(len(typical_day_id)):
            days_list = list(map(int, days_str[i].split(",")))
            self.typical_days[typical_day_id[i]] = days_list
        name = 'MyProblem'  # 初始化name（函数名称，可以随意设置）
        M = 2  # 初始化M（目标维数）
        Dim = 9  # 初始化Dim（决策变量维数）
        maxormins = [1] * M  # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        varTypes = [0] * Dim  # 初始化varTypes（决策变量的类型，0：实数；1：整数）
        lb = [0] * Dim  # 决策变量下界
        ub = [10000, 10000, 10000, 3000, 1000, 1000, 20000, 6000, 2000]  # 决策变量上界
        lbin = [1] * Dim  # 决策变量下边界（0表示不包含该变量的下边界，1表示包含）
        ubin = [1] * Dim  # 决策变量上边界（0表示不包含该变量的上边界，1表示包含）
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
        # 设置用多线程还是多进程
        self.PoolType = PoolType
        if self.PoolType == 'Thread':
            self.pool = ThreadPool(4)  # 设置池的大小
        elif self.PoolType == 'Process':
            num_cores = int(mp.cpu_count())  # 获得计算机的核心数
            print("num_cores:" + str(num_cores))
            self.pool = ProcessPool(num_cores)  # 设置池的大小

    def aimFunc(self, pop):  # 目标函数
        # 获取决策变量值
        Vars = pop.Phen  # 得到决策变量矩阵
        args = list(
            zip(
                list(range(pop.sizes)),
                [Vars] * pop.sizes,
                [self.operation_list] * pop.sizes,
                [self.typical_days] * pop.sizes
            )
        )
        if self.PoolType == 'Thread':
            pop.ObjV = np.array(list(self.pool.map(subAimFunc, args)))
        elif self.PoolType == 'Process':
            result = self.pool.map_async(subAimFunc, args)
            result.wait()
            pop.ObjV = np.array(result.get())

    def kill_pool(self):
        self.pool.close()


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


def subAimFunc(args):
    i = args[0]
    Vars = args[1]
    operation_list = args[2]
    typical_days = args[3]
    ppv = Vars[i, 0]  # 光伏额定功率
    pwt = Vars[i, 1]  # 风电额定功率
    pgt = Vars[i, 2]  # 燃气轮机额定功率
    php = Vars[i, 3]  # 电热泵额定功率
    pec = Vars[i, 4]  # 电制冷额定功率
    pac = Vars[i, 5]  # 吸收式制冷额定功率
    pes = Vars[i, 6]  # 电储能额定功率
    phs = Vars[i, 7]  # 热储能额定功率
    pcs = Vars[i, 8]  # 冷储能额定功率
    oc = 0
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
            oc += operation_model.get_objective_value() * len(typical_days[cluster_medoid])
            complementary_results = operation_model.get_complementary_results()
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

    # 计算上层模型目标函数值
    if is_success:
        # 经济目标
        economic_obj_i = 76.44188371 * ppv + 110.4233218 * pwt + 50.32074101 * pgt + 21.21527903 * php \
                         + 22.85563566 * pec + 21.81674313 * pac + 35.11456751 * pes + 1.689590459 * (phs + pcs) \
                         + 520 * (phs > 0.1) + 520 * (pcs > 0.1) + oc
        # 源荷匹配目标                 
        complementary_obj_i = np.std(net_ele_load) + np.std(net_heat_load) + np.std(net_cool_load)
    else:
        economic_obj_i = float('inf')
        complementary_obj_i = float('inf')
    print("[economic:%f] [complementary:%f] \n "
          "[ppv:%f] [pwt:%f] [pgt:%f] [php:%f] [pec:%f] [pac:%f] [pes:%f] [phs:%f] [pcs:%f]"
          % (economic_obj_i, complementary_obj_i, ppv, pwt, pgt, php, pec, pac, pes, phs, pcs))
    return [economic_obj_i, complementary_obj_i]