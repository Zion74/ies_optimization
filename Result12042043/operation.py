"""
综合能源系统调度
-------------------
园区能源拓扑结构：
                  input/output  ele_bus  heat_bus  cool_bus  gas_bus
                        |          |         |         |         |
 wt(FixedSource)        |--------->|         |         |         |
                        |          |         |         |         |
 pv(FixedSource)        |--------->|         |         |         |
                        |          |         |         |         |
 gas(Commodity)         |--------------------------------------->|
                        |          |         |         |         |
 grid(Commodity)        |--------->|         |         |         |
                        |          |         |         |         |
 ele_demand(Sink)       |<---------|         |         |         |
                        |          |         |         |         |
 heat_demand(Sink)      |<-------------------|         |         |
                        |          |         |         |         |
 cool_demand(Sink)      |<-----------------------------|         |
                        |          |         |         |         |
                        |<---------------------------------------|
 gt(Transformer)        |--------->|         |         |         |
                        |------------------->|         |         |
                        |          |         |         |         |
                        |<---------|         |         |         |
 ac(Transformer)        |<-------------------|         |         |
                        |----------------------------->|         |
                        |          |         |         |         |
 ehp(Transformer)       |<---------|         |         |         |
                        |------------------->|         |         |
                        |          |         |         |         |
 ec(Transformer)        |<---------|         |         |         |
                        |----------------------------->|         |
                        |          |         |         |         |
 es(Storage)            |<---------|         |         |         |
                        |--------->|         |         |         |
                        |          |         |         |         |
 hs(Storage)            |<-------------------|         |         |
                        |------------------->|         |         |
                        |          |         |         |         |
 cs(Storage)            |<-----------------------------|         |
                        |----------------------------->|         |
                  input/output  ele_bus  heat_bus  cool_bus  gas_bus
符号说明：
wt - wind turbine
pv - photovoltaic
gas - natural gas
grid - grid electricity
ele_demand - electricity demand
heat_demand - heat demand
cool_demand - cool demand
gt - gas turbine
ehp - electricity heat pump
ac - absorption chiller
ec - electricity chiller
es - electricity storage
hs - heat storage
cs - cool storage
"""
import logging
import numpy as np
import pandas as pd
import pprint as pp
import pyomo.environ as po
import oemof.solph as solph
import matplotlib.pyplot as plt


class OperationModel:
    # 初始化模型
    def __init__(self, local_time, time_step, ele_price, gas_price,
                 ele_load, heat_demand, cool_demand, wt_output, pv_output,
                 gt_capacity, ehp_capacity, ec_capacity, ac_capacity,
                 ele_storage_io, heat_storage_io, cool_storage_io):
        # 初始化能源系统模型
        logging.info("Initialize the energy system")
        self.date_time_index = pd.date_range(local_time, periods=time_step, freq="H")
        self.energy_system = solph.EnergySystem(timeindex=self.date_time_index)
        ##########################################################################
        # 创建能源系统设备对象
        ##########################################################################
        logging.info("Create oemof objects")
        # 创建电母线
        ele_bus = solph.Bus(label="electricity bus")
        # 创建热母线
        heat_bus = solph.Bus(label="heat bus")
        # 创建冷母线
        cool_bus = solph.Bus(label="cool bus")
        # 创建气母线
        gas_bus = solph.Bus(label="gas bus")
        # 将母线添加到模型中
        self.energy_system.add(ele_bus, heat_bus, cool_bus, gas_bus)
        # 添加电负荷
        self.energy_system.add(
            solph.Sink(
                label="electricity demand",
                inputs={ele_bus: solph.Flow(fix=ele_load, nominal_value=1)},
            )
        )
        # 添加热负荷
        self.energy_system.add(
            solph.Sink(
                label="heat demand",
                inputs={heat_bus: solph.Flow(fix=heat_demand, nominal_value=1)},
            )
        )
        # 添加冷负荷
        self.energy_system.add(
            solph.Sink(
                label="cool demand",
                inputs={cool_bus: solph.Flow(fix=cool_demand, nominal_value=1)},
            )
        )
        # 设置电源参数
        grid = solph.Source(
            label="grid",
            outputs={ele_bus: solph.Flow(nominal_value=10000000, variable_costs=ele_price)},
        )
        # 设置热源参数
        heat_source = solph.Source(
            label="heat source",
            outputs={heat_bus: solph.Flow(nominal_value=1000000, variable_costs=10000000)},
        )
        # 设置冷源参数
        cool_source = solph.Source(
            label="cool source",
            outputs={cool_bus: solph.Flow(nominal_value=1000000, variable_costs=10000000)},
        )
        # 设置气源参数
        gas = solph.Source(
            label="gas",
            outputs={gas_bus: solph.Flow(nominal_value=10000000, variable_costs=gas_price)},
        )
        # 设置风电机组参数
        wt = solph.Source(
            label="wind turbine",
            outputs={ele_bus: solph.Flow(fix=wt_output, nominal_value=1)},
        )
        # 设置光伏机组参数
        pv = solph.Source(
            label="photovoltaic",
            outputs={ele_bus: solph.Flow(fix=pv_output, nominal_value=1)},
        )
        # 设置燃气轮机参数
        gt = solph.Transformer(
            label='gas turbine',
            inputs={gas_bus: solph.Flow()},
            outputs={ele_bus: solph.Flow(nominal_value=gt_capacity),
                     heat_bus: solph.Flow(nominal_value=gt_capacity*1.5)},
            conversion_factors={ele_bus: 0.33, heat_bus: 0.5}
        )
        # 设置AC机组参数
        ac = solph.Transformer(
            label='absorption chiller',
            inputs={heat_bus: solph.Flow(), ele_bus: solph.Flow()},
            outputs={cool_bus: solph.Flow(nominal_value=ac_capacity)},
            conversion_factors={cool_bus: 0.75, heat_bus: 0.983, ele_bus: 0.017}
        )
        # 设置电热泵机组参数
        ehp = solph.Transformer(
            label='electricity heat pump',
            inputs={ele_bus: solph.Flow()},
            outputs={heat_bus: solph.Flow(nominal_value=ehp_capacity)},
            conversion_factors={heat_bus: 4.44}
        )
        # 设置电制冷机组参数
        ec = solph.Transformer(
            label='electricity chiller',
            inputs={ele_bus: solph.Flow()},
            outputs={cool_bus: solph.Flow(nominal_value=ec_capacity)},
            conversion_factors={cool_bus: 2.87}
        )
        # 设置多余电出口参数
        ele_overflow = solph.Sink(
            label="electricity overflow",
            inputs={ele_bus: solph.Flow(nominal_value=100000, variable_costs=0)},
        )
        # 设置多余热出口参数
        heat_overflow = solph.Sink(
            label="heat overflow",
            inputs={heat_bus: solph.Flow(nominal_value=100000, variable_costs=0)},
        )
        # 设置多余冷出口参数
        cool_overflow = solph.Sink(
            label="cool overflow",
            inputs={cool_bus: solph.Flow(nominal_value=100000, variable_costs=0)},
        )
        # 设置蓄电池机组参数
        ele_storage = solph.components.GenericStorage(
            nominal_storage_capacity=ele_storage_io*2,
            label="electricity storage",
            inputs={ele_bus: solph.Flow(nominal_value=ele_storage_io)},
            outputs={ele_bus: solph.Flow(nominal_value=ele_storage_io)},
            loss_rate=0.000125,
            initial_storage_level=None,
            inflow_conversion_factor=0.95,
            outflow_conversion_factor=0.90
        )
        # 设置蓄热罐参数
        heat_storage = solph.components.GenericStorage(
            nominal_storage_capacity=heat_storage_io*4/3,
            label="heat storage",
            inputs={heat_bus: solph.Flow(nominal_value=heat_storage_io)},
            outputs={heat_bus: solph.Flow(nominal_value=heat_storage_io)},
            loss_rate=0.001,
            initial_storage_level=None,
            inflow_conversion_factor=0.9,
            outflow_conversion_factor=0.9,
        )
        # 设置蓄冷罐参数
        cool_storage = solph.components.GenericStorage(
            nominal_storage_capacity=cool_storage_io*4/3,
            label="cool storage",
            inputs={cool_bus: solph.Flow(nominal_value=cool_storage_io)},
            outputs={cool_bus: solph.Flow(nominal_value=cool_storage_io)},
            loss_rate=0.001,
            initial_storage_level=None,
            inflow_conversion_factor=0.9,
            outflow_conversion_factor=0.9,
        )
        # 将以上设备添加到系统中
        self.energy_system.add(wt, pv, gt, grid, ac, ehp, ec, gas, heat_source, cool_source,
                          ele_overflow, heat_overflow, cool_overflow,
                          ele_storage, heat_storage, cool_storage)
        # 初始化模型
        model = solph.Model(self.energy_system)
        # # 创建求解计算电网电功率最大值的子模型
        # max_ele_load_block = po.Block()
        # # 将子模型添加到主模型中
        # model.add_component("max_ele_load_block", max_ele_load_block)
        # # 设置变量
        # model.max_ele_load_block.max_load = po.Var(model.TIMESTEPS, domain=po.NonNegativeReals)
        # model.max_ele_load_block.max_load_upper_switch = po.Var(model.TIMESTEPS, within=po.Binary)
        # big_number = 100000000
        #
        # # 设置计算功率下限约束规则
        # def max_ele_load_lower(m, s, e, t):
        #     return model.max_ele_load_block.max_load[0] >= model.flow[s, e, t]
        #
        # # 设置计算功率上限约束规则
        # def max_ele_load_upper(m, s, e, t):
        #     return model.max_ele_load_block.max_load[0] <= \
        #            model.flow[s, e, t] + big_number * (1 - model.max_ele_load_block.max_load_upper_switch[t])
        #
        # # 添加求解计算功率最大值的约束
        # model.max_ele_load_block.max_load_lower_constr = po.Constraint(
        #     [(grid, ele_bus)], model.TIMESTEPS, rule=max_ele_load_lower)
        # model.max_ele_load_block.max_load_upper_constr = po.Constraint(
        #     [(grid, ele_bus)], model.TIMESTEPS, rule=max_ele_load_upper)
        # model.max_ele_load_block.max_load_upper_switch_constr = po.Constraint(
        #     rule=(sum(model.max_ele_load_block.max_load_upper_switch[t] for t in model.TIMESTEPS) >= 1))
        #
        # # 定义考虑容量费的目标函数
        # objective_expr = 0
        # for t in model.TIMESTEPS:
        #     objective_expr += (model.flows[gas, gas_bus].variable_costs[t] * model.flow[gas, gas_bus, t]
        #                        + model.flows[grid, ele_bus].variable_costs[t] * model.flow[grid, ele_bus, t]
        #                        + model.flows[heat_source, heat_bus].variable_costs[t] * model.flow[heat_source, heat_bus, t]
        #                        + model.flows[cool_source, cool_bus].variable_costs[t] * model.flow[cool_source, cool_bus, t])
        # objective_expr += 16.7 * model.max_ele_load_block.max_load[0]
        # model.del_component('objective')
        # model.objective = po.Objective(expr=objective_expr)
        self.model = model

    # 模型优化与储存
    def optimise(self):
        solver = "glpk"  # 选择求解器
        solver_verbose = False  # 是否输出求解器信息
        self.model.solve(solver=solver, solve_kwargs={"tee": solver_verbose})
        self.energy_system.results["main"] = solph.processing.results(self.model)
        self.energy_system.results["meta"] = solph.processing.meta_results(self.model)

    # 返回优化结果
    def get_objective_value(self):
        return self.energy_system.results["meta"]["objective"]

    # 返回设备出力数据
    def get_complementary_results(self):
        complementary_results = dict()
        results = self.energy_system.results["main"]
        symbols = ["grid", "electricity overflow", "heat source",
                   "heat overflow", "cool source", "cool overflow"]
        for symbol in symbols:
            node = solph.views.node(results, symbol)
            flows = node["sequences"].columns
            flow_list = []
            for f in flows:
                flow_list = np.array(node["sequences"][f]).tolist()
            complementary_results[symbol] = flow_list
        return complementary_results

    # 备份结果
    def dump_result(self):
        # 保存结果
        logging.info("Store the energy system with the results.")
        self.energy_system.dump(dpath=self.log_path, filename=None)

    # 结果展示
    def result_process(self):
        results = self.energy_system.results["main"]
        # 获取需要展示的节点
        ele_bus = solph.views.node(results, "electricity bus")
        ele_storage = solph.views.node(results, "electricity storage")
        # 绘制电母线输入输出图像
        ele_flows = ele_bus["sequences"].columns
        bottom1 = [0] * len(self.date_time_index)
        bottom2 = [0] * len(self.date_time_index)
        fig, ax = plt.subplots(figsize=(8, 5))
        for flow in ele_flows:
            if flow[0][0] != 'electricity bus':
                ax.bar(self.date_time_index, ele_bus["sequences"][flow],
                       0.03, bottom=bottom1, label=flow[0][0] + ' to ' + flow[0][1])
                bottom1 = [(a + b) for a, b in zip(ele_bus["sequences"][flow], bottom1)]
            else:
                bottom2 = [(a - b) for a, b in zip(bottom2, ele_bus["sequences"][flow])]
                ax.bar(self.date_time_index, ele_bus["sequences"][flow],
                       0.03, bottom=bottom2, label=flow[0][0] + ' to ' + flow[0][1])
        plt.legend(
            loc="upper center",
            prop={"size": 7},
            bbox_to_anchor=(0.5, 1.25),
            ncol=3,
        )
        ax.set_yticks(np.linspace(-900, 900, 13, endpoint=True))
        plt.xlabel('t/h')
        plt.ylabel('P/kW')
        plt.show()
        # 绘制蓄电池机组状态曲线
        fig, ax = plt.subplots(figsize=(8, 5))
        ele_storage["sequences"].plot(
            ax=ax, kind="line", drawstyle="steps-post"
        )
        plt.legend(
            loc="upper center",
            prop={"size": 8},
            bbox_to_anchor=(0.5, 1.15),
            ncol=2,
        )
        plt.xlabel('t/h')
        plt.ylabel('P/kW')
        plt.show()
        # 输出求解结果
        print("********* Meta results *********")
        pp.pprint(self.energy_system.results["meta"])
        print("")
        # 输出各个母线的输入输出总量
        print("********* Main results *********")
        print(ele_bus["sequences"].sum(axis=0))
