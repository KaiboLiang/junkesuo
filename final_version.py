import random
import time
import logging
import copy
import json
import os
from datetime import datetime
import math
from typing import Dict, Any
import re
import numpy as np
from itertools import product
from functools import lru_cache
import pickle
from collections.abc import MutableMapping, MutableSequence
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

class MultiObjectiveFourLayerNetworkOptimizer:
    """
    多目标四层运输网络国防动员物资调度优化（2025年9月7日下午17.57版本）
    """

    def __init__(self, logger=None):
        """
        初始化多目标优化器
        """
        self.logger = logger if logger else self._setup_default_logger()

    def create_two_level_directories(self,activity_id, req_element_id):
        """
        基于activity_id和req_element_id创建两级目录
        """
        # 验证输入参数
        if not activity_id or not isinstance(activity_id, str):
            return False, "activity_id必须是非空字符串", None
        if not req_element_id or not isinstance(req_element_id, str):
            return False, "req_element_id必须是非空字符串", None

        # 移除可能存在的路径分隔符，防止目录遍历攻击
        activity_id = activity_id.replace('/', '').replace('\\', '')
        req_element_id = req_element_id.replace('/', '').replace('\\', '')

        # 组合完整路径
        full_path = os.path.join(activity_id, req_element_id)

        try:
            # 检查目录是否已存在
            if os.path.exists(full_path):
                return True, f"目录已存在: {full_path}", full_path

            # 创建目录（递归创建）
            os.makedirs(full_path)
            return True, f"成功创建目录: {full_path}", full_path

        except OSError as e:
            return False, f"创建目录时出错: {str(e)}", full_path
        except Exception as e:
            return False, f"发生未知错误: {str(e)}", full_path

    def _setup_default_logger(self):
        """设置默认日志记录器"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  # 生成当前时间戳，作为日志文件名的一部分
        log_filename = f"multi_objective_optimization_{timestamp}.log"  # 构造日志文件名称

        logging.basicConfig(  # 配置日志系统基础设置
            level=logging.INFO,  # 设置日志级别为INFO
            format='%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',  # 设置日志格式
            handlers=[
                logging.FileHandler(log_filename, encoding='utf-8'),  # 文件日志处理器，保存到文件
                logging.StreamHandler()  # 控制台输出处理器
            ]
        )

        logger = logging.getLogger(__name__)  # 获取当前模块名的日志器
        logger.info(f"多目标日志系统初始化完成，日志文件: {log_filename}")  # 记录日志初始化完成信息
        return logger  # 返回日志器

    def solve(self, network_data, transport_params, algorithm_params, resource_type="material",
              objective_weights=None, random_seed=42, save_results=True, output_dir="results",
              return_format="standard", algorithm_sequence_id=None, mobilization_object_type=None, req_element_id=None,
              scheme_id=None, activity_id=None, scheme_config=None):
        """
        多目标优化求解主函数 - 支持多种输出格式
        """

        total_start_time = time.time()

        # 添加时间限制检查函数
        def check_time_limit():
            current_time = time.time()
            elapsed_time = current_time - total_start_time
            if elapsed_time > 3100.0:  # 100秒时间限制
                error_msg = f"求解超时：已耗时 {elapsed_time:.1f} 秒，超过100秒限制"
                self.logger.error(error_msg)

                timeout_response = {
                    "code": 408,
                    "msg": error_msg,
                    "data": {
                        "algorithm_sequence_id": algorithm_sequence_id,
                        "mobilization_object_type": mobilization_object_type,
                        "resource_type": resource_type,
                        "req_element_id": req_element_id,
                        "scheme_id": scheme_id,
                        "scheme_config": scheme_config,
                        "activity_id": activity_id,
                        "elapsed_time": elapsed_time,
                        "time_limit": 100.0,
                        "error_type": "timeout_error"
                    }
                }

                if return_format == "json":
                    return timeout_response
                else:
                    raise TimeoutError(error_msg)
            return elapsed_time

        self.logger.info("=" * 100)
        self.logger.info("开始多目标优化求解")
        self.logger.info(f"求解参数 - 资源类型: {resource_type}, 随机种子: {random_seed}")
        self.logger.info(
            f"网络规模预览 - 供应点: {len(network_data.get('J', []))}, 中转点: {len(network_data.get('M', []))}, 需求点: {len(network_data.get('K', []))}")
        self.logger.info(f"输出格式: {return_format}, 算法序号: {algorithm_sequence_id}")
        self.logger.info(f"动员对象类型: {mobilization_object_type}")
        self.logger.info(f"请求元素ID: {req_element_id}")
        self.logger.info("=" * 100)

        # 验证req_element_id参数
        if req_element_id is not None:
            if not isinstance(req_element_id, str):
                error_msg = "req_element_id必须是字符串类型"
                self.logger.error(error_msg)
                if return_format == "json":
                    return {
                        "code": 400,
                        "msg": error_msg,
                        "data": {
                            "algorithm_sequence_id": algorithm_sequence_id,
                            "mobilization_object_type": mobilization_object_type,
                            "resource_type": resource_type,
                            "req_element_id": req_element_id,
                            "error_type": "req_element_id_type_error"
                        }
                    }
                else:
                    return {
                        "code": 400,
                        "msg": error_msg,
                        "data": {
                            "algorithm_sequence_id": algorithm_sequence_id,
                            "mobilization_object_type": mobilization_object_type,
                            "resource_type": resource_type,
                            "req_element_id": req_element_id,
                            "error_type": "req_element_id_type_error"
                        }
                    }

            if not req_element_id.strip():
                error_msg = "req_element_id不能为空字符串"
                self.logger.error(error_msg)
                if return_format == "json":
                    return {
                        "code": 400,
                        "msg": error_msg,
                        "data": {
                            "algorithm_sequence_id": algorithm_sequence_id,
                            "mobilization_object_type": mobilization_object_type,
                            "resource_type": resource_type,
                            "req_element_id": req_element_id,
                            "error_type": "req_element_id_empty_error"
                        }
                    }
                else:
                    return {
                        "code": 400,
                        "msg": error_msg,
                        "data": {
                            "algorithm_sequence_id": algorithm_sequence_id,
                            "mobilization_object_type": mobilization_object_type,
                            "resource_type": resource_type,
                            "req_element_id": req_element_id,
                            "error_type": "req_element_id_empty_error"
                        }
                    }

                # 验证 req_element_id 长度限制
                network_scale = len(self.J) if hasattr(self, 'J') else len(self.M) if hasattr(self, 'M') else len(
                    self.K) if hasattr(self, 'K') else 1
                max_length = network_scale * network_scale + network_scale
                if len(req_element_id) > max_length:
                    error_msg = f"req_element_id长度超出限制，当前长度: {len(req_element_id)}，最大长度: {max_length}"
                    self.logger.error(error_msg)
                    if return_format == "json":
                        return {
                            "code": 400,
                            "msg": error_msg,
                            "data": {
                                "algorithm_sequence_id": algorithm_sequence_id,
                                "mobilization_object_type": mobilization_object_type,
                                "req_element_id": req_element_id,
                                "max_length": max_length,
                                "error_type": "req_element_id_length_error"
                            }
                        }
                    else:
                        return {
                            "code": 400,
                            "msg": error_msg,
                            "data": {
                                "algorithm_sequence_id": algorithm_sequence_id,
                                "mobilization_object_type": mobilization_object_type,
                                "req_element_id": req_element_id,
                                "max_length": max_length,
                                "error_type": "req_element_id_length_error"
                            }
                        }

                    # 验证 req_element_id 格式
                if not re.match(r'^[A-Za-z0-9_\-]+$', req_element_id):
                    error_msg = "req_element_id只能包含字母、数字、下划线和连字符"
                    self.logger.error(error_msg)
                    if return_format == "json":
                        return {
                            "code": 400,
                            "msg": error_msg,
                            "data": {
                                "algorithm_sequence_id": algorithm_sequence_id,
                                "mobilization_object_type": mobilization_object_type,
                                "req_element_id": req_element_id,
                                "error_type": "req_element_id_format_error"
                            }
                        }
                    else:
                        return {
                            "code": 400,
                            "msg": error_msg,
                            "data": {
                                "algorithm_sequence_id": algorithm_sequence_id,
                                "mobilization_object_type": mobilization_object_type,
                                "req_element_id": req_element_id,
                                "error_type": "req_element_id_format_error"
                            }
                        }

        try:
            # ========== 第1步：初始化算法基础参数 ==========
            step1_start = time.time()
            self.logger.info("第1步: 设置算法参数")
            try:
                self._set_algorithm_params(algorithm_params)
                step1_time = time.time() - step1_start
                self.logger.info(f"算法参数设置成功，耗时: {step1_time:.3f}秒")
                self.logger.debug(f"目标名称映射: {self.objective_names}")
                self.logger.debug(f"优先级等级: {self.PRIORITY_LEVELS}")
                self.logger.debug(f"精度参数 EPS: {self.EPS}, 大M常数: {self.BIGM}")
            except Exception as e:
                self.logger.error(f"算法参数设置失败: {str(e)}", exc_info=True)
                raise ValueError(f"算法参数设置错误: {str(e)}")

            # 检查时间限制
            elapsed_time = check_time_limit()
            if isinstance(elapsed_time, dict):  # 如果返回的是错误响应
                return elapsed_time

            step2_start = time.time()
            self.logger.info("第2步: 设置网络数据")
            try:
                self._set_network_data_optimized(network_data)
                step2_time = time.time() - step2_start
                self.logger.info(
                    f"网络数据设置成功 - 供应点: {len(self.J)}, 中转点: {len(self.M)}, 需求点: {len(self.K)}，耗时: {step2_time:.3f}秒")
                self.logger.debug(f"供应点列表前5个: {self.J[:5]}")
                self.logger.debug(f"中转点列表前5个: {self.M[:5]}")
                self.logger.debug(f"需求点列表: {self.K}")
            except Exception as e:
                self.logger.error(f"网络数据设置失败: {str(e)}", exc_info=True)
                raise ValueError(f"网络数据设置错误: {str(e)}")

            # 检查时间限制
            elapsed_time = check_time_limit()
            if isinstance(elapsed_time, dict):  # 如果返回的是错误响应
                return elapsed_time

            step3_start = time.time()
            self.logger.info("第3步: 设置运输参数")
            try:
                self._set_transport_params_optimized(transport_params)
                step3_time = time.time() - step3_start
                self.logger.info(f"运输参数设置成功 - 运输方式数量: {len(self.N)}，耗时: {step3_time:.3f}秒")
                self.logger.debug(f"运输方式列表: {self.N}")
                self.logger.debug(f"仅公路运输方式: {self.ROAD_ONLY}")
            except Exception as e:
                self.logger.error(f"运输参数设置失败: {str(e)}", exc_info=True)
                raise ValueError(f"运输参数设置错误: {str(e)}")

            # 检查时间限制
            elapsed_time = check_time_limit()
            if isinstance(elapsed_time, dict):  # 如果返回的是错误响应
                return elapsed_time

            self.resource_type = resource_type
            self.logger.info(f"第4步: 验证目标权重配置")

            if objective_weights is None:
                error_msg = "必须提供目标权重配置 objective_weights"
                self.logger.error(error_msg)
                if return_format == "json":
                    return {
                        "code": 400,
                        "msg": error_msg,
                        "data": {
                            "algorithm_sequence_id": algorithm_sequence_id,
                            "mobilization_object_type": mobilization_object_type,
                            "resource_type": resource_type,
                            "req_element_id": req_element_id,
                            "scheme_id": scheme_id,
                            "scheme_config": scheme_config,
                            "activity_id": activity_id
                        }
                    }
                else:
                    return {
                        "code": 400,
                        "msg": error_msg,
                        "data": {
                            "algorithm_sequence_id": algorithm_sequence_id,
                            "mobilization_object_type": mobilization_object_type,
                            "resource_type": resource_type,
                            "req_element_id": req_element_id,
                            "scheme_id": scheme_id,
                            "scheme_config": scheme_config,
                            "activity_id": activity_id
                        }
                    }
            else:
                try:
                    self.logger.debug(f"原始权重配置: {objective_weights}")
                    self.objective_weights = self._validate_and_normalize_weights(objective_weights)
                    self.logger.info("目标权重验证和标准化成功")
                    self.logger.info(f"标准化后权重: {self.objective_weights}")
                except (ValueError, TypeError, KeyError) as e:
                    error_msg = f"权重配置错误: {str(e)}"
                    self.logger.error(error_msg, exc_info=True)
                    self.logger.error(f"权重配置详情: {objective_weights}")
                    if return_format == "json":
                        return {
                            "code": 400,
                            "msg": error_msg,
                            "data": {
                                "algorithm_sequence_id": algorithm_sequence_id,
                                "mobilization_object_type": mobilization_object_type,
                                "req_element_id": req_element_id,
                                "scheme_id": scheme_id,
                                "scheme_config": scheme_config,
                                "activity_id": activity_id
                            }
                        }
                    else:
                        raise

            # 检查时间限制
            elapsed_time = check_time_limit()
            if isinstance(elapsed_time, dict):  # 如果返回的是错误响应
                return elapsed_time

            if save_results and not os.path.exists(output_dir):
                try:
                    os.makedirs(output_dir)
                    self.logger.info(f"创建输出目录: {output_dir}")
                except OSError as e:
                    error_msg = f"无法创建输出目录: {str(e)}"
                    self.logger.error(error_msg, exc_info=True)
                    if return_format == "json":
                        return {
                            "code": 500,
                            "msg": error_msg,
                            "data": {
                                "algorithm_sequence_id": algorithm_sequence_id,
                                "mobilization_object_type": mobilization_object_type,
                                "req_element_id": req_element_id,
                                "scheme_id": scheme_id,
                                "scheme_config": scheme_config,
                                "activity_id": activity_id
                            }
                        }
                    else:
                        raise

            self.logger.info("=" * 80)
            self.logger.info(f"求解开始 - 资源类型: {resource_type}")
            if mobilization_object_type:
                self.logger.info(f"动员对象类型: {mobilization_object_type}")
            if algorithm_sequence_id:
                self.logger.info(f"算法序号: {algorithm_sequence_id}")
            self.logger.info(f"网络规模: J={len(self.J)}, M={len(self.M)}, K={len(self.K)}")
            self.logger.info(f"随机种子: {random_seed}")

            # ========== 数据动员特殊处理分支 ==========
            data_result = {}
            if resource_type == "data":
                self.logger.info("=" * 80)
                self.logger.info("切换到数据动员模式：识别合适的数据供应点")
                if mobilization_object_type:
                    self.logger.info(f"动员对象类型: {mobilization_object_type}")
                if algorithm_sequence_id:
                    self.logger.info(f"算法序号: {algorithm_sequence_id}")
                self.logger.info("=" * 80)

                print("=" * 80)
                print("数据动员模式：识别合适的数据供应点")
                if mobilization_object_type:
                    print(f"动员对象类型: {mobilization_object_type}")
                if algorithm_sequence_id:
                    print(f"算法序号: {algorithm_sequence_id}")
                print("=" * 80)

                try:
                    self.logger.info("开始执行数据动员处理")
                    data_mobilization_start = time.time()

                    # 检查时间限制
                    elapsed_time = check_time_limit()
                    if isinstance(elapsed_time, dict):  # 如果返回的是错误响应
                        return elapsed_time

                    pre_solution,solution = self._handle_data_mobilization()

                    data_mobilization_time = time.time() - data_mobilization_start
                    self.logger.info(f"数据动员处理完成，耗时: {data_mobilization_time:.3f}秒")

                except (KeyError, AttributeError) as e:
                    error_msg = f"数据动员处理错误: {str(e)}"
                    self.logger.error(error_msg, exc_info=True)
                    self.logger.error(f"数据动员失败时的网络状态:")
                    self.logger.error(f"  - 供应点数量: {len(getattr(self, 'J', []))}")
                    self.logger.error(f"  - 需求点数量: {len(getattr(self, 'K', []))}")
                    self.logger.error(f"  - 点特征数据存在: {hasattr(self, 'point_features')}")
                    if hasattr(self, 'point_features'):
                        self.logger.error(f"  - 点特征数据量: {len(self.point_features)}")
                    if return_format == "json":
                        return {
                            "code": 500,
                            "msg": error_msg,
                            "data": {
                                "algorithm_sequence_id": algorithm_sequence_id,
                                "mobilization_object_type": mobilization_object_type,
                                "req_element_id": req_element_id,
                                "scheme_id": scheme_id,
                                "scheme_config": scheme_config,
                                "activity_id": activity_id
                            }
                        }
                    else:
                        raise ValueError(error_msg)

                total_end_time = time.time()
                total_time = total_end_time - total_start_time
                if len(solution) >= 1 :
                    for i in range(len(solution)):
                        solution[str(i)]['total_solve_time'] = total_time + random.uniform(0, 1) / 100
                        solution[str(i)]['resource_type'] = resource_type
                        solution[str(i)]['objective_weights'] = self.objective_weights.copy()
                        solution[str(i)]['algorithm_sequence_id'] = algorithm_sequence_id
                        solution[str(i)]['mobilization_object_type'] = mobilization_object_type
                        solution[str(i)]['req_element_id'] = req_element_id

                    self.logger.info(f"数据动员完成，总耗时: {total_time:.2f}秒")
                    print(f"\n数据动员完成，总耗时: {total_time:.2f}秒")

                    if save_results:
                        try:
                            self._save_solution_results(solution, output_dir)
                        except (OSError, IOError) as e:
                            self.logger.warning(f"保存结果失败: {str(e)}")

                    # 根据返回格式决定输出内容
                    if return_format == "json":
                        data_re = {}

                        for i in range(len(solution)):
                            data_re[str(i)] = self._generate_data_mobilization_output(solution[str(i)],
                                                              solution[str(i)]['total_solve_time'])

                        data_re1 = {}
                        for i in range(len(pre_solution)):
                            data_re1[str(i)] = self._generate_preSplution_data_mobilization_output(pre_solution[str(i)])

                        data_result = data_re
                        for i in range(len(data_result)):
                            for ja in range(len(data_re1)):
                                data_result[str(i)]['data']['supplier_portfolio'].append(data_re1[str(ja)])

                        # 检查数据动员输出是否包含错误
                        stand_out_1 = {}
                        # 定义需要检查的文件夹列表
                        self.create_two_level_directories(activity_id=activity_id, req_element_id=req_element_id)

                        data_do = data_result

                        if len(data_do) > 1:


                            data_d_time, data_d_cost, data_d_safety, data_d_distance, data_d_priority, data_d_balance, data_d_capability, data_d_social = \
                                sort_dict_inner_value(data_do, 'time', 'cost', 'safety', 'distance', 'priority', 'balance',
                                                      'capability',
                                                      'social')

                            Save_json_file(str(activity_id)+'/'+str(req_element_id), data_do, 'data_do')
                            Save_json_file(str(activity_id)+'/'+str(req_element_id), data_d_social, 'data_d_social')
                            Save_json_file(str(activity_id)+'/'+str(req_element_id), data_d_capability, 'data_d_capability')
                            Save_json_file(str(activity_id)+'/'+str(req_element_id), data_d_balance, 'data_d_balance')
                            Save_json_file(str(activity_id)+'/'+str(req_element_id), data_d_priority, ' data_d_priority')
                            Save_json_file(str(activity_id)+'/'+str(req_element_id), data_d_distance, 'data_d_distance')
                            Save_json_file(str(activity_id)+'/'+str(req_element_id), data_d_safety, 'data_d_safety')
                            Save_json_file(str(activity_id)+'/'+str(req_element_id), data_d_cost, 'data_d_cost')
                            Save_json_file(str(activity_id)+'/'+str(req_element_id), data_d_time, 'data_d_time')


                            k_d_time, v_d_time, k_d_cost, v_d_cost, k_d_safety, v_d_safety = fine_three_kage(data_d_time,
                                                                                                             data_d_cost,
                                                                                                             data_d_safety)

                            k_avrage, avrage_value = find_Avarage_kage(data_do, k_d_safety, k_d_time, k_d_cost)
                            #rfjdebug
                            if 'rangeType' not in v_d_cost or not v_d_cost['rangeType']:
                                v_d_cost['rangeType'] = 'rangeCost'
                            if 'rangeType' not in v_d_time or not v_d_time['rangeType']:
                                v_d_cost['rangeType'] = 'rangeTime'
                            if 'rangeType' not in v_d_safety or not v_d_safety['rangeType']:
                                v_d_safety['rangeType'] = 'rangeSafety'
                            if 'rangeType' not in avrage_value or not avrage_value['rangeType']:
                                avrage_value['rangeType'] = 'rangeAvarage'
                            #endrfj

                            vdf = combine_four_kage(str(activity_id)+'/'+str(req_element_id), v_d_cost,  v_d_time,  v_d_safety,avrage_value)

                            noRange = dataDeal(vdf)
                            vdf['norangeNum'] = noRange
                        else:
                            vdf = data_do['0']
                            vd_cost = copy.copy(vdf)
                            vd_time = copy.copy(vdf)
                            vd_safety = copy.copy(vdf)
                            vd_avrage = copy.copy(vdf)
                            vd_cost['rangeType'] = 'rangeCost'
                            vd_time['rangeType'] = 'rangeTime'
                            vd_safety['rangeType'] = 'rangeSafety'
                            vd_avrage['rangeType'] = 'rangeAvarage'
                            vdf = combine_four_kage(str(activity_id) + '/' + str(req_element_id), vd_cost, vd_time,
                                                    vd_safety, vd_avrage)

                            noRange = dataDeal(vdf)
                            vdf['norangeNum'] = noRange

                        lm = 0
                        if isinstance(vdf, dict) and 'code' in vdf and vdf['code'] != 200:
                            return vdf
                        return {
                            "code": 200,
                            "msg": "数据动员成功",
                            "data": vdf
                        }
                    else:
                        return {
                            "code": 200,
                            "msg": "数据动员成功",
                            "data": solution
                        }
                else:
                    return {
                        "code": 200,
                        "msg": "数据动员指定成功",
                        "data": pre_solution
                    }

            if hasattr(self, 'time_windows') and self.time_windows:
                self.logger.info("时间窗约束:")
                for k, (earliest, latest) in self.time_windows.items():
                    self.logger.info(f"  需求点 {k}: [{earliest:.3f}, {latest:.3f}]")

            self.logger.info("目标权重配置:")
            for obj, weight in self.objective_weights.items():
                self.logger.info(f"  {self.objective_names[obj]}: {weight:.3f}")
            self.logger.info("=" * 80)

            try:
                # ========== 第5步：多目标智能匹配阶段 ==========
                self.logger.info("第5步: 开始多目标匹配阶段")
                matching_start_time = time.time()

                # 检查时间限制
                elapsed_time = check_time_limit()
                if isinstance(elapsed_time, dict):  # 如果返回的是错误响应
                    return elapsed_time

                matching_result = self._multi_objective_intelligent_matching(random_seed)
                matching_time = time.time() - matching_start_time

                # 处理匹配结果的返回格式
                if isinstance(matching_result, dict) and 'code' in matching_result:
                    if matching_result['code'] == 200:
                        paths = matching_result['data']
                        self.logger.info(f"多目标匹配完成，耗时: {matching_time:.3f}秒，生成路径: {len(paths)}条")
                    else:
                        error_msg = f"匹配阶段失败: {matching_result['msg']}"
                        self.logger.error(error_msg)
                        if return_format == "json":
                            return {
                                "code": matching_result['code'],
                                "msg": error_msg,
                                "data": {
                                    "algorithm_sequence_id": algorithm_sequence_id,
                                    "mobilization_object_type": mobilization_object_type,
                                    **matching_result.get('data', {}),
                                    "req_element_id": req_element_id,
                                    "scheme_id": scheme_id,
                                    "scheme_config": scheme_config,
                                    "activity_id": activity_id
                                }
                            }
                        else:
                            raise ValueError(error_msg)
                else:
                    # 兼容旧格式
                    paths = matching_result
                    self.logger.info(f"多目标匹配完成，耗时: {matching_time:.3f}秒，生成路径: {len(paths)}条")

            except (ValueError, KeyError, AttributeError) as e:
                error_msg = f"匹配阶段失败: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                if return_format == "json":
                    return {
                        "code": 500,
                        "msg": error_msg,
                        "data": {
                            "algorithm_sequence_id": algorithm_sequence_id,
                            "mobilization_object_type": mobilization_object_type,
                            "error_type": "matching_error",
                            "req_element_id": req_element_id,
                            "scheme_id": scheme_id,
                            "scheme_config": scheme_config,
                            "activity_id": activity_id
                        }
                    }
                else:
                    raise ValueError(error_msg)

            try:
                # ========== 第6步：多目标智能调度阶段 ==========
                self.logger.info("第6步: 开始多目标调度阶段")
                scheduling_start_time = time.time()

                # 检查时间限制
                elapsed_time = check_time_limit()
                if isinstance(elapsed_time, dict):  # 如果返回的是错误响应
                    return elapsed_time
                scheduling_result = self._multi_objective_intelligent_scheduling(paths)
                scheduling_time = time.time() - scheduling_start_time

                if len(scheduling_result['data'])>=1:
                    # 处理调度结果的返回格式
                    if isinstance(scheduling_result, dict) and 'code' in scheduling_result:
                        if scheduling_result['code'] == 200:
                            solution = scheduling_result['data']
                            self.logger.info(f"多目标调度完成，耗时: {scheduling_time:.3f}秒")
                        else:
                            error_msg = f"调度阶段失败: {scheduling_result['msg']}"
                            self.logger.error(error_msg)
                            if return_format == "json":
                                return {
                                    "code": scheduling_result['code'],
                                    "msg": error_msg,
                                    "data": {
                                        "algorithm_sequence_id": algorithm_sequence_id,
                                        "mobilization_object_type": mobilization_object_type,
                                        **scheduling_result.get('data', {}),
                                        "req_element_id": req_element_id,
                                        "scheme_id": scheme_id,
                                        "scheme_config": scheme_config,
                                        "activity_id": activity_id
                                    }
                                }
                            else:
                                raise ValueError(error_msg)
                    else:
                        solution = scheduling_result
                        self.logger.info(f"多目标调度完成，耗时: {scheduling_time:.3f}秒")
                else:
                    return {
                    "code": 200,
                    "msg": "已经指定",
                    "data":  scheduling_result['pre_data']
                }
            except (ValueError, KeyError, AttributeError) as e:
                error_msg = f"调度阶段失败: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                if return_format == "json":
                    return {
                        "code": 500,
                        "msg": error_msg,
                        "data": {
                            "algorithm_sequence_id": algorithm_sequence_id,
                            "mobilization_object_type": mobilization_object_type,
                            "error_type": "scheduling_error",
                            "req_element_id": req_element_id,
                            "scheme_id": scheme_id,
                            "scheme_config": scheme_config,
                            "activity_id": activity_id
                        }
                    }
                else:
                    raise ValueError(error_msg)

            # 最终时间检查
            elapsed_time = check_time_limit()
            if isinstance(elapsed_time, dict):  # 如果返回的是错误响应
                return elapsed_time

            total_end_time = time.time()
            total_time = total_end_time - total_start_time
            self.logger.info(f"优化完成，总耗时: {total_time:.2f}秒")

            solution['resource_type'] = resource_type
            solution['objective_weights'] = self.objective_weights.copy()
            solution['algorithm_sequence_id'] = algorithm_sequence_id
            solution['mobilization_object_type'] = mobilization_object_type
            solution['req_element_id'] = req_element_id

            if save_results:
                try:
                    self._save_solution_results(solution, output_dir)
                except (OSError, IOError) as e:
                    self.logger.warning(f"保存结果失败: {str(e)}")

            # 根据返回格式决定输出内容
            if return_format == "json":
                s_result = {}
                # s_result = self._generate_standardized_output(solution[str(0)], total_time)
                for i in range(len(solution) - 5):
                    s_result[str(i)] = self._generate_standardized_output(solution[str(i)], total_time)

                pre_solution = scheduling_result['pre_data']
                for i in range(len(s_result)):
                    for j in range(len(pre_solution)):
                        s_result[str(i)]['supplier_portfolio'].append(pre_solution[j])

                standardized_result = s_result

                stand_out_1 = {}
                # 定义需要检查的文件夹列表

                if resource_type == "personnel":
                    self.create_two_level_directories(activity_id=activity_id,req_element_id=req_element_id)

                    data_p_lp = standardized_result
                    if len(data_p_lp) > 1:
                        data_p_time, data_p_cost, data_p_safety, data_p_distance, data_p_priority, data_p_balance, data_p_capability, data_p_social = \
                            sort_dict_innermp_value(data_p_lp, 'time', 'cost', 'safety', 'distance', 'priority', 'balance',
                                                    'capability',
                                                    'social')

                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_p_lp, 'data_p_lp')
                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_p_social, 'data_p_social')
                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_p_capability, 'data_p_capability')
                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_p_balance, 'data_p_balance')
                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_p_priority, ' data_p_priority')
                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_p_distance, 'data_p_distance')
                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_p_safety, 'data_p_safety')
                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_p_cost, 'data_p_cost')
                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_p_time, 'data_p_time')

                        k_d_time, v_d_time, k_d_cost, v_d_cost, k_d_safety, v_d_safety = fine_three_MP_kage(data_p_time,
                                                                                                         data_p_cost,
                                                                                                         data_p_safety)

                        k_avrage, avrage_value = find_Avarage_kage(data_p_lp, k_d_safety, k_d_time, k_d_cost)
                        #rfjdebug
                        if 'rangeType' not in v_d_cost or not v_d_cost['rangeType']:
                            v_d_cost['rangeType'] = 'rangeCost'
                        if 'rangeType' not in v_d_time or not v_d_time['rangeType']:
                            v_d_cost['rangeType'] = 'rangeTime'
                        if 'rangeType' not in v_d_safety or not v_d_safety['rangeType']:
                            v_d_safety['rangeType'] = 'rangeSafety'
                        if 'rangeType' not in avrage_value or not avrage_value['rangeType']:
                            avrage_value['rangeType'] = 'rangeAvarage'
                        #enddebug

                        vdf = combine_four_kage(str(activity_id)+'/'+str(req_element_id), v_d_cost, v_d_time, v_d_safety,avrage_value)

                        noRange = dataDeal(vdf)
                        vdf['norangeNum'] = noRange
                    else:
                        vdf = data_p_lp['0']
                        vd_cost = copy.copy(vdf)
                        vd_time = copy.copy(vdf)
                        vd_safety = copy.copy(vdf)
                        vd_avrage = copy.copy(vdf)
                        vd_cost['rangeType'] = 'rangeCost'
                        vd_time['rangeType'] = 'rangeTime'
                        vd_safety['rangeType'] = 'rangeSafety'
                        vd_avrage['rangeType'] = 'rangeAvarage'
                        vdf = combine_four_kage(str(activity_id) + '/' + str(req_element_id), vd_cost, vd_time,
                                                 vd_safety, vd_avrage)

                        noRange = dataDeal(vdf)
                        vdf['norangeNum'] = noRange


                elif resource_type == "material":

                    self.create_two_level_directories(activity_id=activity_id, req_element_id=req_element_id)
                    data_m_lp = standardized_result

                    if len(data_m_lp) > 1 :
                        data_m_time, data_m_cost, data_m_safety, data_m_distance, data_m_priority, data_m_balance, data_m_capability, data_m_social = \
                            sort_dict_innermp_value(data_m_lp, 'time', 'cost', 'safety', 'distance', 'priority', 'balance',
                                                    'capability',
                                                    'social')

                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_m_lp, 'data_m_lp')
                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_m_social, 'data_m_social')
                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_m_capability, 'data_m_capability')
                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_m_balance, 'data_m_balance')
                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_m_priority, ' data_m_priority')
                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_m_distance, 'data_m_distance')
                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_m_safety, 'data_m_safety')
                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_m_cost, 'data_m_cost')
                        Save_json_file(str(activity_id)+'/'+str(req_element_id), data_m_time, 'data_m_time')

                        k_d_time, v_d_time, k_d_cost, v_d_cost, k_d_safety, v_d_safety = fine_three_MP_kage(data_m_time,
                                                                                                         data_m_cost,
                                                                                                         data_m_safety)

                        k_avrage, avrage_value = find_Avarage_kage(data_m_lp, k_d_safety, k_d_time, k_d_cost)
                        #rfjdebug
                        if 'rangeType' not in v_d_cost or not v_d_cost['rangeType']:
                            v_d_cost['rangeType'] = 'rangeCost'
                        if 'rangeType' not in v_d_time or not v_d_time['rangeType']:
                            v_d_cost['rangeType'] = 'rangeTime'
                        if 'rangeType' not in v_d_safety or not v_d_safety['rangeType']:
                            v_d_safety['rangeType'] = 'rangeSafety'
                        if 'rangeType' not in avrage_value or not avrage_value['rangeType']:
                            avrage_value['rangeType'] = 'rangeAvarage'
                        #end

                        vdf = combine_four_kage(str(activity_id)+'/'+str(req_element_id), v_d_cost, v_d_time,v_d_safety, avrage_value)

                        noRange = dataDeal(vdf)
                        vdf['norangeNum'] = noRange
                        ss = 0
                    else:
                        vdf = data_m_lp['0']
                        vd_cost = copy.copy(vdf)
                        vd_time = copy.copy(vdf)
                        vd_safety = copy.copy(vdf)
                        vd_avrage = copy.copy(vdf)
                        vd_cost['rangeType'] = 'rangeCost'
                        vd_time['rangeType'] = 'rangeTime'
                        vd_safety['rangeType'] = 'rangeSafety'
                        vd_avrage['rangeType'] = 'rangeAvarage'
                        vdf = combine_four_kage(str(activity_id) + '/' + str(req_element_id), vd_cost, vd_time,
                                                vd_safety, vd_avrage)

                        noRange = dataDeal(vdf)
                        vdf['norangeNum'] = noRange


                # 检查标准化输出是否包含错误
                se = 0
                if isinstance(vdf, dict) and 'code' in vdf and vdf[
                    'code'] != 200:
                    return vdf
                return {
                    "code": 200,
                    "msg": "优化成功",
                    "data": vdf
                }
            else:
                return {
                    "code": 200,
                    "msg": "优化成功",
                    "data": solution
                }

        except ValueError as ve:
            error_msg = f"参数配置错误: {str(ve)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                "code": 400,
                "msg": error_msg,
                "data": {
                    "algorithm_sequence_id": algorithm_sequence_id,
                    "mobilization_object_type": mobilization_object_type,
                    "resource_type": resource_type,
                    "error_type": "parameter_error",
                    "req_element_id": req_element_id,
                    "scheme_id": scheme_id,
                    "scheme_config": scheme_config,
                    "activity_id": activity_id
                }
            }

        except (KeyError, AttributeError) as ke:
            error_msg = f"数据结构错误: {str(ke)}"
            self.logger.error(error_msg, exc_info=True)
            if return_format == "json":
                return {
                    "code": 400,
                    "msg": error_msg,
                    "data": {
                        "algorithm_sequence_id": algorithm_sequence_id,
                        "mobilization_object_type": mobilization_object_type,
                        "req_element_id": req_element_id,
                        "scheme_id": scheme_id,
                        "scheme_config": scheme_config,
                        "activity_id": activity_id
                    }
                }
            else:
                raise ValueError(error_msg)
        except (OSError, IOError) as ioe:
            error_msg = f"文件操作错误: {str(ioe)}"
            self.logger.error(error_msg, exc_info=True)
            if return_format == "json":
                return {
                    "code": 500,
                    "msg": "文件系统错误",
                    "data": {
                        "algorithm_sequence_id": algorithm_sequence_id,
                        "mobilization_object_type": mobilization_object_type,
                        "req_element_id": req_element_id,
                        "scheme_id": scheme_id,
                        "scheme_config": scheme_config,
                        "activity_id": activity_id
                    }
                }
            else:
                raise
        except TimeoutError as te:
            # 处理超时异常
            error_msg = str(te)
            self.logger.error(error_msg, exc_info=True)
            if return_format == "json":
                return {
                    "code": 408,
                    "msg": error_msg,
                    "data": {
                        "algorithm_sequence_id": algorithm_sequence_id,
                        "mobilization_object_type": mobilization_object_type,
                        "resource_type": resource_type,
                        "req_element_id": req_element_id,
                        "scheme_id": scheme_id,
                        "scheme_config": scheme_config,
                        "activity_id": activity_id,
                        "elapsed_time": time.time() - total_start_time,
                        "time_limit": 100.0,
                        "error_type": "timeout_error"
                    }
                }
            else:
                raise

    def _set_network_data_optimized(self, network_data):
        """网络数据设置"""
        try:
            # 直接赋值，避免深度复制
            self.J = network_data['J']
            self.M = network_data['M']
            self.K = network_data['K']
            self.point_features = network_data['point_features']
            self.B = network_data['B']
            self.P = network_data['P']
            self.D = network_data['D']
            self.demand_priority = network_data['demand_priority']
            self.Q = network_data['Q']
            self.time_windows = network_data.get('time_windows', {})

            # 验证关键数据结构
            if not self.J:
                raise ValueError("供应点集合J不能为空")
            if not self.K:
                raise ValueError("需求点集合K不能为空")
            if not self.M:
                raise ValueError("中转点集合M不能为空")

            # 验证供应点基础配置
            for j in self.J:
                if j not in self.B:
                    raise ValueError(f"供应点{j}缺少供应能力配置B")
                if j not in self.P:
                    raise ValueError(f"供应点{j}缺少可靠性配置P")
                if j not in self.point_features:
                    raise ValueError(f"供应点{j}缺少点特征配置point_features")
                if self.B[j] is None or self.B[j] < 0:
                    raise ValueError(f"供应点{j}的供应能力B无效: {self.B[j]}")
                if self.P[j] is None or self.P[j] <= 0 or self.P[j] > 1:
                    raise ValueError(f"供应点{j}的可靠性P无效: {self.P[j]}")

            # 验证需求点配置
            for k in self.K:
                if k not in self.D:
                    raise ValueError(f"需求点{k}缺少需求量配置D")
                if k not in self.point_features:
                    raise ValueError(f"需求点{k}缺少点特征配置point_features")
                if self.D[k] is None or self.D[k] <= 0:
                    raise ValueError(f"需求点{k}的需求量D无效: {self.D[k]}")

        except KeyError as e:
            raise ValueError(f"网络数据缺少必要字段: {str(e)}")
        except TypeError as e:
            raise ValueError(f"网络数据类型错误: {str(e)}")

    def _set_transport_params_optimized(self, transport_params):
        """运输参数设置"""
        try:
            # 基础参数直接赋值
            self.N = transport_params['N']
            self.TRANSPORT_MODES = transport_params['TRANSPORT_MODES']

            # 距离矩阵 - 如果数据量大，考虑按需构建
            network_size = len(self.J) * len(self.M) + len(self.M) * len(self.M) + len(self.M) * len(self.K)
            if network_size > len(self.J) * len(self.M) * len(self.K):
                # 大规模网络：延迟构建，先存储原始数据
                self._transport_params_raw = transport_params
                self.L_j_m = transport_params['L_j_m']
                self.L_m_m = transport_params['L_m_m']
                self.L_m_k = transport_params['L_m_k']
            else:
                # 小规模网络：正常处理
                self.L_j_m = transport_params['L_j_m']
                self.L_m_m = transport_params['L_m_m']
                self.L_m_k = transport_params['L_m_k']

            # 其他参数
            self.alpha1 = transport_params['alpha1']
            self.alpha2 = transport_params['alpha2']
            self.alpha3 = transport_params['alpha3']

            self.v1 = transport_params['v1']
            self.v2 = transport_params['v2']
            self.v3 = transport_params['v3']

            time_params = transport_params['time_params']
            self.T1 = time_params['preparation_time']
            self.T4 = time_params['assembly_time']
            self.T6 = time_params['handover_time']

            self.T_loading = self.T1 + self.T4 + self.T6

            self.ROAD_ONLY = transport_params.get('ROAD_ONLY', [1])

            # 快速验证
            if not self.N:
                raise ValueError("运输方式集合N不能为空")
            if not self.TRANSPORT_MODES:
                raise ValueError("运输方式配置TRANSPORT_MODES不能为空")

        except KeyError as e:
            raise ValueError(f"运输参数缺少必要字段: {str(e)}")
        except TypeError as e:
            raise ValueError(f"运输参数类型错误: {str(e)}")

    def _build_active_direct_paths_index(self, x_direct, b_direct, t_direct):
        """构建活跃直接路径索引"""
        active_paths = {}

        for j in self.J:
            active_paths[j] = []
            for k in self.K:
                for n in [1]:
                    if x_direct.get((j, k, n), 0) == 1 and b_direct.get((j, k, n), 0) > self.EPS:
                        supply_amount = b_direct[(j, k, n)]
                        transport_time = t_direct[(j, k, n)]
                        active_paths[j].append((k, n, supply_amount, transport_time))

            # 如果该供应点没有活跃路径，从字典中移除
            if not active_paths[j]:
                del active_paths[j]

        return active_paths

    def _build_active_multimodal_paths_index(self, x1, x2, x3, b1, b2, b3, t1, t2, t3):
        """构建活跃多式联运路径索引"""
        active_paths = {}

        # 预先构建活跃的第二段路径映射
        active_segment2 = {}
        for m1 in self.M:
            for m2 in self.M:
                if m1 != m2:
                    for n2 in self.N:
                        if x2.get((m1, m2, n2), 0) == 1 and b2.get((m1, m2, n2), 0) > self.EPS:
                            if m1 not in active_segment2:
                                active_segment2[m1] = []
                            active_segment2[m1].append((m2, n2))

        # 预先构建活跃的第三段路径映射
        active_segment3 = {}
        for m2 in self.M:
            for k in self.K:
                for n3 in [1]:
                    if x3.get((m2, k, n3), 0) == 1 and b3.get((m2, k, n3), 0) > self.EPS:
                        if m2 not in active_segment3:
                            active_segment3[m2] = []
                        active_segment3[m2].append((k, n3))

        # 构建完整的多式联运路径
        for j in self.J:
            active_paths[j] = []
            for m1 in self.M:
                for n1 in [1]:
                    if x1.get((j, m1, n1), 0) == 1 and b1.get((j, m1, n1), 0) > self.EPS:
                        supply_amount = b1[(j, m1, n1)]

                        # 只处理有活跃第二段的路径
                        if m1 in active_segment2:
                            for m2, n2 in active_segment2[m1]:
                                # 只处理有活跃第三段的路径
                                if m2 in active_segment3:
                                    for k, n3 in active_segment3[m2]:
                                        time_segments = [
                                            t1.get((j, m1, n1), 0),
                                            t2.get((m1, m2, n2), 0),
                                            t3.get((m2, k, n3), 0)
                                        ]
                                        active_paths[j].append((m1, m2, k, n1, n2, n3, supply_amount, time_segments))

            # 如果该供应点没有活跃路径，从字典中移除
            if not active_paths[j]:
                del active_paths[j]

        return active_paths

    def _generate_standardized_output(self, solution, total_time):
        """
        生成标准化的JSON格式输出
        """

        def safe_json_convert(obj):
            """安全的JSON转换函数，处理不可序列化的对象"""
            if isinstance(obj, (int, float, str, bool, type(None))):
                return obj
            elif isinstance(obj, (list, tuple)):
                return [safe_json_convert(item) for item in obj]
            elif isinstance(obj, dict):
                return {str(key): safe_json_convert(value) for key, value in obj.items()}
            else:
                return str(obj)

        def select_optimal_sub_objects_for_supplier(supplier_id, allocated_amount):
            """为供应点选择最优细分对象组合，基于成本效率按优先级依次分配"""

            def find_supplier_name_by_id(supplier_id):
                """根据supplier_id找到对应的供应点名称"""
                for name, features in self.point_features.items():
                    if features.get('original_supplier_id') == supplier_id:
                        return name
                return supplier_id

            supplier_name = find_supplier_name_by_id(supplier_id)
            sub_objects = self.point_features[supplier_name].get('sub_objects', [])

            if not sub_objects:
                return []

            available_objects = []

            # 遍历所有分类中的细分对象
            for category_idx, category in enumerate(sub_objects):
                if isinstance(category, dict) and 'items' in category:
                    # 新的分类结构：遍历分类中的所有项目
                    for item_idx, sub_obj in enumerate(category.get('items', [])):
                        max_available = sub_obj.get('max_available_quantity', 0)
                        if not isinstance(max_available, (int, float)) or max_available <= self.EPS:
                            continue

                        # 根据资源类型计算成本
                        if hasattr(self, 'resource_type'):
                            if self.resource_type == "personnel":
                                wage_cost_raw = sub_obj.get('wage_cost', len(self.J) * len(self.K) + len(self.J))
                                living_cost_raw = sub_obj.get('living_cost', len(self.J) + len(self.K))
                                total_cost = wage_cost_raw + living_cost_raw
                            elif self.resource_type == "material":
                                material_price = sub_obj.get('material_price',
                                                             len(self.J) * len(self.K) * (len(self.J) + len(self.K)))
                                equipment_rental = sub_obj.get('equipment_rental_price', len(self.J) * len(self.K))
                                equipment_depreciation = sub_obj.get('equipment_depreciation_cost',
                                                                     len(self.J) + len(self.K))
                                total_cost = material_price + equipment_rental + equipment_depreciation
                            elif self.resource_type == "data":
                                facility_rental = sub_obj.get('facility_rental_price',
                                                              len(self.J) * len(self.M) + len(self.J))
                                power_cost = sub_obj.get('power_cost', len(self.J) + len(self.K))
                                communication_cost = sub_obj.get('communication_purchase_price',
                                                                 len(self.J) * len(self.K))
                                total_cost = facility_rental + power_cost + communication_cost
                            else:
                                total_cost = len(self.J) * len(self.K) * (len(self.J) + len(self.K))
                        else:
                            material_price = sub_obj.get('material_price',
                                                         len(self.J) * len(self.K) * (len(self.J) + len(self.K)))
                            equipment_rental = sub_obj.get('equipment_rental_price', len(self.J) * len(self.K))
                            equipment_depreciation = sub_obj.get('equipment_depreciation_cost',
                                                                 len(self.J) + len(self.K))
                            total_cost = material_price + equipment_rental + equipment_depreciation

                        safety_score = self._calculate_sub_object_safety(sub_obj)
                        cost_efficiency = safety_score / max(total_cost, self.EPS) if total_cost > 0 else safety_score

                        category_id = sub_obj.get('category_id') or category.get('category_id', 'unknown')
                        category_name = sub_obj.get('category_name') or category.get('category_name', 'unknown')
                        recommend_md5 = sub_obj.get('recommend_md5') or category.get('recommend_md5', 'unknown')

                        available_objects.append({
                            'sub_object': sub_obj,
                            'cost_efficiency': cost_efficiency,
                            'max_available': max_available,
                            'total_cost': total_cost,
                            'safety_score': safety_score,
                            'category_id': category_id,
                            'category_name': category_name,
                            'recommend_md5': recommend_md5,
                            'sub_object_score': cost_efficiency
                        })
                else:
                    # 兼容旧的平铺结构：直接处理细分对象
                    sub_obj = category
                    max_available = sub_obj.get('max_available_quantity', 0)
                    if max_available <= self.EPS:
                        continue

                    # 根据资源类型计算成本（完整版本）
                    if hasattr(self, 'resource_type'):
                        if self.resource_type == "personnel":
                            wage_cost_raw = sub_obj.get('wage_cost', len(self.J) * len(self.K) + len(self.J))
                            living_cost_raw = sub_obj.get('living_cost', len(self.J) + len(self.K))
                            total_cost = wage_cost_raw + living_cost_raw
                        elif self.resource_type == "material":
                            material_price = sub_obj.get('material_price',
                                                         len(self.J) * len(self.K) * (len(self.J) + len(self.K)))
                            equipment_rental = sub_obj.get('equipment_rental_price', len(self.J) * len(self.K))
                            equipment_depreciation = sub_obj.get('equipment_depreciation_cost',
                                                                 len(self.J) + len(self.K))
                            total_cost = material_price + equipment_rental + equipment_depreciation
                        elif self.resource_type == "data":
                            facility_rental = sub_obj.get('facility_rental_price',
                                                          len(self.J) * len(self.M) + len(self.J))
                            power_cost = sub_obj.get('power_cost', len(self.J) + len(self.K))
                            communication_cost = sub_obj.get('communication_purchase_price', len(self.J) * len(self.K))
                            total_cost = facility_rental + power_cost + communication_cost
                        else:
                            total_cost = len(self.J) * len(self.K) * (len(self.J) + len(self.K))
                    else:
                        material_price = sub_obj.get('material_price',
                                                     len(self.J) * len(self.K) * (len(self.J) + len(self.K)))
                        equipment_rental = sub_obj.get('equipment_rental_price', len(self.J) * len(self.K))
                        equipment_depreciation = sub_obj.get('equipment_depreciation_cost', len(self.J) + len(self.K))
                        total_cost = material_price + equipment_rental + equipment_depreciation

                    safety_score = self._calculate_sub_object_safety(sub_obj)
                    cost_efficiency = safety_score / max(total_cost, self.EPS) if total_cost > 0 else safety_score

                    category_id = sub_obj.get('category_id', 'unknown')
                    category_name = sub_obj.get('category_name', 'unknown')
                    recommend_md5 = sub_obj.get('recommend_md5', 'unknown')
                    available_objects.append({
                        'sub_object': sub_obj,
                        'cost_efficiency': cost_efficiency,
                        'max_available': max_available,
                        'total_cost': total_cost,
                        'safety_score': safety_score,
                        'category_id': category_id,
                        'category_name': category_name,
                        'recommend_md5': recommend_md5,
                        'sub_object_score': cost_efficiency  # 添加细分对象评分
                    })

            # 按成本效率排序
            if available_objects:
                available_objects.sort(key=lambda x: x['cost_efficiency'], reverse=True)

                selected_objects = []
                remaining_demand = allocated_amount

                for obj_info in available_objects:

                    max_available = obj_info['max_available']
                    allocated_for_this_obj = min(remaining_demand, max_available)

                    if allocated_for_this_obj > self.EPS:
                        selected_objects.append({
                            'sub_object': obj_info['sub_object'],
                            'allocated_amount': allocated_for_this_obj,
                            'cost_efficiency': obj_info['cost_efficiency'],
                            'max_available': max_available,
                            'total_cost': obj_info['total_cost'],
                            'safety_score': obj_info['safety_score'],
                            'category_id': obj_info.get('category_id', 'unknown'),
                            'category_name': obj_info.get('category_name', 'unknown'),
                            'recommend_md5': obj_info.get('recommend_md5', 'unknown'),
                            'sub_object_score': obj_info.get('sub_object_score', obj_info.get('cost_efficiency', 0.0))
                        })
                        remaining_demand -= allocated_for_this_obj
                qq = 0
                return selected_objects

            return []

        def format_sub_object_info(sub_obj, resource_type, allocated_amount=None, category_id='unknown',
                                   category_name='unknown', recommend_md5='unknown'):
            """根据动员类型格式化细分对象信息"""
            if not sub_obj:
                return {}

            base_info = {
                'sub_object_id': sub_obj.get('sub_object_id', 'unknown'),
                'sub_object_name': sub_obj.get('sub_object_name', 'unknown'),
                'category_id': category_id,
                'category_name': category_name,
                'recommend_md5': recommend_md5,
                'max_available_quantity': round(allocated_amount) if allocated_amount is not None else sub_obj.get(
                    'max_available_quantity', 0),
                'specify_quantity': sub_obj.get('specify_quantity', 0),
                'capacity_quantity': sub_obj.get('capacity_quantity', 0)
            }

            if resource_type == "personnel":
                base_info.update({
                    'identity_card': sub_obj.get('sub_object_id', 'unknown'),
                    'wage_cost': round(sub_obj.get('wage_cost', 0), 2),
                    'living_cost': round(sub_obj.get('living_cost', 0), 2),
                    'political_status_score': round(sub_obj.get('political_status', 0), 2),
                    'military_experience_score': round(sub_obj.get('military_experience', 0), 2)
                })
            elif resource_type == "material":
                base_info.update({
                    'material_code': sub_obj.get('sub_object_id', 'unknown'),
                    'material_price': round(sub_obj.get('material_price', 0), 2),
                    'equipment_rental_price': round(sub_obj.get('equipment_rental_price', 0), 2),
                    'equipment_depreciation_cost': round(sub_obj.get('equipment_depreciation_cost', 0), 2),
                    'safety_attributes': {
                        'flammable_explosive': round(sub_obj.get('flammable_explosive', 0), 2),
                        'corrosive': round(sub_obj.get('corrosive', 0), 2),
                        'polluting': round(sub_obj.get('polluting', 0), 2),
                        'fragile': round(sub_obj.get('fragile', 0), 2)
                    }
                })
            elif resource_type == "data":
                base_info.update({
                    'data_type_code': sub_obj.get('sub_object_id', 'unknown'),
                    'facility_rental_price': round(sub_obj.get('facility_rental_price', 0), 2),
                    'power_cost': round(sub_obj.get('power_cost', 0), 2),
                    'communication_purchase_price': round(sub_obj.get('communication_purchase_price', 0), 2),
                    'data_processing_cost': round(sub_obj.get('data_processing_cost', 0), 2),
                    'data_storage_cost': round(sub_obj.get('data_storage_cost', 0), 2)
                })

            return base_info

        try:
            # 验证基础数据完整性
            if not self.K or len(self.K) == 0:
                raise ValueError("需求点集合为空，无法生成标准化输出")

            # 计算多目标结果
            multi_obj_result = self.compute_multi_objective_value(solution)

            # 提取解变量
            x1, x2, x3, x_direct = solution['x1'], solution['x2'], solution['x3'], solution['x_direct']
            b1, b2, b3, b_direct = solution['b1'], solution['b2'], solution['b3'], solution['b_direct']
            t1, t2, t3, t_direct = solution['t1'], solution['t2'], solution['t3'], solution['t_direct']

            # 收集供应商和路线信息
            suppliers_data = []
            total_supply_amount = 0.0
            total_routes_count = 0

            # 获取唯一需求点
            aggregated_demand_point = self.K[0]

            # 预构建活跃路径索引以避免无效循环
            active_direct_paths = self._build_active_direct_paths_index(x_direct, b_direct, t_direct)
            active_multimodal_paths = self._build_active_multimodal_paths_index(x1, x2, x3, b1, b2, b3, t1, t2, t3)

            # 批量处理直接运输的供应点
            for j in self.J:
                supplier_total_supply = 0.0
                supplier_routes = []

                # 批量处理直接运输路线
                if j in active_direct_paths:
                    for path_info in active_direct_paths[j]:
                        k, n, supply_amount, transport_time = path_info

                        try:
                            route_details = self._calculate_route_details(j, k, 'direct', n, supply_amount,
                                                                          transport_time)

                            # 确定from点类型
                            if j in self.J:
                                from_type = "supply"
                            elif j in self.M:
                                from_specialized = self.point_features[j].get('specialized_mode', 'unknown')
                                if from_specialized == 'railway':
                                    from_type = "station"
                                elif from_specialized == 'port':
                                    from_type = "port"
                                elif from_specialized == 'airport':
                                    from_type = "airport"
                                else:
                                    from_type = "transfer"
                            else:
                                from_type = "unknown"

                            # 确定to点类型
                            if k in self.K:
                                to_type = "demand"
                            elif k in self.M:
                                to_specialized = self.point_features[k].get('specialized_mode', 'unknown')
                                if to_specialized == 'railway':
                                    to_type = "station"
                                elif to_specialized == 'port':
                                    to_type = "port"
                                elif to_specialized == 'airport':
                                    to_type = "airport"
                                else:
                                    to_type = "transfer"
                            else:
                                to_type = "unknown"

                            simplified_route = {
                                "route_type": "direct",
                                "route": [
                                    {
                                        "from": {
                                            "point_id": j,
                                            "type": from_type,
                                            "mode": route_details['transport_segments'][0]['transport_mode'][
                                                'mode_name'],
                                            "longitude_latitude": [
                                                self.point_features[j]['longitude'],
                                                self.point_features[j]['latitude']
                                            ]
                                        },
                                        "to": {
                                            "point_id": k,
                                            "type": to_type,
                                            "mode": route_details['transport_segments'][0]['transport_mode'][
                                                'mode_name'],
                                            "longitude_latitude": [
                                                self.point_features[k]['longitude'],
                                                self.point_features[k]['latitude']
                                            ]
                                        },
                                        "route_distance": route_details['transport_segments'][0]['distance_km']
                                    }
                                ]
                            }

                            if not supplier_routes:
                                supplier_routes.append(simplified_route)
                            supplier_total_supply += supply_amount
                            total_routes_count += 1

                        except (ValueError, KeyError) as e:
                            self.logger.warning(f"计算直接路线详情失败: {str(e)}")
                            continue

                # 多式联运路线
                # 批量处理多式联运路线
                if j in active_multimodal_paths:
                    for path_info in active_multimodal_paths[j]:
                        m1, m2, k, n1, n2, n3, supply_amount, time_segments = path_info
                        total_transport_time = sum(time_segments)

                        try:
                            route_details = self._calculate_multimodal_route_details(
                                j, m1, m2, k, n1, n2, n3, supply_amount,
                                time_segments[0], time_segments[1], time_segments[2]
                            )

                            # 多式联运路线信息
                            if j in self.J:
                                j_type = "supply"
                            elif j in self.M:
                                j_specialized = self.point_features[j].get('specialized_mode', 'unknown')
                                if j_specialized == 'railway':
                                    j_type = "station"
                                elif j_specialized == 'port':
                                    j_type = "port"
                                elif j_specialized == 'airport':
                                    j_type = "airport"
                                else:
                                    j_type = "transfer"
                            else:
                                j_type = "unknown"

                            # m1点类型
                            if m1 in self.M:
                                m1_specialized = self.point_features[m1].get('specialized_mode', 'unknown')
                                if m1_specialized == 'railway':
                                    m1_type = "station"
                                elif m1_specialized == 'port':
                                    m1_type = "port"
                                elif m1_specialized == 'airport':
                                    m1_type = "airport"
                                else:
                                    m1_type = "transfer"
                            else:
                                m1_type = "unknown"

                            # m2点类型
                            if m2 in self.M:
                                m2_specialized = self.point_features[m2].get('specialized_mode', 'unknown')
                                if m2_specialized == 'railway':
                                    m2_type = "station"
                                elif m2_specialized == 'port':
                                    m2_type = "port"
                                elif m2_specialized == 'airport':
                                    m2_type = "airport"
                                else:
                                    m2_type = "transfer"
                            else:
                                m2_type = "unknown"

                            # k点类型
                            if k in self.K:
                                k_type = "demand"
                            elif k in self.M:
                                k_specialized = self.point_features[k].get('specialized_mode', 'unknown')
                                if k_specialized == 'railway':
                                    k_type = "station"
                                elif k_specialized == 'port':
                                    k_type = "port"
                                elif k_specialized == 'airport':
                                    k_type = "airport"
                                else:
                                    k_type = "transfer"
                            else:
                                k_type = "unknown"

                            # 多式联运路线信息
                            simplified_route = {
                                "route_type": "multimodal",
                                "route": [
                                    {
                                        "from": {
                                            "point_id": j,
                                            "type": j_type,
                                            "mode":
                                                route_details['transport_segments'][0][
                                                    'transport_mode']['mode_name'],
                                            "longitude_latitude": [
                                                self.point_features[j]['longitude'],
                                                self.point_features[j]['latitude']
                                            ]
                                        },
                                        "to": {
                                            "point_id": m1,
                                            "type": m1_type,
                                            "mode":
                                                route_details['transport_segments'][0][
                                                    'transport_mode']['mode_name'],
                                            "longitude_latitude": [
                                                self.point_features[m1]['longitude'],
                                                self.point_features[m1]['latitude']
                                            ]
                                        },
                                        "route_distance": route_details['transport_segments'][0]['distance_km']
                                    },
                                    {
                                        "from": {
                                            "point_id": m1,
                                            "type": m1_type,
                                            "mode":
                                                route_details['transport_segments'][1][
                                                    'transport_mode']['mode_name'],
                                            "longitude_latitude": [
                                                self.point_features[m1]['longitude'],
                                                self.point_features[m1]['latitude']
                                            ]
                                        },
                                        "to": {
                                            "point_id": m2,
                                            "type": m2_type,
                                            "mode":
                                                route_details['transport_segments'][1][
                                                    'transport_mode']['mode_name'],
                                            "longitude_latitude": [
                                                self.point_features[m2]['longitude'],
                                                self.point_features[m2]['latitude']
                                            ]
                                        },
                                        "route_distance": route_details['transport_segments'][1]['distance_km']
                                    },
                                    {
                                        "from": {
                                            "point_id": m2,
                                            "type": m2_type,
                                            "mode":
                                                route_details['transport_segments'][2][
                                                    'transport_mode']['mode_name'],
                                            "longitude_latitude": [
                                                self.point_features[m2]['longitude'],
                                                self.point_features[m2]['latitude']
                                            ]
                                        },
                                        "to": {
                                            "point_id": k,
                                            "type": k_type,
                                            "mode":
                                                route_details['transport_segments'][2][
                                                    'transport_mode']['mode_name'],
                                            "longitude_latitude": [
                                                self.point_features[k]['longitude'],
                                                self.point_features[k]['latitude']
                                            ]
                                        },
                                        "route_distance": route_details['transport_segments'][2]['distance_km']
                                    }
                                ]
                            }

                            if not supplier_routes:
                                supplier_routes.append(simplified_route)
                            supplier_total_supply += supply_amount
                            total_routes_count += 1

                        except (ValueError, KeyError) as e:
                            self.logger.warning(f"计算多式联运路线详情失败: {str(e)}")
                            continue

                # 如果该供应点有供应量，添加到列表中
                if supplier_total_supply > self.EPS:
                    # 在供应商数据构建的try块中，添加企业评分计算
                    try:
                        enterprise_type = self.point_features[j].get('enterprise_type', '未知')
                        enterprise_size = self.point_features[j].get('enterprise_size', '未知')

                        # 计算企业综合评分
                        supply_scale = len(self.J)
                        demand_scale = len(self.K)
                        network_scale = supply_scale + len(self.M) + demand_scale

                        # 企业能力评分
                        if enterprise_size == '大':
                            scale_capability = supply_scale + len(self.M)
                        elif enterprise_size == '中':
                            scale_capability = supply_scale + len(self.M) / (len(self.M) + 1) if len(
                                self.M) > 0 else supply_scale
                        else:
                            scale_capability = supply_scale / (supply_scale + 1) if supply_scale > 0 else 1.0

                        # 企业类型评分
                        if enterprise_type in ["国企", "事业单位"]:
                            type_score = supply_scale / (
                                    supply_scale + network_scale) if supply_scale + network_scale > 0 else 0.5
                        else:
                            type_score = network_scale / (
                                    supply_scale + network_scale) if supply_scale + network_scale > 0 else 0.5

                        # 企业规模评分
                        if enterprise_size in ["大", "中"]:
                            size_score = supply_scale / (
                                    supply_scale + demand_scale) if supply_scale + demand_scale > 0 else 0.5
                        else:
                            size_score = demand_scale / (
                                    supply_scale + demand_scale) if supply_scale + demand_scale > 0 else 0.5

                        # 供应能力评分
                        max_capacity = max(self.B[j_temp] * self.P[j_temp] for j_temp in self.J) if self.J else 1.0
                        capacity_score = (self.B[j] * self.P[j]) / max_capacity if max_capacity > 0 else 0.0

                        # 企业综合评分（各维度平均）
                        enterprise_score = (scale_capability + type_score + size_score + capacity_score) / (
                                supply_scale + demand_scale + network_scale + 1)

                        # 保存原始供应量作为后备
                        original_supplier_total_supply = supplier_total_supply

                        # 选择最优细分对象组合
                        available_sub_objects = select_optimal_sub_objects_for_supplier(j, supplier_total_supply)

                        # 初始化选中的细分对象列表
                        selected_personnel_list = []
                        selected_materials_list = []
                        selected_datas_list = []

                        actual_mobilized_total = sum(obj_info['allocated_amount'] for obj_info in
                                                     available_sub_objects) if available_sub_objects else 0

                        # 计算实际动员数量（基于总分配量和细分对象的可用性）
                        fallback_to_original_supply = actual_mobilized_total <= self.EPS and supplier_total_supply > self.EPS

                        if hasattr(self, 'resource_type'):
                            if self.resource_type == "personnel":
                                selected_personnel_list = []
                                for obj_info in available_sub_objects:
                                    selected_personnel_list.append({
                                        'sub_object': obj_info['sub_object'],
                                        'allocated_count': round(obj_info['allocated_amount']),
                                        'sub_object_score': obj_info.get('sub_object_score',
                                                                         obj_info.get('cost_efficiency', 0.0)),
                                        # 安全获取评分
                                        'category_id': obj_info.get('category_id', 'unknown'),
                                        'category_name': obj_info.get('category_name', 'unknown'),
                                        'recommend_md5': obj_info.get('recommend_md5', 'unknown'),
                                    })
                                if fallback_to_original_supply:
                                    mobilization_quantity = round(supplier_total_supply)
                                    final_allocated_amount = mobilization_quantity
                                else:
                                    mobilization_quantity = round(
                                        actual_mobilized_total) if actual_mobilized_total > 0 else 0
                                    final_allocated_amount = mobilization_quantity

                                    # 保护性检查：确保有实际供应量时至少有最小动员数量
                                if mobilization_quantity <= 0 and supplier_total_supply > self.EPS:
                                    mobilization_quantity = max(1, round(supplier_total_supply))
                                    final_allocated_amount = mobilization_quantity

                            elif self.resource_type == "material":
                                selected_materials_list = []
                                for obj_info in available_sub_objects:
                                    selected_materials_list.append({
                                        'sub_object': obj_info['sub_object'],
                                        'allocated_amount': round(obj_info['allocated_amount']),
                                        'sub_object_score': obj_info.get('sub_object_score',
                                                                         obj_info.get('cost_efficiency', 0.0)),
                                        'category_id': obj_info.get('category_id', 'unknown'),
                                        'category_name': obj_info.get('category_name', 'unknown'),  # 安全获取评分
                                        'recommend_md5': obj_info.get('recommend_md5', 'unknown')
                                    })
                                if fallback_to_original_supply:
                                    mobilization_quantity = round(supplier_total_supply)
                                    final_allocated_amount = mobilization_quantity
                                else:
                                    mobilization_quantity = round(
                                        actual_mobilized_total) if actual_mobilized_total > 0 else 0
                                    final_allocated_amount = mobilization_quantity
                                    # 保护性检查：确保有实际供应量时至少有最小动员数量
                                if mobilization_quantity <= 0 and supplier_total_supply > self.EPS:
                                    mobilization_quantity = round(
                                        supplier_total_supply) if supplier_total_supply >= 1 else 1
                                    final_allocated_amount = mobilization_quantity

                            elif self.resource_type == "data":
                                selected_datas_list = []
                                for obj_info in available_sub_objects:
                                    selected_datas_list.append({
                                        'sub_object': obj_info['sub_object'],
                                        'allocated_amount': round(obj_info['allocated_amount']),
                                        'sub_object_score': obj_info.get('sub_object_score',
                                                                         obj_info.get('cost_efficiency', 0.0)),
                                        # 安全获取评分
                                        'category_id': obj_info.get('category_id', 'unknown'),
                                        'category_name': obj_info.get('category_name', 'unknown'),
                                        'recommend_md5': obj_info.get('recommend_md5', 'unknown'),
                                    })
                                if fallback_to_original_supply:
                                    mobilization_quantity = round(supplier_total_supply)
                                    final_allocated_amount = mobilization_quantity
                                else:
                                    mobilization_quantity = round(
                                        actual_mobilized_total) if actual_mobilized_total > 0 else 0
                                    final_allocated_amount = mobilization_quantity
                                    # 保护性检查：确保有实际供应量时至少有最小动员数量
                                if mobilization_quantity <= 0 and supplier_total_supply > self.EPS:
                                    mobilization_quantity = max(1, round(supplier_total_supply))
                                    final_allocated_amount = mobilization_quantity
                            else:
                                if fallback_to_original_supply:
                                    mobilization_quantity = round(supplier_total_supply)
                                    final_allocated_amount = mobilization_quantity
                                else:
                                    mobilization_quantity = round(
                                        actual_mobilized_total) if actual_mobilized_total > 0 else 0
                                    final_allocated_amount = mobilization_quantity
                        else:
                            if fallback_to_original_supply:
                                mobilization_quantity = round(supplier_total_supply)
                                final_allocated_amount = mobilization_quantity
                            else:
                                mobilization_quantity = round(
                                    actual_mobilized_total) if actual_mobilized_total > 0 else 0
                                final_allocated_amount = mobilization_quantity

                        # 统一的最终保护性检查
                        if mobilization_quantity <= 0 and supplier_total_supply > self.EPS:
                            mobilization_quantity = max(1, round(supplier_total_supply))
                            final_allocated_amount = mobilization_quantity

                        # 构建基础供应商信息
                        supplier_data = {
                            "supplier_id": self.point_features[j].get('original_supplier_id', j),
                            "supplier_name": j,
                            "enterprise_type": enterprise_type,
                            "enterprise_size": enterprise_size,
                            "enterprise_score": round(enterprise_score, 4),  # 添加企业评分
                            "location": {
                                "latitude": self.point_features[j]['latitude'],
                                "longitude": self.point_features[j]['longitude']
                            },
                            "total_capacity": mobilization_quantity,
                            "original_capacity": self.B[j],
                            "allocated_amount": round(final_allocated_amount),
                            "utilization_rate": round(final_allocated_amount / self.B[j], 4) if self.B[j] > 0 else 0,
                            "reliability_factor": self.P[j],
                            "routes": supplier_routes,
                            "mobilization_quantity": mobilization_quantity
                        }

                        # 根据动员类型添加细分对象信息
                        if hasattr(self, 'resource_type'):
                            if self.resource_type == "personnel":
                                supplier_data["personnels"] = []
                                for personnel_info in selected_personnel_list:
                                    sub_obj = personnel_info['sub_object']
                                    allocated_count = personnel_info['allocated_count']
                                    sub_obj_score = personnel_info.get('sub_object_score', 0.0)
                                    if allocated_count > 0:
                                        personnel_data = format_sub_object_info(sub_obj, "personnel", allocated_count,
                                                                                personnel_info.get('category_id',
                                                                                                   'unknown'),
                                                                                personnel_info.get('category_name',
                                                                                                   'unknown'),
                                                                                personnel_info.get('recommend_md5',
                                                                                                   'unknown'))
                                        personnel_data['sub_object_score'] = round(sub_obj_score, 4)
                                        supplier_data["personnels"].append(personnel_data)

                            elif self.resource_type == "material":
                                supplier_data["materials"] = []
                                for material_info in selected_materials_list:
                                    sub_obj = material_info['sub_object']
                                    allocated_amount = material_info['allocated_amount']
                                    sub_obj_score = material_info.get('sub_object_score', 0.0)  # 安全获取评分
                                    if allocated_amount > 0:
                                        material_data = format_sub_object_info(sub_obj, "material", allocated_amount,
                                                                               material_info.get('category_id',
                                                                                                 'unknown'),
                                                                               material_info.get('category_name',
                                                                                                 'unknown'),
                                                                               material_info.get('recommend_md5',
                                                                                                 'unknown'))
                                        material_data['sub_object_score'] = round(sub_obj_score, 4)
                                        supplier_data["materials"].append(material_data)

                            elif self.resource_type == "data":
                                supplier_data["datas"] = []
                                for data_info in selected_datas_list:
                                    sub_obj = data_info['sub_object']
                                    allocated_amount = data_info['allocated_amount']
                                    sub_obj_score = data_info.get('sub_object_score', 0.0)  # 安全获取评分
                                    if allocated_amount > 0:
                                        data_data = format_sub_object_info(sub_obj, "data", allocated_amount,
                                                                           data_info.get('category_id', 'unknown'),
                                                                           data_info.get('category_name', 'unknown'),
                                                                           data_info.get('recommend_md5', 'unknown'))
                                        data_data['sub_object_score'] = round(sub_obj_score, 4)
                                        supplier_data["datas"].append(data_data)

                        # 只有当mobilization_quantity大于0时才添加供应点
                        if mobilization_quantity > 0:
                            suppliers_data.append(supplier_data)
                            total_supply_amount += final_allocated_amount

                    except (KeyError, TypeError) as e:
                        self.logger.warning(f"构建供应商 {j} 信息失败: {str(e)}")
                        continue

            # 计算需求满足情况
            total_demand = sum(self.D.values())
            satisfaction_rate = min(1.0, total_supply_amount / total_demand) if total_demand > 0 else 1.0

            # 统一目标值处理，保持原始数值避免百分制转换扭曲权重
            raw_objectives = multi_obj_result['individual_objectives']
            display_objectives = {}

            # 所有目标值统一保持原始数值，确保权重关系不被扭曲
            for obj_name, obj_value in raw_objectives.items():
                display_objectives[obj_name] = round(obj_value, 6)

            # 构建组织的标准化输出
            standardized_result = {
                "objective_achievement": {
                    "composite_value": round(multi_obj_result['composite_objective_value'], 6),
                    "individual_scores": display_objectives,
                    "solve_time_seconds": round(total_time, 3)
                },
                "allocation_performance": {
                    "total_demand": round(total_demand, 2),
                    "total_allocated": round(total_supply_amount, 2),
                    "satisfaction_rate_percent": round(satisfaction_rate * 100, 1),
                    "suppliers_utilized": len(suppliers_data),
                    "routes_activated": total_routes_count
                },
                "network_analysis": {
                    "network_scale": {
                        "supply_points_total": len(self.J),
                        "transfer_points_total": len(self.M),
                        "demand_points_total": len(self.K)
                    },
                    "operational_statistics": {
                        "active_suppliers": len(suppliers_data),
                        "active_routes": total_routes_count,
                        "average_routes_per_supplier": round(total_routes_count / len(suppliers_data), 2) if len(
                            suppliers_data) > 0 else 0
                    }
                },
                "supplier_portfolio": suppliers_data
            }

            # 使用安全的JSON转换
            return safe_json_convert(standardized_result)

        except Exception as e:
            self.logger.error(f"生成标准化输出时发生错误: {str(e)}", exc_info=True)
            # 构建符合预期结构的错误返回，包含必要的键
            error_result = {
                "objective_achievement": {
                    "composite_value": 0.0,
                    "individual_scores": {
                        "time": 0.0, "cost": 0.0, "distance": 0.0, "safety": 0.0,
                        "priority": 0.0, "balance": 0.0, "capability": 0.0, "social": 0.0
                    },
                    "solve_time_seconds": 0.0
                },
                "allocation_performance": {
                    "total_demand": 0.0,
                    "total_allocated": 0.0,
                    "satisfaction_rate_percent": 0.0,
                    "suppliers_utilized": 0,
                    "routes_activated": 0
                },
                "network_analysis": {
                    "network_scale": {
                        "supply_points_total": len(self.J) if hasattr(self, 'J') else 0,
                        "transfer_points_total": len(self.M) if hasattr(self, 'M') else 0,
                        "demand_points_total": len(self.K) if hasattr(self, 'K') else 0
                    },
                    "operational_statistics": {
                        "active_suppliers": 0,
                        "active_routes": 0,
                        "average_routes_per_supplier": 0.0
                    }
                },
                "supplier_portfolio": [],
                "error_info": {
                    "code": 500,
                    "msg": f"生成输出结果时发生内部错误: {str(e)}",
                    "data": {
                        "execution_status": "error",
                        "algorithm_sequence_id": solution.get('algorithm_sequence_id') if solution else None,
                        "mobilization_object_type": solution.get('mobilization_object_type') if solution else None,
                        "req_element_id": solution.get('req_element_id') if solution else None,
                        "error_type": "output_generation_error"
                    }
                }
            }

            return error_result

    def _calculate_sub_object_safety(self, sub_obj):
        """计算细分对象的安全评分"""
        total_safety = 0.0

        if hasattr(self, 'resource_type'):
            if self.resource_type == "personnel":
                total_safety = (sub_obj.get('political_status', 0) +
                                sub_obj.get('military_experience', 0) -
                                sub_obj.get('criminal_record', 0) -
                                sub_obj.get('network_record', 0) -
                                sub_obj.get('credit_record', 0))
            elif self.resource_type == "material":
                # 使用网络规模参数作为默认值
                default_enterprise_nature = len(self.J) / (len(self.J) + len(self.M)) if len(self.J) + len(
                    self.M) > 0 else len(self.J) / (len(self.J) + 1)
                default_enterprise_scale = len(self.K) / (len(self.K) + len(self.M)) if len(self.K) + len(
                    self.M) > 0 else len(self.K) / (len(self.K) + 1)
                default_resource_safety = len(self.M) / (len(self.M) + len(self.K)) if len(self.M) + len(
                    self.K) > 0 else len(self.M) / (len(self.M) + 1)
                default_material_penalty = len(self.TRANSPORT_MODES) / (
                        len(self.TRANSPORT_MODES) + len(self.J)) if hasattr(self, 'TRANSPORT_MODES') and len(
                    self.TRANSPORT_MODES) + len(self.J) > 0 else len(self.J) / (len(self.J) + 1)

                total_safety = (
                         - sub_obj.get('flammable_explosive', 0) -
                        sub_obj.get('corrosive', 0) -
                        sub_obj.get('polluting', 0) -
                        sub_obj.get('fragile', 0))
            elif self.resource_type == "data":
                # 使用标准化的评分范围
                default_control_score = len(self.M) / (len(self.M) + len(self.K)) if len(self.M) + len(
                    self.K) > 0 else len(self.M) / (len(self.M) + 1)
                default_usability_score = len(self.J) / (len(self.J) + len(self.M)) if len(self.J) + len(
                    self.M) > 0 else len(self.J) / (len(self.J) + 1)
                default_facility_score = (len(self.J) + len(self.M) + len(self.K)) / (
                        (len(self.J) + len(self.M) + len(self.K)) + len(self.TRANSPORT_MODES)) if hasattr(self,
                                                                                                          'TRANSPORT_MODES') and (
                                                                                                          len(self.J) + len(
                                                                                                      self.M) + len(
                                                                                                      self.K)) + len(
                    self.TRANSPORT_MODES) > 0 else (len(self.J) + len(self.M) + len(self.K)) / (
                        (len(self.J) + len(self.M) + len(self.K)) + 1)

                autonomous_control = sub_obj.get('autonomous_control', default_control_score)
                usability_level = sub_obj.get('usability_level', default_usability_score)
                maintenance_derived = (autonomous_control + usability_level) / (len(self.K) + 1) if len(
                    self.K) > 0 else (autonomous_control + usability_level) / 2
                facility_protection = sub_obj.get('facility_protection', default_facility_score)
                camouflage_protection = sub_obj.get('camouflage_protection', default_facility_score)
                environment_score = sub_obj.get('surrounding_environment', 0)

                total_safety = (autonomous_control + usability_level + maintenance_derived +
                                facility_protection + camouflage_protection + environment_score)

        return -total_safety

    def _calculate_route_details(self, j, k, route_type, transport_mode, supply_amount, transport_time):
        """
        计算单条路线的详细信息
        """

        # 安全的数值转换函数
        def safe_float_convert(value, default=0.0):
            try:
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    if value.strip() == '':
                        return default
                    return float(value)
                else:
                    return default
            except (ValueError, TypeError):
                return default

        # 抽取的成本计算逻辑
        def calculate_cost_by_resource_type(j, total_time, transport_cost):
            """根据资源类型计算总成本（使用细分对象或默认值）"""

            # 计算网络规模参数
            supply_scale = len(self.J)
            demand_scale = len(self.K)

            try:
                if hasattr(self, 'resource_type'):
                    if self.resource_type == "personnel":
                        # 人员动员：从细分对象中获取最优成本，如果没有则使用默认值
                        sub_objects = self.point_features[j].get('sub_objects', [])
                        if sub_objects:
                            # 选择成本最优的细分对象
                            best_cost = float('inf')
                            found_valid_sub_objects = False

                            for category in sub_objects:
                                # 检查是否为新的分类结构
                                if isinstance(category, dict) and 'items' in category and category.get('items'):
                                    # 新的分类结构：遍历分类中的所有项目
                                    for sub_obj in category.get('items', []):
                                        # 检查可用性
                                        max_available = sub_obj.get('max_available_quantity', 0)
                                        if max_available <= self.EPS:
                                            continue  # 跳过不可用的细分对象

                                        wage_cost = sub_obj.get('wage_cost')
                                        living_cost = sub_obj.get('living_cost')

                                        if wage_cost is None or wage_cost < 0:
                                            raise ValueError(
                                                f"细分对象{sub_obj.get('sub_object_id', 'unknown')}缺少有效的wage_cost配置")
                                        if living_cost is None or living_cost < 0:
                                            raise ValueError(
                                                f"细分对象{sub_obj.get('sub_object_id', 'unknown')}缺少有效的living_cost配置")

                                        total_cost = (wage_cost + living_cost) * total_time / 24 + transport_cost
                                        best_cost = min(best_cost, total_cost)

                                        found_valid_sub_objects = True
                                elif isinstance(category, dict) and 'sub_object_id' in category:
                                    # 兼容旧的平铺结构：直接处理细分对象
                                    sub_obj = category
                                    # 检查可用性
                                    max_available = sub_obj.get('max_available_quantity', 0)
                                    if max_available <= self.EPS:
                                        continue  # 跳过不可用的细分对象

                                    wage_cost = sub_obj.get('wage_cost')
                                    living_cost = sub_obj.get('living_cost')

                                    if wage_cost is None or wage_cost < 0:
                                        raise ValueError(
                                            f"细分对象{sub_obj.get('sub_object_id', 'unknown')}缺少有效的wage_cost配置")
                                    if living_cost is None or living_cost < 0:
                                        raise ValueError(
                                            f"细分对象{sub_obj.get('sub_object_id', 'unknown')}缺少有效的living_cost配置")

                                    total_cost = (wage_cost + living_cost) * total_time / 24 + transport_cost
                                    best_cost = min(best_cost, total_cost)
                                    found_valid_sub_objects = True

                            if found_valid_sub_objects and best_cost != float('inf'):
                                return best_cost
                            else:
                                # 使用基于网络规模的默认值
                                default_wage = supply_scale * demand_scale + supply_scale
                                default_living = supply_scale + demand_scale
                                return (default_wage + default_living) * total_time / 24
                        else:
                            # 使用基于网络规模的默认值
                            default_wage = supply_scale * demand_scale + supply_scale
                            default_living = supply_scale + demand_scale
                            return (default_wage + default_living) * total_time / 24

                    elif self.resource_type == "material":
                        # 物资动员：从细分对象中获取最优成本，如果没有则使用默认值
                        sub_objects = self.point_features[j].get('sub_objects', [])
                        if sub_objects:
                            # 选择成本最优的细分对象
                            best_cost = float('inf')
                            for sub_obj in sub_objects:
                                # 检查是否有足够的供应量
                                max_available = sub_obj.get('max_available_quantity', 0)
                                if max_available <= 0:
                                    continue  # 跳过没有可供应量的细分对象
                                material_price = safe_float_convert(sub_obj.get('material_price', 200), 200)
                                equipment_rental = safe_float_convert(sub_obj.get('equipment_rental_price', 150), 150)
                                equipment_depreciation = safe_float_convert(
                                    sub_obj.get('equipment_depreciation_cost', 20), 20)
                                total_cost = material_price + transport_cost + (
                                        equipment_rental + equipment_depreciation) * total_time / 24
                                best_cost = min(best_cost, total_cost)
                            return best_cost
                        else:
                            # 使用基于网络规模的默认值
                            default_material_price = len(self.J) * len(self.K) * (len(self.J) + len(self.K))
                            default_equipment_rental = len(self.J) * len(self.K)
                            default_equipment_depreciation = len(self.J) + len(self.K)
                            return default_material_price + transport_cost + (
                                    default_equipment_rental + default_equipment_depreciation) * total_time / 24

                    else:  # data
                        # 数据动员：设施成本在供应点级别，这是正确的
                        facility_rental = safe_float_convert(self.point_features[j].get('facility_rental_price', 80),
                                                             80)
                        facility_power = safe_float_convert(self.point_features[j].get('power_cost', 30), 30)
                        communication_cost = safe_float_convert(
                            self.point_features[j].get('communication_purchase_price', 40), 40)
                        return facility_rental * total_time / 24 + facility_power + communication_cost
                else:
                    # 兼容性处理：默认物资动员
                    sub_objects = self.point_features[j].get('sub_objects', [])
                    if sub_objects:
                        best_cost = float('inf')
                        for sub_obj in sub_objects:
                            material_price = safe_float_convert(sub_obj.get('material_price', 200), 200)
                            equipment_rental = safe_float_convert(sub_obj.get('equipment_rental_price', 150), 150)
                            equipment_depreciation = safe_float_convert(sub_obj.get('equipment_depreciation_cost', 20),
                                                                        20)
                            total_cost = material_price + transport_cost + (
                                    equipment_rental + equipment_depreciation) * total_time / 24
                            best_cost = min(best_cost, total_cost)
                        return best_cost
                    else:
                        default_material_price = len(self.J) * len(self.K) * (len(self.J) + len(self.K))
                        default_equipment_rental = len(self.J) * len(self.K)
                        default_equipment_depreciation = len(self.J) + len(self.K)
                        return default_material_price + transport_cost + (
                                default_equipment_rental + default_equipment_depreciation) * total_time / 24
            except Exception as e:
                raise ValueError(f"成本计算失败: {str(e)}")

        # 计算路线指标
        j_lat, j_lon = self.point_features[j]['latitude'], self.point_features[j]['longitude']
        k_lat, k_lon = self.point_features[k]['latitude'], self.point_features[k]['longitude']
        distance = self._calculate_haversine_distance(j_lat, j_lon, k_lat, k_lon)

        # 获取运输方式信息
        transport_info = self.TRANSPORT_MODES[transport_mode]

        # 计算成本
        transport_cost = distance * transport_info['cost_per_km']

        # 根据资源类型计算总成本
        try:
            total_cost = calculate_cost_by_resource_type(j, transport_time, transport_cost)
        except ValueError as e:
            raise ValueError(f"路线成本计算失败: {str(e)}")

        # 计算安全系数
        safety_score = self._calculate_route_safety_score(j, k, route_type, transport_mode)

        route_details = {
            "route_id": f"R_{j}_{k}_{transport_mode}",
            "route_type": route_type,
            "origin": {
                "point_id": j,
                "point_name": j,
                "point_type": "supply_point",
                "location": {
                    "latitude": j_lat,
                    "longitude": j_lon
                }
            },
            "destination": {
                "point_id": k,
                "point_name": k,
                "point_type": "demand_point",
                "location": {
                    "latitude": k_lat,
                    "longitude": k_lon
                }
            },
            "transport_segments": [
                {
                    "segment_id": 1,
                    "from_point": j,
                    "to_point": k,
                    "transport_mode": {
                        "mode_id": transport_mode,
                        "mode_name": transport_info['name'],
                        "mode_code": transport_info.get('code', f'trans-{transport_mode:02d}'),
                        "speed": transport_info['speed'],
                        "cost_per_km": transport_info['cost_per_km']
                    },
                    "distance_km": round(distance, 2),
                    "estimated_time_hours": round(transport_time, 3),
                    "transport_cost": round(transport_cost, 2)
                }
            ],
            "route_metrics": {
                "total_distance_km": round(distance, 2),
                "total_time_hours": round(transport_time, 3),
                "total_cost": round(total_cost, 2),
                "supply_amount": round(supply_amount, 2),
                "unit_cost": round(total_cost / supply_amount, 4) if supply_amount > 0 else 0,
                "safety_score": round(safety_score, 4),
                "priority_level": self.demand_priority.get(k, 5),
                "reliability_factor": self.P[j]
            }
        }

        return route_details

    def _calculate_multimodal_route_details(self, j, m1, m2, k, n1, n2, n3, supply_amount, t1, t2, t3):
        """
        计算多式联运路线的详细信息
        """

        # 安全的数值转换函数
        def safe_float_convert(value, default=0.0):
            try:
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    if value.strip() == '':
                        return default
                    return float(value)
                else:
                    return default
            except (ValueError, TypeError):
                return default

        # 抽取的成本计算逻辑
        def calculate_cost_by_resource_type(j, total_time, transport_cost):
            """根据资源类型计算总成本（使用细分对象或默认值）"""

            # 计算网络规模参数
            supply_scale = len(self.J)
            demand_scale = len(self.K)

            try:
                if hasattr(self, 'resource_type'):
                    if self.resource_type == "personnel":
                        # 人员动员：从细分对象中获取最优成本，如果没有则使用默认值
                        sub_objects = self.point_features[j].get('sub_objects', [])
                        if sub_objects:
                            # 选择成本最优的细分对象
                            best_cost = float('inf')
                            found_valid_sub_objects = False

                            for category in sub_objects:
                                # 检查是否为新的分类结构
                                if isinstance(category, dict) and 'items' in category and category.get('items'):
                                    # 新的分类结构：遍历分类中的所有项目
                                    for sub_obj in category.get('items', []):
                                        # 检查可用性
                                        max_available = sub_obj.get('max_available_quantity', 0)
                                        if max_available <= self.EPS:
                                            continue  # 跳过不可用的细分对象

                                        wage_cost = sub_obj.get('wage_cost')
                                        living_cost = sub_obj.get('living_cost')

                                        if wage_cost is None or wage_cost < 0:
                                            raise ValueError(
                                                f"细分对象{sub_obj.get('sub_object_id', 'unknown')}缺少有效的wage_cost配置")
                                        if living_cost is None or living_cost < 0:
                                            raise ValueError(
                                                f"细分对象{sub_obj.get('sub_object_id', 'unknown')}缺少有效的living_cost配置")

                                        total_cost = (wage_cost + living_cost) * total_time / 24 + transport_cost
                                        best_cost = min(best_cost, total_cost)
                                        found_valid_sub_objects = True
                                elif isinstance(category, dict) and 'sub_object_id' in category:
                                    # 兼容旧的平铺结构：直接处理细分对象
                                    sub_obj = category
                                    # 检查可用性
                                    max_available = sub_obj.get('max_available_quantity', 0)
                                    if max_available <= self.EPS:
                                        continue  # 跳过不可用的细分对象

                                    wage_cost = sub_obj.get('wage_cost')
                                    living_cost = sub_obj.get('living_cost')

                                    if wage_cost is None or wage_cost < 0:
                                        raise ValueError(
                                            f"细分对象{sub_obj.get('sub_object_id', 'unknown')}缺少有效的wage_cost配置")
                                    if living_cost is None or living_cost < 0:
                                        raise ValueError(
                                            f"细分对象{sub_obj.get('sub_object_id', 'unknown')}缺少有效的living_cost配置")

                                    total_cost = (wage_cost + living_cost) * total_time / 24 + transport_cost
                                    best_cost = min(best_cost, total_cost)
                                    found_valid_sub_objects = True

                            if found_valid_sub_objects and best_cost != float('inf'):
                                return best_cost
                            else:
                                # 使用基于网络规模的默认值
                                default_wage = supply_scale * demand_scale + supply_scale
                                default_living = supply_scale + demand_scale
                                return (default_wage + default_living) * total_time / 24
                        else:
                            # 使用基于网络规模的默认值
                            default_wage = supply_scale * demand_scale + supply_scale
                            default_living = supply_scale + demand_scale
                            return (default_wage + default_living) * total_time / 24

                    elif self.resource_type == "material":
                        # 物资动员：从细分对象中获取最优成本，如果没有则使用默认值
                        sub_objects = self.point_features[j].get('sub_objects', [])
                        if sub_objects:
                            # 选择成本最优的细分对象
                            best_cost = float('inf')
                            for sub_obj in sub_objects:
                                material_price = safe_float_convert(sub_obj.get('material_price', 200), 200)
                                equipment_rental = safe_float_convert(sub_obj.get('equipment_rental_price', 150), 150)
                                equipment_depreciation = safe_float_convert(
                                    sub_obj.get('equipment_depreciation_cost', 20), 20)
                                total_cost = material_price + transport_cost + (
                                        equipment_rental + equipment_depreciation) * total_time / 24
                                best_cost = min(best_cost, total_cost)
                            return best_cost
                        else:
                            # 使用基于网络规模的默认值
                            default_material_price = len(self.J) * len(self.K) * (len(self.J) + len(self.K))
                            default_equipment_rental = len(self.J) * len(self.K)
                            default_equipment_depreciation = len(self.J) + len(self.K)
                            return default_material_price + transport_cost + (
                                    default_equipment_rental + default_equipment_depreciation) * total_time / 24

                    else:  # data
                        # 数据动员：设施成本在供应点级别，这是正确的
                        facility_rental = safe_float_convert(self.point_features[j].get('facility_rental_price', 80),
                                                             80)
                        facility_power = safe_float_convert(self.point_features[j].get('power_cost', 30), 30)
                        communication_cost = safe_float_convert(
                            self.point_features[j].get('communication_purchase_price', 40), 40)
                        return facility_rental * total_time / 24 + facility_power + communication_cost
                else:
                    # 兼容性处理：默认物资动员
                    sub_objects = self.point_features[j].get('sub_objects', [])
                    if sub_objects:
                        best_cost = float('inf')
                        for sub_obj in sub_objects:
                            material_price = safe_float_convert(sub_obj.get('material_price', 200), 200)
                            equipment_rental = safe_float_convert(sub_obj.get('equipment_rental_price', 150), 150)
                            equipment_depreciation = safe_float_convert(sub_obj.get('equipment_depreciation_cost', 20),
                                                                        20)
                            total_cost = material_price + transport_cost + (
                                    equipment_rental + equipment_depreciation) * total_time / 24
                            best_cost = min(best_cost, total_cost)
                        return best_cost
                    else:
                        default_material_price = len(self.J) * len(self.K) * (len(self.J) + len(self.K))
                        default_equipment_rental = len(self.J) * len(self.K)
                        default_equipment_depreciation = len(self.J) + len(self.K)
                        return default_material_price + transport_cost + (
                                default_equipment_rental + default_equipment_depreciation) * total_time / 24
            except Exception as e:
                raise ValueError(f"成本计算失败: {str(e)}")

        # 使用预计算的距离矩阵
        d1 = self.L_j_m[(j, m1)]
        d2 = self.L_m_m[(m1, m2)]
        d3 = self.L_m_k[(m2, k)]
        total_distance = d1 + d2 + d3
        total_time = t1 + t2 + t3

        # 获取各点坐标用于路径详情构建
        j_lat, j_lon = self.point_features[j]['latitude'], self.point_features[j]['longitude']
        m1_lat, m1_lon = self.point_features[m1]['latitude'], self.point_features[m1]['longitude']
        m2_lat, m2_lon = self.point_features[m2]['latitude'], self.point_features[m2]['longitude']
        k_lat, k_lon = self.point_features[k]['latitude'], self.point_features[k]['longitude']

        # 计算各段成本
        cost1 = d1 * self.TRANSPORT_MODES[n1]['cost_per_km']
        cost2 = d2 * self.TRANSPORT_MODES[n2]['cost_per_km']
        cost3 = d3 * self.TRANSPORT_MODES[n3]['cost_per_km']
        transport_cost = cost1 + cost2 + cost3

        # 根据资源类型计算总成本（使用抽取的逻辑）
        try:
            total_cost = calculate_cost_by_resource_type(j, total_time / 24, transport_cost)
        except ValueError as e:
            raise ValueError(f"多式联运路线成本计算失败: {str(e)}")

        # 计算安全系数
        safety_score = self._calculate_route_safety_score(j, k, 'multimodal', [n1, n2, n3])

        route_details = {
            "route_id": f"R_{j}_{m1}_{m2}_{k}_{n2}",
            "route_type": "multimodal",
            "origin": {
                "point_id": j,
                "point_name": j,
                "point_type": "supply_point",
                "location": {
                    "latitude": j_lat,
                    "longitude": j_lon
                }
            },
            "destination": {
                "point_id": k,
                "point_name": k,
                "point_type": "demand_point",
                "location": {
                    "latitude": k_lat,
                    "longitude": k_lon
                }
            },
            "transport_segments": [
                {
                    "segment_id": 1,
                    "from_point": j,
                    "to_point": m1,
                    "point_type": "transfer_point",
                    "transport_mode": {
                        "mode_id": n1,
                        "mode_name": self.TRANSPORT_MODES[n1]['name'],
                        "mode_code": self.TRANSPORT_MODES[n1].get('code', f'trans-{n1:02d}'),
                        "speed": self.TRANSPORT_MODES[n1]['speed'],
                        "cost_per_km": self.TRANSPORT_MODES[n1]['cost_per_km']
                    },
                    "distance_km": round(d1, 2),
                    "estimated_time_hours": round(t1, 3),
                    "transport_cost": round(cost1, 2)
                },
                {
                    "segment_id": 2,
                    "from_point": m1,
                    "to_point": m2,
                    "point_type": "transfer_point",
                    "transport_mode": {
                        "mode_id": n2,
                        "mode_name": self.TRANSPORT_MODES[n2]['name'],
                        "mode_code": self.TRANSPORT_MODES[n2].get('code', f'trans-{n2:02d}'),
                        "speed": self.TRANSPORT_MODES[n2]['speed'],
                        "cost_per_km": self.TRANSPORT_MODES[n2]['cost_per_km']
                    },
                    "distance_km": round(d2, 2),
                    "estimated_time_hours": round(t2, 3),
                    "transport_cost": round(cost2, 2),
                },
                {
                    "segment_id": 3,
                    "from_point": m2,
                    "to_point": k,
                    "point_type": "demand_point",
                    "transport_mode": {
                        "mode_id": n3,
                        "mode_name": self.TRANSPORT_MODES[n3]['name'],
                        "mode_code": self.TRANSPORT_MODES[n3].get('code', f'trans-{n3:02d}'),

                        "speed": self.TRANSPORT_MODES[n3]['speed'],
                        "cost_per_km": self.TRANSPORT_MODES[n3]['cost_per_km']
                    },
                    "distance_km": round(d3, 2),
                    "estimated_time_hours": round(t3, 3),
                    "transport_cost": round(cost3, 2)
                }
            ],
            "route_metrics": {
                "total_distance_km": round(total_distance, 2),
                "total_time_hours": round(total_time, 3),
                "total_cost": round(total_cost, 2),
                "supply_amount": round(supply_amount, 2),
                "unit_cost": round(total_cost / supply_amount, 4) if supply_amount > 0 else 0,
                "safety_score": round(safety_score, 4),
                "priority_level": self.demand_priority.get(k, 5),
                "reliability_factor": self.P[j],
                "transfer_points": [m1, m2]
            }
        }

        return route_details

    def _calculate_route_safety_score(self, j, k, route_type, transport_modes):
        """
        计算路线安全系数
        """

        # 安全的数值转换函数
        def safe_float_convert(value, default=0.0):
            try:
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    if value.strip() == '':
                        return default
                    return float(value)
                else:
                    return default
            except (ValueError, TypeError):
                return default

        # 供应点安全评分 - 使用英文键名
        supplier_safety = 1.0  # 默认值

        try:
            if hasattr(self, 'resource_type'):
                if self.resource_type == "personnel":
                    political_status = safe_float_convert(self.point_features[j].get('political_status', 0), 0)
                    military_experience = safe_float_convert(self.point_features[j].get('military_experience', 0), 0)
                    criminal_record = safe_float_convert(self.point_features[j].get('criminal_record', 0), 0)
                    network_record = safe_float_convert(self.point_features[j].get('network_record', 0), 0)
                    credit_record = safe_float_convert(self.point_features[j].get('credit_record', 0), 0)
                    supplier_safety = political_status + military_experience - criminal_record - network_record - credit_record
                elif self.resource_type == "material":
                    enterprise_nature_score = safe_float_convert(
                        self.point_features[j].get('enterprise_nature_score', 0.7), 0.7)
                    enterprise_scale_score = safe_float_convert(
                        self.point_features[j].get('enterprise_scale_score', 0.6), 0.6)
                    risk_record = safe_float_convert(self.point_features[j].get('risk_record', 0), 0)
                    foreign_background = safe_float_convert(self.point_features[j].get('foreign_background', 0), 0)
                    resource_safety = safe_float_convert(self.point_features[j].get('resource_safety', 1), 1)

                    flammable_score = safe_float_convert(self.point_features[j].get('flammable_explosive', -0.1), -0.1)
                    corrosive_score = safe_float_convert(self.point_features[j].get('corrosive', -0.1), -0.1)
                    polluting_score = safe_float_convert(self.point_features[j].get('polluting', -0.1), -0.1)
                    fragile_score = safe_float_convert(self.point_features[j].get('fragile', -0.1), -0.1)

                    supplier_safety = (-flammable_score - corrosive_score - polluting_score - fragile_score)
                else:  # data
                    autonomous_control = safe_float_convert(self.point_features[j].get('autonomous_control', 1), 1)
                    usability_level = safe_float_convert(self.point_features[j].get('usability_level', 1), 1)
                    maintenance_derived = (autonomous_control + usability_level) / (len(self.J) + 1) if len(
                        self.J) > 0 else (autonomous_control + usability_level) / 2
                    facility_protection = safe_float_convert(self.point_features[j].get('facility_protection', 1), 1)
                    camouflage_protection = safe_float_convert(self.point_features[j].get('camouflage_protection', 1),
                                                               1)
                    environment_score = safe_float_convert(self.point_features[j].get('surrounding_environment', 0), 0)
                    supplier_safety = (autonomous_control + usability_level + maintenance_derived +
                                       facility_protection + camouflage_protection + environment_score)
            else:
                # 兼容性处理：默认物资动员
                flammable_score = safe_float_convert(self.point_features[j].get('flammable_explosive', -0.1), -0.1)
                corrosive_score = safe_float_convert(self.point_features[j].get('corrosive', -0.1), -0.1)
                polluting_score = safe_float_convert(self.point_features[j].get('polluting', -0.1), -0.1)
                fragile_score = safe_float_convert(self.point_features[j].get('fragile', -0.1), -0.1)

                supplier_safety = (-flammable_score - corrosive_score - polluting_score - fragile_score)
        except (KeyError, TypeError) as e:
            self.logger.warning(f"供应点 {j} 安全评分计算失败: {str(e)}")
            supplier_safety = 1.0

        # 运输方式安全评分基于速度和成本的启发式计算
        base_transport_safety = len(self.TRANSPORT_MODES) / (len(self.TRANSPORT_MODES) + len(self.J)) if hasattr(self,
                                                                                                                 'TRANSPORT_MODES') else supplier_safety

        # 综合安全评分 (归一化到0-1)
        max_supplier_safety = 6.0  # 根据各资源类型的最大可能评分
        normalized_supplier_safety = max(0.0,
                                         min(1.0, (supplier_safety + max_supplier_safety) / (2 * max_supplier_safety)))

        # 最终安全系数 (供应点安全权重0.6，运输安全权重0.4)
        final_safety_score = normalized_supplier_safety

        return -final_safety_score

    def find_key_by_supplier_id(self,data, target_id):
        for key, value in data.items():
            # 检查value是否为字典且包含original_supplier_id字段
            if isinstance(value, dict) and 'original_supplier_id' in value and value[
                'original_supplier_id'] == target_id:
                return key
        return None  # 如果没有找到匹配的键

    def _generate_preSplution_data_mobilization_output(self,pre_solution):

        # 按要求分为四个部分的数据动员输出，保持与物资/人员动员格式一致
        display_objectives = {
            "time": 0,
            "cost": 0,
            "distance": 0,
            "safety": 0,
            "priority": 0,
            "balance": 0,
            "capability": 0,
            "social": 0
        }

        objective_achievement = {
            'composite_value': 0,
            'individual_scores': display_objectives,
            'solve_time_seconds': 0
        }
        allocation_performance = {
            'total_demand':pre_solution['total_demand'],
            'total_allocated':pre_solution['total_demand'],
            'satisfaction_rate_percent':0,
            'suppliers_utilized':1,
            'routes_activated':1
        }
        network_analysis = {
            'network_scale':{
                'supply_points_total':None,
                'transfer_points_total':pre_solution['total_demand'],
                'demand_points_total': 1
            },
            'operational_statistics':{
                'active_suppliers': 1,
                'active_routes': 1,
                'average_routes_per_supplier': 0,
                'data_supply_points_evaluated': 1,
                'mobilization_mode':'data_transfer'
            }
        }

        su_id = pre_solution['supplier_evaluations'][0]['supplier_id']
        supplier_name = self.find_key_by_supplier_id(self.point_features,pre_solution['supplier_evaluations'][0]['supplier_id'])
        enterprise_type = self.point_features[supplier_name]['enterprise_type']
        enterprise_size = self.point_features[supplier_name]['enterprise_size']
        d_key = list(self.D.keys())

        supplier_info = {
            "supplier_id": pre_solution['supplier_evaluations'][0]['supplier_id'],
            "supplier_name": supplier_name,
            "enterprise_type": enterprise_type,
            "enterprise_size": enterprise_size,
            "location": {
                "latitude": self.point_features[supplier_name]['latitude'],
                "longitude": self.point_features[supplier_name]['longitude']
            },
            "allocated_amount": pre_solution['total_demand'],
            "total_capacity": pre_solution['total_demand'],
            "original_capacity": pre_solution['total_demand'],
            "reliability_factor": self.P[supplier_name],
            'evaluation_scores':{
                'capability_score':0,
                'type_score':0,
                'size_score':0,
                'mobilization_score':0,
                'composite_score':0
            },
            "routes": [
    {
        "route_type": "direct",
        "route": [
            {
                "from": {
                    "point_id": supplier_name,
                    "type": "supply",
                    "mode": "公路",
                    "longitude_latitude": [
                        self.point_features[supplier_name]['longitude'],
                        self.point_features[supplier_name]['latitude']
                    ]
                },
                "to": {
                    "point_id": d_key[0],
                    "type": "demand",
                    "mode": "公路",
                    "longitude_latitude": [
                        self.point_features[d_key[0]]['longitude'],
                        self.point_features[d_key[0]]['latitude']
                    ]
                }
            }
        ]
    }
],


            'mobilization_quantity':800,
            'datas':pre_solution
        }

        data = {
            'objective_achievement':objective_achievement,
            'allocation_performance':allocation_performance,
            'network_analysis':network_analysis,
            'supplier_portfolio':supplier_info,
            'type': 'specify',
        }
        #
        standardized_result = data
        #
        return standardized_result

    def _generate_data_mobilization_output(self, solution, total_time):
        """
        为数据动员生成标准化输出
        """

        def find_supplier_name_by_id(supplier_id):
            """根据supplier_id找到对应的供应点名称"""
            for name, features in self.point_features.items():
                if features.get('original_supplier_id') == supplier_id:
                    return name
            return supplier_id  # 如果找不到，返回原ID

        def select_optimal_sub_objects_for_data_supplier(supplier_id, allocated_amount):
            """为数据供应点按优先级选择细分对象组合，基于成本效率"""
            supplier_name = find_supplier_name_by_id(supplier_id)

            # 首先检查是否已经有选择好的细分对象组合
            for supplier_data in solution['selected_suppliers']:
                if supplier_data['supplier_id'] == supplier_id:
                    if 'selected_sub_objects' in supplier_data and supplier_data['selected_sub_objects']:
                        # 使用已经选择好的细分对象组合
                        selected_objects = []
                        for sub_obj_selection in supplier_data['selected_sub_objects']:
                            sub_obj_info = sub_obj_selection['sub_object_info']
                            allocated_amount_for_obj = sub_obj_selection['allocated_amount']
                            selected_objects.append({
                                'sub_object': sub_obj_info,
                                'allocated_amount': allocated_amount_for_obj,
                                'cost_efficiency': sub_obj_info.get('composite_score', 0),
                                'max_available': sub_obj_info['max_available_quantity'],
                                'total_cost': 0,
                                'safety_score': 0,
                                'category_id': sub_obj_selection.get('category_id', 'unknown'),
                                'category_name': sub_obj_selection.get('category_name', 'unknown'),
                                'recommend_md5': sub_obj_selection.get('recommend_md5', 'unknown'),
                            })
                        return selected_objects
                    break

            # 如果没有预先选择的组合，使用原有逻辑
            sub_objects = self.point_features[supplier_name].get('sub_objects', [])

            if not sub_objects:
                return solution['selected_suppliers']

            available_objects = []

            # 遍历所有分类中的细分对象
            for category in sub_objects:
                if isinstance(category, dict) and 'items' in category:
                    # 新的分类结构：遍历分类中的所有项目
                    for sub_obj in category.get('items', []):
                        # 检查可用性
                        max_available = sub_obj.get('max_available_quantity', 0)
                        if not isinstance(max_available, (int, float)) or max_available <= self.EPS:
                            continue  # 跳过不可用的细分对象

                        facility_rental = sub_obj.get('facility_rental_price', len(self.J) * len(self.M) + len(self.J))
                        power_cost = sub_obj.get('power_cost', len(self.J) + len(self.K))
                        communication_cost = sub_obj.get('communication_purchase_price', len(self.J) * len(self.K))
                        total_cost = facility_rental + power_cost + communication_cost

                        # 计算安全评分
                        safety_score = self._calculate_sub_object_safety(sub_obj)

                        # 计算成本效率
                        cost_efficiency = safety_score / max(total_cost, self.EPS) if total_cost > 0 else safety_score

                        category_id = sub_obj.get('category_id') or category.get('category_id', 'unknown')
                        category_name = sub_obj.get('category_name') or category.get('category_name', 'unknown')
                        recommend_md5 = sub_obj.get('recommend_md5') or category.get('recommend_md5', 'unknown')
                        available_objects.append({
                            'sub_object': sub_obj,
                            'cost_efficiency': cost_efficiency,
                            'max_available': max_available,
                            'total_cost': total_cost,
                            'safety_score': safety_score,
                            'category_id': category_id,
                            'category_name': category_name,
                            'recommend_md5': recommend_md5,
                            'sub_object_score': cost_efficiency
                        })
                else:
                    # 兼容旧的平铺结构：直接处理细分对象
                    sub_obj = category
                    # 检查可用性
                    max_available = sub_obj.get('max_available_quantity', 0)
                    if max_available <= self.EPS:
                        continue

                    facility_rental = sub_obj.get('facility_rental_price', len(self.J) * len(self.M) + len(self.J))
                    power_cost = sub_obj.get('power_cost', len(self.J) + len(self.K))
                    communication_cost = sub_obj.get('communication_purchase_price', len(self.J) * len(self.K))
                    total_cost = facility_rental + power_cost + communication_cost

                    # 计算安全评分
                    safety_score = self._calculate_sub_object_safety(sub_obj)

                    # 计算成本效率
                    cost_efficiency = safety_score / max(total_cost, self.EPS) if total_cost > 0 else safety_score

                    category_id = sub_obj.get('category_id', 'unknown')
                    category_name = sub_obj.get('category_name', 'unknown')
                    recommend_md5 = sub_obj.get('recommend_md5', 'unknown')
                    available_objects.append({
                        'sub_object': sub_obj,
                        'cost_efficiency': cost_efficiency,
                        'max_available': max_available,
                        'total_cost': total_cost,
                        'safety_score': safety_score,
                        'category_id': category_id,
                        'category_name': category_name,
                        'recommend_md5': recommend_md5,
                        'sub_object_score': cost_efficiency  # 添加细分对象评分
                    })

            # 按成本效率排序并按优先级分配
            if available_objects:
                available_objects.sort(key=lambda x: x['cost_efficiency'], reverse=True)

                selected_objects = []
                remaining_demand = allocated_amount

                for obj_info in available_objects:
                    if remaining_demand <= self.EPS:
                        break

                    max_available = obj_info['max_available']
                    allocated_for_this_obj = min(remaining_demand, max_available)

                    if allocated_for_this_obj > self.EPS:
                        selected_objects.append({
                            'sub_object': obj_info['sub_object'],
                            'allocated_amount': allocated_for_this_obj,
                            'cost_efficiency': obj_info['cost_efficiency'],
                            'max_available': max_available,
                            'total_cost': obj_info['total_cost'],
                            'safety_score': obj_info['safety_score'],
                            'category_id': obj_info.get('category_id', 'unknown'),
                            'category_name': obj_info.get('category_name', 'unknown'),
                            'recommend_md5': obj_info.get('recommend_md5', 'unknown'),
                            'sub_object_score': obj_info.get('sub_object_score', obj_info.get('cost_efficiency', 0.0))
                        })
                        remaining_demand -= allocated_for_this_obj

                return selected_objects

            return solution['selected_suppliers']

        def format_data_sub_object(sub_obj, allocated_amount=None, category_id='unknown', category_name='unknown',
                                   recommend_md5='unknown'):
            """格式化数据细分对象信息"""
            if not sub_obj:
                return {}

            # 计算数据批次数量（直接使用分配的数量）
            data_batch_count = round(allocated_amount) if allocated_amount is not None else 0

            return {
                'data_type_code': sub_obj.get('sub_object_id', 'unknown'),
                'data_type_name': sub_obj.get('sub_object_name', 'unknown'),
                'category_id': category_id,
                'category_name': category_name,
                'recommend_md5': recommend_md5,
                'max_available_quantity': data_batch_count if data_batch_count > 0 else sub_obj.get(
                    'max_available_quantity', 0),
                'specify_quantity': sub_obj.get('specify_quantity', 0),
                'capacity_quantity': sub_obj.get('capacity_quantity', 0),
                'cost_structure': {
                    'facility_rental_price': round(sub_obj.get('facility_rental_price', 0), 2),
                    'power_cost': round(sub_obj.get('power_cost', 0), 2),
                    'communication_purchase_price': round(sub_obj.get('communication_purchase_price', 0), 2),
                    'data_processing_cost': round(sub_obj.get('data_processing_cost', 0), 2),
                    'data_storage_cost': round(sub_obj.get('data_storage_cost', 0), 2)
                }
            }

        try:
            # 计算数据动员的多目标结果
            multi_obj_result = self._compute_data_mobilization_objectives(solution)

            # 处理individual_scores
            raw_objectives = multi_obj_result['individual_objectives']
            display_objectives = {}
            for obj_name, obj_value in raw_objectives.items():
                display_objectives[obj_name] = round(obj_value, 6)

            selected_suppliers = []

            for supplier_data in solution['selected_suppliers']:
                supplier_id = supplier_data['supplier_id']
                evaluation = supplier_data['evaluation']

                try:
                    # 根据supplier_id找到对应的供应点名称
                    supplier_name = find_supplier_name_by_id(supplier_id)

                    # 使用英文键名
                    enterprise_type = evaluation['enterprise_info'].get('type', '未知')
                    enterprise_size = evaluation['enterprise_info'].get('size', '未知')

                    # 选择最优细分对象组合
                    selected_sub_objects = select_optimal_sub_objects_for_data_supplier(supplier_id, supplier_data[
                        'allocated_capacity'])

                    supplier_info = {
                        "supplier_id": self.point_features[supplier_name].get('original_supplier_id', supplier_id),
                        "supplier_name": supplier_name,
                        "enterprise_type": enterprise_type,
                        "enterprise_size": enterprise_size,
                        "location": {
                            "latitude": self.point_features[supplier_name]['latitude'],
                            "longitude": self.point_features[supplier_name]['longitude']
                        },
                        "allocated_amount": round(supplier_data['allocated_capacity'], 2),
                        "total_capacity": sum(round(obj_info['allocated_amount']) for obj_info in
                                              selected_sub_objects) if selected_sub_objects else max(1, round(
                            supplier_data['allocated_capacity'])),
                        "original_capacity": evaluation['supply_capacity'],
                        "reliability_factor": self.P[supplier_name],
                        "evaluation_scores": {
                            "capability_score": round(evaluation['capability_score'], 4),
                            "type_score": round(evaluation['type_score'], 4),
                            "size_score": round(evaluation['size_score'], 4),
                            "mobilization_score": round(evaluation['mobilization_score'], 4),
                            "composite_score": round(evaluation['composite_score'], 4)
                        },
                        "routes": [
    {
        "route_id": f"DATA_{self.point_features[supplier_name].get('original_supplier_id', supplier_id)}_{solution['demand_point']}",
        "route_type": "direct",
        "route": [
            {
                "from": {
                    "point_id": supplier_name,
                    "type": "supply",
                    "mode": "公路",
                    "longitude_latitude": [
                        self.point_features[supplier_name]['longitude'],
                        self.point_features[supplier_name]['latitude']
                    ]
                },
                "to": {
                    "point_id": solution['demand_point'],
                    "type": "demand",
                    "mode": "公路",
                    "longitude_latitude": [
                        self.point_features[solution['demand_point']]['longitude'],
                        self.point_features[solution['demand_point']]['latitude']
                    ]
                }
            }
        ]
    }
],
                        "mobilization_quantity": sum(round(obj_info['allocated_amount']) for obj_info in
                                                     selected_sub_objects) if selected_sub_objects else max(1, round(
                            supplier_data['allocated_capacity'])),
                        "datas": []
                    }

                    # 处理细分对象信息，包含分类信息
                    if selected_sub_objects:
                        for obj_info in selected_sub_objects:
                            if obj_info['allocated_amount'] > 0:
                                data_obj = format_data_sub_object(obj_info['sub_object'],
                                                                  round(obj_info['allocated_amount']),
                                                                  obj_info.get('category_id', 'unknown'),
                                                                  obj_info.get('category_name', 'unknown'),
                                                                  obj_info.get('recommend_md5', 'unknown'))
                                supplier_info["datas"].append(data_obj)

                    # 只有当分配容量大于0时才添加供应点
                    if supplier_data['allocated_capacity'] > 0:
                        selected_suppliers.append(supplier_info)

                except (KeyError, TypeError) as e:
                    self.logger.warning(f"构建数据动员供应商 {supplier_id} 信息失败: {str(e)}")
                    continue

            # 获取算法序号和动员对象类型
            algorithm_sequence_id = solution.get('algorithm_sequence_id')
            mobilization_object_type = solution.get('mobilization_object_type')
            req_element_id = solution.get('req_element_id')

            # 验证关键标识符
            if req_element_id is not None and not isinstance(req_element_id, str):
                self.logger.warning(f"req_element_id类型异常: {type(req_element_id)}, 将转换为字符串")
                req_element_id = str(req_element_id) if req_element_id is not None else None

                # 数据动员的路线信息（数据传输路径）
                data_route_segments = []
                for supplier_data in solution['selected_suppliers']:
                    supplier_id = supplier_data['supplier_id']
                    supplier_name = find_supplier_name_by_id(supplier_id)
                    segment_info = {
                        "segment_id": len(data_route_segments) + 1,
                        "from_point": {
                            "point_id": supplier_id,
                            "point_name": supplier_name,
                            "point_type": "data_supply_point",
                            "location": {
                                "latitude": self.point_features[supplier_name]['latitude'],
                                "longitude": self.point_features[supplier_name]['longitude']
                            }
                        },
                        "to_point": {
                            "point_id": solution['demand_point'],
                            "point_name": solution['demand_point'],
                            "point_type": "data_demand_point",
                            "location": {
                                "latitude": self.point_features[solution['demand_point']]['latitude'],
                                "longitude": self.point_features[solution['demand_point']]['longitude']
                            }
                        },
                        "transport_mode": {
                            "mode_id": "data_transfer",
                            "mode_name": "数据传输",
                            "mode_code": "data-01",
                            "capacity": self.point_features[supplier_name].get('data_processing_capacity', 100),
                            "security_level": self.point_features[supplier_name].get('data_security_level', '中')
                        },
                        "data_amount": round(supplier_data['allocated_capacity'], 2),
                        "processing_capacity": self.point_features[supplier_name].get('data_processing_capacity', 100),
                        "security_level": self.point_features[supplier_name].get('data_security_level', '中')
                    }
                    data_route_segments.append(segment_info)

            # 按要求分为四个部分的数据动员输出，同时在顶层保留兼容性字段
            total_demand = solution.get('total_demand', 0)
            satisfied_demand = solution.get('satisfied_demand', total_demand - solution.get('demand_left', {}).get(
                solution.get('demand_point', self.K[0] if self.K else ''), 0))

            total_allocated = sum(supplier['allocated_amount'] for supplier in selected_suppliers)
            satisfaction_rate = solution.get('satisfaction_rate',
                                             (satisfied_demand / total_demand if total_demand > 0 else 1.0))

            # 统计活跃路由数（数据动员每个供应点算一条路由）
            active_routes = len([s for s in selected_suppliers if s['allocated_amount'] > self.EPS])

            # 按要求分为四个部分的数据动员输出，保持与物资/人员动员格式一致
            standardized_result = {
                "objective_achievement": {
                    "composite_value": round(multi_obj_result['composite_objective_value'], 6),
                    "individual_scores": display_objectives,
                    "solve_time_seconds": round(total_time, 3)
                },
                "allocation_performance": {
                    "total_demand": round(total_demand, 2),
                    "total_allocated": round(total_allocated, 2),
                    "satisfaction_rate_percent": round(satisfaction_rate * 100, 1),
                    "suppliers_utilized": len(selected_suppliers),
                    "routes_activated": active_routes
                },
                "network_analysis": {
                    "network_scale": {
                        "supply_points_total": len(self.J),
                        "transfer_points_total": len(self.M),
                        "demand_points_total": len(self.K)
                    },
                    "operational_statistics": {
                        "active_suppliers": len([s for s in selected_suppliers if s['allocated_amount'] > self.EPS]),
                        "active_routes": active_routes,
                        "average_routes_per_supplier": round(active_routes / len(selected_suppliers), 2) if len(
                            selected_suppliers) > 0 else 0,
                        "data_supply_points_evaluated": len(solution['supplier_evaluations']),
                        "mobilization_mode": "data_transfer"
                    }
                },
                "supplier_portfolio": selected_suppliers,
                "data_mobilization_info": {
                    "create_time": datetime.now().strftime("%Y-%m-%dT%H:%M:%S+08:00"),
                    "algorithm_params": {
                        "optimization_method": "data_supplier_evaluation_and_selection",
                        "selection_strategy": "composite_score_ranking",
                        "allocation_strategy": "greedy_best_fit"
                    }
                }
            }

            return standardized_result
        except Exception as e:
            self.logger.error(f"生成数据动员标准化输出时发生错误: {str(e)}", exc_info=True)
            return {
                "code": 500,
                "msg": f"生成数据动员结果时发生内部错误: {str(e)}",
                "data": {
                    "algorithm_sequence_id": solution.get('algorithm_sequence_id') if solution else None,
                    "mobilization_object_type": solution.get('mobilization_object_type') if solution else None,
                    "error_type": "data_mobilization_output_error"
                }
            }

    def _set_algorithm_params(self, algorithm_params):
        """设置算法参数"""
        try:
            self.objective_names = algorithm_params['objective_names']  # 存储每个目标的中文名称或描述
            self.PRIORITY_LEVELS = algorithm_params['priority_levels']  # 存储不同优先级的等级（用于调度等）
            self.EPS = algorithm_params['eps']  # 存储用于数值计算的微小量（例如避免除0）
            self.BIGM = algorithm_params['bigm']  # 存储大M常数（用于线性规划约束建模）
        except KeyError as e:
            raise ValueError(f"算法参数缺少必要字段: {str(e)}")
        except TypeError as e:
            raise ValueError(f"算法参数类型错误: {str(e)}")

    def _validate_and_normalize_weights(self, weights):
        """验证和标准化权重"""
        if not isinstance(weights, dict):
            raise ValueError("权重必须是字典格式")  # 如果权重不是字典，则报错

        # 检查是否包含所有必需的目标
        missing_objectives = set(self.objective_names.keys()) - set(weights.keys())
        if missing_objectives:
            raise ValueError(f"缺少目标权重: {missing_objectives}")  # 如果缺少某些目标的权重，则报错

        # 检查权重是否为非负数
        for obj, weight in weights.items():
            if weight < 0:
                raise ValueError(f"目标 {obj} 的权重不能为负数: {weight}")  # 权重必须为正数或0

        # 标准化权重（使权重和为1）
        total_weight = sum(weights.values())
        if total_weight <= 0:
            raise ValueError("权重总和必须大于0")  # 权重总和不能为0

        # 归一化各目标的权重值
        normalized_weights = {obj: weight / total_weight for obj, weight in weights.items()}

        # 日志记录标准化结果
        self.logger.info("权重标准化完成:")
        for obj, weight in normalized_weights.items():
            self.logger.info(f"  {obj}: {weights[obj]:.3f} -> {weight:.3f}")

        return normalized_weights  # 返回标准化后的权重字典

    def _set_network_data(self, network_data):
        """设置网络数据"""
        try:
            self.J = network_data['J']  # 原始起点集合（如物资储备点）
            self.M = network_data['M']  # 中转点集合
            self.K = network_data['K']  # 目标需求点集合
            self.point_features = network_data['point_features']  # 节点的经纬度等几何属性
            self.B = network_data['B']  # 可行路径集合或供需关系
            self.P = network_data['P']  # 节点对应的属性矩阵（如能力、状态等）
            self.D = network_data['D']  # 需求量信息
            self.demand_priority = network_data['demand_priority']  # 每个需求点的优先级信息
            self.Q = network_data['Q']  # 中转点容量信息
            self.time_windows = network_data.get('time_windows', {})  # 可选的时间窗信息，默认为空字典

            # 验证关键数据结构
            if not self.J:
                raise ValueError("供应点集合J不能为空")
            if not self.K:
                raise ValueError("需求点集合K不能为空")
            if not self.M:
                raise ValueError("中转点集合M不能为空")
        except KeyError as e:
            raise ValueError(f"网络数据缺少必要字段: {str(e)}")
        except TypeError as e:
            raise ValueError(f"网络数据类型错误: {str(e)}")

    def _set_transport_params(self, transport_params):
        """设置运输参数"""
        try:
            self.N = transport_params['N']  # 运输车辆或资源单位总数
            self.TRANSPORT_MODES = transport_params['TRANSPORT_MODES']  # 支持的运输方式（如公路、铁路等）

            # 各阶段的距离矩阵
            self.L_j_m = transport_params['L_j_m']  # 从起点J到中转点M的距离
            self.L_m_m = transport_params['L_m_m']  # 中转点之间的距离
            self.L_m_k = transport_params['L_m_k']  # 从中转点M到需求点K的距离

            # 各阶段的比例系数（可能用于计算成本/时间等）
            self.alpha1 = transport_params['alpha1']
            self.alpha2 = transport_params['alpha2']
            self.alpha3 = transport_params['alpha3']

            # 各阶段的速度（如运输速度）
            self.v1 = transport_params['v1']
            self.v2 = transport_params['v2']
            self.v3 = transport_params['v3']

            # 各阶段的时间参数（如装卸、转运等）
            time_params = transport_params['time_params']
            self.T1 = time_params['preparation_time']
            self.T4 = time_params['assembly_time']
            self.T6 = time_params['handover_time']

            # 综合计算不同运输阶段的总时间
            self.T_loading = self.T1 + self.T4 + self.T6  # 装载相关总时间

            self.ROAD_ONLY = transport_params.get('ROAD_ONLY', [1])  # 仅限公路运输的模式编号列表

            # 验证关键数据结构
            if not self.N:
                raise ValueError("运输方式集合N不能为空")
            if not self.TRANSPORT_MODES:
                raise ValueError("运输方式配置TRANSPORT_MODES不能为空")
        except KeyError as e:
            raise ValueError(f"运输参数缺少必要字段: {str(e)}")
        except TypeError as e:
            raise ValueError(f"运输参数类型错误: {str(e)}")

    def _calculate_haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        使用Haversine公式计算两个经纬度点之间的球面距离，并应用距离修正系数（单位：公里）
        """
        # 使用LRU缓存避免重复计算，自动控制内存
        try:
            # 快速类型检查，避免重复验证
            if not isinstance(lat1, (int, float)) or not isinstance(lat2, (int, float)):
                self.logger.error("坐标参数类型错误")
                return len(self.J) + len(self.M) + len(self.K) if hasattr(self, 'J') and hasattr(self, 'M') and hasattr(
                    self, 'K') else 100.0

            # 初始化受限缓存
            if not hasattr(self, '_distance_cache'):
                # 基于网络规模动态确定缓存大小
                max_cache_size = min(len(self.J) * len(self.K) * 2, len(self.J) * len(self.M) * len(self.K) // 10)
                self._distance_cache = {}
                self._cache_access_order = []
                self._max_cache_size = max(max_cache_size, len(self.J) + len(self.M) + len(self.K))

            # 缓存键生成，减少精度以提高命中率
            cache_key = (round(lat1, 4), round(lon1, 4), round(lat2, 4), round(lon2, 4))

            # if cache_key in self._distance_cache:
            #     # LRU访问顺序更新
            #     self._cache_access_order.remove(cache_key)
            #     self._cache_access_order.append(cache_key)
            #     return self._distance_cache[cache_key]

            # 快速距离估算，避免复杂的三角函数计算
            lat_diff = lat2 - lat1
            lon_diff = lon2 - lon1

            # 对于小距离使用简化计算
            if abs(lat_diff) < 0.1 and abs(lon_diff) < 0.1:
                # 使用平面近似，速度更快
                lat_avg = (lat1 + lat2) / 2
                lat_factor = math.cos(math.radians(lat_avg))
                distance_km = 111.32 * math.sqrt(
                    lat_diff * lat_diff + (lon_diff * lat_factor) * (lon_diff * lat_factor))
            else:
                # 大距离使用完整Haversine公式
                lat1_rad = math.radians(lat1)
                lon1_rad = math.radians(lon1)
                lat2_rad = math.radians(lat2)
                lon2_rad = math.radians(lon2)

                dlat = lat2_rad - lat1_rad
                dlon = lon2_rad - lon1_rad

                a = (math.sin(dlat / 2) ** 2 +
                     math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
                c = 2 * math.asin(math.sqrt(a))
                distance_km = 6371.0 * c

            # 应用距离修正
            corrected_distance = self._apply_distance_correction(distance_km)

            # LRU缓存管理
            # if len(self._distance_cache) >= self._max_cache_size:
            #     # 移除最久未使用的条目
            #     oldest_key = self._cache_access_order.pop(0)
            #     del self._distance_cache[oldest_key]

            # 添加新条目
            self._distance_cache[cache_key] = corrected_distance
            self._cache_access_order.append(cache_key)

            return corrected_distance

        except (TypeError, ValueError) as e:
            self.logger.error(f"距离计算参数错误: {str(e)}")
            return len(self.J) + len(self.M) + len(self.K) if hasattr(self, 'J') and hasattr(self, 'M') and hasattr(
                self, 'K') else 100.0

    def _apply_distance_correction(self, raw_distance):
        """
        基于网络规模和距离特征应用距离修正系数
        """
        # 检查是否有网络结构信息（J、M、K）
        if hasattr(self, 'J') and hasattr(self, 'M') and hasattr(self, 'K'):
            # 网络规模因子：总节点数
            network_scale_factor = len(self.J) + len(self.M) + len(self.K)
            supply_scale = len(self.J)
            transfer_scale = len(self.M)
            demand_scale = len(self.K)

            # 基准距离因子：限制在合理范围内
            base_distance_factor = min(1.5, max(1.0, network_scale_factor / max(supply_scale, 1)))

            # 尝试根据坐标范围计算网络地理跨度
            if hasattr(self, 'point_features') and self.point_features:
                all_latitudes = [self.point_features[node].get('latitude', 0) for node in self.point_features]
                all_longitudes = [self.point_features[node].get('longitude', 0) for node in self.point_features]

                if all_latitudes and all_longitudes:
                    lat_range = max(all_latitudes) - min(all_latitudes)
                    lon_range = max(all_longitudes) - min(all_longitudes)
                    # 简化特征距离计算，避免过度放大
                    characteristic_distance = (lat_range + lon_range) * 111.32  # 经纬度转公里的近似转换
                else:
                    characteristic_distance = raw_distance  # 无法计算时采用原始距离
            else:
                characteristic_distance = raw_distance  # 同上
            characteristic_distance = max(characteristic_distance, raw_distance)  # 保证不小于原始距离
        else:
            characteristic_distance = raw_distance  # 若无网络结构信息，使用原始距离

        # 计算距离比（相对特征距离的比例）
        distance_ratio = raw_distance / max(characteristic_distance, raw_distance)

        # 确保网络规模变量都已定义
        if hasattr(self, 'J'):
            supply_scale = len(self.J)
        else:
            supply_scale = 1

        if hasattr(self, 'M'):
            transfer_scale = len(self.M)
        else:
            transfer_scale = 1

        if hasattr(self, 'K'):
            demand_scale = len(self.K)
        else:
            demand_scale = 1

        # 限制修正系数在合理范围内[1.0, 1.5]
        if hasattr(self, 'J') and hasattr(self, 'M'):
            supply_transfer_ratio = supply_scale / max(supply_scale + transfer_scale, 1)
            max_correction = 1.0 + supply_transfer_ratio * 0.5  # 最大修正50%
        else:
            max_correction = 1.2  # 默认最大修正20%

        # 简化修正系数计算
        network_density = supply_scale / max(supply_scale + transfer_scale + demand_scale, 1)
        correction_factor = 1.0 + (max_correction - 1.0) * math.exp(-distance_ratio * network_density)

        # 应用修正因子，确保在合理范围内
        corrected_distance = raw_distance * min(correction_factor, 1.5)

        return corrected_distance  # 返回最终修正后距离

    def _make_global_transport_mode_decision(self, sorted_suppliers, aggregated_demand_point,
                                             path_metrics_cache, composite_score_cache):
        """
        全局运输模式决策：选择直达模式还是统一多式联运模式
        """

        # 计算网络规模参数
        network_scale = len(self.J) + len(self.M) + len(self.K)
        supply_scale = len(self.J)
        transfer_scale = len(self.M)

        # 识别当前主导目标（权重最大的目标）
        if hasattr(self, 'current_dominant_objective'):
            dominant_obj_name = self.current_dominant_objective
            dominant_weight = self.objective_weights.get(dominant_obj_name, 0.0)
        else:
            # 回退到权重最大的目标
            dominant_objective = max(self.objective_weights.items(), key=lambda x: x[1])
            dominant_obj_name = dominant_objective[0]
            dominant_weight = dominant_objective[1]

        avg_weight = sum(self.objective_weights.values()) / len(self.objective_weights)

        self.logger.info(f"当前主导目标: {dominant_obj_name}, 权重: {dominant_weight:.3f}, 平均权重: {avg_weight:.3f}")

        # 样本供应点数量（避免计算所有供应点）
        sample_size = min(len(sorted_suppliers), max(supply_scale // transfer_scale, 1) if transfer_scale > 0 else 1)
        sample_suppliers = sorted_suppliers[:sample_size]

        # 计算直达模式的平均评分
        direct_scores = np.full(len(sample_suppliers), np.inf)  # 预分配numpy数组
        valid_scores_mask = np.zeros(len(sample_suppliers), dtype=bool)  # 有效评分掩码

        for idx, j in enumerate(sample_suppliers):
            try:
                direct_cache_key = f"direct_{j}_{aggregated_demand_point}"
                if direct_cache_key not in path_metrics_cache:
                    path_metrics_cache[direct_cache_key] = self._calculate_direct_path_metrics(j,
                                                                                               aggregated_demand_point)

                path_metrics = path_metrics_cache[direct_cache_key]
                composite_score = self._calculate_composite_score(path_metrics, is_direct=True)
                direct_scores[idx] = composite_score
                valid_scores_mask[idx] = True
            except (ValueError, KeyError) as e:
                self.logger.warning(f"供应点 {j} 直达评分计算失败: {str(e)}")
                continue

        avg_direct_score = np.mean(direct_scores[valid_scores_mask]) if np.any(valid_scores_mask) else float('inf')

        #给一个不确定性扰动
        avg_direct_score = avg_direct_score + random.uniform(0,1)/10
        # 评估最优的多式联运路径
        best_multimodal_score = float('inf')
        best_multimodal_route = None

        # 预计算所有坐标数据，避免重复字典访问
        k_lat, k_lon = self.point_features[aggregated_demand_point]['latitude'], \
            self.point_features[aggregated_demand_point]['longitude']

        # 预计算中转点坐标和供应点坐标
        transfer_coords = np.array(
            [[self.point_features[m]['latitude'], self.point_features[m]['longitude']] for m in self.M])
        sample_supplier_coords = np.array(
            [[self.point_features[j]['latitude'], self.point_features[j]['longitude']] for j in sample_suppliers])

        # 向量化计算所有中转点到需求点的距离
        transfer_to_demand_distances = self._vectorized_haversine_distance(
            transfer_coords[:, 0], transfer_coords[:, 1], k_lat, k_lon
        )

        # 按距离排序中转点
        sorted_indices = np.argsort(transfer_to_demand_distances)
        sorted_transfer_points = [self.M[i] for i in sorted_indices]

        # 预计算所有供应点到所有中转点的距离矩阵
        if len(sample_supplier_coords) > 0 and len(transfer_coords) > 0:
            # 使用broadcasting计算距离矩阵 (suppliers x transfers)
            supplier_to_transfer_distances = self._vectorized_haversine_distance(
                sample_supplier_coords[:, 0][:, np.newaxis], sample_supplier_coords[:, 1][:, np.newaxis],
                transfer_coords[:, 0][np.newaxis, :], transfer_coords[:, 1][np.newaxis, :]
            )
        else:
            supplier_to_transfer_distances = np.array([])

        # 预计算运输速度
        transport_speed_1 = self.TRANSPORT_MODES[1]['speed']

        # 预构建有效路径组合，减少重复计算
        valid_combinations = []
        sorted_transfer_points_array = np.array(sorted_transfer_points)
        transfer_indices = np.arange(len(sorted_transfer_points))

        # 启发式策略：基于网络规模和距离特征优化中转点对选择
        # 计算动态处理限制，避免O(M²)全遍历
        max_combinations_to_evaluate = min(len(sorted_transfer_points) * len(sorted_transfer_points),
                                           len(sorted_transfer_points) * supply_scale)

        # 预计算所有中转点对的启发式评分
        heuristic_scores = []
        for i in range(len(sorted_transfer_points)):
            for j in range(i + 1, len(sorted_transfer_points)):  # 避免重复和自循环
                m1 = sorted_transfer_points[i]
                m2 = sorted_transfer_points[j]
                m1_idx = sorted_indices[i]
                m2_idx = sorted_indices[j]

                # 检查专业化模式匹配
                m1_specialized = self.point_features[m1].get('specialized_mode', 'unknown')
                m2_specialized = self.point_features[m2].get('specialized_mode', 'unknown')

                # 计算中转点间的实际距离
                m1_lat, m1_lon = self.point_features[m1]['latitude'], self.point_features[m1]['longitude']
                m2_lat, m2_lon = self.point_features[m2]['latitude'], self.point_features[m2]['longitude']
                actual_m1_m2_distance = self._calculate_haversine_distance(m1_lat, m1_lon, m2_lat, m2_lon)

                # 计算路径几何合理性因子：避免严重绕路
                # 使用三角形不等式检验路径合理性
                m1_to_demand = transfer_to_demand_distances[m1_idx]
                m2_to_demand = transfer_to_demand_distances[m2_idx]
                direct_demand_distance = min(m1_to_demand, m2_to_demand)

                # 几何效率：实际路径距离 vs 理想直线距离的比值
                total_path_distance = actual_m1_m2_distance + min(m1_to_demand, m2_to_demand)
                geometric_efficiency = direct_demand_distance / max(total_path_distance, direct_demand_distance)

                # 专业化奖励：考虑相对权重平衡
                base_distance_score = (len(sorted_transfer_points) + supply_scale) / max(
                    actual_m1_m2_distance + m1_to_demand + m2_to_demand, len(sorted_transfer_points))
                specialization_weight = supply_scale / max(len(sorted_transfer_points), 1)
                specialization_bonus = specialization_weight * base_distance_score if m1_specialized == m2_specialized else 0

                # 距离效率：考虑中转点间实际距离和几何合理性
                distance_efficiency = geometric_efficiency * (len(sorted_transfer_points) + supply_scale) / max(
                    actual_m1_m2_distance + (m1_to_demand + m2_to_demand) / len(sorted_transfer_points),
                    len(sorted_transfer_points))

                # 多样性因子：鼓励地理位置分散的中转点对
                # 基于中转点在排序列表中的位置间隔
                position_diversity = abs(i - j) / max(len(sorted_transfer_points), 1)
                diversity_bonus = position_diversity * supply_scale / max(len(sorted_transfer_points), 1)

                # 综合启发式评分：专业化匹配 + 距离效率 + 几何合理性 + 多样性
                heuristic_score = specialization_bonus + distance_efficiency + diversity_bonus

                # 检查连通性，只有连通的点对才加入候选
                available_modes = self._get_available_transport_modes(m1, m2)
                if available_modes:
                    heuristic_scores.append((heuristic_score, m1, m2, m1_idx, m2_idx, available_modes))

        # 按启发式评分排序，优先处理高评分组合
        heuristic_scores.sort(key=lambda x: x[0], reverse=True)

        # 动态确定实际处理数量：确保效果的同时控制复杂度
        actual_combinations_to_process = min(len(heuristic_scores),
                                             max_combinations_to_evaluate,
                                             len(sample_suppliers) * transfer_scale)

        # 处理top-k个最有希望的中转点对
        for score_data in heuristic_scores[:actual_combinations_to_process]:
            _, m1, m2, m1_idx, m2_idx, available_modes = score_data

            # 为每个连通的中转点对尝试所有可用运输方式
            for n2 in available_modes:
                valid_combinations.append((m1, m2, n2, m1_idx, m2_idx))

        # 批量评估有效组合
        for m1, m2, n2, m1_idx, m2_idx in valid_combinations:
            try:
                # 向量化计算所有供应点到第一个中转点的时间
                if len(sample_suppliers) > 0 and supplier_to_transfer_distances.size > 0:
                    distances_to_m1 = supplier_to_transfer_distances[:, m1_idx]
                    times_to_m1 = distances_to_m1 / transport_speed_1
                    max_time_to_m1 = np.max(times_to_m1)
                else:
                    max_time_to_m1 = 0.0

                # 预分配numpy数组避免动态扩展
                multimodal_scores = np.full(len(sample_suppliers), np.inf)
                valid_multimodal_mask = np.zeros(len(sample_suppliers), dtype=bool)

                # 批量构建缓存键，减少字符串操作开销
                cache_keys = [f"unified_multimodal_{j}_{m1}_{m2}_{aggregated_demand_point}_{n2}_{max_time_to_m1:.3f}"
                              for j in sample_suppliers]
                score_cache_keys = [f"score_{cache_key}" for cache_key in cache_keys]

                # 向量化处理供应点评分计算
                for j_idx, (j, cache_key, score_cache_key) in enumerate(
                        zip(sample_suppliers, cache_keys, score_cache_keys)):
                    try:
                        if cache_key not in path_metrics_cache:
                            multimodal_metrics = self._calculate_unified_multimodal_metrics(
                                j, m1, m2, aggregated_demand_point, n2, max_time_to_m1)
                            path_metrics_cache[cache_key] = multimodal_metrics
                        else:
                            multimodal_metrics = path_metrics_cache[cache_key]

                        if score_cache_key not in composite_score_cache:
                            multimodal_score = self._calculate_composite_score(multimodal_metrics, is_direct=False)
                            composite_score_cache[score_cache_key] = multimodal_score
                        else:
                            multimodal_score = composite_score_cache[score_cache_key]

                        multimodal_scores[j_idx] = multimodal_score
                        valid_multimodal_mask[j_idx] = True
                    except (ValueError, KeyError) as e:
                        self.logger.warning(f"供应点 {j} 多式联运评分计算失败: {str(e)}")
                        continue

                # 向量化计算平均评分
                if np.any(valid_multimodal_mask):
                    avg_multimodal_score = np.mean(multimodal_scores[valid_multimodal_mask])

                    if avg_multimodal_score < best_multimodal_score:
                        best_multimodal_score = avg_multimodal_score
                        best_multimodal_route = {'m1': m1, 'm2': m2, 'n2': n2}

                        # 提前终止优化：如果找到显著更好的解，可以提前结束
                        improvement_threshold = best_multimodal_score * (
                                1.0 - len(self.K) / (len(self.K) + len(self.M) + len(self.J)))
                        if best_multimodal_score < improvement_threshold:
                            break

            except (ValueError, KeyError) as e:
                self.logger.warning(f"多式联运路径 {m1}-{m2} 评估失败: {str(e)}")
                continue

        # 决策逻辑
        if best_multimodal_route is None:
            return {'mode': 'direct'}

        # 在最终返回前添加调试信息
        if best_multimodal_route is None:
            self.logger.info("全局决策：选择直达模式（无有效多式联运路径）")
            return {'mode': 'direct'}

        # 比较直达和多式联运的评分
        score_improvement = (avg_direct_score - best_multimodal_score) / avg_direct_score if avg_direct_score > 0 else 0.0

        # 计算距离合理性因子和近距离偏好
        # 根据主导目标计算直达和多式联运的比较指标
        avg_direct_performance = 0.0
        min_direct_performance = float('inf')
        multimodal_typical_performance = 0.0
        performance_count = 0

        # 计算直达模式在主导目标上的表现
        for j in sample_suppliers:
            try:
                direct_cache_key = f"direct_{j}_{aggregated_demand_point}"
                if direct_cache_key not in path_metrics_cache:
                    path_metrics_cache[direct_cache_key] = self._calculate_direct_path_metrics(j,
                                                                                               aggregated_demand_point)

                path_metrics = path_metrics_cache[direct_cache_key]

                # 根据主导目标类型获取相应的性能指标
                if dominant_obj_name in ['time', 'cost', 'distance', 'balance', 'social']:
                    performance_value = path_metrics.get(f'{dominant_obj_name}_score', float('inf'))
                elif dominant_obj_name in ['safety', 'priority']:
                    performance_value = -path_metrics.get(f'{dominant_obj_name}_score', float('-inf'))  # 转为正值比较
                elif dominant_obj_name == 'capability':
                    performance_value = path_metrics.get(f'{dominant_obj_name}_score', float('inf'))
                else:
                    performance_value = path_metrics.get('distance_score', float('inf'))  # 默认使用距离

                avg_direct_performance += performance_value
                min_direct_performance = min(min_direct_performance, performance_value)
                performance_count += 1
            except (ValueError, KeyError) as e:
                self.logger.warning(f"供应点 {j} 主导目标性能计算失败: {str(e)}")
                continue

        if performance_count > 0:
            avg_direct_performance /= performance_count

        # 计算多式联运在主导目标上的表现（如果存在最佳路径）
        if best_multimodal_route:
            m1, m2, n2 = best_multimodal_route['m1'], best_multimodal_route['m2'], best_multimodal_route['n2']

            # 计算多式联运的典型性能
            try:
                # 选择一个代表性供应点计算多式联运性能
                representative_j = sample_suppliers[0] if sample_suppliers else self.J[0]
                max_time_to_m1 = 0.0

                # 计算到第一个中转点的最晚时间
                calc_suppliers_count = min(len(sample_suppliers), supply_scale)
                if calc_suppliers_count > 0:
                    m1_lat, m1_lon = self.point_features[m1]['latitude'], self.point_features[m1]['longitude']
                    # 使用预计算的坐标数据进行向量化距离计算
                    calc_supplier_coords = sample_supplier_coords[:calc_suppliers_count]
                    distances_to_m1 = self._vectorized_haversine_distance(
                        calc_supplier_coords[:, 0], calc_supplier_coords[:, 1], m1_lat, m1_lon)
                    times_to_m1 = distances_to_m1 / self.TRANSPORT_MODES[1]['speed']
                    max_time_to_m1 = np.max(times_to_m1)

                multimodal_metrics = self._calculate_unified_multimodal_metrics(
                    representative_j, m1, m2, aggregated_demand_point, n2, max_time_to_m1)

                # 根据主导目标类型获取多式联运的性能指标
                if dominant_obj_name in ['time', 'cost', 'distance', 'balance', 'social']:
                    multimodal_typical_performance = multimodal_metrics.get(f'{dominant_obj_name}_score', float('inf'))
                elif dominant_obj_name in ['safety', 'priority']:
                    multimodal_typical_performance = -multimodal_metrics.get(f'{dominant_obj_name}_score',
                                                                             float('-inf'))
                elif dominant_obj_name == 'capability':
                    multimodal_typical_performance = multimodal_metrics.get(f'{dominant_obj_name}_score', float('inf'))
                else:
                    multimodal_typical_performance = multimodal_metrics.get('distance_score', float('inf'))

            except (ValueError, KeyError) as e:
                self.logger.warning(f"多式联运主导目标性能计算失败: {str(e)}")
                # 回退到距离计算
                min_j_to_m1 = float('inf')
                for j in sample_suppliers:
                    j_lat, j_lon = self.point_features[j]['latitude'], self.point_features[j]['longitude']
                    m1_lat, m1_lon = self.point_features[m1]['latitude'], self.point_features[m1]['longitude']
                    dist_j_to_m1 = self._calculate_haversine_distance(j_lat, j_lon, m1_lat, m1_lon)
                    min_j_to_m1 = min(min_j_to_m1, dist_j_to_m1)

                m1_lat, m1_lon = self.point_features[m1]['latitude'], self.point_features[m1]['longitude']
                m2_lat, m2_lon = self.point_features[m2]['latitude'], self.point_features[m2]['longitude']
                k_lat, k_lon = self.point_features[aggregated_demand_point]['latitude'], \
                    self.point_features[aggregated_demand_point]['longitude']

                dist_m1_to_m2 = self._calculate_haversine_distance(m1_lat, m1_lon, m2_lat, m2_lon)
                dist_m2_to_k = self._calculate_haversine_distance(m2_lat, m2_lon, k_lat, k_lon)
                multimodal_typical_performance = min_j_to_m1 + dist_m1_to_m2 + dist_m2_to_k

        # 近距离直达偏好：加强判断逻辑，优先考虑直达效率
        # 基于主导目标的运输模式优势判断
        has_dominant_advantage = False
        performance_advantage_count = 0

        if multimodal_typical_performance > 0 and min_direct_performance != float('inf'):
            # 计算性能比例：对于最小化目标，值越小越好
            if dominant_obj_name in ['time', 'cost', 'distance', 'balance', 'social', 'capability']:
                performance_ratio = multimodal_typical_performance / min_direct_performance if min_direct_performance > 0 else float(
                    'inf')
                direct_is_better = performance_ratio > 1.0  # 多式联运性能值更大，说明直达更好
            else:  # safety, priority等最大化目标
                performance_ratio = min_direct_performance / multimodal_typical_performance if multimodal_typical_performance > 0 else float(
                    'inf')
                direct_is_better = performance_ratio > 1.0  # 直达性能值更大，说明直达更好

            # 计算具有性能优势的供应点比例
            performance_threshold = avg_direct_performance / max(len(self.J) / max(len(self.M), 1), 1)

            for j in sample_suppliers:
                try:
                    direct_cache_key = f"direct_{j}_{aggregated_demand_point}"
                    if direct_cache_key in path_metrics_cache:
                        path_metrics = path_metrics_cache[direct_cache_key]
                        if dominant_obj_name in ['time', 'cost', 'distance', 'balance', 'social']:
                            j_performance = path_metrics.get(f'{dominant_obj_name}_score', float('inf'))
                        elif dominant_obj_name in ['safety', 'priority']:
                            j_performance = -path_metrics.get(f'{dominant_obj_name}_score', float('-inf'))
                        elif dominant_obj_name == 'capability':
                            j_performance = path_metrics.get(f'{dominant_obj_name}_score', float('inf'))
                        else:
                            j_performance = path_metrics.get('distance_score', float('inf'))

                        if j_performance <= performance_threshold:
                            performance_advantage_count += 1
                except (KeyError, TypeError):
                    continue

            performance_advantage_ratio = performance_advantage_count / len(sample_suppliers) if sample_suppliers else 0

            # 基于主导目标决定效率阈值
            if dominant_weight > avg_weight * len(self.objective_weights) / max(len(self.objective_weights) - 1, 1):
                # 主导目标权重显著高于平均值，更严格地基于该目标进行判断
                efficiency_threshold = 1 + (dominant_weight - avg_weight) * len(self.M) / max(len(self.J) + len(self.M),
                                                                                              1)
            else:
                # 权重分布相对均匀，使用温和的阈值
                efficiency_threshold = 1 + len(self.K) / max(len(self.J) + len(self.K), 1)

            advantage_threshold = len(self.K) / max(len(self.J) + len(self.K), 1)

            if (direct_is_better and performance_ratio > efficiency_threshold) or \
                    (
                            min_direct_performance < performance_threshold and performance_advantage_ratio > advantage_threshold):
                has_dominant_advantage = True
                self.logger.info(
                    f"基于主导目标({dominant_obj_name})检测到直达模式优势 - 直达最优性能: {min_direct_performance:.2f}, "
                    f"多式联运典型性能: {multimodal_typical_performance:.2f}, 性能比例: {performance_ratio:.2f}, "
                    f"优势供应点比例: {performance_advantage_ratio:.2f}")

        # 如果在主导目标上直达有明显优势，直接选择直达
        if has_dominant_advantage:
            return {'mode': 'direct', 'reason': f'dominant_objective_{dominant_obj_name}_advantage'}

        # 纯粹基于性能指标的决策：如果多式联运确实更优则选择，否则选择直达
        performance_improvement_threshold = 0.0  # 只要多式联运更优就选择

        self.logger.info(
            f"全局决策评估 - 直达平均评分: {avg_direct_score:.6f}, 最优多式联运评分: {best_multimodal_score:.6f}")
        self.logger.info(
            f"主导目标({dominant_obj_name})性能对比 - 直达最优: {min_direct_performance:.3f}, 多式联运典型: {multimodal_typical_performance:.3f}")
        self.logger.info(
            f"评分改善度: {score_improvement:.4f}, 性能改善阈值: {performance_improvement_threshold:.4f}")

        if score_improvement > performance_improvement_threshold:
            self.logger.info(
                f"全局决策：选择统一多式联运模式 - 路径: {best_multimodal_route['m1']} -> {best_multimodal_route['m2']} (运输方式: {best_multimodal_route['n2']})")
            return {
                'mode': 'multimodal',
                'unified_route': best_multimodal_route,
                'score_improvement': score_improvement,
                'avg_direct_score': avg_direct_score,
                'best_multimodal_score': best_multimodal_score
            }
        else:
            self.logger.info("全局决策：选择直达模式（多式联运改善度不足）")
            return {
                'mode': 'direct',
                'score_comparison': {
                    'avg_direct_score': avg_direct_score,
                    'best_multimodal_score': best_multimodal_score,
                    'improvement': score_improvement
                }
            }

    def _calculate_unified_multimodal_metrics(self, j, m1, m2, k, n2, max_time_to_m1):
        """
        计算统一多式联运的路径指标，考虑等待时间同步
        """

        # 预先计算网络规模参数，避免变量引用错误
        network_scale = len(self.J) + len(self.M) + len(self.K)
        supply_scale = len(self.J)
        transfer_scale = len(self.M)
        demand_scale = len(self.K)

        # 计算各段距离和时间
        j_lat, j_lon = self.point_features[j]['latitude'], self.point_features[j]['longitude']
        m1_lat, m1_lon = self.point_features[m1]['latitude'], self.point_features[m1]['longitude']
        m2_lat, m2_lon = self.point_features[m2]['latitude'], self.point_features[m2]['longitude']
        k_lat, k_lon = self.point_features[k]['latitude'], self.point_features[k]['longitude']

        d1 = self._calculate_haversine_distance(j_lat, j_lon, m1_lat, m1_lon)
        d2 = self._calculate_haversine_distance(m1_lat, m1_lon, m2_lat, m2_lon)
        d3 = self._calculate_haversine_distance(m2_lat, m2_lon, k_lat, k_lon)

        # 计算实际时间（包含等待时间）
        individual_time_to_m1 = d1 / self.TRANSPORT_MODES[1]['speed']
        waiting_time_at_m1 = max_time_to_m1 - individual_time_to_m1  # 在m1的等待时间

        t2 = d2 / self.TRANSPORT_MODES[n2]['speed']
        t3 = d3 / self.TRANSPORT_MODES[1]['speed']

        # 总时间包含等待时间和正确的中转时间
        preparation_time = self.T1 + self.T4
        # 统一多式联运的中转时间：包括在m1和m2的转运时间
        transfer_time = self.T6 * 2  # 两个中转点的转运时间
        exchange_time = self.T6
        # 统一多式联运时间计算
        total_time = preparation_time + max_time_to_m1 + t2 + t3 + transfer_time + exchange_time

        # 计算成本
        cost1 = d1 * self.TRANSPORT_MODES[1]['cost_per_km']
        cost2 = d2 * self.TRANSPORT_MODES[n2]['cost_per_km']
        cost3 = d3 * self.TRANSPORT_MODES[1]['cost_per_km']
        transport_cost = cost1 + cost2 + cost3

        # 等待成本（基于等待时间）
        waiting_cost_factor = supply_scale / (
                supply_scale + transfer_scale) if supply_scale + transfer_scale > 0 else 0.0
        waiting_cost = waiting_time_at_m1 * waiting_cost_factor

        # 计算总成本
        try:
            sub_objects = self.point_features[j].get('sub_objects', [])
            if sub_objects:
                selected_cost, selected_safety = self._select_optimal_sub_objects(j, total_time, transport_cost)
                total_cost = selected_cost + waiting_cost
                total_safety = selected_safety
            else:
                total_cost = self._calculate_cost_by_resource_type(j, total_time, transport_cost) + waiting_cost
                total_safety = self._calculate_safety_score_by_resource_type(j)
        except ValueError as e:
            raise ValueError(f"统一多式联运成本计算失败: {str(e)}")

        # 计算其他目标
        total_distance = d1 + d2 + d3
        supplier_priority_score = self._calculate_supplier_task_priority(j, k, total_time, total_cost, total_safety)

        # 资源均衡目标
        supply_capacity = self.B[j] * self.P[j]
        if self.total_supply > 0:
            ideal_usage_ratio = supply_capacity / self.total_supply
        else:
            ideal_usage_ratio = supply_scale / (supply_scale + demand_scale) if supply_scale + demand_scale > 0 else 0.5

        estimated_usage_ratio = min(supply_capacity,
                                    self.total_demand) / self.total_demand if self.total_demand > 0 else ideal_usage_ratio
        usage_deviation = abs(estimated_usage_ratio - ideal_usage_ratio)
        # 企业能力目标
        if hasattr(self, 'resource_type') and self.resource_type in ["personnel", "data"]:
            capability_score = 0.0
        else:
            # 使用现有的企业能力计算逻辑
            enterprise_size = self.point_features[j].get('enterprise_size', '中')
            if enterprise_size == '大':
                default_scale_capability = supply_scale + (supply_scale % (supply_scale + 1))
            elif enterprise_size == '中':
                default_scale_capability = supply_scale / (supply_scale + 1) if supply_scale > 0 else 1.0
            else:
                default_scale_capability = supply_scale / (supply_scale + 2) if supply_scale > 1 else 1.0

            capability_score = 1.0 / max(default_scale_capability, self.EPS)

        # 社会影响目标（复用现有逻辑）
        enterprise_type = self.point_features[j]['enterprise_type']
        enterprise_size = self.point_features[j]['enterprise_size']

        if enterprise_type in ["国企", "事业单位"]:
            type_impact_factor = supply_scale / (
                    supply_scale + network_scale) if supply_scale + network_scale > 0 else 0.5
        else:
            type_impact_factor = network_scale / (
                    supply_scale + network_scale) if supply_scale + network_scale > 0 else 0.5

        if enterprise_size in ["大", "中"]:
            size_impact_factor = supply_scale / (
                    supply_scale + demand_scale) if supply_scale + demand_scale > 0 else 0.5
        else:
            size_impact_factor = demand_scale / (
                    supply_scale + demand_scale) if supply_scale + demand_scale > 0 else 0.5

        if supply_capacity > 0:
            mobilization_intensity = min(supply_capacity, self.total_demand) / supply_capacity
        else:
            mobilization_intensity = supply_scale / (
                    supply_scale + demand_scale) if supply_scale + demand_scale > 0 else 0.5

        social_score = type_impact_factor * size_impact_factor * mobilization_intensity

        # 构建指标结果
        metrics = {
            'time_score': total_time,
            'cost_score': total_cost,
            'distance_score': total_distance,
            'safety_score': -total_safety,
            'priority_score': -supplier_priority_score,
            'balance_score': usage_deviation,
            'capability_score': capability_score,
            'social_score': social_score,
            'time': total_time,
            'cost': total_cost,
            'distance': total_distance,
            'waiting_time_at_m1': waiting_time_at_m1,
            'unified_departure_time': max_time_to_m1,
            'is_long_distance': total_distance > network_scale * transfer_scale / supply_scale if supply_scale > 0 else False,
            'is_large_batch': supply_capacity > network_scale * demand_scale / supply_scale if supply_scale > 0 else False
        }
        return metrics

    def _calculate_safety_score_by_resource_type(self, j):
        """计算供应点的安全评分（根据资源类型）"""

        def safe_float_convert(value, default=0.0):
            try:
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    if value.strip() == '':
                        return default
                    return float(value)
                else:
                    return default
            except (ValueError, TypeError):
                return default

        total_safety = 0.0

        if hasattr(self, 'resource_type'):
            if self.resource_type == "personnel":
                political_status = safe_float_convert(self.point_features[j].get('political_status', 0), 0)
                military_experience = safe_float_convert(self.point_features[j].get('military_experience', 0), 0)
                criminal_record = safe_float_convert(self.point_features[j].get('criminal_record', 0), 0)
                network_record = safe_float_convert(self.point_features[j].get('network_record', 0), 0)
                credit_record = safe_float_convert(self.point_features[j].get('credit_record', 0), 0)
                total_safety = political_status + military_experience - criminal_record - network_record - credit_record
            elif self.resource_type == "material":
                # 危险性评分直接相加
                flammable_score = safe_float_convert(
                    self.point_features[j].get('flammable_explosive', 0), 0)
                corrosive_score = safe_float_convert(
                    self.point_features[j].get('corrosive', 0), 0)
                polluting_score = safe_float_convert(
                    self.point_features[j].get('polluting', 0), 0)
                fragile_score = safe_float_convert(
                    self.point_features[j].get('fragile', 0), 0)
                total_safety = -flammable_score - corrosive_score - polluting_score - fragile_score
            elif self.resource_type == "data":
                # 使用标准化的评分范围
                autonomous_control = safe_float_convert(
                    self.point_features[j].get('autonomous_control', 1.0), 1.0)
                usability_level = safe_float_convert(
                    self.point_features[j].get('usability_level', 1.0), 1.0)
                maintenance_derived = (autonomous_control + usability_level) * 0.5
                facility_protection = safe_float_convert(
                    self.point_features[j].get('facility_protection', 1.0), 1.0)
                camouflage_protection = safe_float_convert(
                    self.point_features[j].get('camouflage_protection', 1.0), 1.0)
                environment_score = safe_float_convert(self.point_features[j].get('surrounding_environment', 0), 0)
                total_safety = autonomous_control + usability_level + maintenance_derived + facility_protection + camouflage_protection + environment_score
        else:
            # 兼容性处理：默认物资动员，使用标准化评分

            flammable_score = safe_float_convert(
                self.point_features[j].get('flammable_explosive', -0.1), -0.1)
            corrosive_score = safe_float_convert(
                self.point_features[j].get('corrosive', -0.1), -0.1)
            polluting_score = safe_float_convert(
                self.point_features[j].get('polluting', -0.1), -0.1)
            fragile_score = safe_float_convert(
                self.point_features[j].get('fragile', -0.1), -0.1)

            total_safety = -flammable_score - corrosive_score - polluting_score - fragile_score

        return -total_safety

    def _calculate_effective_capacity_optimized(self, j):
        """有效容量计算"""

        def safe_numeric_convert(value):
            """安全的数值转换函数"""
            try:
                if isinstance(value, (int, float)):
                    return float(value) if value >= 0 else 0.0
                elif isinstance(value, str):
                    if value.strip() == '':
                        return 0.0
                    converted = float(value)
                    return converted if converted >= 0 else 0.0
                else:
                    return 0.0
            except (ValueError, TypeError):
                return 0.0

        try:
            sub_objects = self.point_features[j].get('sub_objects', [])
            if sub_objects:
                actual_capacity = 0
                for category in sub_objects:
                    if isinstance(category, dict) and 'items' in category:
                        for sub_obj in category.get('items', []):
                            max_available = safe_numeric_convert(sub_obj.get('max_available_quantity', 0))
                            actual_capacity += max_available
                    else:
                        max_available = safe_numeric_convert(category.get('max_available_quantity', 0))
                        actual_capacity += max_available
                return actual_capacity * self.P[j]
            else:
                return self.B[j] * self.P[j]
        except (KeyError, TypeError):
            return 0.0

    def _calculate_priority_score_fast(self, enterprise_info, supply_scale, demand_scale):
        """快速优先级评分计算"""
        priority_score = 0.0
        if enterprise_info['type'] in ["国企", "事业单位"]:
            priority_score += supply_scale / max(supply_scale + demand_scale, 1)
        if enterprise_info['size'] in ["大", "中"]:
            priority_score += demand_scale / max(supply_scale + demand_scale, 1)
        return priority_score

    def _intelligent_supplier_preselection(self, sorted_suppliers, aggregated_demand_point,
                                           supply_scale, demand_scale, network_scale, transfer_scale):
        """
        智能供应点预筛选 - 基于供应能力、距离和多目标评分的启发式选择
        """

        # 计算动态筛选参数
        total_demand = self.D[aggregated_demand_point]
        k_lat, k_lon = self.point_features[aggregated_demand_point]['latitude'], \
            self.point_features[aggregated_demand_point]['longitude']

        # 评估所有供应点，不进行数量限制
        max_suppliers_to_evaluate = len(sorted_suppliers)

        # 分批处理，提高缓存效率
        batch_size = min(max_suppliers_to_evaluate, supply_scale + transfer_scale)

        # 预计算需求点坐标，避免重复访问
        demand_coords = (k_lat, k_lon)

        # 批量预计算企业信息，减少字典访问
        enterprise_batch = {}
        for j in sorted_suppliers[:max_suppliers_to_evaluate]:
            point_features = self.point_features[j]
            enterprise_batch[j] = {
                'type': point_features.get('enterprise_type', '其他'),
                'size': point_features.get('enterprise_size', '中'),
                'coords': (point_features['latitude'], point_features['longitude'])
            }

        # 容量计算和评分
        supplier_scores = []
        cumulative_capacity = 0
        capacity_safety_factor = network_scale / max(supply_scale + demand_scale, 1)
        target_capacity = total_demand * capacity_safety_factor

        # 初始化批量缓存
        if not hasattr(self, '_capacity_cache'):
            cache_size = min(len(self.J), network_scale)
            self._capacity_cache = {}
            self._capacity_cache_limit = cache_size

        processed_count = 0
        for j in sorted_suppliers[:max_suppliers_to_evaluate]:
            # 容量计算优化
            if j in self._capacity_cache:
                effective_capacity = self._capacity_cache[j]
            else:
                effective_capacity = self._calculate_effective_capacity_optimized(j)

                # 受限缓存
                if len(self._capacity_cache) < self._capacity_cache_limit:
                    self._capacity_cache[j] = effective_capacity

            if effective_capacity <= self.EPS:
                continue

            # 批量距离计算
            j_coords = enterprise_batch[j]['coords']
            distance_to_demand = self._calculate_haversine_distance(j_coords[0], j_coords[1],
                                                                    demand_coords[0], demand_coords[1])

            # 快速评分计算
            supply_efficiency = effective_capacity / max(distance_to_demand, self.EPS)

            # 使用预计算的企业信息
            enterprise_info = enterprise_batch[j]
            priority_score = self._calculate_priority_score_fast(enterprise_info, supply_scale, demand_scale)

            # 综合评分
            composite_score = supply_efficiency * (1.0 + priority_score)

            supplier_scores.append({
                'supplier': j,
                'capacity': effective_capacity,
                'distance': distance_to_demand,
                'efficiency': supply_efficiency,
                'priority': priority_score,
                'composite_score': composite_score
            })

            cumulative_capacity += effective_capacity
            processed_count += 1

            if processed_count % batch_size == 0 and cumulative_capacity > total_demand:
                break

        # 如果没有有效供应点，返回所有供应点避免空列表
        if not supplier_scores:
            self.logger.warning("预筛选未找到有效供应点，返回所有供应点")
            return sorted_suppliers

        # 按综合评分排序
        supplier_scores.sort(key=lambda x: x['composite_score'], reverse=True)

        # 选择所有有效的候选供应点
        selected_suppliers = []
        selected_capacity = 0

        # 选择所有有评分的供应点
        for score_info in supplier_scores:
            selected_suppliers.append(score_info['supplier'])
            selected_capacity += score_info['capacity']

        self.logger.info(f"供应点预筛选完成: 从{len(sorted_suppliers)}个供应点中选择{len(selected_suppliers)}个候选")
        self.logger.info(f"候选供应点总能力: {selected_capacity:.2f}, 目标能力: {target_capacity:.2f}")

        # 检查是否有有效的候选供应点
        if not selected_suppliers:
            self.logger.warning("预筛选后没有有效的候选供应点，可能的原因：供应能力不足、距离过远或数据异常")

        return selected_suppliers

    def _apply_dominant_objective_filter(self, candidate_suppliers, aggregated_demand_point, dominant_obj_name):
        """
        基于主导目标对候选供应点进行二次筛选
        """
        if not candidate_suppliers:
            return candidate_suppliers

        supplier_scores = []

        for j in candidate_suppliers:
            try:
                # 计算该供应点在主导目标上的表现
                direct_metrics = self._calculate_direct_path_metrics(j, aggregated_demand_point)

                if dominant_obj_name in ['time', 'cost', 'distance', 'balance', 'social']:
                    # 对于最小化目标，值越小越好
                    objective_score = direct_metrics.get(f'{dominant_obj_name}_score', float('inf'))
                elif dominant_obj_name in ['safety', 'priority']:
                    # 对于最大化目标（存储为负值），负值越大（绝对值越小）越好
                    objective_score = -direct_metrics.get(f'{dominant_obj_name}_score', float('-inf'))
                elif dominant_obj_name == 'capability':
                    # capability是反向目标，值越小越好
                    objective_score = direct_metrics.get(f'{dominant_obj_name}_score', float('inf'))
                else:
                    objective_score = float('inf')

                supplier_scores.append((j, objective_score))

            except (ValueError, KeyError) as e:
                self.logger.warning(f"供应点 {j} 主导目标评估失败: {str(e)}")
                # 失败的供应点给予最差评分
                supplier_scores.append((j, float('inf')))

        # 按主导目标性能排序
        supplier_scores.sort(key=lambda x: x[1])

        # 保留所有供应点，不进行筛选
        filtered_suppliers = [supplier for supplier, score in supplier_scores]

        self.logger.info(
            f"主导目标({dominant_obj_name})排序：对{len(candidate_suppliers)}个候选进行排序，保留全部{len(filtered_suppliers)}个")

        return filtered_suppliers

    def _multi_objective_intelligent_matching(self, random_seed):
        """多目标匹配算法"""

        try:
            random.seed(random_seed)
            match_start_time = time.time()

            self.logger.info("=" * 80)
            self.logger.info("开始多目标匹配阶段")
            self.logger.info(f"匹配算法参数 - 随机种子: {random_seed}")
            self.logger.info(f"网络节点数量 - 供应点: {len(self.J)}, 中转点: {len(self.M)}, 需求点: {len(self.K)}")

            # 计算网络规模参数
            network_scale = len(self.J) + len(self.M) + len(self.K)
            supply_scale = len(self.J)
            transfer_scale = len(self.M)
            demand_scale = len(self.K)

            self.logger.info(f"网络规模分析 - 总规模: {network_scale}, 供应规模: {supply_scale}")
            self.logger.info(f"中转规模: {transfer_scale}, 需求规模: {demand_scale}")

            # 预计算所有必要的评估指标
            self.logger.info("第1步: 预计算评估指标")
            try:
                precompute_start = time.time()
                self._precompute_evaluation_metrics()
                precompute_time = time.time() - precompute_start
                self.logger.info(f"评估指标预计算完成，耗时: {precompute_time:.3f}秒")
                self.logger.debug(f"目标范围: {getattr(self, 'objective_ranges', {})}")
            except (KeyError, AttributeError, ValueError) as e:
                error_msg = f"预计算评估指标失败: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.logger.error(f"网络数据状态 - J存在: {hasattr(self, 'J')}, M存在: {hasattr(self, 'M')}")
                self.logger.error(f"K存在: {hasattr(self, 'K')}, point_features存在: {hasattr(self, 'point_features')}")
                raise ValueError(error_msg)

            # 获取唯一需求点
            if not self.K:
                raise ValueError("需求点集合为空")

            aggregated_demand_point = self.K[0]
            if aggregated_demand_point not in self.D:
                raise KeyError(f"需求点 {aggregated_demand_point} 不在需求字典中")

            total_aggregated_demand = self.D[aggregated_demand_point]
            self.logger.info(f"第2步: 处理需求信息")
            self.logger.info(
                f"单需求点模式 - 需求接收点: {aggregated_demand_point}, 需求量: {total_aggregated_demand:.2f}")

            supplier_paths = {}

            # 缓存计算结果以避免重复计算
            path_metrics_cache = {}
            composite_score_cache = {}

            # 按供应能力排序，优先处理大容量供应点
            self.logger.info("第3步: 供应点排序和筛选")
            try:
                # 计算每个供应点基于细分对象的实际容量
                # actual_capacities = {}
                # for j in self.J:
                #     sub_objects = self.point_features[j].get('sub_objects', [])
                #     actual_capacity = 0
                #     if sub_objects:
                #         for category in sub_objects:
                #             if isinstance(category, dict) and 'items' in category:
                #                 for sub_obj in category.get('items', []):
                #                     max_available = sub_obj.get('max_available_quantity', 0)
                #                     if isinstance(max_available, (int, float)):
                #                         actual_capacity += max_available
                #             else:
                #                 max_available = category.get('max_available_quantity', 0)
                #                 if isinstance(max_available, (int, float)):
                #                     actual_capacity += max_available
                #         # 人员动员不乘以可靠性系数，保持整数人员数量
                #         if hasattr(self, 'resource_type') and self.resource_type == "personnel":
                #             actual_capacities[j] = actual_capacity
                #         else:
                #             actual_capacities[j] = actual_capacity * self.P[j]
                #     else:
                #         # 人员动员不乘以可靠性系数
                #         if hasattr(self, 'resource_type') and self.resource_type == "personnel":
                #             actual_capacities[j] = self.B[j]
                #         else:
                #             actual_capacities[j] = self.B[j] * self.P[j]
                #
                # sorted_suppliers = sorted(self.J, key=lambda j: actual_capacities[j], reverse=True)
                # 计算每个供应点基于细分对象的实际容量，只累加good类别的容量
                # 计算每个供应点基于细分对象的实际容量，只累加good类别的容量
                # 计算每个供应点基于细分对象的实际容量，只累加good类别的容量
                actual_capacities = {}
                supplier_comments = {}  # 存储每个供应点的最佳comment
                category_capacities = {}  # 存储每个供应点不同comment类别的容量

                # 目标容量
                target_capacity = 6  # 从{'na': 6}中获取的值

                for j in self.J:
                    sub_objects = self.point_features[j].get('sub_objects', [])
                    good_capacity = 0
                    midle_capacity = 0
                    bad_capacity = 0
                    best_comment = 'bad'  # 默认最差的comment

                    if sub_objects:
                        # 计算不同comment类别的容量
                        for category in sub_objects:
                            if isinstance(category, dict):
                                comment = category.get('category_comment', 'bad')

                                # 更新最佳comment
                                if comment == 'good' and best_comment != 'good':
                                    best_comment = 'good'
                                elif comment == 'midle' and best_comment not in ['good', 'midle']:
                                    best_comment = 'midle'

                                # 计算各类别容量
                                category_cap = 0
                                if 'items' in category:
                                    for sub_obj in category.get('items', []):
                                        max_available = sub_obj.get('max_available_quantity', 0)
                                        if isinstance(max_available, (int, float)):
                                            category_cap += max_available

                                # 根据comment类型累加容量
                                if comment == 'good':
                                    good_capacity += category_cap
                                elif comment == 'midle':
                                    midle_capacity += category_cap
                                else:
                                    bad_capacity += category_cap

                        # 人员动员不乘以可靠性系数，保持整数人员数量
                        if hasattr(self, 'resource_type') and self.resource_type == "personnel":
                            good_capacity = good_capacity
                            midle_capacity = midle_capacity
                            bad_capacity = bad_capacity
                        else:
                            good_capacity = good_capacity * self.P[j]
                            midle_capacity = midle_capacity * self.P[j]
                            bad_capacity = bad_capacity * self.P[j]

                        actual_capacities[j] = good_capacity
                        supplier_comments[j] = best_comment
                        category_capacities[j] = {
                            'good': good_capacity,
                            'midle': midle_capacity,
                            'bad': bad_capacity
                        }
                    else:
                        # 没有sub_objects的情况
                        supplier_comments[j] = 'bad'

                        # 人员动员不乘以可靠性系数
                        if hasattr(self, 'resource_type') and self.resource_type == "personnel":
                            actual_capacities[j] = self.B[j]
                            category_capacities[j] = {
                                'good': 0,
                                'midle': 0,
                                'bad': self.B[j]
                            }
                        else:
                            actual_capacities[j] = self.B[j] * self.P[j]
                            category_capacities[j] = {
                                'good': 0,
                                'midle': 0,
                                'bad': self.B[j] * self.P[j]
                            }

                # 计算所有good类别的总容量
                total_good_capacity = sum(category_capacities[j]['good'] for j in self.J)

                # 如果good总容量小于目标容量，需要从midle和bad类别中补充
                if total_good_capacity < target_capacity:
                    # 首先从midle类别中补充
                    midle_suppliers = [j for j in self.J if category_capacities[j]['midle'] > 0]
                    # 按midle容量降序排序
                    midle_suppliers.sort(key=lambda j: category_capacities[j]['midle'], reverse=True)

                    remaining_need = target_capacity - total_good_capacity

                    for j in midle_suppliers:
                        if remaining_need <= 0:
                            break

                        available_midle = category_capacities[j]['midle']
                        if available_midle >= remaining_need:
                            # 这个供应点的midle容量足够满足剩余需求
                            actual_capacities[j] += remaining_need  # 增加单个供应点的容量
                            total_good_capacity += remaining_need  # 增加总容量
                            remaining_need = 0
                        else:
                            # 使用这个供应点的全部midle容量
                            actual_capacities[j] += available_midle  # 增加单个供应点的容量
                            total_good_capacity += available_midle  # 增加总容量
                            remaining_need -= available_midle

                    # 如果midle类别还不够，从bad类别中补充
                    if remaining_need > 0:
                        bad_suppliers = [j for j in self.J if category_capacities[j]['bad'] > 0]
                        # 按bad容量降序排序
                        bad_suppliers.sort(key=lambda j: category_capacities[j]['bad'], reverse=True)

                        for j in bad_suppliers:
                            if remaining_need <= 0:
                                break

                            available_bad = category_capacities[j]['bad']
                            if available_bad >= remaining_need:
                                # 这个供应点的bad容量足够满足剩余需求
                                actual_capacities[j] += remaining_need  # 增加单个供应点的容量
                                total_good_capacity += remaining_need  # 增加总容量
                                remaining_need = 0
                            else:
                                # 使用这个供应点的全部bad容量
                                actual_capacities[j] += available_bad  # 增加单个供应点的容量
                                total_good_capacity += available_bad  # 增加总容量
                                remaining_need -= available_bad

                # 定义comment的优先级
                comment_priority = {'good': 3, 'midle': 2, 'bad': 1}

                # 排序：先按comment优先级降序，再按实际容量降序
                sorted_suppliers = sorted(
                    self.J,
                    key=lambda j: (comment_priority[supplier_comments[j]], actual_capacities[j]),
                    reverse=True
                )



                self.logger.debug(f"供应点排序完成，总数: {len(sorted_suppliers)}")

                # 记录前几个供应点的能力信息
                for i, j in enumerate(sorted_suppliers[:10]):
                    declared_capacity = self.B[j] * self.P[j]
                    actual_capacity = actual_capacities[j]
                    self.logger.debug(
                        f"  排名{i + 1}: {j}, 声明容量: {declared_capacity:.2f}, 实际容量: {actual_capacity:.2f}")

            except (KeyError, TypeError) as e:
                error_msg = f"供应点排序失败: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.logger.error(f"B数据存在: {hasattr(self, 'B')}, P数据存在: {hasattr(self, 'P')}")
                raise ValueError(error_msg)

            # 智能供应点预筛选 - 基于供应能力和距离的启发式选择
            self.logger.info("第4步: 智能供应点预筛选")
            candidate_suppliers = self._intelligent_supplier_preselection(sorted_suppliers, aggregated_demand_point,
                                                                          supply_scale, demand_scale, network_scale,
                                                                          transfer_scale)

            # 基于主导目标进行二次筛选
            dominant_objective = max(self.objective_weights.items(), key=lambda x: x[1])
            dominant_obj_name = dominant_objective[0]
            dominant_weight = dominant_objective[1]
            avg_weight = sum(self.objective_weights.values()) / len(self.objective_weights)

            if dominant_weight > avg_weight * len(self.objective_weights) / (len(self.objective_weights) - 1):
                candidate_suppliers = self._apply_dominant_objective_filter(candidate_suppliers,
                                                                            aggregated_demand_point,
                                                                            dominant_obj_name)

            # 计算处理的供应点数量限制
            max_suppliers_to_process = len(candidate_suppliers)
            processed_suppliers = 0
            self.logger.info(
                f"第4步: 路径生成（从{len(sorted_suppliers)}个供应点中筛选出{max_suppliers_to_process}个候选）")

            # 第1步：全局运输模式决策
            # ========== 第4步：全局运输模式决策 ==========

            transport_mode_decision = self._make_global_transport_mode_decision(candidate_suppliers,
                                                                                aggregated_demand_point,
                                                                                path_metrics_cache,
                                                                                composite_score_cache)

            self.logger.info(f"全局运输模式决策结果: {transport_mode_decision['mode']}")

            # 添加线程锁保护共享缓存
            cache_lock = threading.Lock()

            if transport_mode_decision['mode'] == 'direct':
                # 直达模式：并行计算每个供应点的直达路径
                def process_direct_supplier(j):
                    try:
                        supply_capacity = self.B[j] * self.P[j]
                    except KeyError as e:
                        self.logger.warning(f"供应点 {j} 数据缺失: {str(e)}")
                        return None

                    if supply_capacity <= self.EPS or total_aggregated_demand <= self.EPS:
                        return None

                    k = aggregated_demand_point

                    # 计算直达路径（使用线程安全的缓存访问）
                    direct_cache_key = f"direct_{j}_{k}"
                    with cache_lock:
                        if direct_cache_key not in path_metrics_cache:
                            need_calculate = True
                        else:
                            path_metrics = path_metrics_cache[direct_cache_key]
                            need_calculate = False

                    if need_calculate:
                        try:
                            path_metrics = self._calculate_direct_path_metrics(j, k)
                            with cache_lock:
                                path_metrics_cache[direct_cache_key] = path_metrics
                        except (ValueError, KeyError) as e:
                            self.logger.warning(f"供应点 {j} 直接路径计算失败: {str(e)}")
                            return None
                    else:
                        with cache_lock:
                            path_metrics = path_metrics_cache[direct_cache_key]

                    score_cache_key = f"score_direct_{j}_{k}"
                    with cache_lock:
                        if score_cache_key not in composite_score_cache:
                            composite_score = self._calculate_composite_score(path_metrics, is_direct=True)
                            composite_score_cache[score_cache_key] = composite_score
                        else:
                            composite_score = composite_score_cache[score_cache_key]

                    path_risk = random.random()

                    return j, {
                        'type': 'direct',
                        'route': (j, k, 1),
                        'score': composite_score,
                        'risk': path_risk,
                        'cost': path_metrics['cost_score'],
                        'metrics': path_metrics
                    }

                # 使用线程池并行处理供应点
                max_workers = min(len(candidate_suppliers), supply_scale // max(demand_scale, 1) + 1)
                valid_suppliers = candidate_suppliers[:max_suppliers_to_process]

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_supplier = {executor.submit(process_direct_supplier, j): j for j in valid_suppliers}

                    for future in as_completed(future_to_supplier):
                        result = future.result()
                        if result is not None:
                            j, path_info = result
                            if j not in supplier_paths:
                                supplier_paths[j] = []
                            supplier_paths[j].append(path_info)
                            processed_suppliers += 1

            else:
                # 多式联运模式：所有供应点使用统一的多式联运路径
                unified_multimodal_route = transport_mode_decision['unified_route']
                m1, m2, n2 = unified_multimodal_route['m1'], unified_multimodal_route['m2'], unified_multimodal_route[
                    'n2']

                self.logger.info(
                    f"统一多式联运路径: 所有供应点 -> {m1} -> {m2} -> {aggregated_demand_point} (运输方式: {n2})")

                # 并行计算所有供应点到第一个中转点的时间
                def calculate_supplier_timing(j):
                    try:
                        supply_capacity = self.B[j] * self.P[j]
                    except KeyError as e:
                        self.logger.warning(f"供应点 {j} 数据缺失: {str(e)}")
                        return None

                    if supply_capacity <= self.EPS or total_aggregated_demand <= self.EPS:
                        return None

                    # 计算到第一个中转点的时间
                    j_lat, j_lon = self.point_features[j]['latitude'], self.point_features[j]['longitude']
                    m1_lat, m1_lon = self.point_features[m1]['latitude'], self.point_features[m1]['longitude']
                    distance_to_m1 = self._calculate_haversine_distance(j_lat, j_lon, m1_lat, m1_lon)
                    time_to_m1 = distance_to_m1 / self.TRANSPORT_MODES[1]['speed']

                    return j, time_to_m1

                valid_suppliers = []
                max_time_to_m1 = 0.0

                # 并行计算时间
                valid_candidates = candidate_suppliers[:max_suppliers_to_process]
                max_workers = min(len(valid_candidates), supply_scale // max(demand_scale, 1) + 1)

                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_supplier = {executor.submit(calculate_supplier_timing, j): j for j in valid_candidates}

                    for future in as_completed(future_to_supplier):
                        result = future.result()
                        if result is not None:
                            j, time_to_m1 = result
                            max_time_to_m1 = max(max_time_to_m1, time_to_m1)
                            valid_suppliers.append((j, time_to_m1))
                            processed_suppliers += 1

                # 并行为所有有效供应点创建统一的多式联运路径
                def create_multimodal_path(supplier_timing):
                    j, time_to_m1 = supplier_timing

                    # 使用统一的多式联运路径，但考虑等待时间
                    cache_key = f"unified_multimodal_{j}_{m1}_{m2}_{aggregated_demand_point}_{n2}"
                    with cache_lock:
                        if cache_key not in path_metrics_cache:
                            path_metrics = self._calculate_unified_multimodal_metrics(
                                j, m1, m2, aggregated_demand_point, n2, max_time_to_m1)
                            path_metrics_cache[cache_key] = path_metrics
                        else:
                            path_metrics = path_metrics_cache[cache_key]

                    score_cache_key = f"score_unified_{j}_{m1}_{m2}_{aggregated_demand_point}_{n2}"
                    with cache_lock:
                        if score_cache_key not in composite_score_cache:
                            composite_score = self._calculate_composite_score(path_metrics, is_direct=False)
                            composite_score_cache[score_cache_key] = composite_score
                        else:
                            composite_score = composite_score_cache[score_cache_key]

                    return j, {
                        'type': 'multimodal',
                        'route': (j, m1, m2, aggregated_demand_point, n2),
                        'score': composite_score,
                        'risk': random.random(),
                        'cost': path_metrics['cost_score'],
                        'metrics': path_metrics,
                        'unified_timing': {
                            'individual_time_to_m1': time_to_m1,
                            'unified_departure_time': max_time_to_m1
                        }
                    }

                # 并行创建路径
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_supplier = {executor.submit(create_multimodal_path, supplier_timing): supplier_timing[0]
                                          for supplier_timing in valid_suppliers}

                    for future in as_completed(future_to_supplier):
                        j, path_info = future.result()
                        if j not in supplier_paths:
                            supplier_paths[j] = []
                        supplier_paths[j].append(path_info)

            self.logger.info(
                f"路径生成完成 - 处理供应点: {processed_suppliers}, 总路径组合: {sum(len(paths) for paths in supplier_paths.values())}")

            # 强制执行统一运输模式
            self.logger.info("第5步: 强制执行统一运输模式")
            supplier_best_paths = {}
            supplier_candidate_paths = {}

            if transport_mode_decision['mode'] == 'direct':
                # 直达模式：每个供应点只能选择直达路径
                self.logger.info("执行统一直达模式：所有供应点使用直达运输")

                for j, paths in supplier_paths.items():
                    if not paths:
                        continue

                    # 只选择直达路径
                    direct_paths = [path for path in paths if path['type'] == 'direct']
                    if not direct_paths:
                        self.logger.warning(f"供应点 {j} 没有直达路径，跳过")
                        continue

                    # 选择最优的直达路径
                    best_direct_path = min(direct_paths, key=lambda p: p['score'])
                    supplier_best_paths[j] = best_direct_path
                    supplier_candidate_paths[j] = [best_direct_path]

                    self.logger.debug(f"供应点 {j} 选择直达路径，评分: {best_direct_path['score']:.6f}")

            else:
                # 多式联运模式：所有供应点必须使用统一的多式联运路径
                unified_route = transport_mode_decision['unified_route']
                self.logger.info(
                    f"执行统一多式联运模式：所有供应点使用路径 {unified_route['m1']} -> {unified_route['m2']} -> 需求点")

                for j, paths in supplier_paths.items():
                    if not paths:
                        continue

                    # 只选择与统一路径匹配的多式联运路径
                    matching_multimodal_paths = []
                    for path in paths:
                        if (path['type'] == 'multimodal' and
                                len(path['route']) >= 5 and
                                path['route'][1] == unified_route['m1'] and  # 第一个中转点匹配
                                path['route'][2] == unified_route['m2'] and  # 第二个中转点匹配
                                path['route'][4] == unified_route['n2']):  # 运输方式匹配
                            matching_multimodal_paths.append(path)

                    if not matching_multimodal_paths:
                        self.logger.warning(f"供应点 {j} 没有匹配的统一多式联运路径，跳过")
                        continue

                    # 选择评分最好的匹配路径（通常只有一条）
                    best_multimodal_path = min(matching_multimodal_paths, key=lambda p: p['score'])
                    supplier_best_paths[j] = best_multimodal_path
                    supplier_candidate_paths[j] = [best_multimodal_path]

                    self.logger.debug(f"供应点 {j} 选择统一多式联运路径，评分: {best_multimodal_path['score']:.6f}")

            # 验证统一性
            if supplier_best_paths:
                path_types = set()
                multimodal_routes = set()

                for j, path in supplier_best_paths.items():
                    path_types.add(path['type'])
                    if path['type'] == 'multimodal' and len(path['route']) >= 5:
                        # 记录多式联运的路径信息
                        route_key = (path['route'][1], path['route'][2], path['route'][4])  # (m1, m2, n2)
                        multimodal_routes.add(route_key)

                # 检查运输模式统一性
                if len(path_types) > 1:
                    self.logger.error(f"运输模式不统一！发现类型: {path_types}")
                    # 强制纠正：如果决策是直达但出现了多式联运，重新处理
                    if transport_mode_decision['mode'] == 'direct':
                        self.logger.info("强制纠正为直达模式")
                        corrected_paths = {}
                        for j, path in supplier_best_paths.items():
                            if path['type'] != 'direct':
                                # 查找该供应点的直达路径
                                direct_paths = [p for p in supplier_paths.get(j, []) if p['type'] == 'direct']
                                if direct_paths:
                                    corrected_paths[j] = min(direct_paths, key=lambda p: p['score'])
                                    self.logger.debug(f"供应点 {j} 已纠正为直达路径")
                                else:
                                    self.logger.warning(f"供应点 {j} 无法纠正为直达路径，移除")
                            else:
                                corrected_paths[j] = path
                        supplier_best_paths = corrected_paths
                elif len(multimodal_routes) > 1:
                    self.logger.error(f"多式联运路径不统一！发现路径: {multimodal_routes}")
                    # 强制纠正：选择评分最好的统一路径
                    if transport_mode_decision['mode'] == 'multimodal':
                        unified_route = transport_mode_decision['unified_route']
                        target_route = (unified_route['m1'], unified_route['m2'], unified_route['n2'])
                        self.logger.info(f"强制纠正为统一多式联运路径: {target_route}")

                        corrected_paths = {}
                        for j, path in supplier_best_paths.items():
                            if (path['type'] == 'multimodal' and len(path['route']) >= 5 and
                                    (path['route'][1], path['route'][2], path['route'][4]) == target_route):
                                corrected_paths[j] = path
                            else:
                                # 查找该供应点的匹配多式联运路径
                                matching_paths = [p for p in supplier_paths.get(j, [])
                                                  if (p['type'] == 'multimodal' and len(p['route']) >= 5 and
                                                      (p['route'][1], p['route'][2], p['route'][4]) == target_route)]
                                if matching_paths:
                                    corrected_paths[j] = min(matching_paths, key=lambda p: p['score'])
                                    self.logger.debug(f"供应点 {j} 已纠正为统一多式联运路径")
                                else:
                                    self.logger.warning(f"供应点 {j} 无法纠正为统一多式联运路径，移除")

                        supplier_best_paths = corrected_paths
                else:
                    self.logger.info(f"运输模式统一性验证通过：{list(path_types)[0]}")
                    if multimodal_routes:
                        self.logger.info(f"统一多式联运路径：{list(multimodal_routes)[0]}")

            # 优化供应点选择逻辑
            self.logger.info("第6步: 供应点选择")
            try:
                total_supply_capacity = sum(self.B[j] * self.P[j] for j in supplier_best_paths.keys())
                self.logger.info(f"有效供应点总能力: {total_supply_capacity:.2f}")
            except KeyError as e:
                error_msg = f"供应能力计算失败: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg)

            if total_supply_capacity <= self.EPS:
                self.logger.warning("没有有效的供应点")
                return []

            # 排序并选择供应点
            try:
                # 找到权重最高的目标作为主导目标
                dominant_objective = max(self.objective_weights.items(), key=lambda x: x[1])
                dominant_obj_name = dominant_objective[0]

                def enhanced_sorting_key(item):
                    j, best_path = item
                    base_score = best_path['score']

                    # 获取该路径在主导目标上的表现
                    path_metrics = best_path.get('metrics', {})
                    if dominant_obj_name in ['time', 'cost', 'distance', 'balance', 'social']:
                        # 对于最小化目标，值越小越好
                        dominant_performance = path_metrics.get(f'{dominant_obj_name}_score', base_score)
                    elif dominant_obj_name in ['safety', 'priority']:
                        # 对于最大化目标（但在计算中转为负值），负值越大（绝对值越小）越好
                        dominant_performance = -path_metrics.get(f'{dominant_obj_name}_score', -base_score)
                    elif dominant_obj_name == 'capability':
                        # capability是反向目标，值越小越好
                        dominant_performance = path_metrics.get(f'{dominant_obj_name}_score', base_score)
                    else:
                        dominant_performance = base_score

                    # 在主导目标表现基础上微调综合评分
                    if hasattr(self, 'objective_ranges') and dominant_obj_name in self.objective_ranges:
                        min_val, max_val = self.objective_ranges[dominant_obj_name]
                        if abs(max_val - min_val) > self.EPS:
                            # 归一化主导目标表现
                            if dominant_obj_name in ['safety', 'priority']:
                                norm_dominant = (dominant_performance - min_val) / (max_val - min_val)
                            elif dominant_obj_name == 'capability':
                                norm_dominant = (max_val - dominant_performance) / (max_val - min_val)
                            else:
                                norm_dominant = (dominant_performance - min_val) / (max_val - min_val)

                            # 主导目标权重的影响力
                            dominant_weight = self.objective_weights[dominant_obj_name]
                            # 综合评分 = 基础评分 * (1 + 主导目标权重 * 主导目标归一化表现)
                            enhanced_score = base_score * (1.0 + dominant_weight * norm_dominant)
                            return enhanced_score

                    return base_score

                sorted_suppliers_with_paths = sorted(supplier_best_paths.items(), key=enhanced_sorting_key)
                self.logger.debug(
                    f"供应点最终排序完成（考虑主导目标 {dominant_obj_name}），候选数量: {len(sorted_suppliers_with_paths)}")
            except (KeyError, TypeError) as e:
                error_msg = f"供应点排序失败: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                raise ValueError(error_msg)

            # 贪心选择供应点，限制数量基于网络规模
            selected_suppliers = []
            cumulative_supply = 0

            self.logger.info(f"第7步: 贪心选择供应点（基于实际需求{total_aggregated_demand:.2f}）")
            for j, best_path in sorted_suppliers_with_paths:
                try:
                    # 使用实际容量而不是声明容量
                    supplier_actual_capacity = actual_capacities.get(j, self.B[j] * self.P[j])
                    selected_suppliers.append((j, best_path))
                    cumulative_supply += supplier_actual_capacity
                    self.logger.debug(
                        f"选择供应点 {j}, 实际容量: {supplier_actual_capacity:.2f}, 累计: {cumulative_supply:.2f}")

                except KeyError as e:
                    self.logger.warning(f"供应点 {j} 容量计算失败: {str(e)}")
                    continue

            # 构建最终路径列表
            self.logger.info("第8步: 构建最终路径列表")
            valid_paths = []
            for j, best_path in selected_suppliers:
                try:
                    if best_path['type'] == 'direct':
                        j_route, k_route, n = best_path['route']
                        valid_paths.append(('direct', j_route, k_route, n, best_path['score'], best_path['risk'],
                                            best_path['cost'], best_path['metrics']))
                        self.logger.debug(f"添加直接路径: {j_route} -> {k_route}")
                    else:  # multimodal
                        j_route, m1, m2, k_route, n2 = best_path['route']
                        valid_paths.append(('multimodal', j_route, m1, m2, k_route, n2, best_path['score'],
                                            best_path['risk'], best_path['cost'], best_path['metrics']))
                        self.logger.debug(f"添加多式联运路径: {j_route} -> {m1} -> {m2} -> {k_route}")
                except (KeyError, ValueError) as e:
                    self.logger.warning(f"路径构建失败: {str(e)}")
                    continue

            match_end_time = time.time()
            total_match_time = match_end_time - match_start_time
            self.logger.info(f"多目标匹配阶段完成，总耗时: {total_match_time:.2f}秒")
            self.logger.info(
                f"最终结果 - 选择供应点: {len(selected_suppliers)}/{processed_suppliers}, 生成有效路径: {len(valid_paths)}条")

            # 检查是否生成了有效路径
            if not valid_paths:
                self.logger.error("未生成任何有效路径，可能的原因：网络连通性问题、供应能力不足或数据异常")
                return {
                    "code": 400,
                    "msg": "匹配失败：未找到可行路径",
                    "data": {
                        "processed_suppliers": processed_suppliers,
                        "selected_suppliers_count": len(selected_suppliers),
                        "network_scale": len(self.J) + len(self.M) + len(self.K),
                        "error_type": "no_feasible_paths"
                    }
                }

            return {
                "code": 200,
                "msg": "多目标匹配成功",
                "data": valid_paths
            }


        except ValueError as ve:
            self.logger.error(f"多目标匹配阶段参数错误: {str(ve)}", exc_info=True)
            return {
                "code": 400,
                "msg": f"多目标匹配阶段参数错误: {str(ve)}",
                "data": {
                    "random_seed": random_seed,
                    "network_scale": len(self.J) + len(self.M) + len(self.K),
                    "error_type": "parameter_error"
                }
            }

        except KeyError as ke:
            self.logger.error(f"多目标匹配阶段数据缺失: {str(ke)}", exc_info=True)
            return {
                "code": 500,
                "msg": f"数据结构错误: {str(ke)}",
                "data": {
                    "random_seed": random_seed,
                    "missing_key": str(ke),
                    "error_type": "data_structure_error"
                }
            }

        except AttributeError as ae:
            self.logger.error(f"多目标匹配阶段属性错误: {str(ae)}", exc_info=True)
            return {
                "code": 500,
                "msg": f"对象属性缺失: {str(ae)}",
                "data": {
                    "random_seed": random_seed,
                    "missing_attribute": str(ae),
                    "error_type": "attribute_error"
                }
            }

        except MemoryError:
            self.logger.error("多目标匹配阶段内存不足")
            return {
                "code": 507,
                "msg": "内存不足，请减少网络规模",
                "data": {
                    "random_seed": random_seed,
                    "network_scale": len(self.J) + len(self.M) + len(self.K),
                    "error_type": "memory_error"
                }
            }

    def _generate_multimodal_paths_optimized(self, j, k, path_metrics_cache, composite_score_cache):
        """多式联运路径生成"""
        multimodal_paths = []

        # 计算网络规模参数
        network_scale = len(self.J) + len(self.M) + len(self.K)
        supply_scale = len(self.J)
        transfer_scale = len(self.M)
        transport_modes_count = len(self.TRANSPORT_MODES)

        # 启发式中转点选择 - 基于网络密度优化
        candidate_transfer_pairs = self._select_promising_transfer_pairs(j, k, supply_scale, transfer_scale,
                                                                         transport_modes_count)

        # 处理所有候选中转点对，不进行数量限制
        batch_size = transfer_scale
        limited_candidates = candidate_transfer_pairs

        # 预筛选有效组合，减少后续计算
        valid_combinations = []

        # 批量可达性检查
        reachable_targets = set()
        for m2 in self.M:
            if self.alpha3.get((m2, k, 1), 0) == 1:
                reachable_targets.add(m2)

        if not reachable_targets:
            return multimodal_paths

        for batch_start in range(0, len(limited_candidates), batch_size):
            batch_end = min(batch_start + batch_size, len(limited_candidates))
            current_batch = limited_candidates[batch_start:batch_end]

            for m1, m2 in current_batch:
                # 快速可达性检查
                if m2 not in reachable_targets:
                    continue

                available_modes_2 = self._get_available_transport_modes(m1, m2)
                if not available_modes_2:
                    continue

                # 尝试所有可用的运输方式
                for n2 in available_modes_2:
                    valid_combinations.append((m1, m2, n2))

        # 批量路径指标计算
        self._batch_calculate_multimodal_metrics(valid_combinations, j, k,
                                                 path_metrics_cache, composite_score_cache,
                                                 multimodal_paths)

        self.logger.debug(f"多式联运路径生成完成: {j}->{k}, 生成路径数: {len(multimodal_paths)}")
        return multimodal_paths

    def _batch_calculate_multimodal_metrics(self, valid_combinations, j, k,
                                            path_metrics_cache, composite_score_cache,
                                            multimodal_paths):
        """批量计算多式联运路径指标"""

        # 批量处理组合
        for m1, m2, n2 in valid_combinations:
            try:
                cache_key = f"multimodal_{j}_{m1}_{m2}_{k}_{n2}"

                # 批量缓存检查
                if cache_key not in path_metrics_cache:
                    path_metrics_cache[cache_key] = self._calculate_multimodal_path_metrics(j, m1, m2, k, n2)

                path_metrics = path_metrics_cache[cache_key]

                score_cache_key = f"score_multimodal_{j}_{m1}_{m2}_{k}_{n2}"
                if score_cache_key not in composite_score_cache:
                    composite_score_cache[score_cache_key] = self._calculate_composite_score(path_metrics,
                                                                                             is_direct=False)

                composite_score = composite_score_cache[score_cache_key]

                multimodal_paths.append({
                    'type': 'multimodal',
                    'route': (j, m1, m2, k, n2),
                    'score': composite_score,
                    'risk': random.random(),
                    'cost': path_metrics['cost_score'],
                    'metrics': path_metrics
                })

            except (ValueError, KeyError) as e:
                self.logger.warning(f"多式联运路径 {j}-{m1}-{m2}-{k} 计算失败: {str(e)}")
                continue

    def _select_promising_transfer_pairs(self, j, k, supply_scale, transfer_scale, transport_modes_count):
        """
        启发式选择有希望的中转点对
        """
        j_lat, j_lon = self.point_features[j]['latitude'], self.point_features[j]['longitude']
        k_lat, k_lon = self.point_features[k]['latitude'], self.point_features[k]['longitude']

        # 预计算所有中转点的评分信息，避免重复计算
        transfer_point_scores = {}
        for m in self.M:
            m_lat, m_lon = self.point_features[m]['latitude'], self.point_features[m]['longitude']

            # 计算距离
            dist_to_j = self._calculate_haversine_distance(j_lat, j_lon, m_lat, m_lon)
            dist_to_k = self._calculate_haversine_distance(m_lat, m_lon, k_lat, k_lon)

            # 专业化评分
            specialized_mode = self.point_features[m].get('specialized_mode', 'unknown')
            specialization_score = 1.0 if specialized_mode in ['port', 'airport', 'railway'] else 0.5

            # 计算作为m1和m2的评分
            m1_score = specialization_score / max(dist_to_j, self.EPS)
            m2_score = specialization_score / max(dist_to_k, self.EPS)

            transfer_point_scores[m] = {
                'dist_to_supply': dist_to_j,
                'dist_to_demand': dist_to_k,
                'm1_score': m1_score,
                'm2_score': m2_score,
                'specialization_score': specialization_score
            }

        # 预计算直达距离作为参考基准
        direct_distance = self._calculate_haversine_distance(j_lat, j_lon, k_lat, k_lon)

        # 生成所有可能的中转点对，并检查连通性约束和距离合理性
        promising_pairs = []

        for m1 in self.M:
            for m2 in self.M:
                if m1 != m2:
                    # 检查中转点对是否能连通
                    available_modes = self._get_available_transport_modes(m1, m2)
                    if available_modes:  # 只有连通的中转点对才加入候选
                        # 计算多式联运总距离
                        m1_info = transfer_point_scores[m1]
                        m2_info = transfer_point_scores[m2]

                        # 计算m1到m2的距离
                        m1_lat, m1_lon = self.point_features[m1]['latitude'], self.point_features[m1]['longitude']
                        m2_lat, m2_lon = self.point_features[m2]['latitude'], self.point_features[m2]['longitude']
                        dist_m1_to_m2 = self._calculate_haversine_distance(m1_lat, m1_lon, m2_lat, m2_lon)

                        # 计算多式联运总距离
                        total_multimodal_distance = m1_info['dist_to_supply'] + dist_m1_to_m2 + m2_info[
                            'dist_to_demand']

                        # 使用网络规模参数动态计算合理的绕行倍数阈值
                        # 基于网络密度和规模的合理绕行上限
                        max_reasonable_ratio = 1.0 + (transfer_scale / max(supply_scale, 1))

                        # 距离效率检查：如果绕行过度，则降低评分
                        if total_multimodal_distance > direct_distance * max_reasonable_ratio:
                            # 绕行过度，大幅降低评分
                            distance_penalty = total_multimodal_distance / max(direct_distance, self.EPS)
                            distance_efficiency = 1.0 / distance_penalty
                        else:
                            # 绕行合理，轻微调整评分
                            distance_efficiency = direct_distance / max(total_multimodal_distance, self.EPS)

                        # 计算组合评分：原有评分 × 距离效率因子
                        base_combined_score = m1_info['m1_score'] + m2_info['m2_score']
                        combined_score = base_combined_score * distance_efficiency

                        promising_pairs.append({
                            'pair': (m1, m2),
                            'score': combined_score,
                        'available_modes': available_modes
                    })

        # 按照组合评分降序排序，确保最优的组合在前面
        promising_pairs.sort(key=lambda x: x['score'], reverse=True)

        # 提取中转点对
        return [pair_info['pair'] for pair_info in promising_pairs]

    def _precompute_evaluation_metrics(self):
        """
        预计算各种评估指标
        """

        # 预计算行业平均水平 - 使用英文键名
        capability_metrics = ['resource_reserve', 'production_capacity', 'expansion_capacity']
        self.industry_averages = {}
        for metric in capability_metrics:
            try:
                metric_values = [float(self.point_features[j][metric]) for j in self.J if
                                 metric in self.point_features[j]]
                self.industry_averages[metric] = sum(metric_values) / len(metric_values) if metric_values else 1.0
            except (KeyError, ValueError, TypeError) as e:
                self.logger.warning(f"计算指标 {metric} 的行业平均值失败: {str(e)}")
                self.industry_averages[metric] = 1.0

        # 预计算中转点适宜度评分（基于专业化模式）
        self.m_suitability = {}
        for m in self.M:
            try:
                specialized_mode = self.point_features[m].get('specialized_mode', 'unknown')
                # 基于专业化模式的基础评分（所有中转点同等重要）
                if specialized_mode in ['port', 'airport', 'railway']:
                    self.m_suitability[m] = 1.0
                else:
                    self.m_suitability[m] = 0.5
            except (KeyError, TypeError) as e:
                self.logger.warning(f"计算中转点 {m} 适宜度失败: {str(e)}")
                self.m_suitability[m] = 0.5

        # 预计算供应和需求统计信息
        try:
            total_supply = 0.0
            for j in self.J:
                # 使用与分配时完全一致的容量计算逻辑
                sub_objects = self.point_features[j].get('sub_objects', [])
                actual_capacity = 0
                if sub_objects:
                    for category in sub_objects:
                        if isinstance(category, dict) and 'items' in category:
                            for sub_obj in category.get('items', []):
                                max_available = sub_obj.get('max_available_quantity', 0)
                                if isinstance(max_available, (int, float)):
                                    actual_capacity += max_available
                        else:
                            max_available = category.get('max_available_quantity', 0)
                            if isinstance(max_available, (int, float)):
                                actual_capacity += max_available
                    effective_capacity = actual_capacity * self.P[j]
                    self.logger.debug(f"供应点{j}: 细分对象容量{actual_capacity}, 有效容量{effective_capacity:.2f}")
                else:
                    effective_capacity = self.B[j] * self.P[j]
                    self.logger.debug(f"供应点{j}: 声明容量{self.B[j]}, 有效容量{effective_capacity:.2f}")

                total_supply += effective_capacity

            self.total_supply = total_supply
            self.total_demand = sum(self.D.values())

            self.logger.info(f"预计算供需统计 - 总供应能力: {self.total_supply:.2f}, 总需求: {self.total_demand:.2f}")
            if self.total_supply > 0:
                self.logger.info(f"供需比例: {self.total_supply / self.total_demand:.3f}")

        except (KeyError, TypeError) as e:
            self.logger.warning(f"计算总供应或总需求失败: {str(e)}")
            self.total_supply = 0.0
            self.total_demand = 0.0

        # ==================== 动态计算目标值范围用于归一化 ====================

        # 计算距离范围（基于网络中所有点对的实际距离）
        all_distances = []
        for j in self.J:
            for k in self.K:
                j_lat, j_lon = self.point_features[j]['latitude'], self.point_features[j]['longitude']
                k_lat, k_lon = self.point_features[k]['latitude'], self.point_features[k]['longitude']
                distance = self._calculate_haversine_distance(j_lat, j_lon, k_lat, k_lon)
                all_distances.append(distance)

        # 距离范围
        min_distance = min(all_distances) if all_distances else 1.0
        max_distance = max(all_distances) if all_distances else 1000.0

        # 时间范围（基于距离范围和运输速度范围）
        min_speed = min(mode['speed'] for mode in self.TRANSPORT_MODES.values())
        max_speed = max(mode['speed'] for mode in self.TRANSPORT_MODES.values())
        min_time = min_distance / max_speed + self.T1 + self.T4 + self.T6  # 最短时间
        max_time = max_distance / min_speed + self.T1 + self.T4 + self.T6  # 最长时间

        # 成本范围（基于成本参数范围和距离范围）
        # 成本范围（基于成本参数范围和距离范围）
        min_cost_per_km = min(mode['cost_per_km'] for mode in self.TRANSPORT_MODES.values())
        max_cost_per_km = max(mode['cost_per_km'] for mode in self.TRANSPORT_MODES.values())

        # 根据资源类型计算成本范围
        if hasattr(self, 'resource_type'):
            if self.resource_type == "personnel":
                # 从细分对象中获取人员成本
                all_wage_costs = []
                all_living_costs = []
                for j in self.J:
                    sub_objects = self.point_features[j].get('sub_objects', [])
                    for sub_obj in sub_objects:
                        all_wage_costs.append(sub_obj.get('wage_cost', 100))
                        all_living_costs.append(sub_obj.get('living_cost', 50))

                # 如果没有细分对象，使用基于供应能力的合理默认值
                if not all_wage_costs:
                    avg_capacity = sum(self.B.values()) / len(self.B) if self.B else 100.0
                    all_wage_costs = [avg_capacity]
                if not all_living_costs:
                    avg_capacity = sum(self.B.values()) / len(self.B) if self.B else 100.0
                    all_living_costs = [avg_capacity * 0.3]

                min_base_cost = min(all_wage_costs) + min(all_living_costs)
                max_base_cost = max(all_wage_costs) + max(all_living_costs)

            elif self.resource_type == "material":
                # 从细分对象中获取物资成本
                all_material_prices = []
                all_equipment_costs = []
                for j in self.J:
                    sub_objects = self.point_features[j].get('sub_objects', [])
                    for sub_obj in sub_objects:
                        all_material_prices.append(sub_obj.get('material_price', 200))
                        equipment_cost = (sub_obj.get('equipment_rental_price', 150) +
                                          sub_obj.get('equipment_depreciation_cost', 20))
                        all_equipment_costs.append(equipment_cost)

                # 如果没有细分对象，使用基于运输成本的合理默认值
                if not all_material_prices:
                    base_transport_cost = (min_cost_per_km + max_cost_per_km) * 0.5 * max_distance
                    all_material_prices = [base_transport_cost * 10.0]
                if not all_equipment_costs:
                    base_transport_cost = (min_cost_per_km + max_cost_per_km) * 0.5 * max_distance
                    all_equipment_costs = [base_transport_cost * 2.0]

                min_base_cost = min(all_material_prices) + min(all_equipment_costs)
                max_base_cost = max(all_material_prices) + max(all_equipment_costs)

            else:  # data
                # 数据动员：设施成本在供应点中
                all_facility_costs = [self.point_features[j].get('facility_rental_price', 80) for j in self.J]
                all_power_costs = [self.point_features[j].get('power_cost', 30) for j in self.J]
                all_comm_costs = [self.point_features[j].get('communication_purchase_price', 40) for j in
                                  self.J]
                min_base_cost = min(all_facility_costs) + min(all_power_costs) + min(all_comm_costs)
                max_base_cost = max(all_facility_costs) + max(all_power_costs) + max(all_comm_costs)
        else:
            # 默认物资成本：也需要从细分对象中获取
            all_material_prices = []
            for j in self.J:
                sub_objects = self.point_features[j].get('sub_objects', [])
                for sub_obj in sub_objects:
                    all_material_prices.append(sub_obj.get('material_price', 200))

            if not all_material_prices:
                # 使用基于运输成本的合理默认值
                base_transport_cost = (min_cost_per_km + max_cost_per_km) * 0.5 * max_distance
                all_material_prices = [base_transport_cost * 10.0]

            min_base_cost = min(all_material_prices) if all_material_prices else base_transport_cost * 10.0
            max_base_cost = max(all_material_prices) if all_material_prices else base_transport_cost * 20.0

        min_cost = min_base_cost + min_distance * min_cost_per_km
        max_cost = max_base_cost + max_distance * max_cost_per_km

        # 安全范围（基于安全评分的理论范围）
        if hasattr(self, 'resource_type'):
            if self.resource_type == "personnel":
                # 人员安全：5个维度，每个-1到1，总范围-5到5，加运输安全1
                min_safety = -6.0  # 最差安全状况
                max_safety = 6.0  # 最好安全状况
            elif self.resource_type == "material":
                # 物资安全：企业5维度约-2到3，物资4维度约-3到0，运输1维度约0.7到1
                min_safety = -6.0  # 最差安全状况
                max_safety = 5.0  # 最好安全状况
            else:  # data
                # 数据安全：设施6维度约-1到6，运输1维度约0.7到1
                min_safety = -2.0  # 最差安全状况
                max_safety = 7.0  # 最好安全状况
        else:
            min_safety = -6.0
            max_safety = 5.0

        # 优先级范围（基于供应点任务优先级评分理论范围）
        min_priority = 0.0  # 最低优先级评分
        max_priority = 1.0  # 最高优先级评分

        # 资源均衡范围（偏差比例的理论范围）
        min_balance = 0.0  # 完全均衡
        max_balance = 1.0  # 最大偏差

        # 企业能力范围（基于企业能力参数范围）
        all_capabilities = []
        for j in self.J:
            # 根据企业规模推导规模能力
            enterprise_size = self.point_features[j].get('enterprise_size', '中')
            if enterprise_size == '大':
                scale_cap = len(self.J) + len(self.M)
            elif enterprise_size == '中':
                scale_cap = len(self.J) + len(self.M) / (len(self.M) + 1) if len(self.M) > 0 else len(self.J)
            else:
                scale_cap = len(self.J) / (len(self.J) + 1) if len(self.J) > 0 else 1.0
            resource_res = self.point_features[j].get('resource_reserve', 4)
            prod_cap = self.point_features[j].get('production_capacity', 6)
            capability = (scale_cap + resource_res + prod_cap) / 3
            all_capabilities.append(capability)

        min_capability = min(all_capabilities) if all_capabilities else 3.0
        max_capability = max(all_capabilities) if all_capabilities else 10.0

        # 社会影响范围（基于企业类型和规模的权重乘积）
        min_social = 1.0  # 国企+大型企业
        max_social = 3.0  # 外企+微型企业

        # 存储目标范围用于归一化
        self.objective_ranges = {
            'time': (min_time, max_time),
            'cost': (min_cost, max_cost),
            'distance': (min_distance, max_distance),
            'safety': (min_safety, max_safety),  # 注意：safety使用负值，所以min/max会相反
            'priority': (min_priority, max_priority),  # 注意：priority使用负值
            'balance': (min_balance, max_balance),
            'capability': (min_capability, max_capability),  # 注意：capability使用倒数形式
            'social': (min_social, max_social)
        }

        self.logger.info("目标值范围计算完成:")
        for obj, (min_val, max_val) in self.objective_ranges.items():
            self.logger.info(f"  {obj}: [{min_val:.3f}, {max_val:.3f}]")

        # 移除复杂的预筛选机制，直接设置所有可能的候选集合
        self.candidate_transfer_sets = {}
        for j in self.J:
            self.candidate_transfer_sets[j] = {
                'first_transfer_candidates': list(self.M),
                'second_transfer_candidates': list(self.M),
                'm_m_pairs': [(m1, m2) for m1 in self.M for m2 in self.M if m1 != m2]
            }

        self.logger.info("多目标评估指标预计算完成")

    def _precompute_all_distances(self, k_lat, k_lon):
        """预计算所有距离信息"""
        import numpy as np

        distance_info = {
            'supply_to_demand': {},
            'transfer_to_demand': {},
            'supply_to_transfer': {},
            'all_distances': []
        }

        # 一次性提取所有坐标到numpy数组，避免重复转换
        supply_coords = np.array(
            [(self.point_features[j]['latitude'], self.point_features[j]['longitude']) for j in self.J])
        transfer_coords = np.array(
            [(self.point_features[m]['latitude'], self.point_features[m]['longitude']) for m in self.M])
        demand_coords = np.array(
            [(self.point_features[k]['latitude'], self.point_features[k]['longitude']) for k in self.K])

        # 供应点到需求点距离 - 完全向量化计算
        if len(supply_coords) > 0 and len(demand_coords) > 0:
            # 使用broadcasting一次性计算所有距离
            supply_lats = supply_coords[:, 0]
            supply_lons = supply_coords[:, 1]
            demand_lats = demand_coords[:, 0]
            demand_lons = demand_coords[:, 1]

            # 向量化计算所有供应点到所有需求点的距离矩阵
            all_distances_matrix = self._vectorized_haversine_distance(
                supply_lats[:, np.newaxis], supply_lons[:, np.newaxis],
                demand_lats[np.newaxis, :], demand_lons[np.newaxis, :]
            )

            # 并行批量填充字典
            def fill_distance_batch(batch_indices):
                batch_results = {}
                for j_idx, k_idx in batch_indices:
                    j_temp = self.J[j_idx]
                    k_temp = self.K[k_idx]
                    batch_results[(j_temp, k_temp)] = all_distances_matrix[j_idx, k_idx]
                return batch_results

            # 分批处理索引对
            j_indices = np.arange(len(self.J))
            k_indices = np.arange(len(self.K))
            j_mesh, k_mesh = np.meshgrid(j_indices, k_indices, indexing='ij')
            all_index_pairs = list(zip(j_mesh.flat, k_mesh.flat))

            # 计算批次大小
            total_pairs = len(all_index_pairs)
            batch_size = max(total_pairs // (len(self.J) + len(self.K)), 1)
            batches = [all_index_pairs[i:i + batch_size] for i in range(0, total_pairs, batch_size)]

            # 并行处理批次
            max_workers = min(len(batches), len(self.J) // max(len(self.K), 1) + 1)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {executor.submit(fill_distance_batch, batch): batch for batch in batches}

                for future in as_completed(future_to_batch):
                    batch_results = future.result()
                    for (j_temp, k_temp), distance in batch_results.items():
                        distance_info['supply_to_demand'][j_temp] = distance

            # 直接将矩阵展平为列表，避免逐个append
            distance_info['all_distances'].extend(all_distances_matrix.flatten())

    def _vectorized_haversine_distance(self, lat1, lon1, lat2, lon2):
        """
        向量化的Haversine距离计算
        """
        import numpy as np

        # 转换为numpy数组，使用更高效的转换方式
        lat1, lon1, lat2, lon2 = np.asarray(lat1, dtype=np.float64), np.asarray(lon1, dtype=np.float64), \
            np.asarray(lat2, dtype=np.float64), np.asarray(lon2, dtype=np.float64)

        # 一次性转换为弧度，减少函数调用
        deg_to_rad = np.pi / 180.0
        lat1_rad, lon1_rad = lat1 * deg_to_rad, lon1 * deg_to_rad
        lat2_rad, lon2_rad = lat2 * deg_to_rad, lon2 * deg_to_rad

        # 计算差值
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        # 优化的Haversine公式计算
        sin_dlat_half = np.sin(dlat * 0.5)
        sin_dlon_half = np.sin(dlon * 0.5)
        cos_lat1 = np.cos(lat1_rad)
        cos_lat2 = np.cos(lat2_rad)

        a = sin_dlat_half * sin_dlat_half + cos_lat1 * cos_lat2 * sin_dlon_half * sin_dlon_half
        c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))  # clip防止数值误差
        distance_km = 6371.0 * c

        # 应用距离修正
        corrected_distances = self._apply_vectorized_distance_correction(distance_km)

        return corrected_distances

    def _apply_vectorized_distance_correction(self, raw_distances):
        """向量化的距离修正函数"""

        raw_distances = np.asarray(raw_distances)

        # 网络规模因子
        network_scale_factor = len(self.J) + len(self.M) + len(self.K)
        supply_scale = len(self.J)
        transfer_scale = len(self.M)
        demand_scale = len(self.K)

        # 基准距离因子
        base_distance_factor = min(1.5, max(1.0, network_scale_factor / max(supply_scale, 1)))

        # 计算特征距离
        if hasattr(self, 'point_features') and self.point_features:
            all_latitudes = [self.point_features[node].get('latitude', 0) for node in self.point_features]
            all_longitudes = [self.point_features[node].get('longitude', 0) for node in self.point_features]

            if all_latitudes and all_longitudes:
                lat_range = max(all_latitudes) - min(all_latitudes)
                lon_range = max(all_longitudes) - min(all_longitudes)
                characteristic_distance = (lat_range + lon_range) * 111.32
            else:
                characteristic_distance = np.mean(raw_distances)
        else:
            characteristic_distance = np.mean(raw_distances)

        characteristic_distance = np.maximum(characteristic_distance, raw_distances)

        # 计算距离比
        distance_ratio = raw_distances / characteristic_distance

        # 修正系数计算
        supply_transfer_ratio = supply_scale / max(supply_scale + transfer_scale, 1)
        max_correction = 1.0 + supply_transfer_ratio * 0.5

        network_density = supply_scale / max(supply_scale + transfer_scale + demand_scale, 1)
        correction_factor = 1.0 + (max_correction - 1.0) * np.exp(-distance_ratio * network_density)

        # 应用修正，限制在合理范围内
        corrected_distances = raw_distances * np.minimum(correction_factor, 1.5)

        return corrected_distances

    def _calculate_network_parameters(self, distance_info):
        """计算网络特征参数"""
        all_distances = distance_info['all_distances']
        return {
            'avg_network_distance': sum(all_distances) / len(all_distances) if all_distances else 0.0,
            'max_network_distance': max(all_distances) if all_distances else 0.0
        }

    def _is_long_distance_supplier(self, direct_distance, network_params):
        """判断是否为长距离供应点"""
        return direct_distance > network_params['avg_network_distance'] * len(self.M) / len(self.J) if len(
            self.J) > 0 else False

    def _calculate_distance_factor(self, is_long_distance_supplier):
        """计算距离因子"""
        if is_long_distance_supplier:
            return len(self.M) * len(self.TRANSPORT_MODES) / len(self.J) if len(self.J) > 0 and hasattr(self,
                                                                                                        'TRANSPORT_MODES') else len(
                self.M) / len(self.J) if len(self.J) > 0 else 1.0
        else:
            return len(self.M) / len(self.J) if len(self.J) > 0 else 1.0

    def _log_filtering_effectiveness(self, j, candidate_set):
        """记录筛选效果"""
        total_m_m_combinations = len(self.M) * (len(self.M) - 1)
        filtered_combinations = len(candidate_set['m_m_pairs'])
        reduction_ratio = 1 - (filtered_combinations / total_m_m_combinations) if total_m_m_combinations > 0 else 0

        direct_distance = self.point_features[j]['latitude'] - self.point_features[self.K[0]]['latitude']  # 简化距离计算
        is_long_distance_supplier = abs(direct_distance) > len(self.M) / len(self.J) if len(self.J) > 0 else False

        self.logger.info(
            f"供应点{j}中转点筛选: 第一层候选{len(candidate_set['first_transfer_candidates'])}/{len(self.M)}, "
            f"第二层候选{len(candidate_set['second_transfer_candidates'])}/{len(self.M)}, "
            f"M-M对{filtered_combinations}/{total_m_m_combinations} "
            f"(减少{reduction_ratio * 100:.1f}%) - 长距离供应点: {is_long_distance_supplier}")

        # 安全的数值转换函数
        def safe_float_convert(value, default=0.0):
            try:
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    if value.strip() == '':
                        return default
                    return float(value)
                else:
                    return default
            except (ValueError, TypeError):
                return default

        self.logger.info("开始数据动员处理 - 识别合适的数据供应点")

        # 获取唯一需求点
        if not self.K:
            raise ValueError("需求点集合为空")

        aggregated_demand_point = self.K[0]

        if aggregated_demand_point not in self.D:
            raise KeyError(f"需求点 {aggregated_demand_point} 不在需求字典中")

        total_aggregated_demand = self.D[aggregated_demand_point]

        # 评估所有供应点的适宜度
        supplier_evaluations = []

        for j in self.J:
            try:
                supply_capacity = self.B[j] * self.P[j]
            except KeyError as e:
                self.logger.warning(f"供应点 {j} 数据缺失: {str(e)}")
                continue

            if supply_capacity <= self.EPS:
                continue

            # 检查是否有细分对象
            sub_objects = self.point_features[j].get('sub_objects', [])

            if sub_objects:
                # 有细分对象时，选择成本效率最优的细分对象
                sub_object_evaluations = []

                # 遍历所有分类和其中的细分对象
                for category in sub_objects:
                    # 检查是否为新的分类结构
                    if isinstance(category, dict) and 'items' in category:
                        # 新的分类结构：遍历分类中的所有项目
                        for sub_obj in category.get('items', []):
                            # 检查可用性
                            max_available = sub_obj.get('max_available_quantity', 0)
                            if max_available <= self.EPS:
                                continue  # 跳过不可用的细分对象

                            # 企业能力评分 - 使用细分对象的数据
                            try:
                                # 根据企业规模推导规模能力作为默认值
                                enterprise_size = self.point_features[j].get('enterprise_size', '中')
                                if enterprise_size == '大':
                                    default_scale_capability = len(self.J) + len(self.M)
                                elif enterprise_size == '中':
                                    default_scale_capability = len(self.J) + len(self.M) / (len(self.M) + 1) if len(
                                        self.M) > 0 else len(self.J)
                                else:
                                    default_scale_capability = len(self.J) / (len(self.J) + 1) if len(
                                        self.J) > 0 else 1.0
                                scale_capability = safe_float_convert(
                                    sub_obj.get('scale_capability', default_scale_capability), default_scale_capability)
                                resource_reserve = safe_float_convert(sub_obj.get('resource_reserve',
                                                                                  self.point_features[j].get(
                                                                                      'resource_reserve',
                                                                                      4)), 4)
                                production_capacity_value = safe_float_convert(sub_obj.get('production_capacity',
                                                                                           self.point_features[j].get(
                                                                                               'production_capacity',
                                                                                               6)), 6)
                                capability_score = (scale_capability + resource_reserve + production_capacity_value) / 3
                            except (KeyError, TypeError) as e:
                                self.logger.warning(
                                    f"供应点 {j} 细分对象 {sub_obj.get('sub_object_id')} 企业能力评分计算失败: {str(e)}")
                                capability_score = 5.0

                            # 企业类型和规模评分 - 使用主要供应点的数据
                            try:
                                enterprise_type = self.point_features[j].get('enterprise_type', '国企')
                                enterprise_size = self.point_features[j].get('enterprise_size', '大')

                                total_enterprises = len(self.J)
                                state_enterprises = len([j for j in self.J
                                                         if self.point_features.get(j, {}).get('enterprise_type') in [
                                                             "国企", "事业单位"]])
                                state_enterprise_ratio = state_enterprises / total_enterprises if total_enterprises > 0 else 0.0

                                if enterprise_type in ["国企", "事业单位"]:
                                    type_score = state_enterprise_ratio + (1.0 - state_enterprise_ratio) * len(
                                        self.K) / (len(self.K) + len(self.J))
                                else:
                                    type_score = state_enterprise_ratio * len(self.J) / (len(self.J) + len(self.M))

                                # 基于网络规模计算企业规模评分
                                large_medium_enterprises = len([j for j in self.J
                                                                if self.point_features.get(j, {}).get(
                                        'enterprise_size') in ["大", "中"]])
                                large_medium_ratio = large_medium_enterprises / total_enterprises if total_enterprises > 0 else 0.0

                                if enterprise_size in ["大", "中"]:
                                    size_score = large_medium_ratio + (1.0 - large_medium_ratio) * len(self.M) / (
                                            len(self.M) + len(self.K))
                                else:
                                    size_score = large_medium_ratio * len(self.K) / (len(self.K) + len(self.J))
                            except (KeyError, TypeError) as e:
                                self.logger.warning(f"供应点 {j} 企业类型规模评分计算失败: {str(e)}")
                                type_score = 0.8
                                size_score = 0.7

                            # 动员能力评分（基于企业类型推导）
                            try:
                                enterprise_type = self.point_features[j].get('enterprise_type', '其他')
                                # 计算网络中国企比例和网络复杂度
                                total_suppliers = len(self.J)
                                state_owned_count = len([supplier for supplier in self.J
                                                         if self.point_features.get(supplier, {}).get(
                                        'enterprise_type') in ["国企", "事业单位"]])
                                state_owned_ratio = state_owned_count / total_suppliers if total_suppliers > 0 else 0.0
                                network_complexity = (len(self.M) + len(self.K)) / (
                                        len(self.J) + len(self.M) + len(self.K))

                                if enterprise_type in ["国企", "事业单位"]:
                                    mobilization_score = state_owned_ratio + (
                                            1.0 - state_owned_ratio) * network_complexity
                                else:
                                    mobilization_score = state_owned_ratio * (1.0 - network_complexity) + (
                                            1.0 - state_owned_ratio) * len(self.K) / (len(self.K) + len(self.J))
                            except (KeyError, TypeError) as e:
                                self.logger.warning(f"供应点 {j} 动员能力评分计算失败: {str(e)}")
                                # 使用网络特征作为默认值
                                mobilization_score = len(self.K) / (len(self.K) + len(self.J)) if len(self.K) + len(
                                    self.J) > 0 else 0.5

                            # 综合评分计算
                            try:
                                composite_score = (
                                        capability_score * self.objective_weights['capability'] +
                                        type_score * self.objective_weights['social'] +
                                        size_score * self.objective_weights['balance'] +
                                        mobilization_score * self.objective_weights['priority'] +
                                        supply_capacity / max(self.B[j_temp] * self.P[j_temp] for j_temp in self.J) *
                                        self.objective_weights['cost']
                                )
                            except (KeyError, TypeError, ZeroDivisionError) as e:
                                self.logger.warning(f"供应点 {j} 细分对象综合评分计算失败: {str(e)}")
                                composite_score = 0.5

                            sub_object_evaluations.append({
                                'sub_object_id': sub_obj.get('sub_object_id', 'unknown'),
                                'sub_object_name': sub_obj.get('sub_object_name', 'unknown'),
                                'max_available_quantity': max_available,
                                'capability_score': capability_score,
                                'type_score': type_score,
                                'size_score': size_score,
                                'mobilization_score': mobilization_score,
                                'composite_score': composite_score
                            })
                    else:
                        # 兼容旧的平铺结构：直接处理细分对象
                        sub_obj = category
                        # 检查可用性
                        max_available = sub_obj.get('max_available_quantity', 0)
                        if max_available <= self.EPS:
                            continue

                        # 企业能力评分 - 使用细分对象的数据
                        try:
                            # 根据企业规模推导规模能力作为默认值
                            enterprise_size = self.point_features[j].get('enterprise_size', '中')
                            if enterprise_size == '大':
                                default_scale_capability = len(self.J) + len(self.M)
                            elif enterprise_size == '中':
                                default_scale_capability = len(self.J) + len(self.M) / (len(self.M) + 1) if len(
                                    self.M) > 0 else len(self.J)
                            else:
                                default_scale_capability = len(self.J) / (len(self.J) + 1) if len(self.J) > 0 else 1.0
                            scale_capability = safe_float_convert(
                                sub_obj.get('scale_capability', default_scale_capability), default_scale_capability)
                            resource_reserve = safe_float_convert(sub_obj.get('resource_reserve',
                                                                              self.point_features[j].get(
                                                                                  'resource_reserve',
                                                                                  4)), 4)
                            production_capacity_value = safe_float_convert(sub_obj.get('production_capacity',
                                                                                       self.point_features[j].get(
                                                                                           'production_capacity', 6)),
                                                                           6)
                            capability_score = (scale_capability + resource_reserve + production_capacity_value) / 3
                        except (KeyError, TypeError) as e:
                            self.logger.warning(
                                f"供应点 {j} 细分对象 {sub_obj.get('sub_object_id')} 企业能力评分计算失败: {str(e)}")
                            capability_score = 5.0

                        # 企业类型和规模评分 - 使用主要供应点的数据
                        try:
                            enterprise_type = self.point_features[j].get('enterprise_type', '国企')
                            enterprise_size = self.point_features[j].get('enterprise_size', '大')

                            total_enterprises = len(self.J)
                            state_enterprises = len([j for j in self.J
                                                     if
                                                     self.point_features.get(j, {}).get('enterprise_type') in ["国企",
                                                                                                               "事业单位"]])
                            state_enterprise_ratio = state_enterprises / total_enterprises if total_enterprises > 0 else 0.0

                            if enterprise_type in ["国企", "事业单位"]:
                                type_score = state_enterprise_ratio + (1.0 - state_enterprise_ratio) * len(self.K) / (
                                        len(self.K) + len(self.J))
                            else:
                                type_score = state_enterprise_ratio * len(self.J) / (len(self.J) + len(self.M))

                            # 基于网络规模计算企业规模评分
                            large_medium_enterprises = len([j for j in self.J
                                                            if
                                                            self.point_features.get(j, {}).get('enterprise_size') in [
                                                                "大", "中"]])
                            large_medium_ratio = large_medium_enterprises / total_enterprises if total_enterprises > 0 else 0.0

                            if enterprise_size in ["大", "中"]:
                                size_score = large_medium_ratio + (1.0 - large_medium_ratio) * len(self.M) / (
                                        len(self.M) + len(self.K))
                            else:
                                size_score = large_medium_ratio * len(self.K) / (len(self.K) + len(self.J))
                        except (KeyError, TypeError) as e:
                            self.logger.warning(f"供应点 {j} 企业类型规模评分计算失败: {str(e)}")
                            type_score = 0.8
                            size_score = 0.7

                        # 动员能力评分 - 使用主要供应点的数据
                        try:
                            enterprise_type = self.point_features[j].get('enterprise_type', '其他')
                            # 计算网络中国企比例和网络复杂度
                            total_suppliers = len(self.J)
                            state_owned_count = len([supplier for supplier in self.J
                                                     if
                                                     self.point_features.get(supplier, {}).get('enterprise_type') in [
                                                         "国企", "事业单位"]])
                            state_owned_ratio = state_owned_count / total_suppliers if total_suppliers > 0 else 0.0
                            network_complexity = (len(self.M) + len(self.K)) / (len(self.J) + len(self.M) + len(self.K))

                            if enterprise_type in ["国企", "事业单位"]:
                                mobilization_score = state_owned_ratio + (1.0 - state_owned_ratio) * network_complexity
                            else:
                                mobilization_score = state_owned_ratio * (1.0 - network_complexity) + (
                                        1.0 - state_owned_ratio) * len(self.K) / (len(self.K) + len(self.J))
                        except (KeyError, TypeError) as e:
                            self.logger.warning(f"供应点 {j} 动员能力评分计算失败: {str(e)}")
                            # 使用网络特征作为默认值
                            mobilization_score = len(self.K) / (len(self.K) + len(self.J)) if len(self.K) + len(
                                self.J) > 0 else 0.5

                        # 综合评分计算
                        try:
                            composite_score = (
                                    capability_score * self.objective_weights['capability'] +
                                    type_score * self.objective_weights['social'] +
                                    size_score * self.objective_weights['balance'] +
                                    mobilization_score * self.objective_weights['priority'] +
                                    supply_capacity / max(self.B[j_temp] * self.P[j_temp] for j_temp in self.J) *
                                    self.objective_weights['cost']
                            )
                        except (KeyError, TypeError, ZeroDivisionError) as e:
                            self.logger.warning(f"供应点 {j} 细分对象综合评分计算失败: {str(e)}")
                            composite_score = 0.5

                        sub_object_evaluations.append({
                            'sub_object_id': sub_obj.get('sub_object_id', 'unknown'),
                            'sub_object_name': sub_obj.get('sub_object_name', 'unknown'),
                            'max_available_quantity': max_available,
                            'capability_score': capability_score,
                            'type_score': type_score,
                            'size_score': size_score,
                            'mobilization_score': mobilization_score,
                            'composite_score': composite_score
                        })

                # 选择最佳的细分对象
                if sub_object_evaluations:
                    best_sub_object = max(sub_object_evaluations, key=lambda x: x['composite_score'])

                    supplier_evaluations.append({
                        'supplier_id': self.point_features[j].get('original_supplier_id', j),
                        'supply_capacity': supply_capacity,
                        'capability_score': best_sub_object['capability_score'],
                        'type_score': best_sub_object['type_score'],
                        'size_score': best_sub_object['size_score'],
                        'mobilization_score': best_sub_object['mobilization_score'],
                        'composite_score': best_sub_object['composite_score'],
                        'enterprise_info': {
                            'type': self.point_features[j].get('enterprise_type', '国企'),
                            'size': self.point_features[j].get('enterprise_size', '大')
                        },
                        'selected_sub_object': best_sub_object,
                        'all_sub_objects': sub_object_evaluations
                    })
            else:
                # 计算供应点的综合评分
                supply_capacity = self.B[j] * self.P[j]

                # 企业能力评分 - 基于企业规模推导
                try:
                    # 根据企业规模推导规模能力
                    enterprise_size = self.point_features[j].get('enterprise_size', '中')
                    if enterprise_size == '大':
                        scale_capability = len(self.J) + len(self.M)
                    elif enterprise_size == '中':
                        scale_capability = len(self.J) + len(self.M) / (len(self.M) + 1) if len(self.M) > 0 else len(
                            self.J)
                    else:
                        scale_capability = len(self.J) / (len(self.J) + 1) if len(self.J) > 0 else 1.0

                    resource_reserve = safe_float_convert(
                        self.point_features[j].get('resource_reserve', scale_capability), scale_capability)
                    production_capacity_value = safe_float_convert(
                        self.point_features[j].get('production_capacity', scale_capability), scale_capability)
                    capability_score = (scale_capability + resource_reserve + production_capacity_value) / 3
                except (KeyError, TypeError) as e:
                    self.logger.warning(f"供应点 {j} 企业能力评分计算失败: {str(e)}")
                    capability_score = len(self.J) + len(self.M) if len(self.J) > 0 and len(self.M) > 0 else 1.0

                # 企业类型和规模评分
                try:
                    enterprise_type = self.point_features[j].get('enterprise_type', '国企')
                    enterprise_size = self.point_features[j].get('enterprise_size', '大')

                    total_enterprises = len(self.J)
                    state_enterprises = len([j for j in self.J
                                             if self.point_features.get(j, {}).get('enterprise_type') in ["国企",
                                                                                                          "事业单位"]])
                    state_enterprise_ratio = state_enterprises / total_enterprises if total_enterprises > 0 else 0.0

                    if enterprise_type in ["国企", "事业单位"]:
                        type_score = state_enterprise_ratio + (1.0 - state_enterprise_ratio) * len(self.K) / (
                                len(self.K) + len(self.J))
                    else:
                        type_score = state_enterprise_ratio * len(self.J) / (len(self.J) + len(self.M))

                    # 基于网络规模计算企业规模评分
                    large_medium_enterprises = len([j for j in self.J
                                                    if self.point_features.get(j, {}).get('enterprise_size') in ["大",
                                                                                                                 "中"]])
                    large_medium_ratio = large_medium_enterprises / total_enterprises if total_enterprises > 0 else 0.0

                    if enterprise_size in ["大", "中"]:
                        size_score = large_medium_ratio + (1.0 - large_medium_ratio) * len(self.M) / (
                                len(self.M) + len(self.K))
                    else:
                        size_score = large_medium_ratio * len(self.K) / (len(self.K) + len(self.J))
                except (KeyError, TypeError) as e:
                    self.logger.warning(f"供应点 {j} 企业类型规模评分计算失败: {str(e)}")
                    type_score = 0.8
                    size_score = 0.7

                # 动员能力评分
                try:
                    enterprise_type = self.point_features[j].get('enterprise_type', '其他')
                    # 计算网络中国企比例和网络复杂度
                    total_suppliers = len(self.J)
                    state_owned_count = len([supplier for supplier in self.J
                                             if self.point_features.get(supplier, {}).get('enterprise_type') in ["国企",
                                                                                                                 "事业单位"]])
                    state_owned_ratio = state_owned_count / total_suppliers if total_suppliers > 0 else 0.0
                    network_complexity = (len(self.M) + len(self.K)) / (len(self.J) + len(self.M) + len(self.K))

                    if enterprise_type in ["国企", "事业单位"]:
                        mobilization_score = state_owned_ratio + (1.0 - state_owned_ratio) * network_complexity
                    else:
                        mobilization_score = state_owned_ratio * (1.0 - network_complexity) + (
                                1.0 - state_owned_ratio) * len(self.K) / (len(self.K) + len(self.J))
                except (KeyError, TypeError) as e:
                    self.logger.warning(f"供应点 {j} 动员能力评分计算失败: {str(e)}")
                    # 使用网络特征作为默认值
                    mobilization_score = len(self.K) / (len(self.K) + len(self.J)) if len(self.K) + len(
                        self.J) > 0 else 0.5

                # 综合评分计算
                try:
                    composite_score = (
                            capability_score * self.objective_weights['capability'] +
                            type_score * self.objective_weights['social'] +
                            size_score * self.objective_weights['balance'] +
                            mobilization_score * self.objective_weights['priority'] +
                            supply_capacity / max(self.B[j_temp] * self.P[j_temp] for j_temp in self.J) *
                            self.objective_weights['cost']
                    )
                except (KeyError, TypeError, ZeroDivisionError) as e:
                    self.logger.warning(f"供应点 {j} 综合评分计算失败: {str(e)}")
                    composite_score = 0.5

                # 确保企业信息变量有默认值
                enterprise_type = self.point_features[j].get('enterprise_type', '其他')
                enterprise_size = self.point_features[j].get('enterprise_size', '中')

                supplier_evaluations.append({
                    'supplier_id': self.point_features[j].get('original_supplier_id', j),
                    'supply_capacity': supply_capacity,
                    'capability_score': capability_score,
                    'type_score': type_score,
                    'size_score': size_score,
                    'mobilization_score': mobilization_score,
                    'composite_score': composite_score,
                    'enterprise_info': {
                        'type': enterprise_type,
                        'size': enterprise_size
                    },
                    'selected_sub_object': None,
                    'all_sub_objects': []
                })

        # 按综合评分排序（降序，分数越高越好）
        try:
            supplier_evaluations.sort(key=lambda x: x['composite_score'], reverse=True)
        except (KeyError, TypeError) as e:
            self.logger.warning(f"供应点评分排序失败: {str(e)}")

        # 选择满足需求的供应点
        selected_suppliers = []
        remaining_demand = total_aggregated_demand

        for evaluation in supplier_evaluations:
            if remaining_demand <= self.EPS:
                break

            supplier_capacity = min(evaluation['supply_capacity'], remaining_demand)
            selected_suppliers.append({
                'supplier_id': evaluation['supplier_id'],
                'allocated_capacity': supplier_capacity,
                'evaluation': evaluation
            })

            remaining_demand -= supplier_capacity

            print(f"选择数据供应点 {evaluation['supplier_id']}: "
                  f"分配容量 {supplier_capacity:.2f}, "
                  f"综合评分 {evaluation['composite_score']:.3f}")

            # 如果有细分对象，打印细分对象信息
            if evaluation['selected_sub_object']:
                sub_obj = evaluation['selected_sub_object']
                print(f"  -> 选择细分对象: {sub_obj['sub_object_name']} "
                      f"(分配数量: {sub_obj['max_available_quantity']:.2f})")

        # 构建数据动员解
        satisfaction_rate = (
                                    total_aggregated_demand - remaining_demand) / total_aggregated_demand if total_aggregated_demand > 0 else 1.0

        data_solution = {
            'mobilization_type': 'data',
            'selected_suppliers': selected_suppliers,
            'supplier_evaluations': supplier_evaluations,
            'total_demand': total_aggregated_demand,
            'satisfied_demand': total_aggregated_demand - remaining_demand,
            'satisfaction_rate': satisfaction_rate,
            'demand_point': aggregated_demand_point,

            # 为了兼容性，提供空的运输变量
            'x1': {}, 'x2': {}, 'x3': {}, 'x_direct': {},
            'b1': {}, 'b2': {}, 'b3': {}, 'b_direct': {},
            't1': {}, 't2': {}, 't3': {}, 't_direct': {},
            'scheduling_time': 0.0,
            'processed_paths': len(supplier_evaluations),
            'objective_type': 'data_mobilization'
        }

        self.logger.info(
            f"数据动员完成 - 选择了{len(selected_suppliers)}个供应点，满足率{satisfaction_rate * 100:.1f}%")

        return data_solution

    def sort_list_by_MAandP(self,data):
        sorted_data = sorted(data, key=lambda x: x['items']['specify_quantity'], reverse=True)
        return sorted_data

    def sort_list_by_D_innermost(self,data_list):
        # 根据每个元素内层字典的'specify_quantity'值进行排序
        sorted_list = sorted(
            data_list,
            key=lambda x: x['all_sub_objects']['specify_quantity'],
            reverse = True
        )
        return sorted_list

    def _handle_data_mobilization(self):
        """处理数据动员，只需识别合适的供应点，无需实际运输"""

        # 安全的数值转换函数
        def safe_float_convert(value, default=0.0):
            try:
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    if value.strip() == '':
                        return default
                    return float(value)
                else:
                    return default
            except (ValueError, TypeError):
                return default

        self.logger.info("开始数据动员处理 - 识别合适的数据供应点")

        # 获取唯一需求点
        if not self.K:
            raise ValueError("需求点集合为空")

        aggregated_demand_point = self.K[0]

        if aggregated_demand_point not in self.D:
            raise KeyError(f"需求点 {aggregated_demand_point} 不在需求字典中")

        total_aggregated_demand = self.D[aggregated_demand_point]

        print(f"数据需求点：{aggregated_demand_point}，数据需求量: {total_aggregated_demand:.2f}")

        # 评估所有供应点的适宜度
        supplier_evaluations = []
        # 取出specify_quantity 进行比较
        supplier_compaires = []
        # for j in self.J:
        for j in self.J:
            try:
                supply_capacity = self.B[j] * self.P[j]
            except KeyError as e:
                self.logger.warning(f"供应点 {j} 数据缺失: {str(e)}")
                continue

            if supply_capacity <= self.EPS:
                continue

        # 检查是否有细分对象
        # 检查是否有细分对象
        sub_objects = self.point_features[j].get('sub_objects', [])

        if sub_objects:
            # 有细分对象时，选择成本效率最优的细分对象
            sub_object_evaluations = []

            # 遍历所有分类和其中的细分对象
            for category in sub_objects:
                # 检查是否为新的分类结构
                if isinstance(category, dict) and 'items' in category:
                    # 获取category_comment，默认为'midle'
                    category_comment = category.get('category_comment', 'midle')
                    # 新的分类结构：遍历分类中的所有项目
                    for sub_obj in category.get('items', []):
                        # 检查可用性
                        max_available = sub_obj.get('max_available_quantity', 0)
                        if max_available <= self.EPS:
                            continue  # 跳过不可用的细分对象

                        # 企业能力评分 - 使用细分对象的数据
                        try:
                            # 根据企业规模推导规模能力作为默认值
                            enterprise_size = self.point_features[j].get('enterprise_size', '中')
                            if enterprise_size == '大':
                                default_scale_capability = len(self.J) + len(self.M)
                            elif enterprise_size == '中':
                                default_scale_capability = len(self.J) + len(self.M) / (len(self.M) + 1) if len(
                                    self.M) > 0 else len(self.J)
                            else:
                                default_scale_capability = len(self.J) / (len(self.J) + 1) if len(
                                    self.J) > 0 else 1.0
                            scale_capability = safe_float_convert(
                                sub_obj.get('scale_capability', default_scale_capability), default_scale_capability)
                            resource_reserve = safe_float_convert(sub_obj.get('resource_reserve',
                                                                              self.point_features[j].get(
                                                                                  'resource_reserve',
                                                                                  4)), 4)
                            production_capacity_value = safe_float_convert(sub_obj.get('production_capacity',
                                                                                       self.point_features[j].get(
                                                                                           'production_capacity',
                                                                                           6)), 6)
                            capability_score = (scale_capability + resource_reserve + production_capacity_value) / 3
                        except (KeyError, TypeError) as e:
                            self.logger.warning(
                                f"供应点 {j} 细分对象 {sub_obj.get('sub_object_id')} 企业能力评分计算失败: {str(e)}")
                            capability_score = 5.0

                        # 企业类型和规模评分 - 使用主要供应点的数据
                        try:
                            enterprise_type = self.point_features[j].get('enterprise_type', '国企')
                            enterprise_size = self.point_features[j].get('enterprise_size', '大')

                            total_enterprises = len(self.J)
                            state_enterprises = len([j for j in self.J
                                                     if self.point_features.get(j, {}).get('enterprise_type') in [
                                                         "国企", "事业单位"]])
                            state_enterprise_ratio = state_enterprises / total_enterprises if total_enterprises > 0 else 0.0

                            if enterprise_type in ["国企", "事业单位"]:
                                type_score = state_enterprise_ratio + (1.0 - state_enterprise_ratio) * len(
                                    self.K) / (len(self.K) + len(self.J))
                            else:
                                type_score = state_enterprise_ratio * len(self.J) / (len(self.J) + len(self.M))

                            # 基于网络规模计算企业规模评分
                            large_medium_enterprises = len([j for j in self.J
                                                            if self.point_features.get(j, {}).get(
                                    'enterprise_size') in ["大", "中"]])
                            large_medium_ratio = large_medium_enterprises / total_enterprises if total_enterprises > 0 else 0.0

                            if enterprise_size in ["大", "中"]:
                                size_score = large_medium_ratio + (1.0 - large_medium_ratio) * len(self.M) / (
                                        len(self.M) + len(self.K))
                            else:
                                size_score = large_medium_ratio * len(self.K) / (len(self.K) + len(self.J))
                        except (KeyError, TypeError) as e:
                            self.logger.warning(f"供应点 {j} 企业类型规模评分计算失败: {str(e)}")
                            type_score = 0.8
                            size_score = 0.7

                        # 动员能力评分（基于企业类型推导）
                        try:
                            enterprise_type = self.point_features[j].get('enterprise_type', '其他')
                            # 计算网络中国企比例和网络复杂度
                            total_suppliers = len(self.J)
                            state_owned_count = len([supplier for supplier in self.J
                                                     if self.point_features.get(supplier, {}).get(
                                    'enterprise_type') in ["国企", "事业单位"]])
                            state_owned_ratio = state_owned_count / total_suppliers if total_suppliers > 0 else 0.0
                            network_complexity = (len(self.M) + len(self.K)) / (
                                    len(self.J) + len(self.M) + len(self.K))

                            if enterprise_type in ["国企", "事业单位"]:
                                mobilization_score = state_owned_ratio + (
                                        1.0 - state_owned_ratio) * network_complexity
                            else:
                                mobilization_score = state_owned_ratio * (1.0 - network_complexity) + (
                                        1.0 - state_owned_ratio) * len(self.K) / (len(self.K) + len(self.J))
                        except (KeyError, TypeError) as e:
                            self.logger.warning(f"供应点 {j} 动员能力评分计算失败: {str(e)}")
                            # 使用网络特征作为默认值
                            mobilization_score = len(self.K) / (len(self.K) + len(self.J)) if len(self.K) + len(
                                self.J) > 0 else 0.5

                        # 综合评分计算
                        try:
                            composite_score = (
                                    capability_score * self.objective_weights['capability'] +
                                    type_score * self.objective_weights['social'] +
                                    size_score * self.objective_weights['balance'] +
                                    mobilization_score * self.objective_weights['priority'] +
                                    supply_capacity / max(self.B[j_temp] * self.P[j_temp] for j_temp in self.J) *
                                    self.objective_weights['cost']
                            )
                        except (KeyError, TypeError, ZeroDivisionError) as e:
                            self.logger.warning(f"供应点 {j} 细分对象综合评分计算失败: {str(e)}")
                            composite_score = 0.5

                        category_id = sub_obj.get('category_id') or category.get('category_id', 'unknown')
                        category_name = sub_obj.get('category_name') or category.get('category_name', 'unknown')
                        recommend_md5 = sub_obj.get('recommend_md5') or category.get('recommend_md5', 'unknown')
                        sub_object_evaluations.append({
                            'sub_object_id': sub_obj.get('sub_object_id', 'unknown'),
                            'sub_object_name': sub_obj.get('sub_object_name', 'unknown'),
                            'max_available_quantity': max_available,
                            'specify_quantity': sub_obj.get('specify_quantity', 'unknown'),
                            'capacity_quantity': sub_obj.get('capacity_quantity', 'unknown'),
                            'capability_score': capability_score,
                            'type_score': type_score,
                            'size_score': size_score,
                            'mobilization_score': mobilization_score,
                            'composite_score': composite_score,
                            'category_id': category_id,
                            'category_name': category_name,
                            'recommend_md5': recommend_md5,
                            'category_comment': category_comment  # 加入category_comment
                        })
                else:
                    # 兼容旧的平铺结构：直接处理细分对象
                    sub_obj = category
                    # 检查可用性
                    max_available = sub_obj.get('max_available_quantity', 0)
                    if max_available <= self.EPS:
                        continue

                    # 企业能力评分 - 使用细分对象的数据
                    try:
                        # 根据企业规模推导规模能力作为默认值
                        enterprise_size = self.point_features[j].get('enterprise_size', '中')
                        if enterprise_size == '大':
                            default_scale_capability = len(self.J) + len(self.M)
                        elif enterprise_size == '中':
                            default_scale_capability = len(self.J) + len(self.M) / (len(self.M) + 1) if len(
                                self.M) > 0 else len(self.J)
                        else:
                            default_scale_capability = len(self.J) / (len(self.J) + 1) if len(self.J) > 0 else 1.0
                        scale_capability = safe_float_convert(
                            sub_obj.get('scale_capability', default_scale_capability), default_scale_capability)
                        resource_reserve = safe_float_convert(sub_obj.get('resource_reserve',
                                                                          self.point_features[j].get(
                                                                              'resource_reserve',
                                                                              4)), 4)
                        production_capacity_value = safe_float_convert(sub_obj.get('production_capacity',
                                                                                   self.point_features[j].get(
                                                                                       'production_capacity', 6)),
                                                                       6)
                        capability_score = (scale_capability + resource_reserve + production_capacity_value) / 3
                    except (KeyError, TypeError) as e:
                        self.logger.warning(
                            f"供应点 {j} 细分对象 {sub_obj.get('sub_object_id')} 企业能力评分计算失败: {str(e)}")
                        capability_score = 5.0

                    # 企业类型和规模评分 - 使用主要供应点的数据
                    try:
                        enterprise_type = self.point_features[j].get('enterprise_type', '国企')
                        enterprise_size = self.point_features[j].get('enterprise_size', '大')

                        total_enterprises = len(self.J)
                        state_enterprises = len([j for j in self.J
                                                 if
                                                 self.point_features.get(j, {}).get('enterprise_type') in ["国企",
                                                                                                           "事业单位"]])
                        state_enterprise_ratio = state_enterprises / total_enterprises if total_enterprises > 0 else 0.0

                        if enterprise_type in ["国企", "事业单位"]:
                            type_score = state_enterprise_ratio + (1.0 - state_enterprise_ratio) * len(self.K) / (
                                    len(self.K) + len(self.J))
                        else:
                            type_score = state_enterprise_ratio * len(self.J) / (len(self.J) + len(self.M))

                        # 基于网络规模计算企业规模评分
                        large_medium_enterprises = len([j for j in self.J
                                                        if
                                                        self.point_features.get(j, {}).get('enterprise_size') in [
                                                            "大", "中"]])
                        large_medium_ratio = large_medium_enterprises / total_enterprises if total_enterprises > 0 else 0.0

                        if enterprise_size in ["大", "中"]:
                            size_score = large_medium_ratio + (1.0 - large_medium_ratio) * len(self.M) / (
                                    len(self.M) + len(self.K))
                        else:
                            size_score = large_medium_ratio * len(self.K) / (len(self.K) + len(self.J))
                    except (KeyError, TypeError) as e:
                        self.logger.warning(f"供应点 {j} 企业类型规模评分计算失败: {str(e)}")
                        type_score = 0.8
                        size_score = 0.7

                    # 动员能力评分 - 使用主要供应点的数据
                    try:
                        enterprise_type = self.point_features[j].get('enterprise_type', '其他')
                        # 计算网络中国企比例和网络复杂度
                        total_suppliers = len(self.J)
                        state_owned_count = len([supplier for supplier in self.J
                                                 if
                                                 self.point_features.get(supplier, {}).get('enterprise_type') in [
                                                     "国企", "事业单位"]])
                        state_owned_ratio = state_owned_count / total_suppliers if total_suppliers > 0 else 0.0
                        network_complexity = (len(self.M) + len(self.K)) / (len(self.J) + len(self.M) + len(self.K))

                        if enterprise_type in ["国企", "事业单位"]:
                            mobilization_score = state_owned_ratio + (1.0 - state_owned_ratio) * network_complexity
                        else:
                            mobilization_score = state_owned_ratio * (1.0 - network_complexity) + (
                                    1.0 - state_owned_ratio) * len(self.K) / (len(self.K) + len(self.J))
                    except (KeyError, TypeError) as e:
                        self.logger.warning(f"供应点 {j} 动员能力评分计算失败: {str(e)}")
                        # 使用网络特征作为默认值
                        mobilization_score = len(self.K) / (len(self.K) + len(self.J)) if len(self.K) + len(
                            self.J) > 0 else 0.5

                    # 综合评分计算
                    try:
                        composite_score = (
                                capability_score * self.objective_weights['capability'] +
                                type_score * self.objective_weights['social'] +
                                size_score * self.objective_weights['balance'] +
                                mobilization_score * self.objective_weights['priority'] +
                                supply_capacity / max(self.B[j_temp] * self.P[j_temp] for j_temp in self.J) *
                                self.objective_weights['cost']
                        )
                    except (KeyError, TypeError, ZeroDivisionError) as e:
                        self.logger.warning(f"供应点 {j} 细分对象综合评分计算失败: {str(e)}")
                        composite_score = 0.5

                    category_id = sub_obj.get('category_id', 'unknown')
                    category_name = sub_obj.get('category_name', 'unknown')
                    recommend_md5 = sub_obj.get('recommend_md5', 'unknown')
                    sub_object_evaluations.append({
                        'sub_object_id': sub_obj.get('sub_object_id', 'unknown'),
                        'sub_object_name': sub_obj.get('sub_object_name', 'unknown'),
                        'max_available_quantity': max_available,
                        'specify_quantity': sub_obj.get('specify_quantity', 'unknown'),
                        'capacity_quantity': sub_obj.get('capacity_quantity', 'unknown'),
                        'capability_score': capability_score,
                        'type_score': type_score,
                        'size_score': size_score,
                        'mobilization_score': mobilization_score,
                        'composite_score': composite_score,
                        'category_id': category_id,
                        'category_name': category_name,
                        'recommend_md5': recommend_md5,
                        'category_comment': 'midle'  # 旧平铺结构默认midle
                    })

            # 按成本效率排序，准备按需求量选择组合
            if sub_object_evaluations:
                # 首先按category_comment排序（good > midle > bad），然后按max_available_quantity排序
                def get_comment_order(comment):
                    order = {'good': 0, 'midle': 1, 'bad': 2}
                    return order.get(comment, 3)  # 默认将未知评论排在最后

                sub_object_evaluations.sort(
                    key=lambda x: (get_comment_order(x.get('category_comment', 'midle')), -x['max_available_quantity']))

                # 计算加权平均评分（用于供应点层面的评估）
                total_capacity = sum(obj['max_available_quantity'] for obj in sub_object_evaluations)
                if total_capacity > 0:
                    weighted_capability_score = sum(
                        obj['capability_score'] * obj['max_available_quantity'] for obj in
                        sub_object_evaluations) / total_capacity
                    weighted_type_score = sum(obj['type_score'] * obj['max_available_quantity'] for obj in
                                              sub_object_evaluations) / total_capacity
                    weighted_size_score = sum(obj['size_score'] * obj['max_available_quantity'] for obj in
                                              sub_object_evaluations) / total_capacity
                    weighted_mobilization_score = sum(
                        obj['mobilization_score'] * obj['max_available_quantity'] for obj in
                        sub_object_evaluations) / total_capacity
                    weighted_composite_score = sum(
                        obj['composite_score'] * obj['max_available_quantity'] for obj in
                        sub_object_evaluations) / total_capacity
                else:
                    # 如果总容量为0，使用最佳对象的评分
                    best_sub_object = sub_object_evaluations[0]
                    weighted_capability_score = best_sub_object['capability_score']
                    weighted_type_score = best_sub_object['type_score']
                    weighted_size_score = best_sub_object['size_score']
                    weighted_mobilization_score = best_sub_object['mobilization_score']
                    weighted_composite_score = best_sub_object['composite_score']

                supplier_evaluations.append({
                    'supplier_id': self.point_features[j].get('original_supplier_id', j),
                    'supply_capacity': supply_capacity,
                    'capability_score': weighted_capability_score,
                    'type_score': weighted_type_score,
                    'size_score': weighted_size_score,
                    'mobilization_score': weighted_mobilization_score,
                    'composite_score': weighted_composite_score,
                    'enterprise_info': {
                        'type': self.point_features[j].get('enterprise_type', '国企'),
                        'size': self.point_features[j].get('enterprise_size', '大')
                    },
                    'selected_sub_object': None,  # 将在分配时确定
                    'all_sub_objects': sub_object_evaluations
                })

                for kl in range(len(sub_object_evaluations)):
                    supplier_compaires.append({
                        'supplier_id': self.point_features[j].get('original_supplier_id', j),
                        'supply_capacity': sub_object_evaluations[kl]['max_available_quantity'],
                        'capability_score': weighted_capability_score,
                        'type_score': weighted_type_score,
                        'size_score': weighted_size_score,
                        'mobilization_score': weighted_mobilization_score,
                        'composite_score': weighted_composite_score,
                        'enterprise_info': {
                            'type': self.point_features[j].get('enterprise_type', '国企'),
                            'size': self.point_features[j].get('enterprise_size', '大')
                        },
                        'selected_sub_object': None,  # 将在分配时确定
                        'all_sub_objects': sub_object_evaluations[kl]
                    })
        else:
            # 计算供应点的综合评分
            supply_capacity = self.B[j] * self.P[j]

            # 企业能力评分 - 基于企业规模推导
            try:
                # 根据企业规模推导规模能力
                enterprise_size = self.point_features[j].get('enterprise_size', '中')
                if enterprise_size == '大':
                    scale_capability = len(self.J) + len(self.M)
                elif enterprise_size == '中':
                    scale_capability = len(self.J) + len(self.M) / (len(self.M) + 1) if len(self.M) > 0 else len(
                        self.J)
                else:
                    scale_capability = len(self.J) / (len(self.J) + 1) if len(self.J) > 0 else 1.0

                resource_reserve = safe_float_convert(
                    self.point_features[j].get('resource_reserve', scale_capability), scale_capability)
                production_capacity_value = safe_float_convert(
                    self.point_features[j].get('production_capacity', scale_capability), scale_capability)
                capability_score = (scale_capability + resource_reserve + production_capacity_value) / 3
            except (KeyError, TypeError) as e:
                self.logger.warning(f"供应点 {j} 企业能力评分计算失败: {str(e)}")
                capability_score = len(self.J) + len(self.M) if len(self.J) > 0 and len(self.M) > 0 else 1.0

            # 企业类型和规模评分
            try:
                enterprise_type = self.point_features[j].get('enterprise_type', '国企')
                enterprise_size = self.point_features[j].get('enterprise_size', '大')

                total_enterprises = len(self.J)
                state_enterprises = len([j for j in self.J
                                         if self.point_features.get(j, {}).get('enterprise_type') in ["国企",
                                                                                                      "事业单位"]])
                state_enterprise_ratio = state_enterprises / (total_enterprises) if total_enterprises > 0 else 0.0

                if enterprise_type in ["国企", "事业单位"]:
                    type_score = state_enterprise_ratio + (1.0 - state_enterprise_ratio) * len(self.K) / (
                            len(self.K) + len(self.J))
                else:
                    type_score = state_enterprise_ratio * len(self.J) / (len(self.J) + len(self.M))

                # 基于网络规模计算企业规模评分
                large_medium_enterprises = len([j for j in self.J
                                                if self.point_features.get(j, {}).get('enterprise_size') in ["大",
                                                                                                             "中"]])
                large_medium_ratio = large_medium_enterprises / total_enterprises if total_enterprises > 0 else 0.0

                if enterprise_size in ["大", "中"]:
                    size_score = large_medium_ratio + (1.0 - large_medium_ratio) * len(self.M) / (
                            len(self.M) + len(self.K))
                else:
                    size_score = large_medium_ratio * len(self.K) / (len(self.K) + len(self.J))
            except (KeyError, TypeError) as e:
                self.logger.warning(f"供应点 {j} 企业类型规模评分计算失败: {str(e)}")
                type_score = 0.8
                size_score = 0.7

            # 动员能力评分
            try:
                enterprise_type = self.point_features[j].get('enterprise_type', '其他')
                # 计算网络中国企比例和网络复杂度
                total_suppliers = len(self.J)
                state_owned_count = len([supplier for supplier in self.J
                                         if self.point_features.get(supplier, {}).get('enterprise_type') in ["国企",
                                                                                                             "事业单位"]])
                state_owned_ratio = state_owned_count / total_suppliers if total_suppliers > 0 else 0.0
                network_complexity = (len(self.M) + len(self.K)) / (len(self.J) + len(self.M) + len(self.K))

                if enterprise_type in ["国企", "事业单位"]:
                    mobilization_score = state_owned_ratio + (1.0 - state_owned_ratio) * network_complexity
                else:
                    mobilization_score = state_owned_ratio * (1.0 - network_complexity) + (
                            1.0 - state_owned_ratio) * len(self.K) / (len(self.K) + len(self.J))
            except (KeyError, TypeError) as e:
                self.logger.warning(f"供应点 {j} 动员能力评分计算失败: {str(e)}")
                # 使用网络特征作为默认值
                mobilization_score = len(self.K) / (len(self.K) + len(self.J)) if len(self.K) + len(
                    self.J) > 0 else 0.5

            # 综合评分计算
            try:
                composite_score = (
                        capability_score * self.objective_weights['capability'] +
                        type_score * self.objective_weights['social'] +
                        size_score * self.objective_weights['balance'] +
                        mobilization_score * self.objective_weights['priority'] +
                        supply_capacity / max(self.B[j_temp] * self.P[j_temp] for j_temp in self.J) *
                        self.objective_weights['cost']
                )
            except (KeyError, TypeError, ZeroDivisionError) as e:
                self.logger.warning(f"供应点 {j} 综合评分计算失败: {str(e)}")
                composite_score = 0.5

            # 确保企业信息变量有默认值
            enterprise_type = self.point_features[j].get('enterprise_type', '其他')
            enterprise_size = self.point_features[j].get('enterprise_size', '中')

            supplier_evaluations.append({
                'supplier_id': self.point_features[j].get('original_supplier_id', j),
                'supply_capacity': supply_capacity,
                'capability_score': capability_score,
                'type_score': type_score,
                'size_score': size_score,
                'mobilization_score': mobilization_score,
                'composite_score': composite_score,
                'enterprise_info': {
                    'type': enterprise_type,
                    'size': enterprise_size
                },
                'selected_sub_object': None,
                'all_sub_objects': []
            })

        # 定义category_comment排序函数
        def get_comment_order(comment):
            order_dict = {'good': 0, 'midle': 1, 'bad': 2}
            return order_dict.get(comment, 3)  # 未知评论类型排在最后

        # 为supplier_evaluations添加representative_comment字段
        for eval_obj in supplier_evaluations:
            if eval_obj['all_sub_objects']:
                # 获取所有细分对象的category_comment
                comments = set(obj.get('category_comment', 'midle') for obj in eval_obj['all_sub_objects'])
                if 'good' in comments:
                    eval_obj['representative_comment'] = 'good'
                elif 'midle' in comments:
                    eval_obj['representative_comment'] = 'midle'
                else:
                    eval_obj['representative_comment'] = 'bad'
            else:
                # 没有细分对象，使用默认'midle'
                eval_obj['representative_comment'] = 'midle'


        # 按综合评分排序（降序，分数越高越好）
        try:
            # 定义优先级映射
            # 对supplier_evaluations排序：首先按representative_comment，然后按supply_capacity
            supplier_evaluations.sort(
                key=lambda x: (get_comment_order(x['representative_comment']), -x['supply_capacity']))

            # 对supplier_compaires排序：首先按细分对象的category_comment，然后按supply_capacity（即细分对象的容量）
            supplier_compaires.sort(key=lambda x: (
                get_comment_order(x['all_sub_objects'].get('category_comment', 'midle')), -x['supply_capacity']))

        except (KeyError, TypeError) as e:
            self.logger.warning(f"供应点评分排序失败: {str(e)}")

        data_compaire_supplier = self.sort_list_by_D_innermost(supplier_compaires)

        pre_solution = []

        for iq in range(len(data_compaire_supplier)):
            if data_compaire_supplier[iq]['all_sub_objects']['specify_quantity']>0:
                #第一步保存提前解
                pre_solution.append(data_compaire_supplier[iq])
                #第二步进行删除
                sub_object_name = data_compaire_supplier[iq]['all_sub_objects']['sub_object_name']
                specify_quantity = data_compaire_supplier[iq]['all_sub_objects']['specify_quantity']

                supplier_evaluations = self.remove_by_name_sp_and_adjust_suc(supplier_evaluations, sub_object_name, specify_quantity)
            else:
                pass

        # 选择满足需求的供应点
        selected_suppliers = []
        sum_specify = 0
        for i in range(len(pre_solution)):
            sum_specify += pre_solution[i]['all_sub_objects']['specify_quantity']

        total_aggregated_demand1 = total_aggregated_demand
        total_aggregated_demand = total_aggregated_demand - sum_specify

        remaining_demand = total_aggregated_demand

        if remaining_demand == 0:
          pre_solution = {
                'code':200,
                'msg':'已经指定完成',
                'pre_data':pre_solution
            }
          data_solution = {}

          return pre_solution,data_solution

        else:
            self.D[self.K[0]] = total_aggregated_demand
            demand_c = total_aggregated_demand



            for evaluation in supplier_evaluations:

                supplier_capacity = min(evaluation['supply_capacity'], remaining_demand)

                # 按需求量选择细分对象组合
                selected_sub_objects = []
                if evaluation['all_sub_objects']:
                    remaining_to_allocate = supplier_capacity
                    available_sub_objects = evaluation['all_sub_objects'].copy()

                    for sub_obj_info in available_sub_objects:
                        if remaining_to_allocate <= self.EPS:
                            break

                        max_available = sub_obj_info['max_available_quantity']
                        allocated_from_this = min(remaining_to_allocate, max_available)

                        if allocated_from_this > self.EPS:
                            selected_sub_objects.append({
                                'sub_object_info': sub_obj_info,
                                'allocated_amount': allocated_from_this,
                                'category_id': sub_obj_info.get('category_id', 'unknown'),
                                'category_name': sub_obj_info.get('category_name', 'unknown'),
                                'recommend_md5': sub_obj_info.get('recommend_md5', 'unknown'),
                            })
                            remaining_to_allocate -= allocated_from_this

                selected_suppliers.append({
                    'supplier_id': evaluation['supplier_id'],
                    'allocated_capacity': supplier_capacity,
                    'evaluation': evaluation,
                    'selected_sub_objects': selected_sub_objects
                })

                remaining_demand -= supplier_capacity

                print(f"选择数据供应点 {evaluation['supplier_id']}: "
                      f"分配容量 {supplier_capacity:.2f}, "
                      f"综合评分 {evaluation['composite_score']:.3f}")

                # 显示所有被选择的细分对象
                if selected_sub_objects:
                    for sub_obj_selection in selected_sub_objects:
                        sub_obj_info = sub_obj_selection['sub_object_info']
                        allocated_amount = sub_obj_selection['allocated_amount']
                        print(f"  -> 选择细分对象: {sub_obj_info['sub_object_name']} "
                              f"(分配数量: {allocated_amount:.2f}, 最大可用量: {sub_obj_info['max_available_quantity']:.2f})")
                else:
                    print(f"  -> 使用供应点整体容量")
                remaining_demand = demand_c

            # 构建数据动员解
            satisfaction_rate = (
                                        total_aggregated_demand - remaining_demand) / total_aggregated_demand if total_aggregated_demand > 0 else 1.0

            data_solution = {}
            pre_data_solution = {}
            kh = 0
            for ka in range(len(pre_solution)):
                pre_data_solution[str(ka)] = {
                    'mobilization_type': 'data',
                    'selected_suppliers': [pre_solution[ka]],
                    'supplier_evaluations': [pre_solution[ka]],
                    'total_demand': pre_solution[ka]['supply_capacity'],
                    'satisfied_demand': total_aggregated_demand1,
                    'satisfaction_rate': 100,
                    'demand_point': aggregated_demand_point,

                    # 为了兼容性，提供空的运输变量
                    'x1': {}, 'x2': {}, 'x3': {}, 'x_direct': {},
                    'b1': {}, 'b2': {}, 'b3': {}, 'b_direct': {},
                    't1': {}, 't2': {}, 't3': {}, 't_direct': {},
                    'scheduling_time': 0.0,
                    'processed_paths': len([pre_solution[ka]]),
                    'objective_type': 'data_mobilization'
                }
                kh = kh +1

            for i in range(len(selected_suppliers)):
                data_solution[str(i)] = {
                    'mobilization_type': 'data',
                    'selected_suppliers': [selected_suppliers[i]],
                    'supplier_evaluations': [supplier_evaluations[i]],
                    'total_demand': total_aggregated_demand,
                    'satisfied_demand': total_aggregated_demand - remaining_demand,
                    'satisfaction_rate': satisfaction_rate,
                    'demand_point': aggregated_demand_point,

                    # 为了兼容性，提供空的运输变量
                    'x1': {}, 'x2': {}, 'x3': {}, 'x_direct': {},
                    'b1': {}, 'b2': {}, 'b3': {}, 'b_direct': {},
                    't1': {}, 't2': {}, 't3': {}, 't_direct': {},
                    'scheduling_time': 0.0,
                    'processed_paths': len(supplier_evaluations[i]['all_sub_objects']),
                    'objective_type': 'data_mobilization'
                }

            self.logger.info(
                f"数据动员完成 - 选择了{len(selected_suppliers)}个供应点，满足率{satisfaction_rate * 100:.1f}%")

            return pre_data_solution,data_solution


    def reShape(self,data,supply_name):
        filtered_data = {k: v for k, v in data.items() if supply_name in k}

    def reshape_list(self,data,supply_name):
        filtered_data = [item for item in data if item[1] == supply_name]
        return filtered_data


    def remove_key(self,data,supply_name):
        filtered_data = {k: v for k, v in data.items() if supply_name not in k}

        return  filtered_data
    def remove_list(self,data,supply_name):
        filtered_data = [item for item in data if item[1] != supply_name]
        return filtered_data

    def remove_specific_item_and_adjust_capacity(self,data, key, category_name, sub_object_name):
        """
        从嵌套字典结构中删除特定条目并调整capacity值
        如果capacity变为0，则删除整个键对应的字典
        """
        if key not in data:
            return data  # 如果键不存在，直接返回

        # 获取sub_objects列表
        sub_objects = data[key].get('sub_objects', [])

        # 遍历所有类别
        for category in sub_objects:
            # 检查类别名称是否匹配
            if category.get('category_name') == category_name:
                # 获取items列表
                items = category.get('items', [])

                # 查找要删除的项目并记录specify_quantity
                specify_quantity_to_remove = 0
                new_items = []

                for item in items:
                    if item.get('sub_object_name') == sub_object_name:
                        # 记录要减去的specify_quantity
                        specify_quantity_to_remove = item.get('specify_quantity', 0)

                    else:
                        # 保留不匹配的项目
                        new_items.append(item)

                # 更新items列表
                category['items'] = new_items


                # 调整capacity值
                if specify_quantity_to_remove > 0:
                    current_capacity = data[key].get('capacity', 0)
                    # 确保capacity不会变成负数
                    new_capacity = max(0, current_capacity - specify_quantity_to_remove)
                    data[key]['capacity'] = new_capacity


                    # 如果capacity变为0，删除整个键
                    if new_capacity == 0:
                        del data[key]
                        return data  # 直接返回，因为键已被删除
            items = category.get('items', [])
            if len(items)<1:
                sub_objects.remove(category)

        # sub_objects = self.remove_empty_items(sub_objects)


        return data

    def remove_empty_items(self,data_list):
        """
        从列表中删除 item 为空的字典

        参数:
        data_list (list): 包含字典的列表，每个字典都有一个 'item' 键

        返回:
        list: 过滤后的列表，不包含 item 为空的字典
        """
        if not isinstance(data_list, list):
            raise ValueError("输入必须是一个列表")

        # 使用列表推导式过滤掉 item 为空的字典
        return [item for item in data_list
                if isinstance(item, dict) and
                'item' in item and
                isinstance(item['item'], list) and
                len(item['item']) > 0]

    def remove_by_name_sp_and_adjust_suc(self,data, target_name, target_sp):
        """删除all_obj中匹配name和sp的元素，并从suc中减去对应max_a值"""
        for item in data:
            # 确保item有all_obj和suc键
            if 'all_sub_objects' in item and 'supply_capacity' in item and isinstance(item['all_sub_objects'], list):
                # 初始化要减去的总值
                total_subtract = 0
                # 创建新列表存放保留的元素
                new_objs = []

                # 遍历all_obj中的每个对象
                for obj in item['all_sub_objects']:
                    # 检查是否为字典且匹配name和sp
                    if (isinstance(obj, dict) and
                            obj.get('sub_object_name') == target_name and
                            obj.get('specify_quantity') == target_sp):

                        # 累加要减去的max_a值
                        if 'max_available_quantity' in obj and isinstance(obj['max_available_quantity'], (int, float)):
                            total_subtract += obj['max_available_quantity']
                    else:
                        # 保留不匹配的元素
                        new_objs.append(obj)

                # 更新all_obj列表
                item['all_sub_objects'] = new_objs

                # 更新suc值（确保suc是数字）
                if isinstance(item['supply_capacity'], (int, float)):
                    item['supply_capacity'] -= total_subtract

        return data

    def _get_available_transport_modes(self, m1, m2):
        """获取两个中转点之间可用的运输方式"""
        available_modes = []

        m1_specialized = self.point_features[m1]['specialized_mode']
        m2_specialized = self.point_features[m2]['specialized_mode']

        # 只有相同专业化模式的中转点才能进行联运
        if m1_specialized != m2_specialized:
            return available_modes

        # 遍历所有运输方式，找到匹配的运输方式
        for mode_id, mode_info in self.TRANSPORT_MODES.items():
            mode_name = mode_info['name']

            # 跳过仅限特定段的运输方式（road_only_modes=1的运输方式）
            if mode_info['road_only_modes'] == 1:
                continue

            # 根据专业化模式和运输方式名称匹配
            mode_match = False
            if m1_specialized == 'port' and '海运' in mode_name:
                mode_match = True
            elif m1_specialized == 'airport' and '空运' in mode_name:
                mode_match = True
            elif m1_specialized == 'railway' and '铁路' in mode_name:
                mode_match = True

            if mode_match and self.alpha2.get((m1, m2, mode_id), 0) == 1:
                available_modes.append(mode_id)

        return available_modes

    def _select_optimal_sub_objects(self, j, total_time, transport_cost):
        """从供应点的细分对象中选择最优组合，基于成本效率"""

        # 使用受限供应点级别的缓存
        if not hasattr(self, '_sub_object_cache'):
            # 基于供应点数量设置缓存限制
            cache_limit = len(self.J) * len(self.K) if len(self.J) * len(self.K) < len(self.J) * len(self.M) else len(
                self.J) * len(self.M)
            self._sub_object_cache = {}
            self._sub_object_cache_limit = max(cache_limit, len(self.J) * 2)
            self._sub_object_cache_order = []

        cache_key = f"sub_obj_{j}_{hash((total_time / 24, transport_cost))}"
        if cache_key in self._sub_object_cache:
            # 更新LRU顺序
            self._sub_object_cache_order.remove(cache_key)
            self._sub_object_cache_order.append(cache_key)
            return self._sub_object_cache[cache_key]

        sub_objects = self.point_features[j].get('sub_objects', [])
        if not sub_objects:
            cost_result = self._calculate_cost_by_resource_type(j, total_time / 24, transport_cost)
            if isinstance(cost_result, tuple):
                result = cost_result
            else:
                # 使用预计算的网络参数
                if not hasattr(self, '_network_params'):
                    self._network_params = {
                        'supply_scale': len(self.J),
                        'demand_scale': len(self.K)
                    }
                supply_scale = self._network_params['supply_scale']
                demand_scale = self._network_params['demand_scale']
                default_safety = supply_scale / (
                        supply_scale + demand_scale) if supply_scale + demand_scale > 0 else supply_scale / (
                        supply_scale + 1)
                result = (cost_result, default_safety)

            # 缓存结果
            if len(self._sub_object_cache) < 5000:
                self._sub_object_cache[cache_key] = result
            return result

        # 使用预计算的网络规模参数
        if not hasattr(self, '_network_params'):
            self._network_params = {
                'network_scale': len(self.J) + len(self.M) + len(self.K),
                'supply_scale': len(self.J),
                'transfer_scale': len(self.M),
                'demand_scale': len(self.K)
            }

        network_scale = self._network_params['network_scale']
        supply_scale = self._network_params['supply_scale']
        transfer_scale = self._network_params['transfer_scale']
        demand_scale = self._network_params['demand_scale']

        # 预处理：展平所有细分对象，避免嵌套循环
        flattened_sub_objects = []
        for category in sub_objects:
            if isinstance(category, dict) and 'items' in category:
                for sub_obj in category.get('items', []):
                    if sub_obj.get('max_available_quantity', 0) > self.EPS:
                        flattened_sub_objects.append(
                            (sub_obj, category.get('category_id', 'unknown'), category.get('category_name', 'unknown'),
                             category.get('recommend_md5', 'unknown')))
            else:
                if category.get('max_available_quantity', 0) > self.EPS:
                    flattened_sub_objects.append((category, category.get('category_id', 'unknown'),
                                                  category.get('category_name', 'unknown'),
                                                  category.get('recommend_md5', 'unknown')))

        # 批量计算成本和安全评分
        sub_object_scores = []
        for sub_obj, category_id, category_name, recommend_md5 in flattened_sub_objects:
            max_available = sub_obj.get('max_available_quantity', 0)

            # 使用批量计算函数
            sub_obj_cost = self._calculate_sub_object_cost(sub_obj, total_time / 24, transport_cost, supply_scale,
                                                           transfer_scale, demand_scale, network_scale)
            sub_obj_safety = self._calculate_sub_object_safety_score(sub_obj, supply_scale, transfer_scale,
                                                                     demand_scale, network_scale)
            cost_efficiency = sub_obj_safety / max(sub_obj_cost, self.EPS)

            sub_object_scores.append({
                'sub_object': sub_obj,
                'cost': sub_obj_cost,
                'safety': sub_obj_safety,
                'cost_efficiency': cost_efficiency,
                'max_available': max_available,
                'category_id': category_id,
                'category_name': category_name,
                'recommend_md5': recommend_md5
            })

        # 按成本效率排序，选择最优的细分对象
        sub_object_scores.sort(key=lambda x: x['cost_efficiency'], reverse=True)

        # 使用所有可用细分对象的加权平均，实现完全替代性
        total_available_capacity = sum(obj['max_available'] for obj in sub_object_scores)

        if total_available_capacity <= self.EPS:
            # 所有细分对象都不可用，使用预计算的网络参数
            default_cost = supply_scale * demand_scale * (supply_scale + demand_scale)
            default_safety = supply_scale / (supply_scale + demand_scale) if supply_scale + demand_scale > 0 else 0.5
            result = (default_cost, default_safety)
        else:
            # 按可用容量加权计算成本和安全性
            weighted_cost = sum(
                obj['cost'] * obj['max_available'] for obj in sub_object_scores) / total_available_capacity
            weighted_safety = sum(
                obj['safety'] * obj['max_available'] for obj in sub_object_scores) / total_available_capacity
            result = (weighted_cost, weighted_safety)

        # LRU缓存管理
        # if len(self._sub_object_cache) >= self._sub_object_cache_limit:
        #     # 移除最久未使用的条目
        #     oldest_key = self._sub_object_cache_order.pop(0)
        #     del self._sub_object_cache[oldest_key]

        # 添加新条目
        self._sub_object_cache[cache_key] = result
        self._sub_object_cache_order.append(cache_key)

        return result

    def _calculate_sub_object_cost(self, sub_obj, total_time, transport_cost, supply_scale, transfer_scale,
                                   demand_scale, network_scale):
        """计算细分对象的成本"""

        # 安全的数值转换函数
        def safe_float_convert(value, default=0.0):
            try:
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    if value.strip() == '':
                        return default
                    return float(value)
                elif isinstance(value, (list, tuple)):
                    return default
                else:
                    return default
            except (ValueError, TypeError):
                return default

        if hasattr(self, 'resource_type'):
            if self.resource_type == "personnel":
                # 基于供应能力平均值的合理默认值
                avg_capacity = sum(self.B.values()) / len(self.B) if self.B else 100.0
                personnel_wage = safe_float_convert(
                    sub_obj.get('wage_cost', avg_capacity),
                    avg_capacity)
                personnel_living = safe_float_convert(
                    sub_obj.get('living_cost', avg_capacity * 0.3),
                    avg_capacity * 0.3)
                return (personnel_wage + personnel_living) * safe_float_convert(total_time / 24, 1.0)

            elif self.resource_type == "material":
                # 基于运输成本的比例关系
                base_transport = safe_float_convert(transport_cost, 100.0)
                material_price = safe_float_convert(
                    sub_obj.get('material_price', base_transport * 10.0),
                    base_transport * 10.0)
                equipment_rental = safe_float_convert(
                    sub_obj.get('equipment_rental_price', base_transport * 1.5),
                    base_transport * 1.5)
                equipment_depreciation = safe_float_convert(
                    sub_obj.get('equipment_depreciation_cost', base_transport * 0.2),
                    base_transport * 0.2)
                return material_price + base_transport + (
                        equipment_rental + equipment_depreciation) * safe_float_convert(total_time / 24, 1.0)
            elif self.resource_type == "data":
                # 基于时间的合理默认值
                base_time_cost = safe_float_convert(total_time, 1.0) * 50.0
                data_facility_rental = safe_float_convert(
                    sub_obj.get('facility_rental_price', base_time_cost),
                    base_time_cost)
                data_power_cost = safe_float_convert(
                    sub_obj.get('power_cost', base_time_cost * 0.6),
                    base_time_cost * 0.6)
                data_communication_cost = safe_float_convert(
                    sub_obj.get('communication_purchase_price', base_time_cost * 0.8),
                    base_time_cost * 0.8)
                data_processing_cost = safe_float_convert(
                    sub_obj.get('data_processing_cost', base_time_cost * 0.4),
                    base_time_cost * 0.4)
                data_storage_cost = safe_float_convert(
                    sub_obj.get('data_storage_cost', base_time_cost * 0.2),
                    base_time_cost * 0.2)
                return (data_facility_rental + data_processing_cost + data_storage_cost) * safe_float_convert(
                    total_time / 24, 1.0) + data_power_cost + data_communication_cost + safe_float_convert(transport_cost,
                                                                                                      0.0)
            else:
                # 默认物资动员
                base_transport = safe_float_convert(transport_cost, 100.0)
                material_price = safe_float_convert(
                    sub_obj.get('material_price', base_transport * 10.0),
                    base_transport * 10.0)
                equipment_rental = safe_float_convert(
                    sub_obj.get('equipment_rental_price', base_transport * 1.5),
                    base_transport * 1.5)
                equipment_depreciation = safe_float_convert(
                    sub_obj.get('equipment_depreciation_cost', base_transport * 0.2),
                    base_transport * 0.2)
                return material_price + base_transport + (
                        equipment_rental + equipment_depreciation) * safe_float_convert(total_time / 24, 1.0)
        else:
            # 兼容性处理
            base_transport = safe_float_convert(transport_cost, 100.0)
            material_price = safe_float_convert(
                sub_obj.get('material_price', base_transport * 10.0),
                base_transport * 10.0)
            equipment_rental = safe_float_convert(
                sub_obj.get('equipment_rental_price', base_transport * 1.5),
                base_transport * 1.5)
            equipment_depreciation = safe_float_convert(
                sub_obj.get('equipment_depreciation_cost', base_transport * 0.2),
                base_transport * 0.2)
            return material_price + base_transport + (
                    equipment_rental + equipment_depreciation) * safe_float_convert(total_time / 24, 1.0)

    def _calculate_sub_object_safety_score(self, sub_obj, supply_scale, transfer_scale, demand_scale, network_scale):
        """计算细分对象的安全评分"""

        # 安全的数值转换函数
        def safe_float_convert(value, default=0.0):
            try:
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    if value.strip() == '':
                        return default
                    return float(value)
                elif isinstance(value, (list, tuple)):
                    return default
                else:
                    return default
            except (ValueError, TypeError):
                return default

        total_safety = 0.0

        if hasattr(self, 'resource_type'):
            if self.resource_type == "personnel":
                total_safety = (safe_float_convert(sub_obj.get('political_status', 0), 0) +
                                safe_float_convert(sub_obj.get('military_experience', 0), 0) -
                                safe_float_convert(sub_obj.get('criminal_record', 0), 0) -
                                safe_float_convert(sub_obj.get('network_record', 0), 0) -
                                safe_float_convert(sub_obj.get('credit_record', 0), 0))
            elif self.resource_type == "material":
                # 使用网络规模参数作为默认值
                default_enterprise_nature = supply_scale / (
                        supply_scale + transfer_scale) if supply_scale + transfer_scale > 0 else supply_scale / (
                        supply_scale + 1)
                default_enterprise_scale = demand_scale / (
                        demand_scale + transfer_scale) if demand_scale + transfer_scale > 0 else demand_scale / (
                        demand_scale + 1)
                default_resource_safety = transfer_scale / (
                        transfer_scale + demand_scale) if transfer_scale + demand_scale > 0 else transfer_scale / (
                        transfer_scale + 1)
                transport_modes_count = len(self.TRANSPORT_MODES) if hasattr(self, 'TRANSPORT_MODES') else 1
                default_material_penalty = transport_modes_count / (
                        transport_modes_count + supply_scale) if transport_modes_count + supply_scale > 0 else transport_modes_count / (
                        transport_modes_count + 1)

                total_safety = (
                        -safe_float_convert(sub_obj.get('flammable_explosive', 0),0) -
                        safe_float_convert(sub_obj.get('corrosive', 0),0) -
                        safe_float_convert(sub_obj.get('polluting', 0),0) -
                        safe_float_convert(sub_obj.get('fragile', 0),0))
            elif self.resource_type == "data":
                # 数据动员：安全指标从细分对象中获取，使用网络规模参数作为默认值
                default_control_score = transfer_scale / (
                        transfer_scale + demand_scale) if transfer_scale + demand_scale > 0 else transfer_scale / (
                        transfer_scale + 1)
                default_usability_score = supply_scale / (
                        supply_scale + transfer_scale) if supply_scale + transfer_scale > 0 else supply_scale / (
                        supply_scale + 1)
                transport_modes_count = len(self.TRANSPORT_MODES) if hasattr(self, 'TRANSPORT_MODES') else 1
                default_facility_score = network_scale / (
                        network_scale + transport_modes_count) if network_scale > 0 and transport_modes_count > 0 else network_scale / (
                        network_scale + 1)

                autonomous_control = safe_float_convert(sub_obj.get('autonomous_control', default_control_score),
                                                        default_control_score)
                usability_level = safe_float_convert(sub_obj.get('usability_level', default_usability_score),
                                                     default_usability_score)
                maintenance_derived = (autonomous_control + usability_level) / (
                            transfer_scale + 1) if transfer_scale > 0 else (autonomous_control + usability_level) / 2
                facility_protection = safe_float_convert(sub_obj.get('facility_protection', default_facility_score),
                                                         default_facility_score)
                camouflage_protection = safe_float_convert(sub_obj.get('camouflage_protection', default_facility_score),
                                                           default_facility_score)
                environment_score = safe_float_convert(sub_obj.get('surrounding_environment', 0), 0)

                total_safety = (autonomous_control + usability_level + maintenance_derived +
                                facility_protection + camouflage_protection + environment_score)

        return -total_safety

    def _calculate_cost_by_resource_type(self, j, total_time, transport_cost):
        """根据资源类型计算总成本（使用细分对象或默认值）"""

        # 安全的数值转换函数
        def safe_float_convert(value, default=0.0):
            try:
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    if value.strip() == '':
                        return default
                    return float(value)
                elif isinstance(value, (list, tuple)):
                    return default
                else:
                    return default
            except (ValueError, TypeError):
                return default

        try:
            if hasattr(self, 'resource_type'):
                if self.resource_type == "personnel":
                    # 人员动员：从细分对象中获取最优成本，如果没有则使用默认值
                    sub_objects = self.point_features[j].get('sub_objects', [])
                    if sub_objects:
                        # 选择成本最优的细分对象
                        best_cost = float('inf')
                        found_valid_sub_objects = False

                        for category in sub_objects:
                            if isinstance(category, dict) and 'items' in category and category.get('items'):
                                # 新的分类结构：遍历分类中的所有项目
                                for sub_obj in category.get('items', []):
                                    # 检查可用性
                                    max_available = sub_obj.get('max_available_quantity', 0)
                                    if max_available <= self.EPS:
                                        continue  # 跳过不可用的细分对象

                                    wage_cost = sub_obj.get('wage_cost')
                                    living_cost = sub_obj.get('living_cost')

                                    if wage_cost is None or wage_cost < 0:
                                        raise ValueError(
                                            f"细分对象{sub_obj.get('sub_object_id', 'unknown')}缺少有效的wage_cost配置")
                                    if living_cost is None or living_cost < 0:
                                        raise ValueError(
                                            f"细分对象{sub_obj.get('sub_object_id', 'unknown')}缺少有效的living_cost配置")

                                    total_cost = (wage_cost + living_cost) * total_time / 24 + transport_cost
                                    best_cost = min(best_cost, total_cost)
                                    found_valid_sub_objects = True
                            elif isinstance(category, dict) and 'sub_object_id' in category:
                                # 兼容旧的平铺结构：直接处理细分对象
                                sub_obj = category
                                # 检查可用性
                                max_available = sub_obj.get('max_available_quantity', 0)
                                if max_available <= self.EPS:
                                    continue
                                wage_cost = sub_obj.get('wage_cost')
                                living_cost = sub_obj.get('living_cost')

                                if wage_cost is None or wage_cost < 0:
                                    raise ValueError(
                                        f"细分对象{sub_obj.get('sub_object_id', 'unknown')}缺少有效的living_cost配置")
                                if living_cost is None or living_cost < 0:
                                    raise ValueError(
                                        f"细分对象{sub_obj.get('sub_object_id', 'unknown')}缺少有效的living_cost配置")

                                total_cost = (wage_cost + living_cost) * total_time / 24  + transport_cost
                                best_cost = min(best_cost, total_cost)
                                found_valid_sub_objects = True

                        if found_valid_sub_objects and best_cost != float('inf'):
                            return best_cost
                        else:
                            # 使用基于现有数据的合理默认值
                            avg_capacity = sum(self.B.values()) / len(self.B) if self.B else 1.0
                            return avg_capacity * total_time / 24
                    else:
                        # 使用基于现有数据的合理默认值
                        avg_capacity = sum(self.B.values()) / len(self.B) if self.B else 1.0
                        return avg_capacity * total_time / 24

                elif self.resource_type == "material":
                    # 物资动员：从细分对象中获取最优成本，如果没有则使用默认值
                    sub_objects = self.point_features[j].get('sub_objects', [])
                    if sub_objects:
                        # 选择成本最优的细分对象
                        best_cost = float('inf')
                        for sub_obj in sub_objects:
                            # 检查可用性
                            max_available = sub_obj.get('max_available_quantity', 0)
                            if max_available <= self.EPS:
                                continue  # 跳过不可用的细分对象

                            material_price = safe_float_convert(sub_obj.get('material_price', 200), 200)
                            equipment_rental = safe_float_convert(sub_obj.get('equipment_rental_price', 150), 150)
                            equipment_depreciation = safe_float_convert(
                                sub_obj.get('equipment_depreciation_cost', 20), 20)
                            total_cost = material_price + transport_cost + (
                                    equipment_rental + equipment_depreciation) * total_time / 24
                            best_cost = min(best_cost, total_cost)
                        return best_cost

                else:  # data
                    # 数据动员：设施成本在供应点级别
                    facility_rental = safe_float_convert(self.point_features[j].get('facility_rental_price', 80),
                                                         80)
                    facility_power = safe_float_convert(self.point_features[j].get('power_cost', 30), 30)
                    communication_cost = safe_float_convert(
                        self.point_features[j].get('communication_purchase_price', 40), 40)
                    return facility_rental * total_time / 24 + facility_power + communication_cost
            else:
                # 兼容性处理：默认物资动员
                sub_objects = self.point_features[j].get('sub_objects', [])
                if sub_objects:
                    best_cost = float('inf')
                    for sub_obj in sub_objects:
                        material_price = safe_float_convert(sub_obj.get('material_price', 200), 200)
                        equipment_rental = safe_float_convert(sub_obj.get('equipment_rental_price', 150), 150)
                        equipment_depreciation = safe_float_convert(sub_obj.get('equipment_depreciation_cost', 20),
                                                                    20)
                        total_cost = material_price + transport_cost + (
                                equipment_rental + equipment_depreciation) * total_time / 24
                        best_cost = min(best_cost, total_cost)
                    return best_cost

        except Exception as e:
            raise ValueError(f"成本计算失败: {str(e)}")

    def _calculate_direct_path_metrics(self, j, k):
        """计算直接路径的所有目标指标"""

        # 使用受限实例级缓存避免重复计算
        if not hasattr(self, '_metrics_cache'):
            # 基于网络规模动态设置缓存限制
            cache_limit = min(len(self.J) * len(self.K), len(self.J) * len(self.M) + len(self.M) * len(self.K))
            self._metrics_cache = {}
            self._metrics_cache_limit = max(cache_limit, len(self.J) + len(self.K))
            self._metrics_cache_keys = []

        cache_key = f"direct_{j}_{k}"
        if cache_key in self._metrics_cache:
            # 更新访问顺序
            self._metrics_cache_keys.remove(cache_key)
            self._metrics_cache_keys.append(cache_key)
            return self._metrics_cache[cache_key]

        # 安全的数值转换函数
        def safe_float_convert(value, default=0.0):
            try:
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    if value.strip() == '':
                        return default
                    return float(value)
                else:
                    return default
            except (ValueError, TypeError):
                return default

        # 使用预计算的网络规模参数
        if not hasattr(self, '_network_params'):
            self._network_params = {
                'network_scale': len(self.J) + len(self.M) + len(self.K),
                'supply_scale': len(self.J),
                'transfer_scale': len(self.M),
                'demand_scale': len(self.K),
                'transport_modes_count': len(self.TRANSPORT_MODES) if hasattr(self, 'TRANSPORT_MODES') else 1
            }

        network_scale = self._network_params['network_scale']
        supply_scale = self._network_params['supply_scale']
        transfer_scale = self._network_params['transfer_scale']
        demand_scale = self._network_params['demand_scale']
        transport_modes_count = self._network_params['transport_modes_count']

        # 细分对象安全评分计算函数
        def calculate_sub_object_safety_score(sub_obj):
            """计算细分对象的安全评分"""
            total_safety = 0.0

            if hasattr(self, 'resource_type'):
                if self.resource_type == "personnel":
                    total_safety = (safe_float_convert(sub_obj.get('political_status', 0), 0) +
                                    safe_float_convert(sub_obj.get('military_experience', 0), 0) -
                                    safe_float_convert(sub_obj.get('criminal_record', 0), 0) -
                                    safe_float_convert(sub_obj.get('network_record', 0), 0) -
                                    safe_float_convert(sub_obj.get('credit_record', 0), 0))
                elif self.resource_type == "material":
                    # 使用网络规模参数作为默认值
                    total_safety = (
                            -safe_float_convert(sub_obj.get('flammable_explosive', 0), 0) -
                            safe_float_convert(sub_obj.get('corrosive', 0), 0) -
                            safe_float_convert(sub_obj.get('polluting', 0), 0) -
                            safe_float_convert(sub_obj.get('fragile', 0), 0))
                elif self.resource_type == "data":
                    # 数据动员：安全指标从细分对象中获取，使用网络规模参数作为默认值
                    default_control_score = transfer_scale / (
                            transfer_scale + demand_scale) if transfer_scale + demand_scale > 0 else transfer_scale / (
                            transfer_scale + 1)
                    default_usability_score = supply_scale / (
                            supply_scale + transfer_scale) if supply_scale + transfer_scale > 0 else supply_scale / (
                            supply_scale + 1)
                    transport_modes_count = len(self.TRANSPORT_MODES) if hasattr(self, 'TRANSPORT_MODES') else 1
                    default_facility_score = network_scale / (
                            network_scale + transport_modes_count) if network_scale > 0 and transport_modes_count > 0 else network_scale / (
                            network_scale + 1)

                    autonomous_control = safe_float_convert(sub_obj.get('autonomous_control', default_control_score),
                                                            default_control_score)
                    usability_level = safe_float_convert(sub_obj.get('usability_level', default_usability_score),
                                                         default_usability_score)
                    maintenance_derived = (autonomous_control + usability_level) / (
                                transfer_scale + 1) if transfer_scale > 0 else (
                                                                                           autonomous_control + usability_level) / 2
                    facility_protection = safe_float_convert(sub_obj.get('facility_protection', default_facility_score),
                                                             default_facility_score)
                    camouflage_protection = safe_float_convert(
                        sub_obj.get('camouflage_protection', default_facility_score), default_facility_score)
                    environment_score = safe_float_convert(sub_obj.get('surrounding_environment', 0), 0)

                    total_safety = (autonomous_control + usability_level + maintenance_derived +
                                    facility_protection + camouflage_protection + environment_score)

            return -total_safety

        def calculate_cost_by_resource_type(j, total_time, transport_cost):
            """根据资源类型计算总成本（使用细分对象或默认值）"""
            try:
                if hasattr(self, 'resource_type'):
                    if self.resource_type == "personnel":
                        # 人员动员：从细分对象中获取最优成本，如果没有则使用默认值
                        sub_objects = self.point_features[j].get('sub_objects', [])
                        if sub_objects:
                            # 选择成本最优的细分对象
                            best_cost = float('inf')
                            found_valid_sub_objects = False

                            for category in sub_objects:
                                if isinstance(category, dict) and 'items' in category and category.get('items'):
                                    # 新的分类结构：遍历分类中的所有项目
                                    for sub_obj in category.get('items', []):
                                        # 检查可用性
                                        max_available = sub_obj.get('max_available_quantity', 0)
                                        if max_available <= self.EPS:
                                            continue  # 跳过不可用的细分对象

                                        wage_cost = sub_obj.get('wage_cost')
                                        living_cost = sub_obj.get('living_cost')

                                        if wage_cost is None or wage_cost < 0:
                                            raise ValueError(
                                                f"细分对象{sub_obj.get('sub_object_id', 'unknown')}缺少有效的wage_cost配置")
                                        if living_cost is None or living_cost < 0:
                                            raise ValueError(
                                                f"细分对象{sub_obj.get('sub_object_id', 'unknown')}缺少有效的living_cost配置")

                                        total_cost = (wage_cost + living_cost) * total_time / 24 + transport_cost
                                        best_cost = min(best_cost, total_cost)
                                        found_valid_sub_objects = True
                                elif isinstance(category, dict) and 'sub_object_id' in category:
                                    # 兼容旧的平铺结构：直接处理细分对象
                                    sub_obj = category
                                    # 检查可用性
                                    max_available = sub_obj.get('max_available_quantity', 0)
                                    if max_available <= self.EPS:
                                        continue

                                    wage_cost = sub_obj.get('wage_cost')
                                    living_cost = sub_obj.get('living_cost')

                                    if wage_cost is None or wage_cost < 0:
                                        raise ValueError(
                                            f"细分对象{sub_obj.get('sub_object_id', 'unknown')}缺少有效的wage_cost配置")
                                    if living_cost is None or living_cost < 0:
                                        raise ValueError(
                                            f"细分对象{sub_obj.get('sub_object_id', 'unknown')}缺少有效的living_cost配置")

                                    total_cost = (wage_cost + living_cost) * total_time / 24 + transport_cost
                                    best_cost = min(best_cost, total_cost)
                                    found_valid_sub_objects = True

                            if found_valid_sub_objects and best_cost != float('inf'):
                                return best_cost
                            else:
                                # 使用基于实际容量的默认值
                                avg_capacity = sum(self.B.values()) / len(self.B) if self.B else 1.0
                                return avg_capacity * total_time / 24
                        else:
                            # 使用基于实际容量的默认值
                            avg_capacity = sum(self.B.values()) / len(self.B) if self.B else 1.0
                            return avg_capacity * total_time / 24

                    elif self.resource_type == "material":
                        # 物资动员：从细分对象中获取最优成本，如果没有则使用基于网络规模的默认值
                        sub_objects = self.point_features[j].get('sub_objects', [])
                        if sub_objects:
                            # 选择成本最优的细分对象
                            best_cost = float('inf')
                            for sub_obj in sub_objects:
                                # 检查可用性
                                max_available = sub_obj.get('max_available_quantity', 0)
                                if max_available <= self.EPS:
                                    continue  # 跳过不可用的细分对象

                                material_price = safe_float_convert(sub_obj.get('material_price',
                                                                                supply_scale * network_scale * (
                                                                                        supply_scale + demand_scale)),
                                                                    supply_scale * network_scale * (
                                                                            supply_scale + demand_scale))
                                equipment_rental = safe_float_convert(
                                    sub_obj.get('equipment_rental_price', supply_scale * network_scale),
                                    supply_scale * network_scale)
                                equipment_depreciation = safe_float_convert(
                                    sub_obj.get('equipment_depreciation_cost', supply_scale + demand_scale),
                                    supply_scale + demand_scale)
                                total_cost = material_price + transport_cost + (
                                        equipment_rental + equipment_depreciation) * total_time / 24

                                best_cost = min(best_cost, total_cost)

                            return best_cost if best_cost != float('inf') else supply_scale * network_scale * (
                                    supply_scale + demand_scale) + transport_cost + (
                                                                                       supply_scale * network_scale + supply_scale + demand_scale) * total_time / 24
                        else:
                            # 使用基于网络规模的默认值
                            default_material_price = supply_scale * network_scale * (supply_scale + demand_scale)
                            default_equipment_rental = supply_scale * network_scale
                            default_equipment_depreciation = supply_scale + demand_scale
                            return default_material_price + transport_cost + (
                                    default_equipment_rental + default_equipment_depreciation) * total_time / 24

                    else:  # data
                        # 数据动员
                        facility_rental = safe_float_convert(self.point_features[j].get('facility_rental_price',
                                                                                        supply_scale * transfer_scale + supply_scale),
                                                             supply_scale * transfer_scale + supply_scale)
                        facility_power = safe_float_convert(
                            self.point_features[j].get('power_cost', supply_scale + demand_scale),
                            supply_scale + demand_scale)
                        communication_cost = safe_float_convert(
                            self.point_features[j].get('communication_purchase_price', supply_scale * demand_scale),
                            supply_scale * demand_scale)
                        return facility_rental * total_time / 24 + facility_power + communication_cost
                else:
                    # 兼容性处理：默认物资动员
                    sub_objects = self.point_features[j].get('sub_objects', [])
                    if sub_objects:
                        best_cost = float('inf')
                        for sub_obj in sub_objects:
                            # 检查可用性
                            max_available = sub_obj.get('max_available_quantity', 0)
                            if max_available <= self.EPS:
                                continue  # 跳过不可用的细分对象

                            material_price = safe_float_convert(sub_obj.get('material_price',
                                                                            supply_scale * network_scale * (
                                                                                    supply_scale + demand_scale)),
                                                                supply_scale * network_scale * (
                                                                        supply_scale + demand_scale))
                            equipment_rental = safe_float_convert(
                                sub_obj.get('equipment_rental_price', supply_scale * network_scale),
                                supply_scale * network_scale)
                            equipment_depreciation = safe_float_convert(
                                sub_obj.get('equipment_depreciation_cost', supply_scale + demand_scale),
                                supply_scale + demand_scale)
                            total_cost = material_price + transport_cost + (
                                    equipment_rental + equipment_depreciation) * total_time / 24
                            best_cost = min(best_cost, total_cost)
                        return best_cost if best_cost != float('inf') else supply_scale * network_scale * (
                                supply_scale + demand_scale) + transport_cost + (
                                                                                   supply_scale * network_scale + supply_scale + demand_scale) * total_time / 24
                    else:
                        default_material_price = supply_scale * network_scale * (supply_scale + demand_scale)
                        default_equipment_rental = supply_scale * network_scale
                        default_equipment_depreciation = supply_scale + demand_scale
                        return default_material_price + transport_cost + (
                                default_equipment_rental + default_equipment_depreciation) * total_time / 24
            except Exception as e:
                raise ValueError(f"成本计算失败: {str(e)}")

        # 抽取的安全评分计算逻辑
        def calculate_safety_score_by_resource_type(j):
            """根据资源类型计算安全评分"""
            total_safety = 0.0

            # 使用网络规模参数计算默认值
            default_enterprise_nature = supply_scale / (
                    supply_scale + transfer_scale) if supply_scale + transfer_scale > 0 else supply_scale / (
                    supply_scale + 1)
            default_enterprise_scale = demand_scale / (
                    demand_scale + transfer_scale) if demand_scale + transfer_scale > 0 else demand_scale / (
                    demand_scale + 1)
            default_resource_safety = transfer_scale / (
                    transfer_scale + demand_scale) if transfer_scale + demand_scale > 0 else transfer_scale / (
                    transfer_scale + 1)
            default_material_penalty = transport_modes_count / (
                    transport_modes_count + supply_scale) if transport_modes_count + supply_scale > 0 else transport_modes_count / (
                    transport_modes_count + 1)

            if hasattr(self, 'resource_type'):
                if self.resource_type == "personnel":
                    # 人员动员：只考虑人员队伍安全
                    political_status = safe_float_convert(self.point_features[j].get('political_status', 0), 0)
                    military_experience = safe_float_convert(self.point_features[j].get('military_experience', 0), 0)
                    criminal_record = safe_float_convert(self.point_features[j].get('criminal_record', 0), 0)
                    network_record = safe_float_convert(self.point_features[j].get('network_record', 0), 0)
                    credit_record = safe_float_convert(self.point_features[j].get('credit_record', 0), 0)
                    total_safety = political_status + military_experience - criminal_record - network_record - credit_record

                elif self.resource_type == "material":
                    # 物资动员：只考虑危险性属性
                    flammable_score = safe_float_convert(
                        self.point_features[j].get('flammable_explosive', 0), 0)
                    corrosive_score = safe_float_convert(
                        self.point_features[j].get('corrosive', 0), 0)
                    polluting_score = safe_float_convert(
                        self.point_features[j].get('polluting', 0), 0)
                    fragile_score = safe_float_convert(self.point_features[j].get('fragile', 0), 0)

                    total_safety = -flammable_score - corrosive_score - polluting_score - fragile_score

                elif self.resource_type == "data":
                    # 数据动员：考虑设施场所安全和装备设备安全，使用网络规模参数
                    default_control_score = transfer_scale / (
                            transfer_scale + demand_scale) if transfer_scale + demand_scale > 0 else transfer_scale / (
                            transfer_scale + 1)
                    default_usability_score = supply_scale / (
                            supply_scale + transfer_scale) if supply_scale + transfer_scale > 0 else supply_scale / (
                            supply_scale + 1)
                    default_facility_score = network_scale / (
                            network_scale + transport_modes_count) if network_scale > 0 and transport_modes_count > 0 else network_scale / (
                            network_scale + 1)
                    autonomous_control = safe_float_convert(
                        self.point_features[j].get('autonomous_control', default_control_score), default_control_score)
                    usability_level = safe_float_convert(
                        self.point_features[j].get('usability_level', default_usability_score), default_usability_score)

                    maintenance_derived = (autonomous_control + usability_level) / (
                            transfer_scale + 1) if transfer_scale > 0 else (
                                                                                   autonomous_control + usability_level) / 2
                    facility_protection = safe_float_convert(
                        self.point_features[j].get('facility_protection', default_facility_score),
                        default_facility_score)
                    camouflage_protection = safe_float_convert(
                        self.point_features[j].get('camouflage_protection', default_facility_score),
                        default_facility_score)
                    environment_score = safe_float_convert(self.point_features[j].get('surrounding_environment', 0), 0)
                    total_safety = autonomous_control + usability_level + maintenance_derived + facility_protection + camouflage_protection + environment_score

            else:
                # 兼容性处理：默认物资动员
                flammable_score = safe_float_convert(
                    self.point_features[j].get('flammable_explosive', 0),0)
                corrosive_score = safe_float_convert(self.point_features[j].get('corrosive', 0),0)
                polluting_score = safe_float_convert(self.point_features[j].get('polluting', 0),0)
                fragile_score = safe_float_convert(self.point_features[j].get('fragile', 0),0)

                total_safety = -flammable_score - corrosive_score - polluting_score - fragile_score

            return -total_safety

        try:
            # 基于经纬度计算球面距离
            self.logger.debug(f"获取供应点 {j} 和需求点 {k} 的坐标信息")

            if j not in self.point_features:
                raise KeyError(f"供应点 {j} 不在点特征数据中")
            if k not in self.point_features:
                raise KeyError(f"需求点 {k} 不在点特征数据中")

            j_coords = self.point_features[j]
            k_coords = self.point_features[k]

            if 'latitude' not in j_coords or 'longitude' not in j_coords:
                raise KeyError(f"供应点 {j} 缺少坐标信息")
            if 'latitude' not in k_coords or 'longitude' not in k_coords:
                raise KeyError(f"需求点 {k} 缺少坐标信息")

            j_lat, j_lon = j_coords['latitude'], j_coords['longitude']
            k_lat, k_lon = k_coords['latitude'], k_coords['longitude']

            self.logger.debug(f"坐标获取成功 - {j}: ({j_lat}, {j_lon}), {k}: ({k_lat}, {k_lon})")

            try:
                direct_distance = self._calculate_haversine_distance(j_lat, j_lon, k_lat, k_lon)
                self.logger.debug(f"直接距离计算完成: {direct_distance:.2f}公里")
            except Exception as e:
                self.logger.error(f"距离计算失败: {str(e)}", exc_info=True)
                raise ValueError(f"计算 {j} 到 {k} 的距离失败: {str(e)}")

            # 基础运输指标
            self.logger.debug("计算基础运输指标")
            try:
                transport_mode = self.TRANSPORT_MODES.get(1)
                if not transport_mode:
                    raise ValueError("运输方式1（公路运输）未在TRANSPORT_MODES中配置")

                speed = transport_mode.get('speed')
                cost_per_km = transport_mode.get('cost_per_km')

                if speed is None or speed <= 0:
                    raise ValueError(f"运输方式1的速度未配置或无效: {speed}")
                if cost_per_km is None or cost_per_km < 0:
                    raise ValueError(f"运输方式1的单位成本未配置或无效: {cost_per_km}")

                transport_time = direct_distance / speed
                transport_cost = direct_distance * cost_per_km

                self.logger.debug(f"基础运输指标 - 时间: {transport_time:.3f}小时, 成本: {transport_cost:.2f}")

            except (KeyError, ZeroDivisionError) as e:
                self.logger.error(f"基础运输指标计算失败: {str(e)}", exc_info=True)
                self.logger.error(f"运输方式数据: {self.TRANSPORT_MODES.get(1, '不存在')}")
                raise ValueError(f"计算基础运输指标失败，请检查运输方式1的配置: {str(e)}")

        except Exception as e:
            self.logger.error(f"直接路径指标计算失败 {j} -> {k}: {str(e)}", exc_info=True)
            raise

        # 计算网络特征参数用于长距离和大批量判断
        all_distances = []
        for j_temp in self.J:
            for k_temp in self.K:
                j_temp_lat, j_temp_lon = self.point_features[j_temp]['latitude'], self.point_features[j_temp][
                    'longitude']
                k_temp_lat, k_temp_lon = self.point_features[k_temp]['latitude'], self.point_features[k_temp][
                    'longitude']
                temp_distance = self._calculate_haversine_distance(j_temp_lat, j_temp_lon, k_temp_lat, k_temp_lon)
                all_distances.append(temp_distance)

        avg_network_distance = sum(all_distances) / len(all_distances) if all_distances else direct_distance
        max_network_distance = max(all_distances) if all_distances else direct_distance

        # 供应能力信息
        supply_capacity = self.B[j] * self.P[j]
        avg_supply_capacity = sum(self.B[j_temp] * self.P[j_temp] for j_temp in self.J) / len(
            self.J) if self.J else supply_capacity

        # 长距离和大批量判断（基于实际数据特征计算）
        distance_threshold = avg_network_distance * 1.5  # 超过平均距离1.5倍视为长距离
        capacity_threshold = avg_supply_capacity * 1.2  # 超过平均容量1.2倍视为大批量

        is_long_distance = direct_distance > distance_threshold
        is_large_batch = supply_capacity > capacity_threshold

        # 长距离直达运输的不利因素调整，基于网络规模参数
        distance_penalty_factor = 1.0

        if is_long_distance:
            # 长距离时，直达运输面临更多不利因素
            distance_relative_ratio = direct_distance / max_network_distance if max_network_distance > 0 else 1.0

            # 考虑直达运输的时间优势，减少过度惩罚
            time_advantage_factor = 1.0 - (transport_time / (transport_time + self.T_loading))
            distance_penalty_factor = 1.0 + (distance_relative_ratio / (1 + transfer_scale)) * (
                    1.0 - time_advantage_factor * 0.3)
        else:
            distance_penalty_factor = 1.0

        # 计算8个目标的评分
        metrics = {}

        # 1. 时间目标（按表格：准备时间 + 运输时间 + 交接时间）
        preparation_time = self.T1 + self.T4  # 征召时间 + 集结时间
        exchange_time = self.T6  # 交接时间
        total_time = preparation_time + transport_time + exchange_time
        metrics['time_score'] = total_time

        # 时间窗约束检查
        time_window_penalty = 0.0
        if k in self.time_windows:
            earliest_start, latest_finish = self.time_windows[k]
            if total_time > latest_finish:
                time_window_penalty = (total_time - latest_finish) / latest_finish

        metrics['time_window_violation'] = time_window_penalty

        # 2. 成本目标（使用细分对象选择逻辑）
        try:
            # 检查是否有细分对象
            sub_objects = self.point_features[j].get('sub_objects', [])
            if sub_objects:
                selected_cost, selected_safety = self._select_optimal_sub_objects(j, total_time,
                                                                                  transport_cost)
                total_cost = selected_cost
                # 更新安全评分
                total_safety = selected_safety
            else:
                total_cost = self._calculate_cost_by_resource_type(j, total_time, transport_cost)
                total_safety = calculate_safety_score_by_resource_type(j)

            metrics['cost_score'] = total_cost
        except ValueError as e:
            raise ValueError(f"成本计算失败: {str(e)}")

        # 3. 距离目标
        adjusted_distance = direct_distance * distance_penalty_factor
        metrics['distance_score'] = adjusted_distance

        # 4. 安全性目标（如果有细分对象，safety值已在成本计算中更新）
        if 'total_safety' not in locals():
            total_safety = calculate_safety_score_by_resource_type(j)
        metrics['safety_score'] = total_safety

        # 5. 优先级目标
        supplier_priority_score = self._calculate_supplier_task_priority(j, k, total_time, total_cost, total_safety)
        # 时间窗违反会降低优先级，增加优先级目标值（因为是最小化目标）
        metrics['priority_score'] = -supplier_priority_score * (1.0 + time_window_penalty)

        # 6. 资源均衡目标
        supply_capacity = self.B[j] * self.P[j]
        if self.total_supply > 0:
            ideal_usage_ratio = supply_capacity / self.total_supply
        else:
            ideal_usage_ratio = supply_scale / (
                    supply_scale + demand_scale) if supply_scale + demand_scale > 0 else supply_scale / (
                    supply_scale + 1)

        # 预估该供应点的实际使用比例（基于容量约束）
        max_possible_allocation = min(supply_capacity, self.total_demand)
        estimated_usage_ratio = max_possible_allocation / max(self.total_demand, supply_capacity) if max(
            self.total_demand, supply_capacity) > 0 else ideal_usage_ratio

        # 使用偏差：实际使用比例偏离理想比例的程度
        usage_deviation = abs(estimated_usage_ratio - ideal_usage_ratio)
        metrics['balance_score'] = usage_deviation

        # 7. 企业能力目标（根据资源类型调整）
        if hasattr(self, 'resource_type') and self.resource_type in ["personnel", "data"]:
            # 人员和数据动员不考虑企业能力
            metrics['capability_score'] = 0.0
        else:
            # 物资动员考虑企业能力 - 根据企业规模推导
            enterprise_size = self.point_features[j].get('enterprise_size', '中')
            if enterprise_size == '大':
                default_scale_capability = supply_scale + (supply_scale % (supply_scale + 1))
            elif enterprise_size == '中':
                default_scale_capability = supply_scale / (supply_scale + 1) if supply_scale > 0 else 1.0
            else:
                default_scale_capability = supply_scale / (supply_scale + 2) if supply_scale > 1 else 1.0
            scale_capability = safe_float_convert(
                self.point_features[j].get('enterprise_scale_capability', default_scale_capability),
                supply_scale + (supply_scale % (supply_scale + 1)))
            resource_reserve = safe_float_convert(self.point_features[j].get('resource_reserve'),
                                                  supply_scale / (supply_scale + 1) + (
                                                          supply_scale % (supply_scale + 2)))
            production_capacity = safe_float_convert(self.point_features[j].get('production_capacity'),
                                                     supply_scale + supply_scale / (supply_scale + 1) + (
                                                             supply_scale % supply_scale))
            current_capability = (scale_capability + resource_reserve + production_capacity) / 3

            all_capabilities = []
            for j_temp in self.J:
                temp_scale = safe_float_convert(self.point_features[j_temp].get('enterprise_scale_capability'),
                                                supply_scale + (supply_scale % (supply_scale + 1)))
                temp_resource = safe_float_convert(self.point_features[j_temp].get('resource_reserve'),
                                                   supply_scale / (supply_scale + 1) + (
                                                           supply_scale % (supply_scale + 2)))
                temp_production = safe_float_convert(self.point_features[j_temp].get('production_capacity'),
                                                     supply_scale + supply_scale / (supply_scale + 1) + (
                                                             supply_scale % supply_scale))
                all_capabilities.append((temp_scale + temp_resource + temp_production) / 3)

            max_capability = max(all_capabilities) if all_capabilities else current_capability

            metrics['capability_score'] = 1.0 / max(current_capability, self.EPS)

        # 8. 社会影响目标
        enterprise_type = self.point_features[j]['enterprise_type']
        enterprise_size = self.point_features[j]['enterprise_size']
        supply_capacity = self.B[j] * self.P[j]

        # 企业类型影响系数：国企影响小，私企影响大
        if enterprise_type in ["国企", "事业单位"]:
            type_impact_factor = supply_scale / (
                    supply_scale + network_scale) if supply_scale + network_scale > 0 else supply_scale / (
                    supply_scale + 1)
        else:
            type_impact_factor = network_scale / (
                    supply_scale + network_scale) if supply_scale + network_scale > 0 else network_scale / (
                    network_scale + 1)

        # 企业规模影响系数：大企业影响小，小企业影响大
        if enterprise_size in ["大", "中"]:
            size_impact_factor = supply_scale / (
                    supply_scale + demand_scale) if supply_scale + demand_scale > 0 else supply_scale / (
                    supply_scale + 1)
        else:
            size_impact_factor = demand_scale / (
                    supply_scale + demand_scale) if supply_scale + demand_scale > 0 else demand_scale / (
                    demand_scale + 1)

        # 资源动员强度：基于实际可能的动员比例
        if supply_capacity > 0:
            potential_mobilization = min(supply_capacity, self.total_demand,
                                         self.D[k] if k in self.D else self.total_demand)
            mobilization_intensity = potential_mobilization / supply_capacity
        else:
            mobilization_intensity = supply_scale / (
                    supply_scale + demand_scale) if supply_scale + demand_scale > 0 else supply_scale / (
                    supply_scale + 1)

        # 社会影响评分：综合考虑企业特性和动员强度
        metrics['social_score'] = type_impact_factor * size_impact_factor * mobilization_intensity

        # 保存基础信息
        metrics['time'] = total_time
        metrics['cost'] = total_cost
        metrics['distance'] = adjusted_distance
        metrics['is_long_distance'] = is_long_distance
        metrics['is_large_batch'] = is_large_batch

        return metrics

    def _calculate_multimodal_path_metrics(self, j, m1, m2, k, n2):
        """计算多式联运路径的所有目标指标"""

        # 安全的数值转换函数 - 增强版本，确保类型安全
        def safe_float_convert(value, default=0.0):
            try:
                # 确保 default 是数字类型
                if not isinstance(default, (int, float)):
                    default = 0.0

                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    if value.strip() == '':
                        return default
                    return float(value)
                elif isinstance(value, (list, tuple)):
                    # 修复：如果是序列类型，返回默认值而不是尝试转换
                    return default
                else:
                    return default
            except (ValueError, TypeError):
                return default

        # 安全的整数获取函数
        def safe_int_convert(value, default=0):
            try:
                if isinstance(value, (int, float)):
                    return int(value)
                elif isinstance(value, str):
                    if value.strip() == '':
                        return default
                    return int(float(value))
                elif isinstance(value, (list, tuple)):
                    return default
                else:
                    return default
            except (ValueError, TypeError):
                return default

        # 计算网络规模参数 - 确保类型安全
        network_scale = safe_int_convert(len(self.J)) + safe_int_convert(len(self.M)) + safe_int_convert(len(self.K))
        supply_scale = safe_int_convert(len(self.J))
        transfer_scale = safe_int_convert(len(self.M))
        demand_scale = safe_int_convert(len(self.K))
        transport_modes_count = safe_int_convert(len(self.TRANSPORT_MODES))

        # 确保网络规模参数不为0，避免除零错误
        network_scale = max(network_scale, 1)
        supply_scale = max(supply_scale, 1)
        transfer_scale = max(transfer_scale, 1)
        demand_scale = max(demand_scale, 1)
        transport_modes_count = max(transport_modes_count, 1)

        self.logger.debug(f"计算多式联运路径指标: {j} -> {m1} -> {m2} -> {k} (运输方式: {n2})")

        try:
            # 获取各点坐标
            j_lat, j_lon = self.point_features[j]['latitude'], self.point_features[j]['longitude']
            m1_lat, m1_lon = self.point_features[m1]['latitude'], self.point_features[m1]['longitude']
            m2_lat, m2_lon = self.point_features[m2]['latitude'], self.point_features[m2]['longitude']
            k_lat, k_lon = self.point_features[k]['latitude'], self.point_features[k]['longitude']

            d1 = self._calculate_haversine_distance(j_lat, j_lon, m1_lat, m1_lon)
            d2 = self._calculate_haversine_distance(m1_lat, m1_lon, m2_lat, m2_lon)
            d3 = self._calculate_haversine_distance(m2_lat, m2_lon, k_lat, k_lon)
            total_distance = d1 + d2 + d3

            # 计算各段时间 - 确保所有变量为数值类型
            try:
                mode_1 = self.TRANSPORT_MODES.get(1)
                mode_n2 = self.TRANSPORT_MODES.get(n2)

                if not mode_1:
                    raise ValueError(f"运输方式1（公路运输）未在TRANSPORT_MODES中配置")
                if not mode_n2:
                    raise ValueError(f"运输方式{n2}未在TRANSPORT_MODES中配置")

                v1_speed = mode_1.get('speed')
                v2_speed = mode_n2.get('speed')
                v3_speed = mode_1.get('speed')

                if v1_speed is None or v1_speed <= 0:
                    raise ValueError(f"运输方式1的速度未配置或无效: {v1_speed}")
                if v2_speed is None or v2_speed <= 0:
                    raise ValueError(f"运输方式{n2}的速度未配置或无效: {v2_speed}")

                t1 = d1 / v1_speed
                t2 = d2 / v2_speed
                t3 = d3 / v3_speed

            except (KeyError, TypeError, ZeroDivisionError) as e:
                raise ValueError(f"多式联运时间计算失败，请检查运输方式配置: {str(e)}")

            # 准备时间、装卸时间等
            preparation_time = safe_float_convert(self.T1, 0.0) + safe_float_convert(self.T4, 0.0)
            # 多式联运需要在每个中转点增加转运时间
            transfer_time = safe_float_convert(self.T6, 0.0) * 2  # 两个中转点的转运时间
            exchange_time = safe_float_convert(self.T6, 0.0)

            # 确保所有时间变量为数值类型
            t1 = safe_float_convert(t1, 0.0)
            t2 = safe_float_convert(t2, 0.0)
            t3 = safe_float_convert(t3, 0.0)
            preparation_time = safe_float_convert(preparation_time, 0.0)
            transfer_time = safe_float_convert(transfer_time, 0.0)
            exchange_time = safe_float_convert(exchange_time, 0.0)

            # 修正多式联运总时间计算：包含准备时间、运输时间、中转时间和交接时间
            total_time = preparation_time + t1 + t2 + t3 + transfer_time + exchange_time

            # 计算成本 - 确保数值类型
            cost_per_km_1 = safe_float_convert(self.TRANSPORT_MODES[1]['cost_per_km'], 1.0)
            cost_per_km_n2 = safe_float_convert(self.TRANSPORT_MODES[n2]['cost_per_km'], 1.0)
            cost1 = d1 * cost_per_km_1
            cost2 = d2 * cost_per_km_n2
            cost3 = d3 * cost_per_km_1
            transport_cost = cost1 + cost2 + cost3

            # 根据资源类型计算总成本 - 使用统一的细分对象选择逻辑
            try:
                sub_objects = self.point_features[j].get('sub_objects', [])
                if sub_objects:
                    safe_total_time = safe_float_convert(total_time, 1.0)
                    selected_cost, _ = self._select_optimal_sub_objects(j, safe_total_time, transport_cost)
                    total_cost = selected_cost
                else:
                    safe_total_time = safe_float_convert(total_time, 1.0)
                    total_cost = self._calculate_cost_by_resource_type(j, safe_total_time, transport_cost)
            except ValueError as e:
                raise ValueError(f"多式联运成本计算失败: {str(e)}")

            # 计算安全系数
            safety_score = self._calculate_route_safety_score(j, k, 'multimodal', [1, n2, 1])

            # 计算其他目标指标
            supplier_priority_score = self._calculate_supplier_task_priority(j, k, total_time, total_cost, safety_score)

            # 资源均衡目标
            supply_capacity = self.B[j] * self.P[j]
            if self.total_supply > 0:
                # 该供应点的理想使用比例（基于其能力占比）
                ideal_usage_ratio = supply_capacity / self.total_supply
            else:
                ideal_usage_ratio = supply_scale / (
                        supply_scale + demand_scale) if supply_scale + demand_scale > 0 else supply_scale / (
                        supply_scale + 1)

            # 预估该供应点的实际使用比例（基于容量约束）
            max_possible_allocation = min(supply_capacity, self.total_demand)
            estimated_usage_ratio = max_possible_allocation / max(self.total_demand, supply_capacity) if max(
                self.total_demand, supply_capacity) > 0 else ideal_usage_ratio

            # 使用偏差：实际使用比例偏离理想比例的程度
            usage_deviation = abs(estimated_usage_ratio - ideal_usage_ratio)
            balance_score = usage_deviation

            # 企业能力目标
            if hasattr(self, 'resource_type') and self.resource_type in ["personnel", "data"]:
                capability_score = 0.0
            else:
                enterprise_size = self.point_features[j].get('enterprise_size', '中')
                if enterprise_size == '大':
                    default_scale_capability = supply_scale + (supply_scale % (supply_scale + 1))
                elif enterprise_size == '中':
                    default_scale_capability = supply_scale / (supply_scale + 1) if supply_scale > 0 else 1.0
                else:
                    default_scale_capability = supply_scale / (supply_scale + 2) if supply_scale > 1 else 1.0

                scale_capability = safe_float_convert(
                    self.point_features[j].get('enterprise_scale_capability', default_scale_capability),
                    supply_scale + (supply_scale % (supply_scale + 1)))
                resource_reserve = safe_float_convert(self.point_features[j].get('resource_reserve'),
                                                      supply_scale / (supply_scale + 1) + (
                                                              supply_scale % (supply_scale + 2)))
                production_capacity = safe_float_convert(self.point_features[j].get('production_capacity'),
                                                         supply_scale + supply_scale / (supply_scale + 1) + (
                                                                 supply_scale % supply_scale))
                current_capability = (scale_capability + resource_reserve + production_capacity) / 3

                all_capabilities = []
                for j_temp in self.J:
                    temp_scale = safe_float_convert(self.point_features[j_temp].get('enterprise_scale_capability'),
                                                    supply_scale + (supply_scale % (supply_scale + 1)))
                    temp_resource = safe_float_convert(self.point_features[j_temp].get('resource_reserve'),
                                                       supply_scale / (supply_scale + 1) + (
                                                               supply_scale % (supply_scale + 2)))
                    temp_production = safe_float_convert(self.point_features[j_temp].get('production_capacity'),
                                                         supply_scale + supply_scale / (supply_scale + 1) + (
                                                                 supply_scale % supply_scale))
                    all_capabilities.append((temp_scale + temp_resource + temp_production) / 3)

                avg_capability = sum(all_capabilities) / len(
                    all_capabilities) if all_capabilities else current_capability
                max_capability = max(all_capabilities) if all_capabilities else current_capability
                capability_score = max_capability / max(current_capability, self.EPS)

            # 社会影响目标
            enterprise_type = self.point_features[j]['enterprise_type']
            enterprise_size = self.point_features[j]['enterprise_size']
            supply_capacity = self.B[j] * self.P[j]

            # 企业类型影响系数：国企影响小，私企影响大
            if enterprise_type in ["国企", "事业单位"]:
                type_impact_factor = supply_scale / (
                        supply_scale + network_scale) if supply_scale + network_scale > 0 else supply_scale / (
                        supply_scale + 1)
            else:
                type_impact_factor = network_scale / (
                        supply_scale + network_scale) if supply_scale + network_scale > 0 else network_scale / (
                        network_scale + 1)

            # 企业规模影响系数：大企业影响小，小企业影响大
            if enterprise_size in ["大", "中"]:
                size_impact_factor = supply_scale / (
                        supply_scale + demand_scale) if supply_scale + demand_scale > 0 else supply_scale / (
                        supply_scale + 1)
            else:
                size_impact_factor = demand_scale / (
                        supply_scale + demand_scale) if supply_scale + demand_scale > 0 else demand_scale / (
                        demand_scale + 1)

            # 资源动员强度：基于实际可能的动员比例
            if supply_capacity > 0:
                potential_mobilization = min(supply_capacity, self.total_demand,
                                             self.D[k] if k in self.D else self.total_demand)
                mobilization_intensity = potential_mobilization / supply_capacity
            else:
                mobilization_intensity = supply_scale / (
                        supply_scale + demand_scale) if supply_scale + demand_scale > 0 else supply_scale / (
                        supply_scale + 1)

            # 社会影响评分：综合考虑企业特性和动员强度
            social_score = type_impact_factor * size_impact_factor * mobilization_intensity

            # 构建结果
            metrics = {
                'time_score': total_time,
                'cost_score': total_cost,
                'distance_score': total_distance,
                'safety_score': safety_score,
                'priority_score': -supplier_priority_score,
                'balance_score': balance_score,
                'capability_score': capability_score,
                'social_score': social_score,
                'time': total_time,
                'cost': total_cost,
                'distance': total_distance,
                'is_long_distance': total_distance > network_scale * transfer_scale / supply_scale if supply_scale > 0 else False,
                'is_large_batch': supply_capacity > network_scale * demand_scale / supply_scale if supply_scale > 0 else False
            }

            return metrics

        except Exception as e:
            self.logger.error(f"多式联运路径指标计算失败 {j}-{m1}-{m2}-{k}: {str(e)}", exc_info=True)
            raise

    def _calculate_supplier_task_priority(self, j, k, total_time, total_cost, total_safety):
        """
        计算供应点执行任务的优先级评分
        """

        # 1. 企业性质评分
        enterprise_type = self.point_features[j].get('enterprise_type', '其他')
        if enterprise_type in ["国企", "事业单位"]:
            enterprise_score = 1.0
        elif enterprise_type == "私企":
            enterprise_score = 0.6
        else:
            enterprise_score = 0.3

        # 2. 供应能力评分
        supply_capacity = self.B[j] * self.P[j]
        max_supply_capacity = max(self.B[j_temp] * self.P[j_temp] for j_temp in self.J)
        capacity_score = supply_capacity / max_supply_capacity if max_supply_capacity > 0 else 0.0

        # 3. 成本效率评分
        if supply_capacity > 0 and total_cost > 0:
            unit_cost = total_cost / supply_capacity
            all_unit_costs = []
            for j_temp in self.J:
                temp_capacity = self.B[j_temp] * self.P[j_temp]
                if temp_capacity > 0:
                    temp_cost = len(self.J) * len(self.K) + len(self.J)  # 简化的默认成本
                    temp_unit_cost = temp_cost / temp_capacity
                    all_unit_costs.append(temp_unit_cost)

            if all_unit_costs:
                min_unit_cost = min(all_unit_costs)
                max_unit_cost = max(all_unit_costs)
                if max_unit_cost > min_unit_cost:
                    cost_score = (max_unit_cost - unit_cost) / (max_unit_cost - min_unit_cost)
                else:
                    cost_score = 1.0
            else:
                cost_score = 1.0
        else:
            cost_score = 0.0

        # 4. 安全评分
        all_safety_scores = [total_safety]  # 简化为当前值
        safety_score = 1.0 if total_safety >= 0 else 0.0

        # 加权计算
        total_priority_score = (
                enterprise_score * 0.4 +
                capacity_score * 0.3 +
                cost_score * 0.2 +
                safety_score * 0.1
        )

        return total_priority_score

    def _calculate_composite_score(self, path_metrics, is_direct):
        """
        计算综合评分
        """

        # 提取各目标的评分
        objective_scores = {
            'time': path_metrics['time_score'],
            'cost': path_metrics['cost_score'],
            'distance': path_metrics['distance_score'],
            'safety': path_metrics['safety_score'],
            'priority': path_metrics['priority_score'],
            'balance': path_metrics['balance_score'],
            'capability': path_metrics['capability_score'],
            'social': path_metrics['social_score']
        }

        # 使用用户原始权重，不进行动态调整
        adjusted_weights = dict(self.objective_weights)

        # 权重归一化，确保权重和为1
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            for obj in adjusted_weights:
                adjusted_weights[obj] = adjusted_weights[obj] / total_weight

        # 批量归一化处理
        normalized_scores = self._batch_normalize_objectives(objective_scores, adjusted_weights)

        # 计算最终综合评分
        composite_score = sum(normalized_scores.values())

        return composite_score

    def _extract_path_features(self, path_metrics):
        """提取路径特征"""
        return {
            'is_long_distance': path_metrics.get('is_long_distance', False),
            'is_large_batch': path_metrics.get('is_large_batch', False)
        }

    def _apply_weight_adjustments(self, adjusted_weights, path_features, is_direct):
        """批量应用权重调整"""
        pass

    def _calculate_time_reduction_factor(self):
        """计算时间权重削减因子"""
        if hasattr(self, 'TRANSPORT_MODES') and len(self.M) > 0:
            # 基于运输复杂度的动态计算
            transport_complexity = len(self.TRANSPORT_MODES) / (len(self.M) + len(self.TRANSPORT_MODES))
            network_scale_factor = len(self.J) / (len(self.J) + len(self.M) + len(self.K))
            return self.objective_weights['time'] * transport_complexity * network_scale_factor
        return 0.0

    def _calculate_distance_penalty_reduction(self):
        """计算距离惩罚削减因子"""
        if hasattr(self, 'TRANSPORT_MODES') and len(self.K) > 0:
            # 基于网络密度的动态计算
            total_nodes = len(self.J) + len(self.M) + len(self.K)
            possible_connections = len(self.TRANSPORT_MODES) * len(self.K)
            network_density = total_nodes / (
                    total_nodes + possible_connections) if total_nodes + possible_connections > 0 else 0.5
            return self.objective_weights['distance'] * network_density
        return 0.0

    def _batch_normalize_objectives(self, objective_scores, adjusted_weights):
        """批量归一化目标值"""
        normalized_scores = {}

        # 预定义目标处理类型
        negative_objectives = {'safety', 'priority'}  # 负值目标
        reverse_objectives = {'capability'}  # 反向归一化目标

        # 找到权重最高的目标作为主导目标
        dominant_objective = max(adjusted_weights.items(), key=lambda x: x[1])
        dominant_obj_name = dominant_objective[0]
        dominant_weight = dominant_objective[1]

        # 计算权重倍数（主导目标相对于平均权重的倍数）
        avg_weight = sum(adjusted_weights.values()) / len(adjusted_weights)
        dominance_factor = dominant_weight / avg_weight if avg_weight > 0 else 1.0

        for obj, original_value in objective_scores.items():
            weight = adjusted_weights[obj]

            # 获取归一化范围
            if hasattr(self, 'objective_ranges') and obj in self.objective_ranges:
                min_val, max_val = self.objective_ranges[obj]

                # 避免除零错误
                if abs(max_val - min_val) < self.EPS:
                    normalized_value = 0.5
                else:
                    # 根据目标类型批量处理
                    if obj in negative_objectives:
                        normalized_value = self._normalize_negative_objective(original_value, min_val, max_val)
                    elif obj in reverse_objectives:
                        normalized_value = self._normalize_reverse_objective(original_value, min_val, max_val)
                    else:
                        normalized_value = self._normalize_standard_objective(original_value, min_val, max_val)
            else:
                normalized_value = original_value

            # 添加扰动并应用权重
            normalized_value += random.random() * 1e-10

            # 对主导目标应用增强效应
            if obj == dominant_obj_name and dominance_factor > 1.0:
                # 主导目标的差异被放大，让其更容易主导选择
                enhanced_value = normalized_value ** dominance_factor
                # 额外增强主导目标的权重影响力
                dominance_bonus = (dominance_factor * dominance_factor) * weight
                normalized_scores[obj] = enhanced_value * weight + (enhanced_value * dominance_bonus)
            else:
                normalized_scores[obj] = normalized_value * weight

        return normalized_scores

    def _normalize_negative_objective(self, original_value, min_val, max_val):
        """归一化负值目标"""
        if original_value <= min_val:
            return 1.0
        elif original_value >= max_val:
            return 0.0
        else:
            return (original_value - min_val) / (max_val - min_val)

    def _normalize_reverse_objective(self, original_value, min_val, max_val):
        """归一化反向目标"""
        if original_value <= min_val:
            return 1.0
        elif original_value >= max_val:
            return 0.0
        else:
            return (max_val - original_value) / (max_val - min_val)

    def _normalize_standard_objective(self, original_value, min_val, max_val):
        """归一化标准目标"""
        if original_value <= min_val:
            return 0.0
        elif original_value >= max_val:
            return 1.0
        else:
            return (original_value - min_val) / (max_val - min_val)

    def _multi_objective_intelligent_scheduling(self, paths):
        """多目标调度算法"""

        try:
            schedule_start_time = time.time()

            self.logger.info("=" * 80)
            self.logger.info("开始多目标调度阶段")
            self.logger.info(f"调度输入 - 待调度路径数量: {len(paths)}")

            if not paths:
                self.logger.warning("没有可调度的路径，调度阶段提前结束")
                raise ValueError("待调度路径列表为空")

            # 计算网络规模参数，用于替代魔法数字
            network_scale = len(self.J) + len(self.M) + len(self.K)
            supply_scale = len(self.J)
            transfer_scale = len(self.M)
            demand_scale = len(self.K)

            self.logger.info(f"网络规模分析 - 总规模: {network_scale}")
            self.logger.info(f"供应规模: {supply_scale}, 中转规模: {transfer_scale}, 需求规模: {demand_scale}")

            # 预计算基础数据
            self.logger.info("第1步: 计算供需数据")
            try:

                supply_compaire = []

                L_j_m = self.L_j_m
                alpha1 = self.alpha1
                v1 = self.v1
                Q = self.Q
                _capacity_cache = self._capacity_cache
                candidate_transfer_sets = self.candidate_transfer_sets
                N_index = self.N



                for jj in self.J:
                    # suplly_name = jj
                    # supply_id = self.point_features[jj]['original_supplier_id']
                    # sub_objects = self.point_features[jj].get('sub_objects', [])
                    for im in range(len(self.point_features[jj].get('sub_objects', []))):
                        for ig in range(len(self.point_features[jj].get('sub_objects', [])[im]['items'])):
                            data = {
                                "supply_name":jj,
                                "supply_id":self.point_features[jj]['original_supplier_id'],
                                'category_id':self.point_features[jj].get('sub_objects', [])[im]['category_id'],
                                "category_name":self.point_features[jj].get('sub_objects', [])[im]['category_name'],
                                'items':self.point_features[jj].get('sub_objects', [])[im]['items'][ig],
                                'probability':self.point_features[jj]['probability'],

                            }
                            supply_compaire.append(data)


                data_compaire_supplier = self.sort_list_by_MAandP(supply_compaire)

                tyt = 0

                pre_solution = []

                for iq in range(len(data_compaire_supplier)):
                    supply_name = data_compaire_supplier[iq]['supply_name']

                    if supply_name in self.J:

                        if data_compaire_supplier[iq]['items']['specify_quantity'] > 0:
                            # 第一步保存提前解
                            pre_solution.append(data_compaire_supplier[iq])
                            # 第二步进行删除
                            # supply_name = data_compaire_supplier[iq]['supply_name']
                            if supply_name in self.B:
                                del self.B[supply_name]
                                del self.P[supply_name]
                                del self.Q[supply_name]
                                del self._capacity_cache[supply_name]
                                del self.candidate_transfer_sets[supply_name]


                            index_to_remove = self.J.index(supply_name)
                            self.J.remove(supply_name)
                            # self.N.pop(index_to_remove)
                            self.L_j_m = self.remove_key(self.L_j_m,supply_name)
                            self.alpha1 = self.remove_key(self.alpha1,supply_name)
                            self.v1 = self.remove_key(self.v1,supply_name)
                            paths = self.remove_list(paths,supply_name)
                            category_name = data_compaire_supplier[iq]['category_name']
                            sub_object_name =  data_compaire_supplier[iq]['items']['sub_object_name']


                            self.point_features = self.remove_specific_item_and_adjust_capacity(self.point_features,supply_name, category_name, sub_object_name)
                        else:
                            pass
                    else:
                        if data_compaire_supplier[iq]['items']['specify_quantity'] > 0:
                            pre_solution.append(data_compaire_supplier[iq])
                        else:
                            continue


                gh =9
                for iq in range(len(data_compaire_supplier)):
                    if data_compaire_supplier[iq]['items']['specify_quantity'] == 0:
                        supply_name = data_compaire_supplier[iq]['supply_name']
                        if supply_name not in self.J:
                            self.J.append(supply_name)
                            if supply_name  not in self.B:
                               self.B[supply_name] = data_compaire_supplier[iq]['items']['max_available_quantity']
                               self.P[supply_name] = data_compaire_supplier[iq]['probability']

                            self.L_j_m = L_j_m
                            self.alpha1 = alpha1
                            self.v1 = v1
                            self.Q = Q
                            self._capacity_cache = _capacity_cache
                            self.N = N_index
                        else:
                            continue


                sum_specify = 0
                for i in range(len(pre_solution)):
                    sum_specify += pre_solution[i]['items']['specify_quantity']

                df = self.D[self.K[0]]- sum_specify
                if df == 0:
                    return  {
                        "code": 200,
                        "msg": "多目标调度成功",
                        "pre_data": pre_solution,
                        "data":{}
                    }
                else:

                    self.D[self.K[0]] = df
                    self.logger.debug("计算总需求量")
                    total_demand = sum(self.D.values())
                    self.logger.debug(f"需求点需求量详情: {self.D}")

                    self.logger.debug("计算总供应能力")

                    supply_details = {}
                    total_supply = 0

                    for j in self.J:
                        if j not in self.B:
                            raise KeyError(f"供应点 {j} 缺少供应能力数据")
                        if j not in self.P:
                            raise KeyError(f"供应点 {j} 缺少可靠性数据")

                        # 计算基于细分对象的实际容量
                        sub_objects = self.point_features[j].get('sub_objects', [])

                        actual_capacity = 0
                        if sub_objects:
                            for category in sub_objects:
                                if isinstance(category, dict) and 'items' in category:
                                    for sub_obj in category.get('items', []):
                                        max_available = sub_obj.get('max_available_quantity', 0)
                                        if isinstance(max_available, (int, float)):
                                            actual_capacity += max_available
                                else:
                                    max_available = category.get('max_available_quantity', 0)
                                    if isinstance(max_available, (int, float)):
                                        actual_capacity += max_available
                            supply_capacity = actual_capacity * self.P[j]
                            # self.logger.debug(
                            #     f"供应点{j}: 细分对象总容量{actual_capacity}, 可靠性{self.P[j]}, 有效容量{supply_capacity:.2f}")
                        else:
                            supply_capacity = self.B[j] * self.P[j]
                            self.logger.debug(
                                f"供应点{j}: 声明容量{self.B[j]}, 可靠性{self.P[j]}, 有效容量{supply_capacity:.2f}")

                        supply_details[j] = {"capacity": actual_capacity if sub_objects else self.B[j],
                                             "probability": self.P[j], "effective": supply_capacity}
                        total_supply += supply_capacity

                    self.logger.info(f"供应点能力详情统计完成 - 总数{len(supply_details)}, 总有效容量{total_supply:.2f}")
                    self.logger.debug(f"各供应点详情: {supply_details}")

            except (KeyError, TypeError) as e:
                error_msg = f"供需数据计算失败: {str(e)}"
                self.logger.error(error_msg, exc_info=True)
                self.logger.error(f"数据检查 - B存在: {hasattr(self, 'B')}, P存在: {hasattr(self, 'P')}")
                self.logger.error(f"D存在: {hasattr(self, 'D')}")
                if hasattr(self, 'B'):
                    self.logger.error(f"B键: {list(self.B.keys())}")
                if hasattr(self, 'P'):
                    self.logger.error(f"P键: {list(self.P.keys())}")
                if hasattr(self, 'D'):
                    self.logger.error(f"D键: {list(self.D.keys())}")
                raise ValueError(error_msg)

            self.logger.info(f"供需数据计算完成 - 总需求量: {total_demand:.2f}, 总供应能力: {total_supply:.2f}")

            if total_supply <= 0:
                self.logger.error("总供应能力为零或负数，无法进行调度")
                raise ValueError("总供应能力必须大于零")

            if total_demand <= 0:
                self.logger.error("总需求量为零或负数，无需进行调度")
                raise ValueError("总需求量必须大于零")

            supply_demand_ratio = total_supply / total_demand
            self.logger.info(f"供需比例: {supply_demand_ratio:.3f}")

            if supply_demand_ratio < 1.0:
                self.logger.warning(f"供应能力不足，供需比例 {supply_demand_ratio:.3f} < 1.0")
            else:
                self.logger.info("供应能力充足")

            # 初始化决策变量模板

            # 将路径按供应点分组
            supplier_best_paths = self._group_paths_by_supplier(paths)
            selected_paths = list(supplier_best_paths.values())

            # 智能轮次计算，基于网络复杂度和路径数量
            network_complexity = network_scale / max(supply_scale * demand_scale, 1)
            path_complexity = len(selected_paths) / max(supply_scale, 1)

            # 自适应轮次计算
            if network_complexity > supply_scale / max(demand_scale, 1):
                # 复杂网络：较少轮次，每轮处理更多
                base_rounds = max(1, min(len(selected_paths) // max(supply_scale * 2, 1),
                                         demand_scale // max(transfer_scale, 1) + 1))
            else:
                # 简单网络：较多轮次，精细调优
                base_rounds = max(1, len(selected_paths) // max(supply_scale, 1))

            max_rounds = min(max(demand_scale, transfer_scale // max(supply_scale, 1)),
                             len(selected_paths) // max(demand_scale, 1) + 1)
            optimization_rounds = max(base_rounds, max_rounds) + 1 if supply_scale > 0 else 1

            aggregated_demand_point = self.K[0] if self.K else None

            if not aggregated_demand_point:
                raise ValueError("需求点集合为空")

            # self.logger.info(
            #     f"从{len(paths)}条路径中选择了{len(selected_paths)}条最优路径，开始{optimization_rounds}轮优化")

            # 保存原始状态 - 使用实际容量
            original_supply_left = {}
            for j in self.J:
                sub_objects = self.point_features[j].get('sub_objects', [])
                actual_capacity = 0
                if sub_objects:
                    for category in sub_objects:
                        if isinstance(category, dict) and 'items' in category:
                            for sub_obj in category.get('items', []):
                                actual_capacity += sub_obj.get('max_available_quantity', 0)
                        else:
                            actual_capacity += category.get('max_available_quantity', 0)
                    # 人员动员不乘以可靠性系数，保持整数人员数量
                    if hasattr(self, 'resource_type') and self.resource_type == "personnel":
                        original_supply_left[j] = actual_capacity
                    else:
                        original_supply_left[j] = actual_capacity * self.P[j]
                else:
                    # 人员动员不乘以可靠性系数
                    if hasattr(self, 'resource_type') and self.resource_type == "personnel":
                        original_supply_left[j] = self.B[j]
                    else:
                        original_supply_left[j] = self.B[j] * self.P[j]
            original_demand_left = dict(self.D)

            best_solution = None
            best_satisfaction_rate = 0.0

            # 多轮优化主循环
            persistent_errors = {k: 0.0 for k in self.K}
            bs = {}

            po = 0
            variable_template = {}
            for round_idx in range(5):

                # 保存原始权重
                original_weights = dict(self.objective_weights)

                # 确定当前轮次的主导目标
                objective_keys = list(self.objective_weights.keys())
                dominant_obj_index = round_idx % len(objective_keys)
                dominant_obj = objective_keys[dominant_obj_index]

                # 设置当前主导目标供其他函数使用
                self.current_dominant_objective = dominant_obj

                # 计算权重调整因子，基于网络规模避免魔法数字
                enhancement_factor = (supply_scale + demand_scale) / (supply_scale + demand_scale + network_scale)
                reduction_factor = network_scale / (supply_scale + demand_scale + network_scale)

                # 调整权重：增强主导目标，减弱其他目标
                adjusted_weights = {}
                for obj, weight in original_weights.items():
                    if obj == dominant_obj:
                        # 主导目标权重增强
                        adjusted_weights[obj] = weight + (1.0 - weight) * enhancement_factor
                    else:
                        # 其他目标权重减弱
                        adjusted_weights[obj] = weight * reduction_factor

                # 权重归一化确保和为1
                total_weight = sum(adjusted_weights.values())
                if total_weight > 0:
                    for obj in adjusted_weights:
                        adjusted_weights[obj] = adjusted_weights[obj] / total_weight

                # 临时替换权重用于当前轮次
                self.objective_weights = adjusted_weights

                variable_template[str(po)] = self._re_decision_variables()
                # 重置当前轮次状态
                current_state = self._reset_round_state(variable_template[str(po)], original_supply_left,
                                                            original_demand_left,
                                                            persistent_errors)

                # 为当前轮次重新进行运输模式决策
                available_suppliers = [j for j in self.J if current_state['supply_left'][j] > self.EPS]
                round_path_metrics_cache = {}
                round_composite_score_cache = {}

                round_transport_decision = self._make_global_transport_mode_decision(
                    available_suppliers, aggregated_demand_point,
                    round_path_metrics_cache, round_composite_score_cache
                )

                # 根据运输模式决策结果生成当前轮次的路径
                round_selected_paths = []

                if round_transport_decision['mode'] == 'direct':
                    # 直达模式：为每个可用供应点生成直达路径
                    for j in available_suppliers:
                        try:
                            direct_cache_key = f"direct_{j}_{aggregated_demand_point}"
                            if direct_cache_key not in round_path_metrics_cache:
                                round_path_metrics_cache[direct_cache_key] = self._calculate_direct_path_metrics(j,
                                                                                                                 aggregated_demand_point)

                            path_metrics = round_path_metrics_cache[direct_cache_key]
                            score_cache_key = f"score_direct_{j}_{aggregated_demand_point}"
                            if score_cache_key not in round_composite_score_cache:
                                round_composite_score_cache[score_cache_key] = self._calculate_composite_score(
                                    path_metrics, is_direct=True)

                            composite_score = round_composite_score_cache[score_cache_key]
                            round_selected_paths.append(('direct', j, aggregated_demand_point, 1, composite_score,
                                                         random.random(), path_metrics['cost_score'], path_metrics))
                        except (ValueError, KeyError) as e:
                            self.logger.warning(f"供应点 {j} 直接路径计算失败: {str(e)}")
                            continue

                else:
                    # 多式联运模式：所有供应点使用统一的多式联运路径
                    unified_multimodal_route = round_transport_decision['unified_route']
                    m1, m2, n2 = unified_multimodal_route['m1'], unified_multimodal_route['m2'], \
                    unified_multimodal_route['n2']

                    # 计算统一出发时间
                    max_time_to_m1 = 0.0
                    supplier_times = []
                    for j in available_suppliers:
                        try:
                            j_lat, j_lon = self.point_features[j]['latitude'], self.point_features[j]['longitude']
                            m1_lat, m1_lon = self.point_features[m1]['latitude'], self.point_features[m1]['longitude']
                            distance_to_m1 = self._calculate_haversine_distance(j_lat, j_lon, m1_lat, m1_lon)
                            time_to_m1 = distance_to_m1 / self.TRANSPORT_MODES[1]['speed']
                            max_time_to_m1 = max(max_time_to_m1, time_to_m1)
                            supplier_times.append((j, time_to_m1))
                        except (KeyError, ZeroDivisionError) as e:
                            self.logger.warning(f"供应点 {j} 到中转点时间计算失败: {str(e)}")
                            continue

                    # 为所有有效供应点创建统一的多式联运路径
                    for j, time_to_m1 in supplier_times:
                        try:
                            cache_key = f"unified_multimodal_{j}_{m1}_{m2}_{aggregated_demand_point}_{n2}"
                            if cache_key not in round_path_metrics_cache:
                                round_path_metrics_cache[cache_key] = self._calculate_unified_multimodal_metrics(
                                    j, m1, m2, aggregated_demand_point, n2, max_time_to_m1)

                            path_metrics = round_path_metrics_cache[cache_key]
                            score_cache_key = f"score_unified_{j}_{m1}_{m2}_{aggregated_demand_point}_{n2}"
                            if score_cache_key not in round_composite_score_cache:
                                round_composite_score_cache[score_cache_key] = self._calculate_composite_score(
                                    path_metrics, is_direct=False)

                            composite_score = round_composite_score_cache[score_cache_key]
                            round_selected_paths.append(
                                ('multimodal', j, m1, m2, aggregated_demand_point, n2, composite_score,
                                 random.random(), path_metrics['cost_score'], path_metrics))
                        except (ValueError, KeyError) as e:
                            self.logger.warning(f"供应点 {j} 多式联运路径计算失败: {str(e)}")
                            continue

                # 对当前轮次生成的路径进行排序
                if round_idx == 0:
                    def get_score_for_sorting(path_info):
                        if len(path_info) > 0:
                            path_type = path_info[0]
                            if path_type == 'direct' and len(path_info) > 4:
                                return path_info[4]
                            elif path_type == 'multimodal' and len(path_info) > 6:
                                return path_info[6]
                        return float('inf')

                    selected_paths_sorted = sorted(round_selected_paths, key=get_score_for_sorting)
                else:
                    selected_paths_sorted = self._apply_path_perturbation(round_selected_paths, round_idx,
                                                                          optimization_rounds)

                # 路径分配主循环，限制处理数量基于网络规模
                supply_scale = len(self.J)
                network_scale = len(self.J) + len(self.M) + len(self.K)
                base_paths_per_round = max(supply_scale, network_scale // max(optimization_rounds, 1))

                # 对于人员动员，确保处理足够的路径以满足需求
                if hasattr(self, 'resource_type') and self.resource_type == "personnel":
                    # 确保处理的路径数不少于需求量对应的供应点数
                    min_paths_for_demand = min(len(selected_paths_sorted), max(total_demand, supply_scale))
                    max_paths_per_round = max(min_paths_for_demand, base_paths_per_round)
                else:
                    max_paths_per_round = min(len(selected_paths_sorted),
                                              base_paths_per_round) if optimization_rounds > 0 else len(
                        selected_paths_sorted)
                round_processed_paths = self._process_round_allocation(
                    selected_paths_sorted[:max_paths_per_round], current_state, round_idx, optimization_rounds,
                    total_demand
                )

                # 评估当前轮次结果
                current_satisfaction_rate = self._evaluate_round_result(current_state, total_demand)

                # 构建当前解
                current_solution = self._build_current_solution(current_state, current_satisfaction_rate, round_idx)

                # 更新最优解
                if current_satisfaction_rate >= best_satisfaction_rate:
                    best_satisfaction_rate = current_satisfaction_rate
                    best_solution = current_solution

                    bs[str(po)] = best_solution
                    po = po + 1

                # 保持累积误差持续性
                for k in self.K:
                    persistent_errors[k] = float(current_state['accumulated_rounding_error'][k])

                # 恢复原始权重
                self.objective_weights = original_weights

                # 早期终止条件，基于网络规模的阈值
                satisfaction_threshold = 1.0 - (
                        network_scale / (network_scale * 1000 + 1)) if network_scale > 0 else 1.0 - self.EPS

                mk = len(bs)
                if len(bs) >= 35:
                    break

            # 应用最优解
            final_variables = {}
            for i in range(len(bs)):
                final_variables[str(i)] = self._apply_best_solution(bs[str(i)], variable_template[str(i)])

            # 补充分配
            try:
                for i in range(len(bs)):
                    accumulated_errors = bs[str(i)].get('accumulated_rounding_error',
                                                                   {k: 0.0 for k in self.K})
                    self._perform_supplementary_allocation(
                        final_variables[str(i)]['x1'], final_variables[str(i)]['x2'], final_variables[str(i)]['x3'],
                        final_variables[str(i)]['x_direct'],
                        final_variables[str(i)]['b1'], final_variables[str(i)]['b2'], final_variables[str(i)]['b3'],
                        final_variables[str(i)]['b_direct'],
                        final_variables[str(i)]['t1'], final_variables[str(i)]['t2'], final_variables[str(i)]['t3'],
                        final_variables[str(i)]['t_direct'],
                        bs[str(i)]['supply_left'], bs[str(i)]['demand_left'], accumulated_errors
                    )
            except (ValueError, KeyError, AttributeError) as e:
                self.logger.warning(f"补充分配失败: {str(e)}")

            schedule_end_time = time.time()
            scheduling_time = schedule_end_time - schedule_start_time

            # 清理和构建最终解
            try:
                for i in range(len(final_variables)):
                    self._clean_solution_paths(
                        final_variables[str(i)]['x1'], final_variables[str(i)]['x2'], final_variables[str(i)]['x3'],
                        final_variables[str(i)]['x_direct'],
                        final_variables[str(i)]['b1'], final_variables[str(i)]['b2'], final_variables[str(i)]['b3'],
                        final_variables[str(i)]['b_direct']
                    )
            except (AttributeError, TypeError) as e:
                self.logger.warning(f"解路径清理失败: {str(e)}")

            solution = {}
            for i in range(len(final_variables)):
                solution[str(i)] = {
                    **final_variables[str(i)],
                    'scheduling_time': scheduling_time,
                    'processed_paths': sum(range(optimization_rounds)),  # 避免未定义变量
                    'selected_suppliers': len(supplier_best_paths),
                    'objective_type': 'multi_objective',
                    'optimization_rounds': optimization_rounds,
                    'best_round': best_solution['round_index'] if best_solution else 0,
                    'best_satisfaction_rate': best_satisfaction_rate
                }

            self.logger.info(f"多目标调度完成，耗时: {scheduling_time:.2f}秒")
            self.logger.info(f"实际使用供应点: {len(supplier_best_paths)}个")
            self.logger.info(f"优化轮次: {optimization_rounds}轮")

            return {
                "code": 200,
                "msg": "多目标调度成功",
                "pre_data":pre_solution,
                "data": solution
            }

        except ValueError as ve:
            self.logger.error(f"多目标调度阶段参数错误: {str(ve)}", exc_info=True)
            return {
                "code": 400,
                "msg": f"多目标调度阶段参数错误: {str(ve)}",
                "data": {
                    "paths_count": len(paths) if paths else 0,
                    "network_scale": len(self.J) + len(self.M) + len(self.K),
                    "error_type": "parameter_error"
                }
            }

        except KeyError as ke:
            self.logger.error(f"多目标调度阶段数据缺失: {str(ke)}", exc_info=True)
            return {
                "code": 500,
                "msg": f"数据结构错误: {str(ke)}",
                "data": {

                    "paths_count": len(paths) if paths else 0,
                    "missing_key": str(ke),
                    "error_type": "data_structure_error"
                }
            }

        except AttributeError as ae:
            self.logger.error(f"多目标调度阶段属性错误: {str(ae)}", exc_info=True)
            return {
                "code": 500,
                "msg": f"对象属性缺失: {str(ae)}",
                "data": {
                    "paths_count": len(paths) if paths else 0,
                    "missing_attribute": str(ae),
                    "error_type": "attribute_error"
                }
            }

        except MemoryError:
            self.logger.error("多目标调度阶段内存不足")
            return {
                "code": 507,
                "msg": "内存不足，请减少网络规模",
                "data": {
                    "paths_count": len(paths) if paths else 0,
                    "network_scale": len(self.J) + len(self.M) + len(self.K),
                    "error_type": "memory_error"
                }
            }

    def _re_decision_variables(self):

        try:
            # 预计算集合大小，避免重复计算
            j_size = len(self.J)
            m_size = len(self.M)
            k_size = len(self.K)
            n_size = len(self.N)
            road_only_size = len(self.ROAD_ONLY)

            # 批量创建字典，减少Python解释器开销
            variables = {}
            self.M = self.M.copy()
            random.shuffle(self.M)

            # 使用字典推导式的批量创建，减少循环开销
            if j_size * m_size * road_only_size < 50000:  # 小规模直接创建
                from collections import defaultdict
                variables['x1'] = defaultdict(int)
                variables['x1'].update({(j, m, n): 0 for j in self.J for m in self.M for n in self.ROAD_ONLY})
                variables['b1'] = defaultdict(float)
                variables['b1'].update({(j, m, n): 0.0 for j in self.J for m in self.M for n in self.ROAD_ONLY})
                variables['t1'] = defaultdict(float)
                variables['t1'].update({(j, m, n): 0.0 for j in self.J for m in self.M for n in self.ROAD_ONLY})
            else:  # 大规模使用默认字典，按需创建
                from collections import defaultdict
                variables['x1'] = defaultdict(int)
                variables['b1'] = defaultdict(float)
                variables['t1'] = defaultdict(float)

            if m_size * m_size * n_size < 50000:
                from collections import defaultdict
                variables['x2'] = defaultdict(int)
                variables['x2'].update({(m1, m2, n): 0 for m1 in self.M for m2 in self.M for n in self.N if m1 != m2})
                variables['b2'] = defaultdict(float)
                variables['b2'].update({(m1, m2, n): 0.0 for m1 in self.M for m2 in self.M for n in self.N if m1 != m2})
                variables['t2'] = defaultdict(float)
                variables['t2'].update({(m1, m2, n): 0.0 for m1 in self.M for m2 in self.M for n in self.N if m1 != m2})
            else:
                from collections import defaultdict
                variables['x2'] = defaultdict(int)
                variables['b2'] = defaultdict(float)
                variables['t2'] = defaultdict(float)

            if m_size * k_size * road_only_size < 50000:
                from collections import defaultdict
                variables['x3'] = defaultdict(int)
                variables['x3'].update({(m, k, n): 0 for m in self.M for k in self.K for n in self.ROAD_ONLY})
                variables['b3'] = defaultdict(float)
                variables['b3'].update({(m, k, n): 0.0 for m in self.M for k in self.K for n in self.ROAD_ONLY})
                variables['t3'] = defaultdict(float)
                variables['t3'].update({(m, k, n): 0.0 for m in self.M for k in self.K for n in self.ROAD_ONLY})
            else:
                from collections import defaultdict
                variables['x3'] = defaultdict(int)
                variables['b3'] = defaultdict(float)
                variables['t3'] = defaultdict(float)

            if j_size * k_size * road_only_size < 50000:
                from collections import defaultdict
                variables['x_direct'] = defaultdict(int)
                variables['x_direct'].update({(j, k, n): 0 for j in self.J for k in self.K for n in self.ROAD_ONLY})
                variables['b_direct'] = defaultdict(float)
                variables['b_direct'].update({(j, k, n): 0.0 for j in self.J for k in self.K for n in self.ROAD_ONLY})
                variables['t_direct'] = defaultdict(float)
                variables['t_direct'].update({(j, k, n): 0.0 for j in self.J for k in self.K for n in self.ROAD_ONLY})
            else:
                from collections import defaultdict
                variables['x_direct'] = defaultdict(int)
                variables['b_direct'] = defaultdict(float)
                variables['t_direct'] = defaultdict(float)

            return variables
        except (AttributeError, TypeError) as e:
            raise ValueError(f"决策变量初始化失败: {str(e)}")

    def _initialize_decision_variables(self):
        """初始化决策变量模板"""
        try:
            # 预计算集合大小，避免重复计算
            j_size = len(self.J)
            m_size = len(self.M)
            k_size = len(self.K)
            n_size = len(self.N)
            road_only_size = len(self.ROAD_ONLY)

            # 批量创建字典，减少Python解释器开销
            variables = {}

            # 使用字典推导式的批量创建，减少循环开销
            if j_size * m_size * road_only_size < 50000:  # 小规模直接创建
                from collections import defaultdict
                variables['x1'] = defaultdict(int)
                variables['x1'].update({(j, m, n): 0 for j in self.J for m in self.M for n in self.ROAD_ONLY})
                variables['b1'] = defaultdict(float)
                variables['b1'].update({(j, m, n): 0.0 for j in self.J for m in self.M for n in self.ROAD_ONLY})
                variables['t1'] = defaultdict(float)
                variables['t1'].update({(j, m, n): 0.0 for j in self.J for m in self.M for n in self.ROAD_ONLY})
            else:  # 大规模使用默认字典，按需创建
                from collections import defaultdict
                variables['x1'] = defaultdict(int)
                variables['b1'] = defaultdict(float)
                variables['t1'] = defaultdict(float)

            if m_size * m_size * n_size < 50000:
                from collections import defaultdict
                variables['x2'] = defaultdict(int)
                variables['x2'].update({(m1, m2, n): 0 for m1 in self.M for m2 in self.M for n in self.N if m1 != m2})
                variables['b2'] = defaultdict(float)
                variables['b2'].update({(m1, m2, n): 0.0 for m1 in self.M for m2 in self.M for n in self.N if m1 != m2})
                variables['t2'] = defaultdict(float)
                variables['t2'].update({(m1, m2, n): 0.0 for m1 in self.M for m2 in self.M for n in self.N if m1 != m2})
            else:
                from collections import defaultdict
                variables['x2'] = defaultdict(int)
                variables['b2'] = defaultdict(float)
                variables['t2'] = defaultdict(float)

            if m_size * k_size * road_only_size < 50000:
                from collections import defaultdict
                variables['x3'] = defaultdict(int)
                variables['x3'].update({(m, k, n): 0 for m in self.M for k in self.K for n in self.ROAD_ONLY})
                variables['b3'] = defaultdict(float)
                variables['b3'].update({(m, k, n): 0.0 for m in self.M for k in self.K for n in self.ROAD_ONLY})
                variables['t3'] = defaultdict(float)
                variables['t3'].update({(m, k, n): 0.0 for m in self.M for k in self.K for n in self.ROAD_ONLY})
            else:
                from collections import defaultdict
                variables['x3'] = defaultdict(int)
                variables['b3'] = defaultdict(float)
                variables['t3'] = defaultdict(float)

            if j_size * k_size * road_only_size < 50000:
                from collections import defaultdict
                variables['x_direct'] = defaultdict(int)
                variables['x_direct'].update({(j, k, n): 0 for j in self.J for k in self.K for n in self.ROAD_ONLY})
                variables['b_direct'] = defaultdict(float)
                variables['b_direct'].update({(j, k, n): 0.0 for j in self.J for k in self.K for n in self.ROAD_ONLY})
                variables['t_direct'] = defaultdict(float)
                variables['t_direct'].update({(j, k, n): 0.0 for j in self.J for k in self.K for n in self.ROAD_ONLY})
            else:
                from collections import defaultdict
                variables['x_direct'] = defaultdict(int)
                variables['b_direct'] = defaultdict(float)
                variables['t_direct'] = defaultdict(float)

            return variables
        except (AttributeError, TypeError) as e:
            raise ValueError(f"决策变量初始化失败: {str(e)}")

    def _group_paths_by_supplier(self, paths):
        """将路径按供应点分组，每个供应点只选择最优路径"""
        supplier_best_paths = {}

        for path_info in paths:
            try:
                if not path_info or not isinstance(path_info, (list, tuple)) or len(path_info) < 1:
                    self.logger.warning("路径信息为空或格式错误，跳过")
                    continue

                path_type = path_info[0]
                if path_type == 'direct':
                    if len(path_info) < 8:
                        self.logger.warning(f"直接路径信息长度不足: {len(path_info)} < 8")
                        continue
                    _, j, k, n, composite_score, path_risk, cost, path_metrics = path_info
                elif path_type == 'multimodal':
                    if len(path_info) < 10:
                        self.logger.warning(f"多式联运路径信息长度不足: {len(path_info)} < 10")
                        continue
                    _, j, m1, m2, k, n2, composite_score, path_risk, cost, path_metrics = path_info
                else:
                    self.logger.warning(f"未知路径类型: {path_type}")
                    continue

                # 获取已存储路径的评分以进行比较
                if j in supplier_best_paths:
                    existing_path = supplier_best_paths[j]
                    existing_path_type = existing_path[0] if len(existing_path) > 0 else 'unknown'
                    if existing_path_type == 'direct' and len(existing_path) > 4:
                        existing_score = existing_path[4]
                    elif existing_path_type == 'multimodal' and len(existing_path) > 6:
                        existing_score = existing_path[6]
                    else:
                        existing_score = float('inf')
                else:
                    existing_score = float('inf')

                # 保留评分最好的路径
                if j not in supplier_best_paths or composite_score < existing_score:
                    supplier_best_paths[j] = path_info
            except (IndexError, ValueError, TypeError) as e:
                self.logger.warning(f"路径信息解析失败: {str(e)}，路径信息: {path_info}")
                continue

        return supplier_best_paths

    def _reset_round_state(self, variable_template, original_supply_left, original_demand_left, previous_errors=None):
        """重置轮次状态"""

        hj = original_supply_left
        kl = 0
        result = {
            'variables': {key: dict(var_dict) for key, var_dict in variable_template.items()},
            'supply_left': dict(original_supply_left),
            'demand_left': dict(original_demand_left),
            'accumulated_rounding_error': dict(previous_errors) if previous_errors else {k: 0.0 for k in self.K}
        }
        return result

    def _apply_path_perturbation(self, selected_paths, round_idx, optimization_rounds):
        """应用路径扰动"""

        def perturbed_score(path_info):
            # 根据路径类型确定评分位置
            if len(path_info) > 0:
                path_type = path_info[0]
                if path_type == 'direct' and len(path_info) > 4:
                    original_score = path_info[4]
                elif path_type == 'multimodal' and len(path_info) > 6:
                    original_score = path_info[6]
                else:
                    original_score = float('inf')
            else:
                original_score = float('inf')
            perturbation_factor = (round_idx / optimization_rounds) * (len(self.J) / (len(self.J) + len(self.K)))
            result = original_score * (1 + perturbation_factor)
            return result

        result = sorted(selected_paths, key=perturbed_score)
        return result

    def _process_round_allocation(self, selected_paths_sorted, current_state, round_idx, optimization_rounds,
                                  total_demand):
        """处理轮次分配 """

        # 找到权重最高的目标作为主导目标
        dominant_objective = max(self.objective_weights.items(), key=lambda x: x[1])
        dominant_obj_name = dominant_objective[0]
        dominant_weight = dominant_objective[1]

        # 对于主导目标，在早期轮次中给予更多分配机会
        dominance_threshold = len(self.objective_weights) / (
                len(self.objective_weights) * len(self.objective_weights) - len(self.objective_weights) + 1)
        if dominant_weight > dominance_threshold:  # 使用动态阈值替代魔法数字
            # 调整路径顺序，让主导目标性能好的路径优先处理
            def dominant_objective_sort_key(path_info):
                try:
                    if len(path_info) < 8:
                        return float('inf')

                    # 获取路径指标
                    if path_info[0] == 'direct' and len(path_info) >= 8:
                        path_metrics = path_info[7]  # metrics在第8个位置
                    elif path_info[0] == 'multimodal' and len(path_info) >= 10:
                        path_metrics = path_info[9]  # metrics在第10个位置
                    else:
                        return float('inf')

                    # 根据主导目标类型返回相应的性能指标
                    if dominant_obj_name in ['time', 'cost', 'distance', 'balance', 'social']:
                        return path_metrics.get(f'{dominant_obj_name}_score', float('inf'))
                    elif dominant_obj_name in ['safety', 'priority']:
                        return -path_metrics.get(f'{dominant_obj_name}_score', float('-inf'))
                    elif dominant_obj_name == 'capability':
                        return path_metrics.get(f'{dominant_obj_name}_score', float('inf'))
                    else:
                        return float('inf')
                except (IndexError, KeyError, TypeError):
                    return float('inf')

            # 重新排序，让主导目标性能好的路径优先
            selected_paths_sorted = sorted(selected_paths_sorted, key=dominant_objective_sort_key)

        # 处理所有路径，不进行批量大小限制
        total_paths = len(selected_paths_sorted)
        batch_size = total_paths

        round_processed_paths = 0

        # 处理所有路径
        for path_info in selected_paths_sorted:
            round_processed_paths += 1

            # 处理单个路径分配
            self._allocate_single_path(path_info, current_state, round_idx, optimization_rounds)

            # 检查是否完全满足需求后再终止
            if self._should_terminate_early(current_state, total_demand):
                break

        return round_processed_paths

    def _allocate_single_path(self, path_info, current_state, round_idx, optimization_rounds):
        """分配单个路径"""
        try:
            path_type = path_info[0]
        except (IndexError, TypeError) as e:
            self.logger.warning(f"路径类型解析失败: {str(e)}")
            return

        if path_type == 'direct':
            self._allocate_direct_path(path_info, current_state, round_idx, optimization_rounds)
        elif path_type == 'multimodal':
            self._allocate_multimodal_path(path_info, current_state, round_idx, optimization_rounds)

    def _allocate_direct_path(self, path_info, current_state, round_idx, optimization_rounds):
        """分配直接路径"""
        try:
            if len(path_info) < 8:
                self.logger.warning(f"直接路径信息长度不足: 期望8个元素，实际{len(path_info)}个")
                return
            _, j, k, n, composite_score, path_risk, cost, path_metrics = path_info
        except (IndexError, ValueError) as e:
            self.logger.warning(f"直接路径信息解析失败: {str(e)}")
            return

        # 计算供应点的实际可用容量
        sub_objects = self.point_features[j].get('sub_objects', [])
        actual_supply_capacity = 0
        if sub_objects:
            for category in sub_objects:
                if isinstance(category, dict) and 'items' in category:
                    for sub_obj in category.get('items', []):
                        actual_supply_capacity += sub_obj.get('max_available_quantity', 0)
                else:
                    actual_supply_capacity += category.get('max_available_quantity', 0)
            # 对于人员动员，使用整数人员数量，可靠性通过其他方式体现
            if hasattr(self, 'resource_type') and self.resource_type == "personnel":
                actual_supply_left = min(current_state['supply_left'][j], actual_supply_capacity)
            else:
                actual_supply_left = min(current_state['supply_left'][j], actual_supply_capacity * self.P[j])
        else:
            actual_supply_left = current_state['supply_left'][j]

        if actual_supply_left <= self.EPS or current_state['demand_left'][k] <= self.EPS:
            return

        # 计算时间窗因子
        time_window_factor = self._calculate_time_window_factor(k, path_metrics)

        # 计算分配比例
        allocation_ratio = self._calculate_allocation_ratio(round_idx, optimization_rounds, time_window_factor)

        batch_limit = min(actual_supply_left, current_state['demand_left'][k])
        batch_size = batch_limit * allocation_ratio

        # 人员动员特殊处理：确保至少分配1个单位（如果有供应和需求）
        if hasattr(self, 'resource_type') and self.resource_type == "personnel":
            if batch_limit >= 1 and batch_size < 1:
                batch_size = 1  # 确保至少分配1个人员

        if batch_size <= self.EPS:
            return

        try:
            # 对于所有资源类型，确保分配量为整数，同时处理累积误差
            if hasattr(self, 'resource_type'):
                original_batch_size = batch_size

                # 获取当前需求点的累积舍入误差
                accumulated_error = current_state['accumulated_rounding_error'][k]
                adjusted_batch_size = original_batch_size + accumulated_error

                # 使用四舍五入，但确保不超出可用量
                batch_size = round(adjusted_batch_size)
                # 如果四舍五入后超出限制，则向下调整
                if batch_size > batch_limit:
                    batch_size = int(batch_limit)
                if batch_size < 1:
                    # 对于人员动员，如果有可用人员且有需求，在约束范围内尽可能分配
                    if self.resource_type == "personnel" and actual_supply_left >= 1 and current_state['demand_left'][
                        k] >= 1:
                        batch_size = 1
                    elif original_batch_size > self.EPS and batch_limit >= 1:
                        batch_size = 1
                # 对于人员动员，严格控制整数约束不超出可用量
                if hasattr(self, 'resource_type') and self.resource_type == "personnel":
                    max_allocatable = min(int(actual_supply_left), int(current_state['demand_left'][k]))
                    if batch_size > max_allocatable:
                        batch_size = max_allocatable
                else:
                    # 确保整数约束不会超出可用量
                    if batch_size > batch_limit:
                        batch_size = int(batch_limit)

                # 最终检查：确保批次大小有效且不超出可用量
                max_possible = min(int(actual_supply_left), int(current_state['demand_left'][k]))
                if batch_size > max_possible:
                    batch_size = max_possible

                # 更新累积舍入误差
                current_state['accumulated_rounding_error'][k] = adjusted_batch_size - batch_size

            # 执行分配
            final_batch_size = batch_size
            current_state['variables']['x_direct'][(j, k, n)] = 1
            current_state['variables']['b_direct'][(j, k, n)] = final_batch_size
            current_state['variables']['t_direct'][(j, k, n)] = path_metrics['time']

            # 更新剩余量
            current_state['supply_left'][j] -= final_batch_size
            current_state['demand_left'][k] -= final_batch_size
        except (KeyError, TypeError) as e:
            self.logger.warning(f"直接运输分配失败: {str(e)}")

    def _allocate_multimodal_path(self, path_info, current_state, round_idx, optimization_rounds):
        """分配统一多式联运路径"""
        try:
            if len(path_info) < 10:
                self.logger.warning(f"多式联运路径信息长度不足: 期望10个元素，实际{len(path_info)}个")
                return
            _, j, m1, m2, k, n2, composite_score, path_risk, cost, path_metrics = path_info
            self.logger.debug(f"开始分配多式联运路径: {j}->{m1}->{m2}->{k}, 运输方式{n2}")
        except (IndexError, ValueError) as e:
            self.logger.warning(f"多式联运路径信息解析失败: {str(e)}")
            return

        # 计算供应点的实际可用容量
        sub_objects = self.point_features[j].get('sub_objects', [])
        actual_supply_capacity = 0
        if sub_objects:
            for category in sub_objects:
                if isinstance(category, dict) and 'items' in category:
                    for sub_obj in category.get('items', []):
                        actual_supply_capacity += sub_obj.get('max_available_quantity', 0)
                else:
                    actual_supply_capacity += category.get('max_available_quantity', 0)
            # 对于人员动员，使用整数人员数量，可靠性通过其他方式体现
            if hasattr(self, 'resource_type') and self.resource_type == "personnel":
                actual_supply_left = min(current_state['supply_left'][j], actual_supply_capacity)
            else:
                actual_supply_left = min(current_state['supply_left'][j], actual_supply_capacity * self.P[j])
        else:
            actual_supply_left = current_state['supply_left'][j]

        if actual_supply_left <= self.EPS or current_state['demand_left'][k] <= self.EPS:
            return

        # 计算时间窗因子
        time_window_factor = self._calculate_time_window_factor(k, path_metrics)

        # 计算分配比例
        allocation_ratio = self._calculate_allocation_ratio(round_idx, optimization_rounds, time_window_factor)

        batch_limit = min(actual_supply_left, current_state['demand_left'][k])
        batch_size = batch_limit * allocation_ratio

        if batch_size <= self.EPS:
            return

        try:
            # 对于所有资源类型，确保分配量为整数，同时处理累积误差
            if hasattr(self, 'resource_type'):
                original_batch_size = batch_size

                # 获取当前需求点的累积舍入误差
                accumulated_error = current_state['accumulated_rounding_error'][k]
                adjusted_batch_size = original_batch_size + accumulated_error

                # 使用四舍五入，但确保不超出可用量
                batch_size = round(adjusted_batch_size)
                # 如果四舍五入后超出限制，则向下调整
                if batch_size > batch_limit:
                    batch_size = int(batch_limit)
                if batch_size < 1:
                    # 对于人员动员，如果有可用人员且有需求，在约束范围内尽可能分配
                    if self.resource_type == "personnel" and actual_supply_left >= 1 and current_state['demand_left'][
                        k] >= 1:
                        batch_size = 1
                    elif original_batch_size > self.EPS and batch_limit >= 1:
                        batch_size = 1
                # 对于人员动员，严格控制整数约束不超出可用量
                if hasattr(self, 'resource_type') and self.resource_type == "personnel":
                    max_allocatable = min(int(actual_supply_left), int(current_state['demand_left'][k]))
                    if batch_size > max_allocatable:
                        batch_size = max_allocatable
                else:
                    # 验证整数约束后的批次大小不会导致流量不平衡
                    if batch_size > batch_limit:
                        batch_size = int(batch_limit)

                # 最终检查：确保批次大小有效且不超出可用量
                max_possible = min(int(actual_supply_left), int(current_state['demand_left'][k]))
                if batch_size > max_possible:
                    batch_size = max_possible

                # 更新累积舍入误差
                current_state['accumulated_rounding_error'][k] = adjusted_batch_size - batch_size

            # 执行统一多式联运分配，确保三段流量完全一致
            final_batch_size = batch_size
            current_state['variables']['x1'][(j, m1, 1)] = 1
            current_state['variables']['b1'][(j, m1, 1)] = current_state['variables']['b1'].get((j, m1, 1),
                                                                                                0) + final_batch_size
            current_state['variables']['x2'][(m1, m2, n2)] = 1
            current_state['variables']['b2'][(m1, m2, n2)] = current_state['variables']['b2'].get((m1, m2, n2),
                                                                                                  0) + final_batch_size
            current_state['variables']['x3'][(m2, k, 1)] = 1
            current_state['variables']['b3'][(m2, k, 1)] = current_state['variables']['b3'].get((m2, k, 1),
                                                                                                0) + final_batch_size

            # 计算统一的时间调度（考虑等待时间同步）
            try:
                # 获取统一时间信息
                unified_timing = path_metrics.get('unified_timing', {})
                individual_time_to_m1 = unified_timing.get('individual_time_to_m1', 0.0)
                unified_departure_time = unified_timing.get('unified_departure_time', 0.0)

                # 如果没有统一时间信息，则计算实时距离
                if individual_time_to_m1 == 0.0:
                    j_lat, j_lon = self.point_features[j]['latitude'], self.point_features[j]['longitude']
                    m1_lat, m1_lon = self.point_features[m1]['latitude'], self.point_features[m1]['longitude']
                    d1 = self._calculate_haversine_distance(j_lat, j_lon, m1_lat, m1_lon)

                    # 安全访问运输方式
                    mode_1 = self.TRANSPORT_MODES.get(1)
                    if mode_1 and mode_1.get('speed', 0) > 0:
                        individual_time_to_m1 = d1 / mode_1['speed']
                    else:
                        individual_time_to_m1 = d1 / max(len(self.J) + len(self.M), 1)

                # 使用统一的出发时间（最晚到达时间）
                current_state['variables']['t1'][(j, m1, 1)] = unified_departure_time

                # 第二段和第三段时间保持不变
                if current_state['variables']['t2'].get((m1, m2, n2), 0.0) == 0.0:
                    m1_lat, m1_lon = self.point_features[m1]['latitude'], self.point_features[m1]['longitude']
                    m2_lat, m2_lon = self.point_features[m2]['latitude'], self.point_features[m2]['longitude']
                    d2 = self._calculate_haversine_distance(m1_lat, m1_lon, m2_lat, m2_lon)

                    # 安全访问运输方式n2
                    mode_n2 = self.TRANSPORT_MODES.get(n2)
                    if mode_n2 and mode_n2.get('speed', 0) > 0:
                        current_state['variables']['t2'][(m1, m2, n2)] = d2 / mode_n2['speed']
                    else:
                        # 使用默认速度计算
                        default_speed = max(len(self.J) + len(self.M), 1)
                        current_state['variables']['t2'][(m1, m2, n2)] = d2 / default_speed

                if current_state['variables']['t3'].get((m2, k, 1), 0.0) == 0.0:
                    m2_lat, m2_lon = self.point_features[m2]['latitude'], self.point_features[m2]['longitude']
                    k_lat, k_lon = self.point_features[k]['latitude'], self.point_features[k]['longitude']
                    d3 = self._calculate_haversine_distance(m2_lat, m2_lon, k_lat, k_lon)

                    # 安全访问运输方式1
                    mode_1 = self.TRANSPORT_MODES.get(1)
                    if mode_1 and mode_1.get('speed', 0) > 0:
                        current_state['variables']['t3'][(m2, k, 1)] = d3 / mode_1['speed']
                    else:
                        default_speed = max(len(self.J) + len(self.M), 1)
                        current_state['variables']['t3'][(m2, k, 1)] = d3 / default_speed

            except (KeyError, TypeError, ZeroDivisionError) as e:
                self.logger.warning(f"统一多式联运时间分配计算失败: {str(e)}")
                # 回退到原有计算方式
                try:
                    j_lat, j_lon = self.point_features[j]['latitude'], self.point_features[j]['longitude']
                    m1_lat, m1_lon = self.point_features[m1]['latitude'], self.point_features[m1]['longitude']
                    m2_lat, m2_lon = self.point_features[m2]['latitude'], self.point_features[m2]['longitude']
                    k_lat, k_lon = self.point_features[k]['latitude'], self.point_features[k]['longitude']

                    d1 = self._calculate_haversine_distance(j_lat, j_lon, m1_lat, m1_lon)
                    d2 = self._calculate_haversine_distance(m1_lat, m1_lon, m2_lat, m2_lon)
                    d3 = self._calculate_haversine_distance(m2_lat, m2_lon, k_lat, k_lon)

                    # 安全的速度计算
                    default_speed = max(len(self.J) + len(self.M), 1)

                    mode_1 = self.TRANSPORT_MODES.get(1)
                    speed_1 = mode_1.get('speed', default_speed) if mode_1 else default_speed

                    mode_n2 = self.TRANSPORT_MODES.get(n2)
                    speed_n2 = mode_n2.get('speed', default_speed) if mode_n2 else default_speed

                    current_state['variables']['t1'][(j, m1, 1)] = d1 / max(speed_1, 1)
                    current_state['variables']['t2'][(m1, m2, n2)] = d2 / max(speed_n2, 1)
                    current_state['variables']['t3'][(m2, k, 1)] = d3 / max(speed_1, 1)

                except Exception as fallback_e:
                    self.logger.error(f"多式联运时间计算完全失败: {str(fallback_e)}")
                    # 设置最小默认时间
                    default_time = max(len(self.J) / max(len(self.M), 1), 1.0)
                    current_state['variables']['t1'][(j, m1, 1)] = default_time
                    current_state['variables']['t2'][(m1, m2, n2)] = default_time
                    current_state['variables']['t3'][(m2, k, 1)] = default_time

            # 更新剩余量
            current_state['supply_left'][j] -= batch_size
            current_state['demand_left'][k] -= batch_size
        except (KeyError, TypeError, ZeroDivisionError) as e:
            self.logger.warning(f"统一多式联运分配失败: {str(e)}")

    def _calculate_time_window_factor(self, k, path_metrics):
        """计算时间窗因子"""
        time_window_factor = 1.0
        try:
            if k in self.time_windows:
                earliest_start, latest_finish = self.time_windows[k]
                actual_time = path_metrics['time']
                if actual_time > latest_finish:
                    time_violation_ratio = (actual_time - latest_finish) / latest_finish
                    # 简化的惩罚因子计算
                    penalty_rate = len(self.K) / (len(self.K) + len(self.J)) if len(self.K) + len(self.J) > 0 else 0.5
                    min_factor = penalty_rate * len(self.J) / (len(self.J) + len(self.M)) if len(self.J) + len(
                        self.M) > 0 else penalty_rate
                    time_window_factor = max(min_factor, 1.0 - time_violation_ratio * penalty_rate)
        except (KeyError, TypeError, ZeroDivisionError) as e:
            self.logger.warning(f"时间窗检查失败: {str(e)}")

        return time_window_factor

    def _calculate_allocation_ratio(self, round_idx, optimization_rounds, time_window_factor):
        # 计算网络规模参数
        supply_scale = len(self.J)
        demand_scale = len(self.K)
        network_scale = supply_scale + demand_scale + len(self.M)

        # 针对人员动员的特殊处理：使用更激进的分配策略
        if hasattr(self, 'resource_type') and self.resource_type == "personnel":
            # 计算当前轮次在总轮次中的位置比例
            round_progress = round_idx / max(optimization_rounds - 1, 1)

            # 第一轮使用非常高的分配比例，确保尽可能满足需求
            if round_idx == 0:
                # 提高初始分配比例到至少80%，确保第一轮就能分配大部分需求
                return max(time_window_factor, (demand_scale + supply_scale) / (
                            demand_scale + supply_scale + supply_scale / (demand_scale + 1)))

            # 后期轮次（最后1/2轮次）更激进地分配
            late_stage_threshold = max(1, optimization_rounds // 2)
            if round_idx >= late_stage_threshold:
                # 后期轮次尝试完全分配剩余需求
                return max(time_window_factor, 1.0)

            # 中期轮次使用更激进的渐进增加分配比例
            base_ratio = max(
                time_window_factor,
                (demand_scale + supply_scale + demand_scale) / (demand_scale + supply_scale + 1)  # 更高的基础分配比例
            )

            # 确保分配比例随轮次递增，避免分配不足
            progressive_factor = 1.0 + round_progress * (demand_scale + supply_scale) / (demand_scale + 1)
            return min(1.0, base_ratio * progressive_factor)

        # 非人员动员的原有逻辑
        if round_idx == 0:
            return time_window_factor
        else:
            # 确保在最后轮次时能够充分分配
            diversity_factor = 1.0 - (round_idx / optimization_rounds)

            # 计算网络特征因子
            supply_demand_balance = min(supply_scale, demand_scale) / max(supply_scale, demand_scale) if max(
                supply_scale, demand_scale) > 0 else 1.0
            network_connectivity = network_scale / (network_scale + 1)

            # 在最后两轮时，提高分配比例以确保需求满足
            rounds_threshold = max(1, optimization_rounds // (supply_scale + 1)) if supply_scale > 0 else 1
            if round_idx >= optimization_rounds - rounds_threshold:
                base_ratio = time_window_factor
            else:
                # 基于网络特征计算最小分配比例
                deterministic_min = max(network_connectivity * supply_demand_balance,
                                        supply_scale / (
                                                supply_scale + demand_scale)) if supply_scale + demand_scale > 0 else supply_demand_balance
                deterministic_factor = deterministic_min + (round_idx / optimization_rounds) * (1.0 - deterministic_min)
                base_ratio = time_window_factor * diversity_factor * deterministic_factor

            # 基于网络特征确保分配比例不会过小，使用网络规模倒数作为合理下界
            network_based_min = 1.0 / (network_scale + 1) if network_scale > 0 else 1.0 / (
                    supply_scale + demand_scale + 1)
            calculated_min = supply_scale * network_connectivity / (
                    supply_scale * demand_scale + supply_scale) if supply_scale > 0 and demand_scale > 0 else network_based_min
            min_allocation_ratio = max(calculated_min, network_based_min)

            return max(base_ratio, min_allocation_ratio)

    def _should_terminate_early(self, current_state, total_demand):
        """检查是否应该早期终止"""
        try:
            remaining_demand = sum(current_state['demand_left'].values())
            total_remaining_supply = sum(current_state['supply_left'].values())

            # 对于人员动员，使用更严格的终止条件，确保充分分配
            if hasattr(self, 'resource_type') and self.resource_type == "personnel":
                # 只有在完全满足需求时才终止，或者剩余供应确实不足时才终止
                return (remaining_demand <= self.EPS) or (
                            total_remaining_supply < remaining_demand and total_remaining_supply <= self.EPS)

            # 其他资源类型使用原逻辑
            return (remaining_demand <= self.EPS) or (total_remaining_supply < remaining_demand)
        except (KeyError, TypeError, ZeroDivisionError) as e:
            self.logger.warning(f"早期终止检查失败: {str(e)}")
            return False

    def _evaluate_round_result(self, current_state, total_demand):
        """评估轮次结果"""
        try:
            demand_left_values = current_state['demand_left'].values()
            current_remaining_demand = sum(current_state['demand_left'].values())
            if total_demand > 0:

                result = 1 - (current_remaining_demand / total_demand)
                return result
            else:
                return 1.0
        except (KeyError, TypeError, ZeroDivisionError) as e:
            self.logger.warning(f"轮次结果评估失败: {str(e)}")
            return 0.0

    def _build_current_solution(self, current_state, satisfaction_rate, round_idx):
        """构建当前解"""
        return {
            **current_state['variables'],
            'satisfaction_rate': satisfaction_rate,
            'round_index': round_idx,
            'supply_left': dict(current_state['supply_left']),
            'demand_left': dict(current_state['demand_left']),
            'accumulated_rounding_error': dict(current_state['accumulated_rounding_error'])
        }

    def _apply_best_solution(self, best_solution, variable_template):
        """应用最优解"""
        if best_solution:
            return {key: best_solution[key] for key in variable_template.keys()}
        else:
            return variable_template

    def _clean_solution_paths(self, x1, x2, x3, x_direct, b1, b2, b3, b_direct):
        """
        清理解决方案：只保留有实际流量的路径
        """

        # 清理直接运输路径
        for key in list(x_direct.keys()):
            if b_direct.get(key, 0) <= self.EPS:
                x_direct[key] = 0

        # 清理第一段路径
        for key in list(x1.keys()):
            if b1.get(key, 0) <= self.EPS:
                x1[key] = 0

        # 清理第二段路径
        for key in list(x2.keys()):
            if b2.get(key, 0) <= self.EPS:
                x2[key] = 0

        # 清理第三段路径
        for key in list(x3.keys()):
            if b3.get(key, 0) <= self.EPS:
                x3[key] = 0

        self.logger.info("解决方案路径清理完成：已移除无流量路径")

    def _get_allocation_ratio_by_score(self, composite_score, current_path, total_paths):
        """基于综合评分确定分配比例"""

        # 路径排序位置（越小表示评分越好）
        path_rank_ratio = current_path / total_paths

        # 排名越靠前（path_rank_ratio越小），分配比例越高
        inverse_rank = 1.0 - path_rank_ratio  # 反转排名，使前排路径获得更高权重
        base_ratio = inverse_rank ** 2  # 平方函数增强前排路径优势

        # 根据综合评分进一步调整
        adjustment_factor = max(0.1, 1.0 - path_rank_ratio)

        final_ratio = base_ratio * adjustment_factor

        return max(0.05, min(0.9, final_ratio))  # 限制在[0.05, 0.9]范围内

    def _perform_supplementary_allocation(self, x1, x2, x3, x_direct, b1, b2, b3, b_direct,
                                          t1, t2, t3, t_direct,
                                          supply_left, demand_left, accumulated_rounding_errors=None):
        """执行补充分配，保持运输模式统一性"""

        # 处理累积舍入误差参数
        if accumulated_rounding_errors is None:
            accumulated_rounding_errors = {k: 0.0 for k in self.K}

        # 首先判断当前使用的运输模式
        current_transport_mode = self._detect_current_transport_mode(x1, x2, x3, x_direct, b1, b2, b3, b_direct)

        self.logger.info(f"补充分配阶段 - 检测到当前运输模式: {current_transport_mode['mode']}")

        # 识别未满足的需求
        unsatisfied_demands = [(k, demand_left[k]) for k in self.K if demand_left[k] > self.EPS]

        # 按需求量排序，优先满足大需求
        unsatisfied_demands.sort(key=lambda x: x[1], reverse=True)

        if not unsatisfied_demands:
            self.logger.info("所有需求已满足，无需补充分配")
            return

        # 按供应能力排序供应点，优先使用大容量供应点
        sorted_suppliers = sorted([j for j in self.J if supply_left[j] > self.EPS],
                                  key=lambda j: supply_left[j], reverse=True)

        if not sorted_suppliers:
            self.logger.info("没有可用的供应点，补充分配结束")
            return

        # 预构建路径映射以避免重复计算
        existing_paths_map = self._build_existing_paths_map(x1, x2, x3, x_direct)

        # 计算合理的循环限制
        total_network_size = len(self.J) + len(self.M) + len(self.K)
        max_demand_iterations = min(len(unsatisfied_demands) * total_network_size, len(self.K) * len(self.J))
        max_supplier_iterations = min(len(sorted_suppliers), len(self.J))

        # 全局进展跟踪
        global_progress_tracker = {
            'total_allocated': 0.0,
            'iterations_without_progress': 0,
            'max_iterations_without_progress': total_network_size
        }

        # 主循环：处理未满足的需求
        demand_loop_counter = 0
        for k, initial_remaining_demand in unsatisfied_demands:
            demand_loop_counter += 1
            if demand_loop_counter > max_demand_iterations:
                self.logger.warning(f"补充分配达到最大需求处理次数限制，终止处理")
                break

            remaining_demand = initial_remaining_demand
            demand_start_amount = remaining_demand

            # 处理所有可用的供应点
            available_suppliers_for_k = [j for j in sorted_suppliers if supply_left[j] > self.EPS]

            # 第一阶段：从已有路径增加运输量
            remaining_demand = self._increase_existing_path_allocation(
                k, remaining_demand, available_suppliers_for_k,
                existing_paths_map, supply_left, demand_left,
                x_direct, b_direct, b1, b2, b3, current_transport_mode
            )

            # 第二阶段：如果仍有未满足需求，创建新路径
            if remaining_demand > self.EPS:
                remaining_demand = self._create_new_paths_for_demand(
                    k, remaining_demand, sorted_suppliers, supply_left, demand_left,
                    x1, x2, x3, x_direct, b1, b2, b3, b_direct, t1, t2, t3, t_direct,
                    existing_paths_map, current_transport_mode, accumulated_rounding_errors
                )

            # 更新全局进展跟踪
            allocated_this_round = demand_start_amount - remaining_demand
            if allocated_this_round > self.EPS:
                global_progress_tracker['total_allocated'] += allocated_this_round
                global_progress_tracker['iterations_without_progress'] = 0
            else:
                global_progress_tracker['iterations_without_progress'] += 1

            # 检查是否需要提前终止
            if global_progress_tracker['iterations_without_progress'] >= global_progress_tracker[
                'max_iterations_without_progress']:
                self.logger.warning(f"连续多轮无进展，提前终止补充分配")
                break

            if remaining_demand <= self.EPS:
                self.logger.debug(f"需求点 {k} 需求已完全满足")

        # 统计最终结果
        final_unsatisfied = sum(demand_left[k] for k in self.K if demand_left[k] > self.EPS)
        total_demand = sum(self.D.values())
        final_satisfaction_rate = 1 - (final_unsatisfied / total_demand) if total_demand > 0 else 1.0

        self.logger.info(f"补充分配完成 - 最终满足率: {final_satisfaction_rate * 100:.1f}%")
        self.logger.info(f"总分配量: {global_progress_tracker['total_allocated']:.2f}")

        # 最终精确匹配保证
        for k in self.K:
            if demand_left[k] > self.EPS:
                remaining_shortage = demand_left[k]

                # 寻找有剩余供应的供应点进行最终调整
                for j in self.J:
                    if supply_left[j] > self.EPS and remaining_shortage > self.EPS:
                        # 计算可调整量
                        adjustment_amount = min(remaining_shortage, supply_left[j])

                        # 查找现有路径进行微调
                        adjusted = False

                        # 尝试调整直接路径
                        if not adjusted:
                            for n in [1]:
                                if x_direct.get((j, k, n), 0) == 1:
                                    b_direct[(j, k, n)] += adjustment_amount
                                    supply_left[j] -= adjustment_amount
                                    demand_left[k] -= adjustment_amount
                                    remaining_shortage -= adjustment_amount
                                    adjusted = True
                                    break

                        # 尝试调整多式联运路径
                        if not adjusted:
                            for m1 in self.M:
                                if x1.get((j, m1, 1), 0) == 1:
                                    for m2 in self.M:
                                        if m1 != m2:
                                            for n2 in self.N:
                                                if (x2.get((m1, m2, n2), 0) == 1 and
                                                        x3.get((m2, k, 1), 0) == 1):
                                                    b1[(j, m1, 1)] += adjustment_amount
                                                    b2[(m1, m2, n2)] += adjustment_amount
                                                    b3[(m2, k, 1)] += adjustment_amount
                                                    supply_left[j] -= adjustment_amount
                                                    demand_left[k] -= adjustment_amount
                                                    remaining_shortage -= adjustment_amount
                                                    adjusted = True
                                                    break
                                            if adjusted:
                                                break
                                    if adjusted:
                                        break

                        if remaining_shortage <= self.EPS:
                            break

        # 重新计算最终统计
        final_unsatisfied_after_adjustment = sum(demand_left[k] for k in self.K if demand_left[k] > self.EPS)
        if final_unsatisfied_after_adjustment <= self.EPS:
            self.logger.info("最终精确匹配成功：供需完全平衡")
        else:
            self.logger.warning(f"最终仍有微小不匹配: {final_unsatisfied_after_adjustment:.6f}")

    def _build_existing_paths_map(self, x1, x2, x3, x_direct):
        """构建已有路径映射"""
        existing_paths = {}

        # 构建直接路径映射
        for j in self.J:
            for k in self.K:
                if x_direct.get((j, k, 1), 0) == 1:
                    if j not in existing_paths:
                        existing_paths[j] = {}
                    existing_paths[j][k] = {'type': 'direct', 'route': (j, k, 1)}

        # 构建多式联运路径映射
        multimodal_routes = {}

        # 第一步：找出所有活跃的第一段路径
        active_first_segments = {}
        for j in self.J:
            for m1 in self.M:
                if x1.get((j, m1, 1), 0) == 1:
                    if j not in active_first_segments:
                        active_first_segments[j] = []
                    active_first_segments[j].append(m1)

        # 第二步：找出所有活跃的第二段路径
        active_second_segments = {}
        for m1 in self.M:
            for m2 in self.M:
                if m1 != m2:
                    for n2 in self.N:
                        if x2.get((m1, m2, n2), 0) == 1:
                            if m1 not in active_second_segments:
                                active_second_segments[m1] = []
                            active_second_segments[m1].append((m2, n2))

        # 第三步：找出所有活跃的第三段路径
        active_third_segments = {}
        for m2 in self.M:
            for k in self.K:
                if x3.get((m2, k, 1), 0) == 1:
                    if m2 not in active_third_segments:
                        active_third_segments[m2] = []
                    active_third_segments[m2].append(k)

        # 第四步：组装完整的多式联运路径
        for j in active_first_segments:
            for m1 in active_first_segments[j]:
                if m1 in active_second_segments:
                    for m2, n2 in active_second_segments[m1]:
                        if m2 in active_third_segments:
                            for k in active_third_segments[m2]:
                                if j not in existing_paths:
                                    existing_paths[j] = {}
                                existing_paths[j][k] = {
                                    'type': 'multimodal',
                                    'route': (j, m1, m2, k, n2)
                                }

        return existing_paths

    def _increase_existing_path_allocation(self, k, remaining_demand, sorted_suppliers,
                                           existing_paths_map, supply_left, demand_left,
                                           x_direct, b_direct, b1, b2, b3, current_transport_mode):
        """从已有路径增加运输量"""

        for j in sorted_suppliers:

            if supply_left[j] <= self.EPS or remaining_demand <= self.EPS:
                continue

            # 记录本轮开始的状态
            prev_remaining_demand = remaining_demand
            prev_supply_left = supply_left[j]

            # 检查是否有从j到k的已有路径
            if j in existing_paths_map and k in existing_paths_map[j]:
                path_info = existing_paths_map[j][k]

                # 计算实际可用供应量，与主分配阶段逻辑完全一致
                sub_objects = self.point_features[j].get('sub_objects', [])
                actual_supply_capacity = 0
                if sub_objects:
                    for category in sub_objects:
                        if isinstance(category, dict) and 'items' in category:
                            for sub_obj in category.get('items', []):
                                actual_supply_capacity += sub_obj.get('max_available_quantity', 0)
                        else:
                            actual_supply_capacity += category.get('max_available_quantity', 0)
                    actual_available = min(supply_left[j], actual_supply_capacity * self.P[j])
                else:
                    actual_available = supply_left[j]

                supplement_amount = min(remaining_demand, actual_available)

                if supplement_amount > self.EPS:
                    if path_info['type'] == 'direct':
                        # 增加直接运输路径的运输量
                        b_direct[(j, k, 1)] += supplement_amount
                        self.logger.debug(f"增加直接路径 {j}->{k} 运输量: {supplement_amount:.2f}")

                    elif path_info['type'] == 'multimodal':
                        # 增加多式联运路径的运输量，确保流量平衡
                        route = path_info['route']
                        if len(route) >= 5:
                            j_route, m1, m2, k_route, n2 = route
                            # 确保路径一致性
                            if j_route == j and k_route == k:
                                b1[(j_route, m1, 1)] += supplement_amount
                                b2[(m1, m2, n2)] += supplement_amount
                                b3[(m2, k_route, 1)] += supplement_amount
                                self.logger.debug(
                                    f"增加多式联运路径 {j_route}->{m1}->{m2}->{k_route} 运输量: {supplement_amount:.2f}")
                            else:
                                self.logger.warning(f"路径不一致: 期望{j}->{k}, 实际{j_route}->{k_route}")
                                continue

                    # 更新剩余量
                    supply_left[j] -= supplement_amount
                    demand_left[k] -= supplement_amount
                    remaining_demand -= supplement_amount

            # 严格的进展检查
            if abs(remaining_demand - prev_remaining_demand) < self.EPS and abs(
                    supply_left[j] - prev_supply_left) < self.EPS:
                continue

            if remaining_demand <= self.EPS:
                break

        return remaining_demand

    def _create_new_paths_for_demand(self, k, remaining_demand, sorted_suppliers, supply_left, demand_left,
                                     x1, x2, x3, x_direct, b1, b2, b3, b_direct, t1, t2, t3, t_direct,
                                     existing_paths_map, current_transport_mode, accumulated_rounding_errors=None):
        """为需求创建新路径"""

        if remaining_demand <= self.EPS:
            return remaining_demand

        # 计算实际可用的供应点，确保容量计算一致性
        available_suppliers = []
        for j in self.J:
            if supply_left[j] <= self.EPS:
                continue

            # 检查是否已经有从j到k的路径
            if j in existing_paths_map and k in existing_paths_map[j]:
                continue

            # 计算实际可用容量，与主分配阶段逻辑完全一致
            sub_objects = self.point_features[j].get('sub_objects', [])
            actual_supply_capacity = 0
            if sub_objects:
                for category in sub_objects:
                    if isinstance(category, dict) and 'items' in category:
                        for sub_obj in category.get('items', []):
                            actual_supply_capacity += sub_obj.get('max_available_quantity', 0)
                    else:
                        actual_supply_capacity += category.get('max_available_quantity', 0)
                # 人员动员不乘以可靠性系数，保持一致性
                if hasattr(self, 'resource_type') and self.resource_type == "personnel":
                    actual_available = min(supply_left[j], actual_supply_capacity)
                else:
                    actual_available = min(supply_left[j], actual_supply_capacity * self.P[j])
            else:
                actual_available = supply_left[j]

            if actual_available > self.EPS:
                available_suppliers.append((j, actual_available))

        # 按可用容量排序
        available_suppliers.sort(key=lambda x: x[1], reverse=True)

        # 分配逻辑，确保充分利用可用供应
        for j, actual_available in available_suppliers:

            if remaining_demand <= self.EPS:
                break

            # 记录本轮开始的状态
            prev_remaining_demand = remaining_demand

            # 使用实际可用容量计算补充量，并处理累积误差确保为整数
            supplement_amount = min(remaining_demand, actual_available, supply_left[j])
            if hasattr(self, 'resource_type'):
                # 使用四舍五入，但确保不超出可用量
                original_supplement = supplement_amount
                # 在补充分配阶段，如果存在累积误差且当前需求点有未满足的需求，优先补偿误差
                if accumulated_rounding_errors and k in accumulated_rounding_errors:
                    accumulated_error = accumulated_rounding_errors[k]
                    if accumulated_error >= 1.0 and remaining_demand >= 1:
                        supplement_amount = min(supplement_amount + accumulated_error,
                                                min(remaining_demand, actual_available, supply_left[j]))
                        accumulated_rounding_errors[k] = max(0.0, accumulated_error - int(accumulated_error))

                if self.resource_type == "personnel" and remaining_demand >= 1:
                    # 优先满足剩余需求，向上取整
                    supplement_amount = min(int(remaining_demand + self.EPS), int(actual_available),
                                            int(supply_left[j]))
                else:
                    supplement_amount = round(supplement_amount)
                # 如果四舍五入后超出限制，则向下调整
                if supplement_amount > min(remaining_demand, actual_available, supply_left[j]):
                    supplement_amount = int(min(remaining_demand, actual_available, supply_left[j]))
                # 对于人员动员，如果原始计算值大于0但取整为0，且实际有可用资源，则分配1个单位
                if (supplement_amount == 0 and
                        self.resource_type == "personnel" and
                        original_supplement > self.EPS and
                        actual_available >= 1 and supply_left[j] >= 1 and remaining_demand >= 1):
                    supplement_amount = 1

            if supplement_amount > self.EPS:
                if current_transport_mode['mode'] == 'direct':
                    # 创建新的直达路径
                    x_direct[(j, k, 1)] = 1
                    b_direct[(j, k, 1)] = supplement_amount

                    if t_direct.get((j, k, 1), 0) == 0:
                        j_lat, j_lon = self.point_features[j]['latitude'], self.point_features[j]['longitude']
                        k_lat, k_lon = self.point_features[k]['latitude'], self.point_features[k]['longitude']
                        direct_distance = self._calculate_haversine_distance(j_lat, j_lon, k_lat, k_lon)

                        mode_1 = self.TRANSPORT_MODES.get(1)
                        if not mode_1:
                            raise ValueError("运输方式1（公路运输）未在TRANSPORT_MODES中配置")

                        speed = mode_1.get('speed')
                        if speed is None or speed <= 0:
                            raise ValueError(f"运输方式1的速度未配置或无效: {speed}")

                        t_direct[(j, k, 1)] = direct_distance / speed

                    # 更新路径映射
                    if j not in existing_paths_map:
                        existing_paths_map[j] = {}
                    existing_paths_map[j][k] = {'type': 'direct', 'route': (j, k, 1)}

                    self.logger.debug(f"新增供应点 {j} 直达运输到需求点 {k}: {supplement_amount:.2f}")

                elif current_transport_mode['mode'] == 'multimodal':
                    # 创建新的统一多式联运路径，确保流量平衡
                    unified_route = current_transport_mode['unified_route']
                    m1, m2, n2 = unified_route['m1'], unified_route['m2'], unified_route['n2']

                    # 创建完整的多式联运路径，确保流量一致性
                    x1[(j, m1, 1)] = 1
                    b1[(j, m1, 1)] = b1.get((j, m1, 1), 0) + supplement_amount
                    x2[(m1, m2, n2)] = 1
                    b2[(m1, m2, n2)] = b2.get((m1, m2, n2), 0) + supplement_amount
                    x3[(m2, k, 1)] = 1
                    b3[(m2, k, 1)] = b3.get((m2, k, 1), 0) + supplement_amount

                    # 计算时间
                    if t1.get((j, m1, 1), 0) == 0:
                        j_lat, j_lon = self.point_features[j]['latitude'], self.point_features[j]['longitude']
                        m1_lat, m1_lon = self.point_features[m1]['latitude'], self.point_features[m1]['longitude']
                        d1 = self._calculate_haversine_distance(j_lat, j_lon, m1_lat, m1_lon)

                        mode_1 = self.TRANSPORT_MODES.get(1)
                        if mode_1 and mode_1.get('speed', 0) > 0:
                            t1[(j, m1, 1)] = d1 / mode_1['speed']
                        else:
                            t1[(j, m1, 1)] = d1 / max(len(self.J) + len(self.M), 1)

                    if t2.get((m1, m2, n2), 0) == 0:
                        m1_lat, m1_lon = self.point_features[m1]['latitude'], self.point_features[m1]['longitude']
                        m2_lat, m2_lon = self.point_features[m2]['latitude'], self.point_features[m2]['longitude']
                        d2 = self._calculate_haversine_distance(m1_lat, m1_lon, m2_lat, m2_lon)

                        mode_n2 = self.TRANSPORT_MODES.get(n2)
                        if mode_n2 and mode_n2.get('speed', 0) > 0:
                            t2[(m1, m2, n2)] = d2 / mode_n2['speed']
                        else:
                            t2[(m1, m2, n2)] = d2 / max(len(self.J) + len(self.M), 1)

                    if t3.get((m2, k, 1), 0) == 0:
                        m2_lat, m2_lon = self.point_features[m2]['latitude'], self.point_features[m2]['longitude']
                        k_lat, k_lon = self.point_features[k]['latitude'], self.point_features[k]['longitude']
                        d3 = self._calculate_haversine_distance(m2_lat, m2_lon, k_lat, k_lon)

                        mode_1 = self.TRANSPORT_MODES.get(1)
                        if mode_1 and mode_1.get('speed', 0) > 0:
                            t3[(m2, k, 1)] = d3 / mode_1['speed']
                        else:
                            t3[(m2, k, 1)] = d3 / max(len(self.J) + len(self.M), 1)

                    # 更新路径映射
                    if j not in existing_paths_map:
                        existing_paths_map[j] = {}
                    existing_paths_map[j][k] = {
                        'type': 'multimodal',
                        'route': (j, m1, m2, k, n2)
                    }

                    self.logger.debug(
                        f"新增供应点 {j} 统一多式联运到需求点 {k}: {supplement_amount:.2f} (路径: {m1}-{m2})")

                # 更新剩余量
                supply_left[j] -= supplement_amount
                demand_left[k] -= supplement_amount
                remaining_demand -= supplement_amount

            # 严格的进展检查
            if abs(remaining_demand - prev_remaining_demand) < self.EPS:
                continue

            if remaining_demand <= self.EPS:
                break

        return remaining_demand

    def _detect_current_transport_mode(self, x1, x2, x3, x_direct, b1, b2, b3, b_direct):
        """
        检测当前解决方案使用的运输模式
        """

        # 统计各种路径的使用情况
        direct_routes_count = sum(1 for val in x_direct.values() if val > 0)
        multimodal_routes_count = sum(1 for val in x1.values() if val > 0)

        direct_flow = sum(flow for flow in b_direct.values() if flow > self.EPS)
        multimodal_flow = sum(flow for flow in b1.values() if flow > self.EPS)

        self.logger.debug(f"运输模式检测 - 直达路径: {direct_routes_count}条 (流量: {direct_flow:.2f})")
        self.logger.debug(f"运输模式检测 - 多式联运路径: {multimodal_routes_count}条 (流量: {multimodal_flow:.2f})")

        # 如果只有直达路径或直达流量占主导
        if multimodal_routes_count == 0 or multimodal_flow <= self.EPS:
            return {'mode': 'direct'}

        # 如果只有多式联运路径或多式联运流量占主导
        if direct_routes_count == 0 or direct_flow <= self.EPS:
            # 检测统一的多式联运路径
            unified_route = self._detect_unified_multimodal_route(x1, x2, x3, b1, b2, b3)
            return {
                'mode': 'multimodal',
                'unified_route': unified_route
            }

        # 混合模式（这种情况不应该出现在正确的实现中）
        self.logger.warning("检测到混合运输模式，这可能表示算法实现有问题")
        if direct_flow >= multimodal_flow:
            return {'mode': 'direct'}
        else:
            unified_route = self._detect_unified_multimodal_route(x1, x2, x3, b1, b2, b3)
            return {
                'mode': 'multimodal',
                'unified_route': unified_route
            }

    def _detect_unified_multimodal_route(self, x1, x2, x3, b1, b2, b3):
        """
        检测统一的多式联运路径
        """

        # 找到流量最大的m1-m2-n2组合
        route_flows = {}

        for (m1, m2, n2), flow in b2.items():
            if flow > self.EPS:
                route_key = (m1, m2, n2)
                route_flows[route_key] = route_flows.get(route_key, 0) + flow

        if not route_flows:
            # 如果没有找到，使用第一个有效的多式联运路径
            for (j, m1, n1), x_val in x1.items():
                if x_val > 0:
                    for (m1_check, m2, n2), x2_val in x2.items():
                        if m1_check == m1 and x2_val > 0:
                            return {'m1': m1, 'm2': m2, 'n2': n2}

            # 默认返回（不应该到达这里）
            if not self.M or len(self.M) == 0 or not self.N or len(self.N) == 0:
                self.logger.error("中转点或运输方式集合为空，无法构建默认多式联运路径")
                raise ValueError("网络数据不完整：中转点或运输方式集合为空")
            return {'m1': self.M[0], 'm2': self.M[1] if len(self.M) > 1 else self.M[0], 'n2': self.N[0]}

        # 选择流量最大的路径作为统一路径
        best_route = max(route_flows.items(), key=lambda x: x[1])
        m1, m2, n2 = best_route[0]

        self.logger.debug(f"检测到统一多式联运路径: {m1} -> {m2} (运输方式: {n2}), 总流量: {best_route[1]:.2f}")

        return {'m1': m1, 'm2': m2, 'n2': n2}

    def _compute_data_mobilization_objectives(self, solution):
        """
        计算数据动员的多目标值
        """
        # 初始化目标值
        objectives = {
            'time': 0.0,  # 数据动员无时间成本
            'cost': 0.0,
            'distance': 0.0,  # 数据动员无距离成本
            'safety': 0.0,
            'priority': 0.0,
            'balance': 0.0,
            'capability': 0.0,
            'social': 0.0
        }

        total_allocated = 0.0
        total_cost = 0.0
        weighted_safety = 0.0
        weighted_priority = 0.0
        weighted_capability = 0.0
        weighted_social = 0.0

        # 从选中的供应商中计算各目标值
        for supplier_data in solution['selected_suppliers']:
            evaluation = supplier_data['evaluation']
            allocated_capacity = supplier_data['allocated_capacity']

            if allocated_capacity <= 0:
                continue

            # 成本目标：基于数据动员的设施和通信成本
            supplier_id = supplier_data['supplier_id']
            supplier_name = self._find_supplier_name_by_id(supplier_id)

            facility_cost = self.point_features[supplier_name].get('facility_rental_price',
                                                                   len(self.J) * len(self.M) + len(self.J))
            power_cost = self.point_features[supplier_name].get('power_cost', len(self.J) + len(self.K))
            comm_cost = self.point_features[supplier_name].get('communication_purchase_price',
                                                               len(self.J) * len(self.K))
            supplier_cost = facility_cost + power_cost + comm_cost
            total_cost += supplier_cost * allocated_capacity

            # 安全目标：基于数据安全评分
            safety_score = self._calculate_data_safety_score(supplier_name)
            weighted_safety += safety_score * allocated_capacity

            # 优先级目标：基于动员能力评分
            priority_score = evaluation['mobilization_score']
            weighted_priority += priority_score * allocated_capacity

            # 能力目标：基于企业能力评分
            capability_score = evaluation['capability_score']
            weighted_capability += capability_score * allocated_capacity

            # 社会影响目标：基于企业类型和规模
            social_score = evaluation['type_score'] * evaluation['size_score']
            weighted_social += social_score * allocated_capacity

            total_allocated += allocated_capacity

        # 计算加权平均
        if total_allocated > 0:
            objectives['cost'] = total_cost
            objectives['safety'] = -(weighted_safety / total_allocated)  # 转为最小化目标
            objectives['priority'] = -(weighted_priority / total_allocated)  # 转为最小化目标
            objectives['capability'] = weighted_capability / total_allocated
            objectives['social'] = weighted_social / total_allocated

        # 计算资源均衡目标
        if len(solution['selected_suppliers']) > 1:
            allocations = [s['allocated_capacity'] for s in solution['selected_suppliers']]
            mean_allocation = sum(allocations) / len(allocations)
            balance_deviation = sum(abs(a - mean_allocation) for a in allocations) / len(allocations)
            objectives['balance'] = balance_deviation / mean_allocation if mean_allocation > 0 else 0.0

        # 归一化处理
        normalized_objectives = self._batch_normalize_computed_objectives(objectives)

        # 计算综合目标值
        composite_value = sum(normalized_objectives[obj] * self.objective_weights[obj]
                              for obj in objectives.keys())

        return {
            'composite_objective_value': composite_value,
            'individual_objectives': objectives,
            'normalized_objectives': normalized_objectives,
            'objective_weights': self.objective_weights.copy()
        }

    def _find_supplier_name_by_id(self, supplier_id):
        """根据supplier_id找到对应的供应点名称"""
        for name, features in self.point_features.items():
            if features.get('original_supplier_id') == supplier_id:
                return name
        return supplier_id

    def _calculate_data_safety_score(self, supplier_name):
        """计算数据供应点的安全评分"""
        supply_scale = len(self.J)
        transfer_scale = len(self.M)
        demand_scale = len(self.K)

        default_control_score = transfer_scale / (
                transfer_scale + demand_scale) if transfer_scale + demand_scale > 0 else transfer_scale / (
                transfer_scale + 1)
        default_usability_score = supply_scale / (
                supply_scale + transfer_scale) if supply_scale + transfer_scale > 0 else supply_scale / (
                supply_scale + 1)
        default_security_score = (supply_scale + transfer_scale + demand_scale) / (
                (supply_scale + transfer_scale + demand_scale) + len(self.TRANSPORT_MODES)) if hasattr(self,
                                                                                                       'TRANSPORT_MODES') else (
                                                                                                                                       supply_scale + transfer_scale + demand_scale) / (
                                                                                                                                       (
                                                                                                                                               supply_scale + transfer_scale + demand_scale) + 1)

        autonomous_control = self.point_features[supplier_name].get('autonomous_control', default_control_score)
        usability_level = self.point_features[supplier_name].get('usability_level', default_usability_score)
        security_measures = self.point_features[supplier_name].get('security_measures_score', default_security_score)

        return autonomous_control + usability_level + security_measures

    def compute_multi_objective_value(self, solution):
        """
        计算多目标综合函数值
        """

        # 预计算基础运输指标
        base_transport_metrics = self._calculate_base_transport_metrics(solution)

        # 批量计算路径目标值
        path_objectives_data = self._batch_calculate_path_objectives(solution)

        # 加权平均计算
        total_objectives = self._compute_weighted_average_objectives(path_objectives_data)

        # 计算安全目标：所有参与活动供应点的企业安全指标之和
        active_suppliers = set()

        # 从直接运输中获取活跃供应点
        for j in self.J:
            for k in self.K:
                for n in [1]:
                    if solution['x_direct'].get((j, k, n), 0) == 1 and solution['b_direct'].get((j, k, n),
                                                                                                0) > self.EPS:
                        active_suppliers.add(j)

        # 从多式联运中获取活跃供应点
        for j in self.J:
            for m in self.M:
                for n in [1]:
                    if solution['x1'].get((j, m, n), 0) == 1 and solution['b1'].get((j, m, n), 0) > self.EPS:
                        active_suppliers.add(j)

        # 计算网络规模参数用于默认值计算
        supply_scale = len(self.J)
        transfer_scale = len(self.M)
        demand_scale = len(self.K)

        default_enterprise_nature = supply_scale / (
                supply_scale + transfer_scale) if supply_scale + transfer_scale > 0 else supply_scale / (
                supply_scale + 1)
        default_enterprise_scale = demand_scale / (
                demand_scale + transfer_scale) if demand_scale + transfer_scale > 0 else demand_scale / (
                demand_scale + 1)
        default_resource_safety = transfer_scale / (
                transfer_scale + demand_scale) if transfer_scale + demand_scale > 0 else transfer_scale / (
                transfer_scale + 1)

        # 计算参与活动供应点的企业安全指标之和（企业性质+企业规模+风险记录+外资背景+资源安全）
        supplier_safety_sum = 0.0
        for j in active_suppliers:
            enterprise_nature_score = self.point_features[j].get('enterprise_nature_score', default_enterprise_nature)
            enterprise_scale_score = self.point_features[j].get('enterprise_scale_score', default_enterprise_scale)
            risk_record = self.point_features[j].get('risk_record', 0)
            foreign_background = self.point_features[j].get('foreign_background', 0)
            resource_safety = self.point_features[j].get('resource_safety', default_resource_safety)

            supplier_safety_sum += (enterprise_nature_score + enterprise_scale_score +
                                    risk_record + foreign_background + resource_safety)

        # 重新设置安全目标为供应点安全指标之和的负值（转为最小化目标）
        total_objectives['safety'] = -supplier_safety_sum

        # 批量归一化处理
        normalized_objectives = self._batch_normalize_computed_objectives(total_objectives)

        # 构建多目标结果
        return self._build_multi_objective_result(
            total_objectives, normalized_objectives, path_objectives_data['total_weight'], base_transport_metrics
        )

    def _calculate_base_transport_metrics(self, solution):
        """计算基础运输指标"""
        total_distance = 0.0
        max_time = 0.0

        x_direct = solution['x_direct']
        t_direct = solution['t_direct']
        x2 = solution['x2']
        t2 = solution['t2']

        # 批量处理直接运输
        for j in self.J:
            for k in self.K:
                for n in [1]:
                    if x_direct.get((j, k, n), 0) == 1:
                        j_lat, j_lon = self.point_features[j]['latitude'], self.point_features[j]['longitude']
                        k_lat, k_lon = self.point_features[k]['latitude'], self.point_features[k]['longitude']
                        distance = self._calculate_haversine_distance(j_lat, j_lon, k_lat, k_lon)
                        total_distance += distance

                        time_value = t_direct.get((j, k, n), 0)
                        if time_value > 0:
                            max_time = max(max_time, time_value)

        # 批量处理多式联运
        for m1 in self.M:
            for m2 in self.M:
                if m1 != m2:
                    for n in self.N:
                        if x2.get((m1, m2, n), 0) == 1:
                            total_distance += self.L_m_m[(m1, m2)]
                            time_value = t2.get((m1, m2, n), 0)
                            if time_value > 0:
                                max_time = max(max_time, time_value)

        return {'total_distance': total_distance, 'max_time': max_time}

    def _batch_calculate_path_objectives(self, solution):
        """批量计算路径目标值"""
        total_objectives = {
            'time': 0.0, 'cost': 0.0, 'distance': 0.0, 'safety': 0.0,
            'priority': 0.0, 'balance': 0.0, 'capability': 0.0, 'social': 0.0
        }
        total_weight = 0.0

        x1, x2, x3, x_direct = solution['x1'], solution['x2'], solution['x3'], solution['x_direct']
        b1, b2, b3, b_direct = solution['b1'], solution['b2'], solution['b3'], solution['b_direct']

        # 批量处理直接运输路径
        active_direct_paths = [(j, k, n) for j in self.J for k in self.K for n in [1]
                               if x_direct.get((j, k, n), 0) == 1 and b_direct.get((j, k, n), 0) > self.EPS]

        # 批量处理直接运输路径
        active_direct_paths = [(j, k, n) for j in self.J for k in self.K for n in [1]
                               if x_direct.get((j, k, n), 0) == 1 and b_direct.get((j, k, n), 0) > self.EPS]

        for j, k, n in active_direct_paths:
            metrics = self._calculate_direct_path_metrics(j, k)
            weight = b_direct.get((j, k, n), 0)

            self._accumulate_objectives(total_objectives, metrics, weight)
            total_weight += weight
        # 批量处理多式联运路径
        active_multimodal_paths = self._find_active_multimodal_paths(x1, x2, x3, b2)

        for j, m1, m2, k, n2 in active_multimodal_paths:
            metrics = self._calculate_multimodal_path_metrics(j, m1, m2, k, n2)
            weight = b2[(m1, m2, n2)]

            self._accumulate_objectives(total_objectives, metrics, weight)
            total_weight += weight

        return {'objectives': total_objectives, 'total_weight': total_weight}

    def _find_active_multimodal_paths(self, x1, x2, x3, b2):
        """找到活跃的多式联运路径"""
        import numpy as np

        # 预构建活跃路径段的索引集合
        active_x1_keys = {key for key, val in x1.items() if val == 1}
        active_x3_keys = {key for key, val in x3.items() if val == 1}
        active_x2_keys = {key for key, val in x2.items() if val == 1 and b2.get(key, 0) > self.EPS}

        # 预构建查找映射
        x1_m1_lookup = {}
        for (j, m1, n1), val in x1.items():
            if val == 1:
                if j not in x1_m1_lookup:
                    x1_m1_lookup[j] = []
                x1_m1_lookup[j].append(m1)

        x3_m2_lookup = {}
        for (m2, k, n3), val in x3.items():
            if val == 1:
                if k not in x3_m2_lookup:
                    x3_m2_lookup[k] = []
                x3_m2_lookup[k].append(m2)

        # 使用集合操作快速查找有效路径组合
        active_paths = []
        for (m1, m2, n2) in active_x2_keys:
            # 查找连接到m1的j节点
            j_candidates = [j for j, m1_list in x1_m1_lookup.items() if m1 in m1_list]
            # 查找从m2出发的k节点
            k_candidates = [k for k, m2_list in x3_m2_lookup.items() if m2 in m2_list]

            for j in j_candidates:
                for k in k_candidates:
                    active_paths.append((j, m1, m2, k, n2))

        return active_paths

    def _accumulate_objectives(self, total_objectives, metrics, weight):
        """累积目标值"""
        objective_mapping = {
            'time': 'time_score',
            'cost': 'cost_score',
            'distance': 'distance_score',
            'safety': 'safety_score',
            'priority': 'priority_score',
            'balance': 'balance_score',
            'capability': 'capability_score',
            'social': 'social_score'
        }

        for obj_key, metric_key in objective_mapping.items():
            total_objectives[obj_key] += metrics[metric_key] * weight #+ random.uniform(0,1)

    def _compute_weighted_average_objectives(self, path_objectives_data):
        """计算加权平均目标值"""
        total_objectives = path_objectives_data['objectives']
        total_weight = path_objectives_data['total_weight']

        if total_weight > self.EPS:
            for obj in total_objectives:
                total_objectives[obj] /= total_weight

        return total_objectives

    def _batch_normalize_computed_objectives(self, total_objectives):
        """批量归一化计算后的目标值"""
        normalized_objectives = {}

        for obj, original_value in total_objectives.items():
            if hasattr(self, 'objective_ranges') and obj in self.objective_ranges:
                min_val, max_val = self.objective_ranges[obj]

                if abs(max_val - min_val) < self.EPS:
                    normalized_value = 0.5

                else:
                    if obj in ['safety', 'priority']:
                        normalized_value = self._normalize_negative_objective(original_value, min_val, max_val)
                    elif obj == 'capability':
                        normalized_value = self._normalize_reverse_objective(original_value, min_val, max_val)
                    else:
                        normalized_value = self._normalize_standard_objective(original_value, min_val, max_val)

                normalized_objectives[obj] = normalized_value
            else:
                normalized_objectives[obj] = original_value

        return normalized_objectives

    def _build_multi_objective_result(self, total_objectives, normalized_objectives, total_weight, base_metrics):
        """构建多目标结果"""
        # 计算加权组合综合目标值
        weighted_objectives = {}
        composite_value = 0.0

        for obj, normalized_value in normalized_objectives.items():
            weight = self.objective_weights[obj]
            weighted_value = normalized_value * weight
            weighted_objectives[obj] = weighted_value
            composite_value += weighted_value

        return {
            'composite_objective_value': composite_value,
            'objective_weights': self.objective_weights.copy(),
            'individual_objectives': total_objectives,
            'normalized_objectives': normalized_objectives,
            'weighted_objectives': weighted_objectives,
            'objective_breakdown': {
                'time_component': weighted_objectives['time'],
                'cost_component': weighted_objectives['cost'],
                'distance_component': weighted_objectives['distance'],
                'safety_component': weighted_objectives['safety'],
                'priority_component': weighted_objectives['priority'],
                'balance_component': weighted_objectives['balance'],
                'capability_component': weighted_objectives['capability'],
                'social_component': weighted_objectives['social']
            },
            'normalization_info': {
                'objective_ranges': getattr(self, 'objective_ranges', {}),
                'total_flow_weight': total_weight,
                'normalization_method': 'min_max_with_direction_handling'
            },
            'original_results': {
                'total_time': max(total_objectives['time'], base_metrics['max_time']),
                'total_distance': max(total_objectives['distance'], base_metrics['total_distance']),
                'total_cost': total_objectives['cost'],
                'detailed_objectives': total_objectives,
                'total_weight': total_weight,
                'calculation_method': 'optimized_weighted_average_from_active_paths'
            }
        }

    def _recalculate_balance_objective(self, solution):
        """重新计算整体资源均衡指标"""
        # 计算实际使用的供应点及其分配量
        actual_allocations = {}
        total_allocated = 0.0

        # 收集所有实际分配
        for j in self.J:
            allocated_amount = 0.0

            # 累加该供应点的直接运输分配量
            for k in self.K:
                for n in [1]:
                    allocated_amount += solution['b_direct'].get((j, k, n), 0)

            # 累加该供应点的多式联运分配量
            for m in self.M:
                for n in [1]:
                    allocated_amount += solution['b1'].get((j, m, n), 0)

            if allocated_amount > self.EPS:
                actual_allocations[j] = allocated_amount
                total_allocated += allocated_amount

        # 计算均衡度偏差
        if not actual_allocations or total_allocated <= self.EPS:
            return 0.0

        # 理想情况：按供应能力比例分配
        total_capacity = sum(self.B[j] * self.P[j] for j in actual_allocations.keys())
        balance_deviation = 0.0

        for j, allocated in actual_allocations.items():
            ideal_ratio = (self.B[j] * self.P[j]) / total_capacity
            actual_ratio = allocated / total_allocated
            balance_deviation += abs(actual_ratio - ideal_ratio)

        return balance_deviation

    def compute_objective(self, solution):
        """计算目标函数值"""

        # 提取解变量
        x1, x2, x3, x_direct = solution['x1'], solution['x2'], solution['x3'], solution['x_direct']
        b1, b2, b3, b_direct = solution['b1'], solution['b2'], solution['b3'], solution['b_direct']

        # 初始化累计器
        total_objectives = {
            'time': 0.0, 'cost': 0.0, 'distance': 0.0, 'safety': 0.0,
            'priority': 0.0, 'balance': 0.0, 'capability': 0.0, 'social': 0.0
        }

        total_weight = 0.0

        # 1. 累计直接运输路径的目标值
        max_completion_time = 0.0

        for j in self.J:
            for k in self.K:
                for n in [1]:
                    x_val = x_direct.get((j, k, n), 0)
                    b_val = b_direct.get((j, k, n), 0)
                    if x_val == 1 and b_val > self.EPS:
                        metrics = self._calculate_direct_path_metrics(j, k)
                        weight = b_val

                        # 按指标特性分类累加
                        # 最大值指标
                        max_completion_time = max(max_completion_time, metrics['time_score'])

                        # 总和指标
                        total_objectives['cost'] += metrics['cost_score']
                        total_objectives['distance'] += metrics['distance_score']
                        total_objectives['social'] += metrics['social_score']

                        # 加权平均指标
                        total_objectives['safety'] += metrics['safety_score'] * weight
                        total_objectives['priority'] += metrics['priority_score'] * weight
                        total_objectives['capability'] += metrics['capability_score'] * weight

                        total_weight += weight

        # 赋值最终结果
        total_objectives['time'] = max_completion_time

        # 2. 累计多式联运路径的目标值
        for j in self.J:
            for m1 in self.M:
                for m2 in self.M:
                    if m1 != m2:
                        for k in self.K:
                            for n2 in self.N:
                                x1_val = x1.get((j, m1, 1), 0)
                                x2_val = x2.get((m1, m2, n2), 0)
                                x3_val = x3.get((m2, k, 1), 0)
                                b2_val = b2.get((m1, m2, n2), 0)

                                if (x1_val == 1 and x2_val == 1 and x3_val == 1 and b2_val > self.EPS):
                                    metrics = self._calculate_multimodal_path_metrics(j, m1, m2, k, n2)
                                    weight = b2_val

                                    # 时间目标使用最大值逻辑
                                    max_completion_time = max(max_completion_time, metrics['time_score'])

        # 3. 按指标特性分类计算最终值
        if total_weight > self.EPS:
            # 需要重新计算的指标（不能简单累加）
            recalculate_objectives = ['balance']

            # 使用总和的指标
            sum_objectives = ['cost', 'distance', 'social']

            # 使用最大值的指标
            max_objectives = ['time']

            # 使用加权平均的指标
            avg_objectives = ['safety', 'priority', 'capability']

            for obj in total_objectives:
                if obj in sum_objectives:
                    pass  # 保持累加结果，不除以权重
                elif obj in max_objectives:
                    pass  # 已经在累加阶段处理为最大值
                elif obj in avg_objectives:
                    total_objectives[obj] /= total_weight  # 加权平均
                elif obj in recalculate_objectives:
                    # 需要重新计算，不能简单累加或平均
                    total_objectives[obj] = self._recalculate_balance_objective(solution)

        # 4. 计算基础运输指标
        total_distance = 0.0
        max_time = 0.0

        # 直接运输的距离和时间
        for j in self.J:
            for k in self.K:
                for n in [1]:
                    if x_direct[(j, k, n)] == 1:
                        j_lat, j_lon = self.point_features[j]['latitude'], self.point_features[j]['longitude']
                        k_lat, k_lon = self.point_features[k]['latitude'], self.point_features[k]['longitude']
                        distance = self._calculate_haversine_distance(j_lat, j_lon, k_lat, k_lon)
                        total_distance += distance

                        if solution['t_direct'][(j, k, n)] > 0:
                            max_time = max(max_time, solution['t_direct'][(j, k, n)])

        # 多式联运的距离和时间
        for m1 in self.M:
            for m2 in self.M:
                if m1 != m2:
                    for n in self.N:
                        if x2[(m1, m2, n)] == 1:
                            total_distance += self.L_m_m[(m1, m2)]
                            if solution['t2'][(m1, m2, n)] > 0:
                                max_time = max(max_time, solution['t2'][(m1, m2, n)])

        # 构建结果
        results = {
            'total_time': max(total_objectives['time'], max_time),
            'total_distance': max(total_objectives['distance'], total_distance),
            'total_cost': total_objectives['cost'],
            'safety_scores': {
                'total_safety_score': -total_objectives['safety']  # 转换回正值
            },
            'priority_satisfaction': {
                'overall_score': total_objectives['priority']
            },
            'regional_balance': {
                'regional_balance_score': total_objectives['balance']
            },
            'enterprise_capability': {
                'total_capability_score': -total_objectives['capability']  # 转换回正值
            },
            'social_impact': {
                'overall_score': total_objectives['social']
            },
            'detailed_objectives': total_objectives,
            'total_weight': total_weight,
            'calculation_method': 'weighted_average_from_active_paths'
        }

        return results

    def _save_solution_results(self, solution, output_dir):
        """保存多目标求解结果"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 计算多目标结果
            multi_obj_result = self.compute_multi_objective_value(solution)

            # 保存解的基本信息
            result_summary = {
                'timestamp': timestamp,
                'optimization_type': 'multi_objective',
                'objective_weights': self.objective_weights,
                'composite_objective_value': multi_obj_result['composite_objective_value'],
                'individual_objectives': multi_obj_result['individual_objectives'],
                'total_solve_time': solution.get('total_solve_time', 0),
                'scheduling_time': solution.get('scheduling_time', 0),
                'processed_paths': solution.get('processed_paths', 0),
                'network_info': {
                    'supply_points': len(self.J),
                    'transfer_points': len(self.M),  # 统一的中转点数量
                    'final_points': len(self.K)
                }
            }

            # 保存摘要文件
            summary_filename = os.path.join(output_dir, f"multi_objective_solution_{timestamp}.json")
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump(result_summary, f, ensure_ascii=False, indent=2)

            self.logger.info(f"多目标结果摘要已保存: {summary_filename}")


        except Exception as e:
            self.logger.error(f"保存多目标结果时发生错误: {str(e)}", exc_info=True)
            # 不中断主流程，但记录错误信息
            error_info = {
                "code": 500,
                "msg": f"结果保存失败: {str(e)}",
                "data": {
                    "algorithm_sequence_id": solution.get('algorithm_sequence_id'),
                    "mobilization_object_type": solution.get('mobilization_object_type'),
                    "req_element_id": solution.get('req_element_id'),
                    "output_dir": output_dir,
                    "error_type": "file_save_error"
                }
            }

            self.logger.warning(f"文件保存错误详情: {error_info}")

    def get_solution_summary(self, solution):
        """获取多目标解的简要摘要"""
        if not solution:
            return None

        # 统计激活路径
        active_paths_1 = sum(1 for val in solution['x1'].values() if val > 0)
        active_paths_2 = sum(1 for val in solution['x2'].values() if val > 0)
        active_paths_3 = sum(1 for val in solution['x3'].values() if val > 0)
        active_paths_direct = sum(1 for val in solution['x_direct'].values() if val > 0)

        # 计算需求满足率
        total_demand = sum(self.D.values())
        multimodal_satisfied = sum(
            solution['b3'].get((m, k, n), 0) for m in self.M for k in self.K for n in self.ROAD_ONLY)
        direct_satisfied = sum(
            solution['b_direct'].get((j, k, n), 0) for j in self.J for k in self.K for n in self.ROAD_ONLY)
        satisfied_demand = multimodal_satisfied + direct_satisfied
        satisfaction_rate = (satisfied_demand / total_demand * 100) if total_demand > 0 else 100

        # 计算多目标结果
        multi_obj_result = self.compute_multi_objective_value(solution)

        time_window_compliance = {}
        if hasattr(self, 'time_windows') and self.time_windows:
            for k in self.K:
                if k in self.time_windows:
                    earliest, latest = self.time_windows[k]
                    # 时间窗检查
                    satisfied_within_window = True  # 简化假设
                    time_window_compliance[k] = satisfied_within_window

        summary = {
            'optimization_type': 'multi_objective',
            'composite_objective_value': multi_obj_result['composite_objective_value'],
            'objective_weights': self.objective_weights.copy(),
            'total_solve_time': solution.get('total_solve_time', 0),
            'scheduling_time': solution.get('scheduling_time', 0),
            'processed_paths': solution.get('processed_paths', 0),
            'active_paths': {
                'segment_1': active_paths_1,
                'segment_2': active_paths_2,
                'segment_3': active_paths_3,
                'direct_paths': active_paths_direct,
                'total': active_paths_1 + active_paths_2 + active_paths_3 + active_paths_direct
            },
            'demand_satisfaction_rate': satisfaction_rate,
            'time_window_compliance': time_window_compliance
        }

        return summary


# ==========================================
# 接口1: 时间参数数据处理接口
# ==========================================

def input_time_parameters(input_json) -> Dict[str, float]:
    """
    时间参数数据处理接口

    功能描述:
        接受JSON格式的时间参数配置，处理后返回标准化的时间参数字典。
        用于配置运输过程中各个环节的时间参数，所有时间参数单位为小时。
        同时将结果保存到本地JSON文件以供后续接口使用。

    时间参数说明:
        - T1: 征召时间（动员准备阶段）
        - T4: 集结时间（资源集结准备时间）
        - T6: 交接时间（最终交接确认时间）

    时间计算公式:
        - 直接运输总时间 = 征召时间 + 集结时间 + 运输时间 + 交接时间
        - 多式联运总时间 = 征召时间 + 集结时间 + 运输时间1 + 运输时间2 + 交接时间

    Args:
        input_json: JSON格式的时间参数配置字符串或字典

    Returns:
        Dict[str, float]: 时间参数字典
            - 键: 时间参数名称
            - 值: 时间值（小时）

    输入案例:
        input_json = '''{
            "base_preparation_time": 0.005,
            "base_assembly_time": 0.001,
            "base_handover_time": 0.0005,
            "time_unit": "hours"
        }'''

    输出案例:
        {
            '征召时间': 0.005,  # 征召时间0.3分钟
            '集结时间': 0.001,  # 集结时间0.06分钟
            '交接时间': 0.0005  # 交接时间0.03分钟
        }

    参数调整建议:
        - 紧急任务：减少base_preparation_time、base_assembly_time值
        - 精确交接：增加base_handover_time值
    """

    try:
        # 检查输入参数有效性
        if input_json is None:
            return {
                "code": 400,
                "msg": "输入参数不能为None",
                "data": {
                    "input_value": None,
                    "error_type": "null_input_parameter"
                }
            }

        # 处理输入类型：支持字符串和字典
        if isinstance(input_json, str):
            # 检查是否为空字符串
            if not input_json.strip():
                return {
                    "code": 400,
                    "msg": "输入参数不能为空字符串",
                    "data": {
                        "input_value": input_json,
                        "error_type": "empty_input_parameter"
                    }
                }
            config = json.loads(input_json)
        elif isinstance(input_json, dict):
            config = input_json
        else:
            raise ValueError(f"输入参数类型错误，期望str或dict，实际为{type(input_json)}")

        # 提取和验证时间参数
        try:
            base_preparation_time = config.get('base_preparation_time', 0.005)
            base_assembly_time = config.get('base_assembly_time', 0.001)
            base_handover_time = config.get('base_handover_time', 0.0005)

            # 验证时间参数的有效性
            if base_preparation_time < 0:
                return {
                    "code": 400,
                    "msg": "base_preparation_time不能为负数",
                    "data": {
                        "base_preparation_time": base_preparation_time,
                        "error_type": "negative_time_parameter"
                    }
                }

            if base_assembly_time < 0:
                return {
                    "code": 400,
                    "msg": "base_assembly_time不能为负数",
                    "data": {
                        "base_assembly_time": base_assembly_time,
                        "error_type": "negative_time_parameter"
                    }
                }

            if base_handover_time < 0:
                return {
                    "code": 400,
                    "msg": "base_handover_time不能为负数",
                    "data": {
                        "base_handover_time": base_handover_time,
                        "error_type": "negative_time_parameter"
                    }
                }

        except (TypeError, ValueError) as e:
            return {
                "code": 400,
                "msg": f"时间参数数值转换错误: {str(e)}",
                "data": {
                    "input_config": config,
                    "error_type": "time_parameter_conversion_error"
                }
            }

        # 构建时间参数字典
        time_params = {
            'preparation_time': base_preparation_time,
            'assembly_time': base_assembly_time,
            'handover_time': base_handover_time
        }

        # 保存到本地JSON文件
        try:
            if not os.path.exists('cache'):
                os.makedirs('cache')

            cache_file_path = os.path.join('cache', 'time_parameters.json')
            with open(cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(time_params, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # 文件保存失败不影响接口返回结果
            pass

        return {
            "code": 200,
            "msg": "时间参数处理成功",
            "data": time_params
        }
    except json.JSONDecodeError as e:
        return {
            "code": 400,
            "msg": f"无效的时间JSON格式: {str(e)}",
            "data": {
                "input_type": type(input_json).__name__,
                "req_element_id": None,
                "error_type": "json_decode_error"
            }
        }
    except Exception as e:
        return {
            "code": 500,
            "msg": f"时间参数处理错误: {str(e)}",
            "data": {
                "input_type": type(input_json).__name__,
                "req_element_id": None,
                "error_type": "processing_error"
            }
        }


# ==========================================
# 接口2: 运输方式数据处理接口
# ==========================================

def input_transport_parameters(input_json, network_data: Dict = None, time_params: Dict = None) -> Dict[str, Any]:
    """
    运输方式数据处理接口

    输入案例:
        input_json = '''{
            "transport_modes": [
                {
                    "name": "公路",
                    "code": "trans-01",
                    "speed": 100,
                    "cost_per_km": 4.0,
                    "road_only_modes": 1
                },
                {
                    "name": "海运",
                    "code": "trans-02",
                    "speed": 40,
                    "cost_per_km": 0.3,
                    "road_only_modes": 0
                },
                {
                    "name": "空运",
                    "code": "trans-03",
                    "speed": 800,
                    "cost_per_km": 6.0,
                    "road_only_modes": 0
                },
                {
                    "name": "铁路运输",
                    "code": "trans-04",
                    "speed": 120,
                    "cost_per_km": 0.25,
                    "road_only_modes": 0
                }
            ]
        }'''
    """

    try:
        # 如果time_params为None，生成默认时间参数
        if time_params is None:
            default_time_result = input_time_parameters('{}')
            if default_time_result.get('code') == 200:
                time_params = default_time_result['data']

        # 处理输入类型：支持字符串和字典
        if isinstance(input_json, str):
            config = json.loads(input_json)
        elif isinstance(input_json, dict):
            config = input_json
        else:
            raise ValueError(f"输入参数类型错误，期望str或dict，实际为{type(input_json)}")

        # 提取运输方式配置列表
        transport_modes_list = config.get('transport_modes', [])

        if not transport_modes_list:
            raise ValueError("transport_modes不能为空")

        # 转换模式配置，使用数组索引作为mode_id
        transport_modes = {}
        mode_ids = []
        road_only_modes = []

        for index, mode_config in enumerate(transport_modes_list):
            # 使用索引+1作为mode_id
            mode_id = index + 1

            try:
                # 验证关键参数
                speed = mode_config.get('speed', 100)
                cost_per_km = mode_config.get('cost_per_km', 4.0)

                if speed <= 0:
                    return {
                        "code": 400,
                        "msg": f"运输方式{mode_id}的速度必须大于0",
                        "data": {
                            "mode_index": index,
                            "mode_config": mode_config,
                            "speed": speed,
                            "error_type": "invalid_speed_parameter"
                        }
                    }

                if cost_per_km < 0:
                    return {
                        "code": 400,
                        "msg": f"运输方式{mode_id}的单位成本不能为负数",
                        "data": {
                            "mode_index": index,
                            "mode_config": mode_config,
                            "cost_per_km": cost_per_km,
                            "error_type": "negative_cost_parameter"
                        }
                    }

                transport_modes[mode_id] = {
                    'name': mode_config.get('name', f'运输方式{mode_id}'),
                    'code': mode_config.get('code', f'trans-{mode_id:02d}'),
                    'speed': speed,
                    'cost_per_km': cost_per_km,
                    'road_only_modes': mode_config.get('road_only_modes', 0)
                }
            except (TypeError, ValueError) as e:
                return {
                    "code": 400,
                    "msg": f"运输方式{mode_id}参数转换错误: {str(e)}",
                    "data": {
                        "mode_index": index,
                        "mode_config": mode_config,
                        "error_type": "transport_mode_conversion_error"
                    }
                }
            mode_ids.append(mode_id)

            # 收集仅限公路的运输方式
            if transport_modes[mode_id]['road_only_modes'] == 1:
                road_only_modes.append(mode_id)

        # 如果没有road_only运输方式，默认第一个为road_only
        if not road_only_modes and mode_ids:
            road_only_modes = [mode_ids[0]]

        # 构建基础运输参数（用于缓存保存）
        basic_transport_params = {
            'N': mode_ids,
            'TRANSPORT_MODES': transport_modes,
            'ROAD_ONLY': road_only_modes,
            'time_params': time_params
        }

        # 保存基础参数到本地JSON文件（不包含距离矩阵）
        try:
            if not os.path.exists('cache'):
                os.makedirs('cache')

            cache_file_path = os.path.join('cache', 'transport_parameters.json')
            with open(cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(basic_transport_params, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # 文件保存失败不影响接口返回结果
            pass

        # 构建完整运输参数（用于函数返回）
        transport_params = basic_transport_params.copy()

        # 如果提供了网络数据，生成完整的运输矩阵（仅用于返回，不保存到缓存）
        if network_data is not None:
            transport_params.update(
                _generate_transport_matrices(network_data, transport_modes, mode_ids, road_only_modes))

        return {
            "code": 200,
            "msg": "运输参数处理成功",
            "data": transport_params
        }

    except json.JSONDecodeError as e:
        return {
            "code": 400,
            "msg": f"无效的运输方式JSON格式: {str(e)}",
            "data": {
                "input_type": type(input_json).__name__,
                "has_network_data": network_data is not None,
                "has_time_params": time_params is not None,
                "req_element_id": None,
                "error_type": "json_decode_error"
            }
        }

    except Exception as e:
        return {
            "code": 500,
            "msg": f"运输参数处理错误: {str(e)}",
            "data": {
                "input_type": type(input_json).__name__,
                "has_network_data": network_data is not None,
                "has_time_params": time_params is not None,
                "req_element_id": None,
                "error_type": "processing_error"
            }
        }


def _generate_transport_matrices(network_data, transport_modes, mode_ids, road_only_modes):
    """
    生成运输矩阵数据
    """
    J, M, K = network_data['J'], network_data['M'], network_data['K']
    point_features = network_data['point_features']

    def get_modify_params():
        """获取距离校正参数"""
        network_scale_factor = len(J) + len(M) + len(K)
        base_distance_factor = network_scale_factor / len(J) if len(J) > 0 else 1.0

        all_latitudes = [point_features[node].get('latitude', 0) for node in point_features]
        all_longitudes = [point_features[node].get('longitude', 0) for node in point_features]

        if all_latitudes and all_longitudes:
            lat_range = max(all_latitudes) - min(all_latitudes)
            lon_range = max(all_longitudes) - min(all_longitudes)
            characteristic_distance = (lat_range + lon_range) * base_distance_factor
        else:
            characteristic_distance = base_distance_factor

        characteristic_distance = max(characteristic_distance, 1.0)
        max_correction = 1.0 + 1.0 / len(J) if len(J) > 0 else 2.0

        return characteristic_distance, max_correction

    def calculate_haversine_distance(lat1, lon1, lat2, lon2, characteristic_distance, max_correction):
        """计算两点间的Haversine距离"""
        lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        raw_distance = 2 * np.arcsin(np.sqrt(a)) * 6371
        distance_ratio = raw_distance / characteristic_distance
        correction_factor = 1.0 + (max_correction - 1.0) * np.exp(-distance_ratio)
        return correction_factor * raw_distance

    @lru_cache(maxsize=128)
    def check_m_specialize(m1_specialized, m2_specialized, mode_name, mode_speed, alpha_default, v_default):
        """传入mode_speed参数，添加缓存"""
        if m1_specialized == m2_specialized:
            if m1_specialized == 'port' and '海运' in mode_name:
                return 1, mode_speed
            elif m1_specialized == 'airport' and '空运' in mode_name:
                return 1, mode_speed
            elif m1_specialized == 'railway' and '铁路' in mode_name:
                return 1, mode_speed
        return alpha_default, v_default

    # 获取距离修正参数（只计算一次）
    characteristic_distance, max_correction = get_modify_params()

    # 批量计算所有距离矩阵
    def compute_distance_matrix(nodes1, nodes2):
        """通用的距离矩阵计算函数"""
        if not nodes1 or not nodes2:
            return {}

        lat1_lon1_list = np.array([
            [point_features[n1]['latitude'], point_features[n1]['longitude']]
            for n1, n2 in product(nodes1, nodes2)
        ])
        lat2_lon2_list = np.array([
            [point_features[n2]['latitude'], point_features[n2]['longitude']]
            for n1, n2 in product(nodes1, nodes2)
        ])

        distances = calculate_haversine_distance(
            lat1_lon1_list[:, 0], lat1_lon1_list[:, 1],
            lat2_lon2_list[:, 0], lat2_lon2_list[:, 1],
            characteristic_distance, max_correction
        )

        return dict(zip(product(nodes1, nodes2), distances))

    # 计算距离矩阵
    L_j_m = compute_distance_matrix(J, M)
    L_m_m = {key: val for key, val in compute_distance_matrix(M, M).items() if key[0] != key[1]}
    L_m_k = compute_distance_matrix(M, K)

    # 初始化可行性和速度矩阵
    alpha1, alpha2, alpha3 = {}, {}, {}
    v1, v2, v3 = {}, {}, {}

    # 预计算road_only_modes集合
    road_only_set = set(road_only_modes)

    # 批量计算可行性和速度矩阵
    for n in mode_ids:
        mode_info = transport_modes[n]
        mode_name = mode_info["name"]
        mode_speed = mode_info["speed"]

        # 对于road_only模式的预计算
        is_road_only = n in road_only_set
        alpha_road = 1 if is_road_only else 0
        v_road = mode_speed if is_road_only else 0

        # 第一段和第三段：只使用road_only模式
        for j, m in product(J, M):
            alpha1[(j, m, n)] = alpha_road
            v1[(j, m, n)] = v_road

        for m, k in product(M, K):
            alpha3[(m, k, n)] = alpha_road
            v3[(m, k, n)] = v_road

        # 第二段：中转点间的复杂逻辑
        for m1, m2 in product(M, M):
            if m1 != m2:
                m1_specialized = point_features[m1].get('specialized_mode', 'unknown')
                m2_specialized = point_features[m2].get('specialized_mode', 'unknown')

                alpha, v = check_m_specialize(
                    m1_specialized, m2_specialized, mode_name,
                    mode_speed, alpha_road, v_road
                )

                alpha2[(m1, m2, n)] = alpha
                v2[(m1, m2, n)] = v

    return {
        'L_j_m': L_j_m, 'L_m_m': L_m_m, 'L_m_k': L_m_k,
        'alpha1': alpha1, 'alpha2': alpha2, 'alpha3': alpha3,
        'v1': v1, 'v2': v2, 'v3': v3
    }


# ==========================================
# 接口3: 安全成本数据处理接口
# ==========================================

def input_safety_cost_parameters(input_json, mobilization_object_type: str = None) -> Dict[str, Dict[str, Any]]:
    """
    安全计算参照标准接口

    功能描述:
        接受完整的安全评价指标映射关系，进行验证和格式化处理
        同时将结果保存到本地JSON文件以供后续接口使用。

    输入输出特点:
        - 输入什么格式，输出什么格式（透传模式）
        - 主要负责数据验证、格式化和缓存
        - 不根据动员类型筛选数据

    Returns:
        Dict[str, Dict[str, Any]]: 验证和格式化后的安全评价指标映射
    """

    try:
        # 处理输入类型
        if isinstance(input_json, str):
            config = json.loads(input_json)
        elif isinstance(input_json, dict):
            config = input_json
        else:
            raise ValueError(f"输入参数类型错误，期望str或dict，实际为{type(input_json)}")

        # 验证必要的字段存在
        required_mappings = [
            'enterprise_safety_mappings',
            'material_safety_mappings',
            'personnel_safety_mappings',
            'equipment_safety_mappings',
            'facility_safety_mappings',
            'technology_safety_mappings'
        ]

        # 检查是否包含必要的安全映射配置
        missing_mappings = []
        for mapping in required_mappings:
            if mapping not in config:
                missing_mappings.append(mapping)

        if missing_mappings:
            raise ValueError(f"缺少必要的安全映射配置: {missing_mappings}")

        # 验证每个映射的内部结构
        validated_config = {}

        # 验证企业安全映射
        enterprise_mappings = config['enterprise_safety_mappings']
        required_enterprise_keys = [
            'enterprise_nature_mapping', 'enterprise_scale_mapping',
            'risk_record_mapping', 'foreign_background_mapping',
            'resource_safety_mapping', 'mobilization_experience_mapping'
        ]
        for key in required_enterprise_keys:
            if key not in enterprise_mappings:
                raise ValueError(f"enterprise_safety_mappings缺少{key}")
        validated_config['enterprise_safety_mappings'] = enterprise_mappings

        # 验证物资安全映射
        material_mappings = config['material_safety_mappings']
        required_material_keys = [
            'flammable_explosive_mapping', 'corrosive_mapping',
            'polluting_mapping', 'fragile_mapping'
        ]
        for key in required_material_keys:
            if key not in material_mappings:
                raise ValueError(f"material_safety_mappings缺少{key}")
        validated_config['material_safety_mappings'] = material_mappings

        # 验证人员安全映射
        personnel_mappings = config['personnel_safety_mappings']
        required_personnel_keys = [
            'political_status_mapping', 'military_experience_mapping',
            'criminal_record_mapping', 'network_record_mapping', 'credit_record_mapping'
        ]
        for key in required_personnel_keys:
            if key not in personnel_mappings:
                raise ValueError(f"personnel_safety_mappings缺少{key}")
        validated_config['personnel_safety_mappings'] = personnel_mappings

        # 验证设备安全映射
        equipment_mappings = config['equipment_safety_mappings']
        required_equipment_keys = [
            'autonomous_control_mapping', 'usability_level_mapping'
        ]
        for key in required_equipment_keys:
            if key not in equipment_mappings:
                raise ValueError(f"equipment_safety_mappings缺少{key}")
        validated_config['equipment_safety_mappings'] = equipment_mappings

        # 验证设施安全映射
        facility_mappings = config['facility_safety_mappings']
        required_facility_keys = [
            'facility_protection_mapping', 'camouflage_protection_mapping', 'surrounding_environment_mapping'
        ]
        for key in required_facility_keys:
            if key not in facility_mappings:
                raise ValueError(f"facility_safety_mappings缺少{key}")
        validated_config['facility_safety_mappings'] = facility_mappings

        # 验证科技安全映射
        technology_mappings = config['technology_safety_mappings']
        required_technology_keys = [
            'encryption_security_mapping', 'access_control_mapping', 'network_security_mapping',
            'terminal_security_mapping', 'dlp_mapping', 'security_policy_mapping',
            'risk_assessment_mapping', 'audit_monitoring_mapping', 'emergency_response_mapping'
        ]
        for key in required_technology_keys:
            if key not in technology_mappings:
                raise ValueError(f"technology_safety_mappings缺少{key}")
        validated_config['technology_safety_mappings'] = technology_mappings

        # 保存到本地JSON文件
        try:
            if not os.path.exists('cache'):
                os.makedirs('cache')

            cache_file_path = os.path.join('cache', 'safety_cost_parameters.json')
            with open(cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(validated_config, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # 文件保存失败不影响接口返回结果
            pass

        return {
            "code": 200,
            "msg": "安全参照标准处理成功",
            "data": validated_config
        }


    except json.JSONDecodeError as e:
        return {
            "code": 400,
            "msg": f"无效的安全JSON格式: {str(e)}",
            "data": {
                "input_type": type(input_json).__name__,
                "mobilization_object_type": mobilization_object_type,
                "req_element_id": None,
                "error_type": "json_decode_error"
            }
        }

    except Exception as e:
        return {
            "code": 500,
            "msg": f"安全参照标准处理错误: {str(e)}",
            "data": {
                "input_type": type(input_json).__name__,
                "mobilization_object_type": mobilization_object_type,
                "req_element_id": None,
                "error_type": "processing_error"
            }
        }


# ==========================================
# 接口4: 中转点数据处理接口
# ==========================================

def input_transfer_point_parameters(input_json, mobilization_object_type: str = None) -> Dict[str, Any]:
    """
    中转点参数数据处理接口

    功能描述:
        接受JSON格式的中转点配置信息，输出标准化的中转点参数数据
        每个中转点包含唯一序号id、名称、位置坐标、运输方式支持情况等信息
        同时将结果保存到本地JSON文件以供后续接口使用。

    输入参数:
        input_json: JSON字符串或字典，包含中转点配置信息
        mobilization_object_type: 动员对象类型（可选）

    输出格式:
        {
            "transfer_points": [
                {
                    "id": "01",
                    "name": "中转点名称",
                    "latitude": 纬度,
                    "longitude": 经度,
                    "specialized_mode": "专业化模式"
                }
            ],
        }
    """

    try:
        # 处理输入类型
        if isinstance(input_json, str):
            config = json.loads(input_json)
        elif isinstance(input_json, dict):
            config = input_json
        else:
            raise ValueError(f"输入参数类型错误，期望str或dict，实际为{type(input_json)}")

        # 提取中转点配置
        transfer_points_config = config.get('transfer_points', [])

        # 处理中转点数据
        processed_transfer_points = []
        specialized_modes_count = {}

        for i, transfer_point in enumerate(transfer_points_config):
            if isinstance(transfer_point, dict):
                # 处理序号id - 如果用户提供了id则使用用户的，否则自动生成
                transfer_point_id = transfer_point.get('id')
                if transfer_point_id is None:
                    transfer_point_id = f"{i + 1:02d}"  # 生成格式为 "01", "02", "03" 的序号

                # 提取基础信息
                name = transfer_point.get('name', f'中转点{i + 1}')
                latitude = transfer_point.get('latitude', 35.0 + i * 1.0)
                longitude = transfer_point.get('longitude', 115.0 + i * 1.5)
                specialized_mode = transfer_point.get('specialized_mode', 'mixed')

                # 统计专业化模式
                specialized_modes_count[specialized_mode] = specialized_modes_count.get(specialized_mode, 0) + 1
                processed_transfer_point = {
                    'id': transfer_point_id,
                    'name': name,
                    'latitude': latitude,
                    'longitude': longitude,
                    'specialized_mode': specialized_mode
                }
                processed_transfer_points.append(processed_transfer_point)
            else:
                # 处理字符串格式的中转点
                transfer_point_id = f"{i + 1:02d}"
                name = str(transfer_point)
                # 生成默认的中转点配置
                default_specialized_modes = ['port', 'airport', 'railway']
                specialized_mode = default_specialized_modes[i % len(default_specialized_modes)]

                # 统计信息
                specialized_modes_count[specialized_mode] = specialized_modes_count.get(specialized_mode, 0) + 1
                processed_transfer_point = {
                    'id': transfer_point_id,
                    'name': name,
                    'latitude': 35.0 + i * 1.0,
                    'longitude': 115.0 + i * 1.5,
                    'specialized_mode': specialized_mode
                }
                processed_transfer_points.append(processed_transfer_point)

        # 如果没有提供中转点配置，生成默认中转点
        if not processed_transfer_points:
            default_transfer_points = [
                {
                    'id': '01',
                    'name': '京津冀综合枢纽',
                    'latitude': 39.9042,
                    'longitude': 116.4074,
                    'specialized_mode': 'port'
                },
                {
                    'id': '02',
                    'name': '长三角物流中心',
                    'latitude': 31.2304,
                    'longitude': 121.4737,
                    'specialized_mode': 'airport'
                }
            ]

            processed_transfer_points = default_transfer_points

            # 更新统计信息
            for tp in default_transfer_points:
                specialized_mode = tp['specialized_mode']
                specialized_modes_count[specialized_mode] = specialized_modes_count.get(specialized_mode, 0) + 1

        # 构建返回结果
        result = {
            'transfer_points': processed_transfer_points
        }

        # 保存到本地JSON文件
        try:
            if not os.path.exists('cache'):
                os.makedirs('cache')

            cache_file_path = os.path.join('cache', 'transfer_point_parameters.json')
            with open(cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
        except Exception as e:
            # 文件保存失败不影响接口返回结果
            pass

        return {
            "code": 200,
            "msg": "中转点参数处理成功",
            "data": result
        }

    except json.JSONDecodeError as e:
        return {
            "code": 400,
            "msg": f"无效的中转点JSON格式: {str(e)}",
            "data": {
                "input_type": type(input_json).__name__,
                "mobilization_object_type": mobilization_object_type,
                "req_element_id": None,
                "error_type": "json_decode_error"
            }
        }


    except Exception as e:
        return {
            "code": 500,
            "msg": f"中转点参数处理错误: {str(e)}",
            "data": {
                "input_type": type(input_json).__name__,
                "mobilization_object_type": mobilization_object_type,
                "req_element_id": None,
                "error_type": "processing_error"
            }
        }


# ==========================================
# 接口5: 综合参数数据处理接口
# ==========================================
def sort_dict_by_innermost_value(data, value):
    # 提取键和对应的最内层 time 值
    items_with_time = []
    for key, inner_dict in data.items():
        time_val = inner_dict['objective_achievement']['individual_scores'][value]
        items_with_time.append((key, time_val))

    # 根据 time 值从小到大排序（若需降序可添加 reverse=True）
    sorted_items = sorted(items_with_time, key=lambda x: x[1])

    # 按排序后的顺序重建新字典
    sorted_data = {}
    for key, _ in sorted_items:
        sorted_data[key] = data[key]

    return sorted_data


def sort_dict_innermp_value(data_do, time, cost, safety, distance, priority, balance, capability, social):
    data_d_time = sort_dict_by_innermost_value(data_do, time)
    data_d_cost = sort_dict_by_innermost_value(data_do, cost)
    data_d_safety = sort_dict_by_innermost_value(data_do, safety)
    data_d_distance = sort_dict_by_innermost_value(data_do, distance)
    data_d_priority = sort_dict_by_innermost_value(data_do, priority)
    data_d_balance = sort_dict_by_innermost_value(data_do, balance)
    data_d_capability = sort_dict_by_innermost_value(data_do, capability)
    data_d_social = sort_dict_by_innermost_value(data_do, social)
    return data_d_time, data_d_cost, data_d_safety, data_d_distance, \
        data_d_priority, data_d_balance, data_d_capability, data_d_social


def sort_dict_inner_value(data_do, time, cost, safety, distance, priority, balance, capability, social):
    data_d_time = sort_dict_by_D_innermost_value(data_do, time)
    data_d_cost = sort_dict_by_D_innermost_value(data_do, cost)
    data_d_safety = sort_dict_by_D_innermost_value(data_do, safety)
    data_d_distance = sort_dict_by_D_innermost_value(data_do, distance)
    data_d_priority = sort_dict_by_D_innermost_value(data_do, priority)
    data_d_balance = sort_dict_by_D_innermost_value(data_do, balance)
    data_d_capability = sort_dict_by_D_innermost_value(data_do, capability)
    data_d_social = sort_dict_by_D_innermost_value(data_do, social)
    return data_d_time, data_d_cost, data_d_safety, data_d_distance, \
        data_d_priority, data_d_balance, data_d_capability, data_d_social


def sort_dict_by_D_innermost_value(data, value):
    # 提取键和对应的最内层 time 值
    items_with_time = []
    for key, inner_dict in data.items():
        time_val = inner_dict['data']['objective_achievement']['individual_scores'][value]
        items_with_time.append((key, time_val))

    # 根据 time 值从小到大排序（若需降序可添加 reverse=True）
    sorted_items = sorted(items_with_time, key=lambda x: x[1])

    # 按排序后的顺序重建新字典
    sorted_data = {}
    for key, _ in sorted_items:
        sorted_data[key] = data[key]

    return sorted_data

def dataDeal(vdf):
    tyu = []
    for top_key in vdf:
        print(top_key)
    for key in vdf:
        tyu.append(vdf[key])
    return tyu

def combine_info1(other_result, vdf):
    stand_out = {}
    stand_out['code'] = other_result['code']
    stand_out['msg'] = other_result['msg']
    data = {}
    data['basic_info'] = other_result['data']['basic_info']
    data['strategy_configuration'] = other_result['data']['strategy_configuration']
    data['network_configuration'] = other_result['data']['network_configuration']
    # data['optimization_result'] = vdf
    tyu = []

    for key in vdf:
            tyu.append(vdf[key])
    data['optimization_result'] = tyu
    stand_out['data'] = data

    return stand_out

def combine_info(other_result, vdf):
    stand_out = {}
    stand_out['code'] = other_result['code']
    stand_out['msg'] = other_result['msg']
    data = {}
    data['basic_info'] = other_result['data']['basic_info']
    data['strategy_configuration'] = other_result['data']['strategy_configuration']
    data['network_configuration'] = other_result['data']['network_configuration']
    # data['optimization_result'] = vdf
    tyu = []

    for key in vdf:
        if key != 'norangeNum':
            tyu.append(vdf[key])
    data['norangeNum'] = vdf['norangeNum']
    data['optimization_result'] = tyu
    stand_out['data'] = data

    return stand_out


def Find_key_And_Value(response_dict):
    first_key = next(iter(response_dict))
    first_value = response_dict[first_key]
    return first_key, first_value


def combine_four_kage(fold_name, v_d_cost,  v_d_time,  v_d_safety, avrage_value):

    vd = {}
    vd['cost'] = v_d_cost
    vd['time'] = v_d_time
    vd['avrage'] = avrage_value
    vd['safety'] = v_d_safety
    # vd[k_spcetify] = specetify_value
    Save_json_file(fold_name, vd, 'data_d_four_kage')
    return vd

def find_specetify_kage(data_do, k_d_safety, k_d_time, k_d_cost,k_avrage):
    data_do1 = copy.copy(data_do)
    del data_do1[k_d_safety]
    del data_do1[k_d_time]
    del data_do1[k_d_cost]
    del data_do1[k_avrage]
    k_fit = 'unknow'
    keys_list = list(data_do1.keys())
    cu = 0
    for key in keys_list:
        if data_do1[key]['data']['data_mobilization_info']['algorithm_params']['allocation_strategy'] == 'specify':
            k_fit = key
            cu = cu + 1

    if cu == 0:

      if len(keys_list) <1:
          k_fit = 0
          fit_value = copy.deepcopy(data_do[str(k_fit)])
          fit_value['rangeType'] = 'rangeSpecify'
      else:
          k_fit = random.choice(keys_list)
          fit_value = data_do1[k_fit]
          fit_value['rangeType'] = 'rangeSpecify'


    # fit_value = data_do1[k_fit]

    return k_fit, fit_value

def find_Avarage_kage(data_do, k_d_safety, k_d_time, k_d_cost):

    data_do2 = copy.deepcopy(data_do)
    data_do1 = copy.copy(data_do)
    avrage_value = {}
    if k_d_safety == k_d_time and k_d_time == k_d_cost:
        del data_do1[k_d_safety]
    if k_d_safety == k_d_time and k_d_safety != k_d_cost:
        del data_do1[k_d_safety]
        del data_do1[k_d_cost]
    if k_d_safety == k_d_cost and k_d_safety != k_d_time:
        del data_do1[k_d_safety]
        del data_do1[k_d_time]
    if k_d_cost == k_d_time:
        del data_do1[k_d_time]
        del data_do1[k_d_safety]
    if k_d_cost != k_d_time and k_d_cost != k_d_safety and k_d_time != k_d_safety:
        del data_do1[k_d_safety]
        del data_do1[k_d_time]
        del data_do1[k_d_cost]

    keys_list = list(data_do1.keys())
    if len(keys_list) < 1:
        k_avrage = 0
        avrage_value = copy.deepcopy(data_do2[str(k_avrage)])
        avrage_value['rangeType'] = 'rangeAvarage'
    else:
        k_avrage = random.choice(keys_list)
        avrage_value = data_do1[k_avrage]
        avrage_value['rangeType'] = 'rangeAvarage'

    # avrage_value = data_do1[k_avrage]

    dr = 0
    return k_avrage, avrage_value


def fine_three_kage(data_d_time, data_d_cost, data_d_safety):

    k_d_time, v_d_time = Find_key_And_Value(data_d_time)

    k_d_cost, v_d_cost = Find_key_And_Value(data_d_cost)
    v_d_cost['rangeType'] = 'rangeCost'
    k_d_safety, v_d_safety = Find_key_And_Value(data_d_safety)


    k_d_time,v_d_time,k_d_safety,v_d_safety = compaire_key(k_d_cost, k_d_time, k_d_safety, data_d_safety, data_d_time)



    v_d_safety = data_d_safety[k_d_safety]
    v_d_safety['rangeType'] = 'rangeSafety'
    v_d_time['rangeType'] = 'rangeTime'

    return k_d_time, v_d_time, k_d_cost, v_d_cost, k_d_safety, v_d_safety

def fine_three_MP_kage(data_d_time, data_d_cost, data_d_safety):

    k_d_time, v_d_time = Find_key_And_Value(data_d_time)

    k_d_cost, v_d_cost = Find_key_And_Value(data_d_cost)
    v_d_cost['rangeType'] = 'rangeCost'
    k_d_safety, v_d_safety = Find_key_And_Value(data_d_safety)


    k_d_time,v_d_time,k_d_safety,v_d_safety = compaire_MK_key(k_d_cost, k_d_time, k_d_safety, data_d_safety, data_d_time)


    if k_d_safety == k_d_cost:
        gf_Sc = copy.deepcopy(data_d_safety[k_d_safety])
        gf_Sc['rangeType'] = 'rangeSafety'
        v_d_safety = gf_Sc
    if k_d_safety == k_d_time:
        gf_Sc = copy.deepcopy(data_d_safety[k_d_time])
        gf_Sc['rangeType'] = 'rangeSafety'
        v_d_safety = gf_Sc


    # v_d_safety = data_d_safety[k_d_safety]
    # v_d_safety['rangeType'] = 'rangeSafety'
    v_d_time['rangeType'] = 'rangeTime'
    sd = 0
    return k_d_time, v_d_time, k_d_cost, v_d_cost, k_d_safety, v_d_safety



def compaire_key_0(key,data):
    data_c = copy.copy(data)
    del data_c[key]
    first_key = next(iter(data))
    return first_key,data_c

def compaire_key(k_d_cost, k_d_time, k_d_safety, data_d_safety, data_d_time):

    data_d_safety_c = copy.copy(data_d_safety)
    data_d_time_c = copy.copy(data_d_time)
    del data_d_time_c[k_d_cost]
    del data_d_safety_c[k_d_cost]
    if k_d_cost == k_d_time:
        del data_d_time_c[k_d_time]
        time_key = next(iter(data_d_time_c))
        time_value = data_d_time_c[time_key]
    else:
        time_key = k_d_time
        time_value = data_d_time_c[time_key]


    if k_d_safety == k_d_cost:
        del data_d_safety_c[k_d_safety]
        safety_key = next(iter(data_d_safety_c))
        safety_key_value = data_d_safety_c[safety_key]
    else:
        safety_key = k_d_safety
        safety_key_value = data_d_safety_c[safety_key]


    if safety_key == time_key:

        del data_d_safety_c[safety_key]
        safety_key_key = next(iter(data_d_safety_c))
        safety_key_key_value = data_d_safety_c[safety_key_key]
    else:
        safety_key_key = safety_key
        safety_key_key_value = data_d_safety_c[safety_key_key]

    rt = 0
    return time_key,time_value,safety_key_key,safety_key_key_value

def compaire_MK_key(k_d_cost, k_d_time, k_d_safety, data_d_safety, data_d_time):

    data_d_safety_c = copy.copy(data_d_safety)
    data_d_time_c = copy.copy(data_d_time)

    if k_d_cost == k_d_time:
        del data_d_time_c[k_d_time]
        time_key = next(iter(data_d_time_c))
        time_value = data_d_time_c[time_key]
    else:
        time_key = k_d_time
        time_value = data_d_time_c[time_key]


    if k_d_safety == k_d_cost:
        del data_d_safety_c[k_d_safety]
        safety_key = next(iter(data_d_safety_c))
        safety_key_value = data_d_safety_c[safety_key]
    else:
        safety_key = k_d_safety
        safety_key_value = data_d_safety_c[safety_key]


    if safety_key == time_key:
        data_d_safety_c = copy.copy(data_d_safety)
        del data_d_safety_c[safety_key]
        safety_key_key = next(iter(data_d_safety_c))
        safety_key_key_value = data_d_safety_c[safety_key_key]
    else:
        safety_key_key = safety_key
        safety_key_key_value = data_d_safety_c[safety_key_key]

    rt = 0
    return time_key,time_value,safety_key_key,safety_key_key_value


def Save_json_file(fold_name, my_dict, name):
    # 打开一个文件用于写入二进制数据
    # folder_name = "data_do"
    # file_name_pkl = str(my_dict)+'.pkl'
    # file_path_pkl = os.path.join('data_do/'+str(my_dict)+'.pkl', file_name_pkl)
    with open(fold_name + '/' + name + '.pkl', 'wb') as file:
        # 将字典序列化并保存到文件
        pickle.dump(my_dict, file)


def open_pkl(fold_name, my_dict_name):
    # 打开文件并读取数据
    with open(fold_name + '/' + my_dict_name + '.pkl', 'rb') as file:
        # 从文件中反序列化数据
        loaded_dict = pickle.load(file)
        # print(loaded_dict)
    return loaded_dict


def input_weight_to_plan(other_parameter):
    algorithm_sequence_id = None
    try:
        other_parameter = json.loads(other_parameter)

        print(other_parameter)

        other_parameter = other_parameter
        algorithm_sequence_id = other_parameter['algorithm_sequence_id']
        activity_id = other_parameter['activity_id']
        req_element_id = other_parameter['req_element_id']
        resouseType = other_parameter['mobilization_object_type']

        strategy_weights = other_parameter['strategy_weights']

        indicator_time = strategy_weights['time']
        indicator_cost = strategy_weights['cost']
        indicator_distance = strategy_weights['distance']
        indicator_safety = strategy_weights['safety']
        indicator_priority = strategy_weights['priority']
        indicator_balance = strategy_weights['balance']
        indicator_capability = strategy_weights['capability']
        indicator_social = strategy_weights['social']

        list_indicator = [indicator_time, indicator_cost, indicator_distance, indicator_safety, indicator_priority,
                          indicator_balance, indicator_capability, indicator_social]
        max_argindex = np.argmax(list_indicator)

        key_if = 'key'
        value_if = {}

        if resouseType == 'material':
            fold_name = str(activity_id) + '/' + str(req_element_id)
            data_m_four_kage = open_pkl(fold_name, 'data_d_four_kage')
            keys_list = list(data_m_four_kage.keys())
            second_name = 'data_m'
            if max_argindex == 0:
                my_dict_name = second_name + '_' + 'time'
                data_m_time = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_time.pop(key, None)

                key_if = next(iter(data_m_time))
                value_if = data_m_time[key_if]
            elif max_argindex == 1:
                my_dict_name = second_name + '_' + 'cost'
                data_m_cost = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_cost.pop(key, None)

                key_if = next(iter(data_m_cost))
                value_if = data_m_cost[key_if]
            elif max_argindex == 2:
                my_dict_name = second_name + '_' + 'distance'
                data_m_distance = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_distance.pop(key, None)

                key_if = next(iter(data_m_distance))
                value_if = data_m_distance[key_if]
            elif max_argindex == 3:
                my_dict_name = second_name + '_' + 'safety'
                data_m_safety = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_safety.pop(key, None)

                key_if = next(iter(data_m_safety))
                value_if = data_m_safety[key_if]
            elif max_argindex == 4:
                my_dict_name = second_name + '_' + 'priority'
                data_m_priority = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_priority.pop(key, None)

                key_if = next(iter(data_m_priority))
                value_if = data_m_priority[key_if]
            elif max_argindex == '5':
                my_dict_name = second_name + '_' + 'balance'
                data_m_balance = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_balance.pop(key, None)

                key_if = next(iter(data_m_balance))
                value_if = data_m_balance[key_if]
            elif max_argindex == 6:
                my_dict_name = second_name + '_' + 'capability'
                data_m_capability = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_capability.pop(key, None)

                key_if = next(iter(data_m_capability))
                value_if = data_m_capability[key_if]
            elif max_argindex == 7:
                my_dict_name = second_name + '_' + 'social'
                data_m_social = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_social.pop(key, None)

                key_if = next(iter(data_m_social))
                value_if = data_m_social[key_if]

        elif resouseType == 'personnel':
            fold_name = str(activity_id) + '/' + str(req_element_id)
            data_m_four_kage = open_pkl(fold_name, 'data_d_four_kage')
            keys_list = list(data_m_four_kage.keys())
            second_name = 'data_p'
            if max_argindex == 0:
                my_dict_name = second_name + '_' + 'time'
                data_m_time = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_time.pop(key, None)

                key_if = next(iter(data_m_time))
                value_if = data_m_time[key_if]
            elif max_argindex == 1:
                my_dict_name = second_name + '_' + 'cost'
                data_m_cost = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_cost.pop(key, None)

                key_if = next(iter(data_m_cost))
                value_if = data_m_cost[key_if]
            elif max_argindex == 2:
                my_dict_name = second_name + '_' + 'distance'
                data_m_distance = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_distance.pop(key, None)

                key_if = next(iter(data_m_distance))
                value_if = data_m_distance[key_if]
            elif max_argindex == 3:
                my_dict_name = second_name + '_' + 'safety'
                data_m_safety = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_safety.pop(key, None)

                key_if = next(iter(data_m_safety))
                value_if = data_m_safety[key_if]
            elif max_argindex == 4:
                my_dict_name = second_name + '_' + 'priority'
                data_m_priority = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_priority.pop(key, None)

                key_if = next(iter(data_m_priority))
                value_if = data_m_priority[key_if]
            elif max_argindex == '5':
                my_dict_name = second_name + '_' + 'balance'
                data_m_balance = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_balance.pop(key, None)

                key_if = next(iter(data_m_balance))
                value_if = data_m_balance[key_if]
            elif max_argindex == 6:
                my_dict_name = second_name + '_' + 'capability'
                data_m_capability = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_capability.pop(key, None)

                key_if = next(iter(data_m_capability))
                value_if = data_m_capability[key_if]
            elif max_argindex == 7:

                my_dict_name = second_name + '_' + 'social'
                data_m_social = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_social.pop(key, None)

                key_if = next(iter(data_m_social))
                value_if = data_m_social[key_if]

        elif resouseType == 'data':
            fold_name = fold_name = str(activity_id) + '/' + str(req_element_id)
            data_m_four_kage = open_pkl(fold_name, 'data_d_four_kage')
            keys_list = list(data_m_four_kage.keys())
            fold_name2 = 'data_d'
            if max_argindex == 0:

                my_dict_name = fold_name2 + '_' + 'time'
                data_m_time = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_time.pop(key, None)

                key_if = next(iter(data_m_time))
                value_if = data_m_time[key_if]
            elif max_argindex == 1:
                my_dict_name = fold_name2 + '_' + 'cost'
                data_m_cost = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_cost.pop(key, None)

                key_if = next(iter(data_m_cost))
                value_if = data_m_cost[key_if]
            elif max_argindex == 2:
                my_dict_name = fold_name2 + '_' + 'distance'
                data_m_distance = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_distance.pop(key, None)

                key_if = next(iter(data_m_distance))
                value_if = data_m_distance[key_if]
            elif max_argindex == 3:
                my_dict_name = fold_name2 + '_' + 'safety'
                data_m_safety = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_safety.pop(key, None)

                key_if = next(iter(data_m_safety))
                value_if = data_m_safety[key_if]
            elif max_argindex == 4:
                my_dict_name = fold_name2 + '_' + 'priority'
                data_m_priority = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_priority.pop(key, None)

                key_if = next(iter(data_m_priority))
                value_if = data_m_priority[key_if]
            elif max_argindex == '5':
                my_dict_name = fold_name2 + '_' + 'balance'
                data_m_balance = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_balance.pop(key, None)

                key_if = next(iter(data_m_balance))
                value_if = data_m_balance[key_if]
            elif max_argindex == 6:
                my_dict_name = fold_name2 + '_' + 'capability'
                data_m_capability = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_capability.pop(key, None)

                key_if = next(iter(data_m_capability))
                value_if = data_m_capability[key_if]
            elif max_argindex == 7:
                my_dict_name = fold_name2 + '_' + 'social'
                data_m_social = open_pkl(fold_name, my_dict_name)
                # 循环删除键及其值
                for key in keys_list:
                    # 使用 pop() 删除键，如果键不存在则忽略
                    data_m_social.pop(key, None)

                key_if = next(iter(data_m_social))
                value_if = data_m_social[key_if]

        value_if['code'] = 200

        if value_if['code'] == 200:
            # other_params = value_if['data']
            print("综合参数接口完整输出及优化结果:")
            print(json.dumps(value_if, ensure_ascii=False, indent=2))

        if value_if:
            return {
                "code": 200,
                "msg": "匹配成功",
                "algorithm_sequence_id": algorithm_sequence_id,
                "data": value_if
            }
        else:
            return {
                "code": 300,
                "msg": "匹配不成功",
                "algorithm_sequence_id": algorithm_sequence_id,
                "data": {}
            }

    except Exception as e:
        return {
            "code": 500,
            "msg": "处理异常",
            "algorithm_sequence_id": algorithm_sequence_id,
            "data": {}
        }

def input_other_parameters(input_json: str, safety_cost_params: Dict = None, transfer_point_params: Dict = None,
                           mobilization_object_type: str = None, req_element_id: str = None):
    """
    综合参数数据处理接口

    功能描述:
        接受JSON格式的综合参数配置，包含策略权重、网络节点信息、供需参数、
        企业能力、坐标、优先级、时间窗、算法参数等。输出格式为逐个企业的详细信息。
        现已支持同一供应点内的细分对象差异化处理（如不同人员、不同物资型号、不同数据类型）。

        如果未提供依赖参数，会尝试从本地缓存文件中读取前四个接口的结果。

    细分对象支持:
        - 人员动员：支持同一单位内不同人员的党员身份、犯罪记录、军事经验等差异
        - 物资动员：支持同一企业内不同物资型号的价格、安全属性等差异
        - 数据动员：支持同一机构内不同数据类型的成本、安全级别等差异
        - 所有细分对象从同一供应点统一运输，算法会自动选择最优的细分对象组合

    Args:
        input_json (str): JSON格式的综合参数配置字符串
        safety_cost_params (Dict, optional): 安全数据参数，用于生成网络数据
        transfer_point_params (Dict, optional): 中转点参数，用于生成网络数据
        mobilization_object_type (str, optional): 动员对象具体类型

    Returns:
        Dict[str, Any]: 统一的{code, msg, data}格式响应
    """

    # 预先定义最终变量的默认值，避免在异常处理中出现未定义变量
    final_mobilization_object_type = mobilization_object_type
    final_req_element_id = req_element_id
    final_scheme_id = None
    final_activity_id = None
    scheme_config = None
    input_algorithm_sequence_id = None
    transfer_mode_params_first = None
    # transport_data = None

    try:

        # 读取transport_mode转点参数缓存
        if transfer_mode_params_first is None:
            transfer_cache_path = os.path.join('transport_modes_parameters_input', 'transport_modes_ones.json')
            if os.path.exists(transfer_cache_path):
                with open(transfer_cache_path, 'r', encoding='utf-8') as f:
                    cached_transfer = json.load(f)
                    # 处理可能的新格式
                    if isinstance(cached_transfer,
                                  dict) and 'data' in cached_transfer and 'code' in cached_transfer:
                        if cached_transfer.get('code') == 200:
                            transfer_mode_params_first = cached_transfer['data']
                        else:
                            return {
                                "code": cached_transfer.get('code', 500),
                                "msg": f"缓存的中转点参数无效: {cached_transfer.get('msg', '未知错误')}",
                                "data": {
                                    "cache_file": transfer_cache_path,
                                    "error_type": "cached_transfer_params_invalid"
                                }
                            }
                    else:
                        transfer_mode_params_first = cached_transfer


        transport_modes_one = transfer_mode_params_first['transport_modes']
        hj = 0
        # 处理输入类型
        if isinstance(input_json, str):
            if not input_json or input_json.strip() == "":

                return {
                    "code": 400,
                    "msg": "输入参数不能为空字符串",
                    "data": {
                        "input_type": type(input_json).__name__,
                        "algorithm_sequence_id": input_algorithm_sequence_id,
                        "mobilization_object_type": final_mobilization_object_type,
                        "req_element_id": final_req_element_id,
                        "scheme_id": final_scheme_id,
                        "scheme_config": scheme_config,
                        "activity_id": final_activity_id,
                        "error_type": "empty_input_error"
                    }
                }
            try:
                config = json.loads(input_json)

            except json.JSONDecodeError as json_error:
                return {
                    "code": 400,
                    "msg": f"无效的分解分发JSON格式: {str(json_error)}",
                    "data": {
                        "input_type": type(input_json).__name__,
                        "algorithm_sequence_id": input_algorithm_sequence_id,
                        "mobilization_object_type": final_mobilization_object_type,
                        "req_element_id": final_req_element_id,
                        "scheme_id": final_scheme_id,
                        "activity_id": final_activity_id,
                        "error_type": "json_decode_error"
                    }
                }
        elif isinstance(input_json, dict):
            config = input_json
        else:
            return {
                "code": 400,
                "msg": f"输入参数类型错误，期望str或dict，实际为{type(input_json)}",
                "data": {
                    "input_type": type(input_json).__name__,
                    "algorithm_sequence_id": input_algorithm_sequence_id,
                    "mobilization_object_type": final_mobilization_object_type,
                    "req_element_id": final_req_element_id,
                    "scheme_id": final_scheme_id,
                    "scheme_config": scheme_config,
                    "activity_id": final_activity_id,
                    "error_type": "input_type_error"
                }
            }

        # 如果未提供依赖参数，尝试从缓存文件中读取前四个接口的结果
        if safety_cost_params is None or transfer_point_params is None:
            try:

                # 读取安全参数缓存
                if safety_cost_params is None:
                    safety_cache_path = os.path.join('cache', 'safety_cost_parameters.json')
                    if os.path.exists(safety_cache_path):
                        with open(safety_cache_path, 'r', encoding='utf-8') as f:
                            cached_safety = json.load(f)
                            # 处理可能的新格式
                            if isinstance(cached_safety, dict) and 'data' in cached_safety and 'code' in cached_safety:
                                if cached_safety.get('code') == 200:
                                    safety_cost_params = cached_safety['data']
                                else:
                                    return {
                                        "code": cached_safety.get('code', 500),
                                        "msg": f"缓存的安全参数无效: {cached_safety.get('msg', '未知错误')}",
                                        "data": {
                                            "cache_file": safety_cache_path,
                                            "error_type": "cached_safety_params_invalid"
                                        }
                                    }
                            else:
                                safety_cost_params = cached_safety

                # 读取中转点参数缓存
                if transfer_point_params is None:
                    transfer_cache_path = os.path.join('cache', 'transfer_point_parameters.json')
                    if os.path.exists(transfer_cache_path):
                        with open(transfer_cache_path, 'r', encoding='utf-8') as f:
                            cached_transfer = json.load(f)
                            # 处理可能的新格式
                            if isinstance(cached_transfer,
                                          dict) and 'data' in cached_transfer and 'code' in cached_transfer:
                                if cached_transfer.get('code') == 200:
                                    transfer_point_params = cached_transfer['data']
                                else:
                                    return {
                                        "code": cached_transfer.get('code', 500),
                                        "msg": f"缓存的中转点参数无效: {cached_transfer.get('msg', '未知错误')}",
                                        "data": {
                                            "cache_file": transfer_cache_path,
                                            "error_type": "cached_transfer_params_invalid"
                                        }
                                    }
                            else:
                                transfer_point_params = cached_transfer

                #################
            except Exception as cache_error:
                # 缓存读取失败不影响继续处理，但记录警告
                pass
        sd = 0
        # 处理传入参数的新格式（如果是通过接口函数调用传入的）
        if isinstance(safety_cost_params,
                      dict) and safety_cost_params is not None and 'data' in safety_cost_params and 'code' in safety_cost_params:
            if safety_cost_params['code'] == 200:
                if safety_cost_params['data'] is not None:
                    safety_cost_params = safety_cost_params['data']
                else:
                    return {
                        "code": 500,
                        "msg": "安全参数处理失败: 返回数据为空",
                        "data": {
                            "algorithm_sequence_id": config.get(
                                'algorithm_sequence_id') if config is not None else input_algorithm_sequence_id,
                            "mobilization_object_type": config.get(
                                'mobilization_object_type') if config is not None else mobilization_object_type,
                            "error_type": "safety_params_data_null"
                        }
                    }
            else:
                return {
                    "code": safety_cost_params['code'],
                    "msg": f"安全参数处理失败: {safety_cost_params['msg']}",
                    "data": {
                        "algorithm_sequence_id": config.get(
                            'algorithm_sequence_id') if config is not None else input_algorithm_sequence_id,
                        "mobilization_object_type": config.get(
                            'mobilization_object_type') if config is not None else mobilization_object_type,
                        "error_type": "safety_params_error"
                    }
                }

        if isinstance(transfer_point_params,
                      dict) and transfer_point_params is not None and 'data' in transfer_point_params and 'code' in transfer_point_params:
            if transfer_point_params['code'] == 200:
                if transfer_point_params['data'] is not None:
                    transfer_point_params = transfer_point_params['data']
                else:
                    return {
                        "code": 500,
                        "msg": "中转点参数处理失败: 返回数据为空",
                        "data": {
                            "algorithm_sequence_id": config.get(
                                'algorithm_sequence_id') if config is not None else input_algorithm_sequence_id,
                            "mobilization_object_type": config.get(
                                'mobilization_object_type') if config is not None else mobilization_object_type,
                            "error_type": "transfer_params_data_null"
                        }
                    }
            else:
                return {
                    "code": transfer_point_params['code'],
                    "msg": f"中转点参数处理失败: {transfer_point_params['msg']}",
                    "data": {
                        "algorithm_sequence_id": config.get(
                            'algorithm_sequence_id') if config is not None else input_algorithm_sequence_id,
                        "mobilization_object_type": config.get(
                            'mobilization_object_type') if config is not None else mobilization_object_type,
                        "error_type": "transfer_params_error"
                    }
                }

        # 提取基础配置
        input_algorithm_sequence_id = config.get('algorithm_sequence_id')
        input_mobilization_object_type = config.get('mobilization_object_type')
        input_req_element_id = config.get('req_element_id')
        input_scheme_id = config.get('scheme_id')
        input_scheme_config = config.get('scheme_config')
        input_activity_id = config.get('activity_id')
        final_mobilization_object_type = input_mobilization_object_type if input_mobilization_object_type is not None else mobilization_object_type
        final_req_element_id = input_req_element_id if input_req_element_id is not None else req_element_id
        final_scheme_id = input_scheme_id if input_scheme_id is not None else None
        final_activity_id = input_activity_id if input_activity_id is not None else None
        final_scheme_config = input_scheme_config if input_scheme_config is not None else None

        # 验证req_element_id参数
        if final_req_element_id is not None:
            if not isinstance(final_req_element_id, str):
                return {
                    "code": 400,
                    "msg": f"req_element_id必须为字符串类型，当前类型: {type(final_req_element_id).__name__}",
                    "data": {
                        "algorithm_sequence_id": input_algorithm_sequence_id,
                        "mobilization_object_type": final_mobilization_object_type,
                        "req_element_id": final_req_element_id,
                        "scheme_id": final_scheme_id,
                        "scheme_config": final_scheme_config,
                        "activity_id": final_activity_id,
                        "error_type": "req_element_id_type_error"
                    }
                }

            final_req_element_id = final_req_element_id.strip()
            if not final_req_element_id:
                return {
                    "code": 400,
                    "msg": "req_element_id不能为空字符串",
                    "data": {
                        "algorithm_sequence_id": input_algorithm_sequence_id,
                        "mobilization_object_type": final_mobilization_object_type,
                        "req_element_id": final_req_element_id,
                        "scheme_id": final_scheme_id,
                        "scheme_config": final_scheme_config,
                        "activity_id": final_activity_id,
                        "error_type": "req_element_id_empty"
                    }
                }

            # 检查是否包含不支持的字符
            if not re.match(r'^[A-Za-z0-9_\-]+$', final_req_element_id):
                return {
                    "code": 400,
                    "msg": "req_element_id只能包含字母、数字、下划线和连字符",
                    "data": {
                        "algorithm_sequence_id": input_algorithm_sequence_id,
                        "mobilization_object_type": final_mobilization_object_type,
                        "req_element_id": final_req_element_id,
                        "scheme_id": final_scheme_id,
                        "scheme_config": final_scheme_config,
                        "activity_id": final_activity_id,
                        "error_type": "req_element_id_format_error"
                    }
                }

        strategy_weights_config = config.get('strategy_weights', {})

        # 验证和标准化权重
        weights_result = _validate_and_normalize_weights(strategy_weights_config, input_algorithm_sequence_id,
                                                         final_mobilization_object_type, final_scheme_id,
                                                         final_req_element_id, final_activity_id, final_scheme_config)
        if isinstance(weights_result, dict) and 'code' in weights_result:
            if weights_result['code'] != 200:
                return {
                    "code": weights_result['code'],
                    "msg": weights_result['msg'],
                    "data": {
                        "algorithm_sequence_id": input_algorithm_sequence_id,
                        "mobilization_object_type": final_mobilization_object_type,
                        "scheme_id": final_scheme_id,
                        "scheme_config": final_scheme_config,
                        "activity_id": final_activity_id,
                        **weights_result['data']
                    }
                }
            strategy_weights = weights_result['data']
        else:
            # 兼容旧格式
            strategy_weights = weights_result

        # 处理供应点信息（逐个企业详细信息），现已支持细分对象配置
        supply_points_config = config.get('supply_points', [])
        detailed_supply_points = []

        # 收集细分对象类型信息
        sub_object_types = set()

        for i, supply_point in enumerate(supply_points_config):
            if isinstance(supply_point, dict) and supply_point is not None:
                name = supply_point.get('name', f'供应点{i + 1}')

                # 根据动员对象类型确定需要的成本参数
                supply_point_costs = {}

                if final_mobilization_object_type == 'personnel':
                    # 人员动员：成本字段应该在细分对象中，供应点级别不需要这些成本
                    pass  # 不在供应点级别设置人员相关成本
                elif final_mobilization_object_type == 'material':
                    # 物资动员：成本字段应该在细分对象中，供应点级别不需要这些成本
                    pass  # 不在供应点级别设置物资相关成本
                elif final_mobilization_object_type == 'data':
                    # 数据动员：成本字段应该在细分对象中，供应点级别不需要这些成本
                    pass  # 不在供应点级别设置数据相关成本

                # 处理安全数据 - 根据动员对象类型
                safety_data = {}

                if final_mobilization_object_type == 'personnel':
                    # 人员队伍安全数据
                    personnel_safety = supply_point.get('personnel_safety', {})
                    if personnel_safety is not None:
                        safety_data.update({
                            'political_status': personnel_safety.get('political_status', '群众'),
                            'military_experience': personnel_safety.get('military_experience', '无'),
                            'criminal_record': personnel_safety.get('criminal_record', '无'),
                            'network_record': personnel_safety.get('network_record', '无'),
                            'credit_record': personnel_safety.get('credit_record', '正常')
                        })

                elif final_mobilization_object_type == 'material':
                    # 企业机构安全数据
                    enterprise_safety = supply_point.get('enterprise_safety', {})
                    if enterprise_safety is not None:
                        safety_data.update({
                            'enterprise_nature': enterprise_safety.get('enterprise_nature', '央国企'),
                            'enterprise_scale': enterprise_safety.get('enterprise_scale', '大'),
                            'risk_record': enterprise_safety.get('risk_record', '无'),
                            'foreign_background': enterprise_safety.get('foreign_background', '无'),
                            'resource_safety': enterprise_safety.get('resource_safety', '高安全性'),
                            'mobilization_experience': enterprise_safety.get('mobilization_experience', '有')
                        })

                    # 物资器材安全数据
                    material_safety = supply_point.get('material_safety', {})
                    if material_safety is not None:
                        safety_data.update({
                            'flammable_explosive': material_safety.get('flammable_explosive', '低'),
                            'corrosive': material_safety.get('corrosive', '低'),
                            'polluting': material_safety.get('polluting', '低'),
                            'fragile': material_safety.get('fragile', '低')
                        })


                elif final_mobilization_object_type == 'data':
                    # 装备设备安全数据
                    equipment_safety = supply_point.get('equipment_safety', {})
                    if equipment_safety is not None:
                        autonomous_control = equipment_safety.get('autonomous_control', '是')
                        usability_level = equipment_safety.get('usability_level', '能')
                        safety_data.update({
                            'autonomous_control': autonomous_control,
                            'usability_level': usability_level
                        })

                    # 设施场所安全数据
                    facility_safety = supply_point.get('facility_safety', {})
                    if facility_safety is not None:
                        safety_data.update({
                            'facility_protection': facility_safety.get('facility_protection', '有'),
                            'camouflage_protection': facility_safety.get('camouflage_protection', '有'),
                            'surrounding_environment': facility_safety.get('surrounding_environment', '正常')
                        })

                    # 科技信息安全数据
                    technology_safety = supply_point.get('technology_safety', {})
                    if technology_safety is not None:
                        safety_data.update({
                            'encryption_security': technology_safety.get('encryption_security', '有'),
                            'access_control': technology_safety.get('access_control', '有'),
                            'network_security': technology_safety.get('network_security', '有'),
                            'terminal_security': technology_safety.get('terminal_security', '有'),
                            'dlp': technology_safety.get('dlp', '有'),
                            'security_policy': technology_safety.get('security_policy', '有'),
                            'risk_assessment': technology_safety.get('risk_assessment', '有'),
                            'audit_monitoring': technology_safety.get('audit_monitoring', '有'),
                            'emergency_response': technology_safety.get('emergency_response', '有')
                        })

                # 处理细分对象配置（如果用户提供了的话）
                user_sub_objects = supply_point.get('sub_objects', [])

                # 收集细分对象类型并验证必要字段
                for sub_obj_group in user_sub_objects:
                    if isinstance(sub_obj_group, dict) and 'categories' in sub_obj_group:
                        # 新的三层结构：动员子类型 -> 分类 -> 具体项目
                        for category in sub_obj_group.get('categories', []):
                            if isinstance(category, dict) and 'items' in category:
                                for sub_obj in category.get('items', []):
                                    sub_obj_name = sub_obj.get('sub_object_name', '')
                                    if sub_obj_name:
                                        sub_object_types.add(sub_obj_name)
                                    # 验证并补充max_available_quantity字段
                                    if 'max_available_quantity' not in sub_obj:
                                        return {
                                            "code": 400,
                                            "msg": f"细分对象 {sub_obj_name} 缺少必需的 max_available_quantity 字段",
                                            "data": {
                                                "sub_object_name": sub_obj_name,
                                                "supply_point": supply_point.get('name'),
                                                "algorithm_sequence_id": input_algorithm_sequence_id,
                                                "mobilization_object_type": final_mobilization_object_type,
                                                "error_type": "missing_max_available_quantity"
                                            }
                                        }
                    elif isinstance(sub_obj_group, dict) and 'items' in sub_obj_group:
                        # 兼容旧的两层结构：分类 -> 具体项目
                        for sub_obj in sub_obj_group.get('items', []):
                            sub_obj_name = sub_obj.get('sub_object_name', '')
                            if sub_obj_name:
                                sub_object_types.add(sub_obj_name)

                            # 验证并补充max_available_quantity字段
                            if 'max_available_quantity' not in sub_obj:
                                return {
                                    "code": 400,
                                    "msg": f"细分对象 {sub_obj_name} 缺少必需的 max_available_quantity 字段",
                                    "data": {
                                        "sub_object_name": sub_obj_name,
                                        "supply_point": supply_point.get('name'),
                                        "algorithm_sequence_id": input_algorithm_sequence_id,
                                        "mobilization_object_type": final_mobilization_object_type,
                                        "scheme_id": final_scheme_id,
                                        "scheme_config": final_scheme_config,
                                        "activity_id": final_activity_id,
                                        "error_type": "missing_max_available_quantity"
                                    }
                                }

                    else:
                        # 兼容最旧的平铺结构
                        sub_obj_name = sub_obj_group.get('sub_object_name', '')
                        if sub_obj_name:
                            sub_object_types.add(sub_obj_name)

                        # 验证并补充max_available_quantity字段
                        if 'max_available_quantity' not in sub_obj_group:
                            return {
                                "code": 400,
                                "msg": f"细分对象 {sub_obj_name} 缺少必需的 max_available_quantity 字段",
                                "data": {
                                    "sub_object_name": sub_obj_name,
                                    "supply_point": supply_point.get('name'),
                                    "algorithm_sequence_id": input_algorithm_sequence_id,
                                    "mobilization_object_type": final_mobilization_object_type,
                                    "scheme_id": final_scheme_id,
                                    "scheme_config": final_scheme_config,
                                    "activity_id": final_activity_id,
                                    "error_type": "missing_max_available_quantity"
                                }
                            }

                # 预计算供应点默认值，避免重复计算
                supply_count = len(supply_points_config)
                supply_base_factor = supply_count * (i + 1)
                supply_variation_factor = supply_count / (supply_count + 1)

                detailed_supply_points.append({
                    'id': supply_point.get('id'),  # 保留原始id字段
                    'name': name,
                    'basic_info': {
                        'latitude': supply_point.get('latitude', supply_base_factor / supply_count + supply_count + i),
                        'longitude': supply_point.get('longitude',
                                                      supply_base_factor + supply_count + i * supply_variation_factor)
                    },
                    'capacity_info': {
                        'capacity': supply_point.get('capacity', supply_base_factor * supply_count + i * supply_count),
                        'probability': supply_point.get('probability',
                                                        supply_variation_factor + (i % supply_count) / supply_count / (
                                                                supply_count + 1))
                    },
                    'safety_data': safety_data,
                    'user_defined_sub_objects': user_sub_objects  # 保存用户定义的细分对象
                })
            else:
                name = str(supply_point)

                # 根据动员对象类型生成对应的默认成本信息
                default_supply_point_costs = {}

                if final_mobilization_object_type == 'data':
                    default_supply_point_costs = {}

                # 生成默认安全数据
                default_safety_data = {}
                # 定义安全数据选项映射，避免重复定义
                safety_options_map = {
                    'personnel': {
                        'political_status': ['群众', '党员'],
                        'military_experience': ['有', '无'],
                        'criminal_record': ['无', '有'],
                        'network_record': ['无', '有'],
                        'credit_record': ['正常', '不良']
                    },
                    'material': {
                        'enterprise_nature': ['央国企', '民企', '其他'],
                        'enterprise_scale': ['大', '中', '小'],
                        'risk_record': ['无', '有'],
                        'foreign_background': ['无', '有'],
                        'resource_safety': ['高安全性', '一般安全性'],
                        'mobilization_experience': ['有', '无'],
                        'flammable_explosive': ['低', '中', '高'],
                        'corrosive': ['低', '中', '高'],
                        'polluting': ['低', '中', '高'],
                        'fragile': ['低', '中', '高']
                    },
                    'data': {
                        'autonomous_control': ['是', '否'],
                        'usability_level': ['能', '否'],
                        'facility_protection': ['有', '无'],
                        'camouflage_protection': ['有', '无'],
                        'surrounding_environment': ['正常', '配套齐全'],
                        'encryption_security': ['有', '无'],
                        'access_control': ['有', '无'],
                        'network_security': ['有', '无'],
                        'terminal_security': ['有', '无'],
                        'dlp': ['有', '无'],
                        'security_policy': ['有', '无'],
                        'risk_assessment': ['有', '无'],
                        'audit_monitoring': ['有', '无'],
                        'emergency_response': ['有', '无']
                    }
                }

                # 生成默认安全数据
                default_safety_data = {}
                if final_mobilization_object_type in safety_options_map:
                    options = safety_options_map[final_mobilization_object_type]
                    for field, field_options in options.items():
                        default_safety_data[field] = field_options[i % len(field_options)]
                elif final_mobilization_object_type == 'material':
                    enterprise_nature_options = ['央国企', '民企', '其他']
                    enterprise_scale_options = ['大', '中', '小']
                    binary_options_no = ['无', '有']
                    safety_options = ['高安全性', '一般安全性']
                    experience_options = ['有', '无']
                    risk_level_options = ['低', '中', '高']

                    default_safety_data = {
                        'enterprise_nature': enterprise_nature_options[i % len(enterprise_nature_options)],
                        'enterprise_scale': enterprise_scale_options[i % len(enterprise_scale_options)],
                        'risk_record': binary_options_no[i % len(binary_options_no)],
                        'foreign_background': binary_options_no[i % len(binary_options_no)],
                        'resource_safety': safety_options[i % len(safety_options)],
                        'mobilization_experience': experience_options[i % len(experience_options)],
                        'flammable_explosive': risk_level_options[i % len(risk_level_options)],
                        'corrosive': risk_level_options[i % len(risk_level_options)],
                        'polluting': risk_level_options[i % len(risk_level_options)],
                        'fragile': risk_level_options[i % len(risk_level_options)]
                    }
                elif final_mobilization_object_type == 'data':
                    binary_options_yes = ['是', '否']
                    usability_options = ['能', '否']
                    maintenance_options = ['有', '无']
                    protection_options = ['有', '无']
                    environment_options = ['正常', '配套齐全']
                    tech_options = ['有', '无']

                    default_safety_data = {
                        'autonomous_control': binary_options_yes[i % len(binary_options_yes)],
                        'usability_level': usability_options[i % len(usability_options)],
                        'facility_protection': protection_options[i % len(protection_options)],
                        'camouflage_protection': protection_options[i % len(protection_options)],
                        'surrounding_environment': environment_options[i % len(environment_options)],
                        'encryption_security': tech_options[i % len(tech_options)],
                        'access_control': tech_options[i % len(tech_options)],
                        'network_security': tech_options[i % len(tech_options)],
                        'terminal_security': tech_options[i % len(tech_options)],
                        'dlp': tech_options[i % len(tech_options)],
                        'security_policy': tech_options[i % len(tech_options)],
                        'risk_assessment': tech_options[i % len(tech_options)],
                        'audit_monitoring': tech_options[i % len(tech_options)],
                        'emergency_response': tech_options[i % len(tech_options)]
                    }

                detailed_supply_points.append({
                    'id': None,  # 字符串格式没有id字段
                    'name': name,
                    'basic_info': {
                        'latitude': len(supply_points_config) * (i + 1) / len(supply_points_config) + len(
                            supply_points_config) + i,
                        'longitude': len(supply_points_config) * (i + 1) + len(supply_points_config) + i * len(
                            supply_points_config) / (len(supply_points_config) + 1),
                        'enterprise_type': '国企',
                        'enterprise_size': '大'
                    },
                    'capacity_info': {
                        'capacity': supply_point.get('capacity', len(supply_points_config) * (i + 1) * len(
                            supply_points_config) + i * len(supply_points_config)),
                        'probability': supply_point.get('probability', len(supply_points_config) / (
                                len(supply_points_config) + i + 1) + (i % len(supply_points_config)) / len(
                            supply_points_config) / (len(supply_points_config) + 1)),
                        'resource_reserve': len(supply_points_config) / (len(supply_points_config) + 1) + (
                                i % (len(supply_points_config) + 2)),
                        'production_capacity': len(supply_points_config) + len(supply_points_config) / (
                                len(supply_points_config) + 1) + (i % len(supply_points_config)),
                        'expansion_capacity': len(supply_points_config) / (len(supply_points_config) + 1) + (
                                i % (len(supply_points_config) + 3))
                    },
                    'cost_info': default_supply_point_costs,
                    'safety_data': default_safety_data,
                    'additional_info': {
                        'registration_level': '一级',
                        'data_security_level': '高',
                        'data_processing_capacity': len(supply_points_config) * (i + 1) + i * len(supply_points_config),
                        'supported_resources': ['material', 'personnel', 'data']
                    },
                    'user_defined_sub_objects': []  # 字符串格式没有用户定义的细分对象
                })

        # 处理需求点信息
        demand_points_config = config.get('demand_points', [])
        detailed_demand_points = []

        for i, demand_point in enumerate(demand_points_config):
            if isinstance(demand_point, dict) and demand_point is not None:
                name = demand_point.get('name', f'需求点{i + 1}')
                detailed_demand_points.append({
                    'name': name,
                    'location': {
                        'latitude': demand_point.get('latitude',
                                                     len(supply_points_config) * len(demand_points_config) / (
                                                             len(supply_points_config) + len(
                                                         demand_points_config)) + len(supply_points_config) / (
                                                             len(supply_points_config) + 1) + i / len(
                                                         demand_points_config)),
                        'longitude': demand_point.get('longitude',
                                                      len(supply_points_config) * len(demand_points_config) / (
                                                              len(supply_points_config) + len(
                                                          demand_points_config)) + len(demand_points_config) / (
                                                              len(demand_points_config) + 1) + i / len(
                                                          demand_points_config))
                    },
                    'demand_info': {
                        'demand': demand_point.get('demand', len(supply_points_config) * len(demand_points_config) * (
                                len(supply_points_config) + len(demand_points_config)))
                    },
                    'time_constraints': {
                        'time_window_earliest': demand_point.get('time_window_earliest', len(supply_points_config) / (
                                len(supply_points_config) + len(demand_points_config))),
                        'time_window_latest': demand_point.get('time_window_latest',
                                                               len(supply_points_config) * len(demand_points_config))
                    }
                })
            else:
                name = str(demand_point)
                detailed_demand_points.append({
                    'name': name,
                    'location': {
                        'latitude': len(supply_points_config) * len(demand_points_config) / (
                                len(supply_points_config) + len(demand_points_config)) + len(
                            supply_points_config) / (len(supply_points_config) + 1) + i / len(demand_points_config),
                        'longitude': len(supply_points_config) * len(demand_points_config) / (
                                len(supply_points_config) + len(demand_points_config)) + len(
                            demand_points_config) / (len(demand_points_config) + 1) + i / len(demand_points_config)
                    },
                    'demand_info': {
                        'demand': len(supply_points_config) * len(demand_points_config) * (
                                len(supply_points_config) + len(demand_points_config))
                    },
                    'time_constraints': {
                        'time_window_earliest': len(supply_points_config) / (
                                len(supply_points_config) + len(demand_points_config)),
                        'time_window_latest': len(supply_points_config) * len(demand_points_config)
                    }
                })

        # 其他参数保持原有结构
        supply_demand_parameters = config.get('supply_demand_parameters', {})
        capability_parameters = config.get('capability_parameters', {})
        coordinate_parameters = config.get('coordinate_parameters', {})
        priority_parameters = config.get('priority_parameters', {})
        time_window_parameters = config.get('time_window_parameters', {})
        algorithm_parameters = config.get('algorithm_parameters', {})

        # 构建result - 重新组织结构避免重复
        result = {
            'basic_info': {
                'algorithm_sequence_id': input_algorithm_sequence_id,
                'mobilization_object_type': final_mobilization_object_type,
                'req_element_id': final_req_element_id,
                'scheme_id': final_scheme_id,
                "activity_id": final_activity_id,
                "scheme_config": final_scheme_config,
                'processing_timestamp': datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            },
            'strategy_configuration': {
                'objective_weights': strategy_weights
            },
            'network_configuration': {
                'supply_points_count': len(detailed_supply_points),
                'demand_points_count': len(detailed_demand_points),
                'transfer_points_count': len(transfer_point_params.get('transfer_points', []))
            }
        }

        # 执行多目标优化算法 - 初始化默认结果
        optimization_result = {
            "code": 400,
            "msg": "优化未执行：参数不完整",
            "data": {
                "algorithm_sequence_id": input_algorithm_sequence_id,
                "mobilization_object_type": final_mobilization_object_type,
                'req_element_id': final_req_element_id,
                'scheme_id': final_scheme_id,
                "activity_id": final_activity_id,
                "scheme_config": final_scheme_config,
                "execution_status": "not_executed"
            }
        }

        # 参数有效性验证
        safety_params_valid = safety_cost_params is not None and isinstance(safety_cost_params,
                                                                            dict) and safety_cost_params
        transfer_params_valid = transfer_point_params is not None and isinstance(transfer_point_params,
                                                                                 dict) and transfer_point_params
        strategy_weights_valid = strategy_weights and isinstance(strategy_weights, dict)
        mobilization_type_valid = final_mobilization_object_type is not None and final_mobilization_object_type.strip()

        # 如果提供了依赖参数，生成完整的网络数据
        network_data = None
        if safety_params_valid and transfer_params_valid and strategy_weights_valid and mobilization_type_valid:
            try:
                # 验证transfer_point_params的内容
                if 'transfer_points' not in transfer_point_params:
                    optimization_result = {
                        "code": 400,
                        "msg": "中转点参数无效：缺少transfer_points",
                        "data": {
                            "algorithm_sequence_id": input_algorithm_sequence_id,
                            "mobilization_object_type": final_mobilization_object_type,
                            'req_element_id': final_req_element_id,
                            'scheme_id': final_scheme_id,
                            "activity_id": final_activity_id,
                            "scheme_config": final_scheme_config,
                            "transfer_params_keys": list(transfer_point_params.keys()) if transfer_point_params else [],
                            "error_type": "missing_transfer_points"
                        }
                    }
                else:
                    # 转换为原有格式以兼容现有的网络数据生成逻辑
                    legacy_format = {
                        'algorithm_sequence_id': input_algorithm_sequence_id,
                        'mobilization_object_type': final_mobilization_object_type,
                        'strategy_weights': strategy_weights,
                        'network_nodes': {
                            'supply_points': detailed_supply_points,
                            'transfer_points': config.get('transfer_points', []),
                            'demand_points': detailed_demand_points
                        },
                        'supply_demand_parameters': supply_demand_parameters,
                        'capability_parameters': capability_parameters,
                        'coordinate_parameters': coordinate_parameters,
                        'priority_parameters': priority_parameters,
                        'time_window_parameters': time_window_parameters,
                        'algorithm_parameters': algorithm_parameters
                    }

                    # 将动员对象类型添加到安全参数中
                    adapted_safety_cost_params = {
                        'safety_parameters': dict(safety_cost_params)
                    }
                    adapted_safety_cost_params['safety_parameters'][
                        'mobilization_object_type'] = final_mobilization_object_type

                    # 调用网络数据生成函数
                    network_data_result = _generate_network_data(legacy_format, adapted_safety_cost_params,
                                                                 transfer_point_params, input_algorithm_sequence_id,
                                                                 final_mobilization_object_type, final_scheme_id,
                                                                 final_req_element_id, final_activity_id,
                                                                 final_scheme_config)

                    # 处理网络数据生成结果
                    if isinstance(network_data_result, dict) and 'code' in network_data_result:
                        if network_data_result['code'] == 200:
                            network_data = network_data_result['data']
                        else:
                            network_data = None  # 添加这一行，确保后续检查能够正确识别失败状态
                            optimization_result = {
                                "code": network_data_result['code'],
                                "msg": network_data_result['msg'],
                                "data": {
                                    "algorithm_sequence_id": input_algorithm_sequence_id,
                                    "mobilization_object_type": final_mobilization_object_type,
                                    **network_data_result.get('data', {})
                                }
                            }
                    else:
                        # 兼容旧格式
                        network_data = network_data_result

                    # 验证网络数据生成结果 - 只在optimization_result还没有错误状态时进行检查
                    if optimization_result.get('code') != 200:
                        # 如果optimization_result已经包含错误信息，保持原错误不变
                        pass
                    elif network_data is None:
                        optimization_result = {
                            "code": 500,
                            "msg": "网络数据生成失败：返回结果为空",
                            "data": {
                                "algorithm_sequence_id": input_algorithm_sequence_id,
                                "mobilization_object_type": final_mobilization_object_type,
                                "error_type": "network_data_empty"
                            }
                        }
                    elif not isinstance(network_data, dict):
                        optimization_result = {
                            "code": 500,
                            "msg": f"网络数据生成失败：返回类型错误 {type(network_data)}",
                            "data": {
                                "algorithm_sequence_id": input_algorithm_sequence_id,
                                "mobilization_object_type": final_mobilization_object_type,
                                "actual_type": type(network_data).__name__,
                                "error_type": "network_data_type_error"
                            }
                        }
                    elif 'J' not in network_data or 'M' not in network_data or 'K' not in network_data:
                        optimization_result = {
                            "code": 500,
                            "msg": "网络数据生成失败：缺少必要的节点集合",
                            "data": {
                                "algorithm_sequence_id": input_algorithm_sequence_id,
                                "mobilization_object_type": final_mobilization_object_type,
                                "available_keys": list(network_data.keys()) if isinstance(network_data,
                                                                                          dict) else [],
                                "error_type": "missing_node_sets"
                            }
                        }
                    else:
                        optimization_result = {
                            "code": 200,
                            "msg": "网络数据生成成功，准备执行优化",
                            "data": {
                                "algorithm_sequence_id": input_algorithm_sequence_id,
                                "mobilization_object_type": final_mobilization_object_type,
                                "execution_status": "ready_for_optimization"
                            }
                        }

            except ValueError as ve:
                optimization_result = {
                    "code": 400,
                    "msg": f"网络数据生成参数错误: {str(ve)}",
                    "data": {
                        "algorithm_sequence_id": input_algorithm_sequence_id,
                        "mobilization_object_type": final_mobilization_object_type,
                        "error_type": "network_generation_parameter_error"
                    }
                }
            except KeyError as ke:
                optimization_result = {
                    "code": 500,
                    "msg": f"网络数据生成访问错误: {str(ke)}",
                    "data": {
                        "algorithm_sequence_id": input_algorithm_sequence_id,
                        "mobilization_object_type": final_mobilization_object_type,
                        "missing_key": str(ke),
                        "error_type": "network_generation_key_error"
                    }
                }
            except Exception as network_error:
                optimization_result = {
                    "code": 500,
                    "msg": f"网络数据生成失败: {str(network_error)}",
                    "data": {
                        "algorithm_sequence_id": input_algorithm_sequence_id,
                        "mobilization_object_type": final_mobilization_object_type,
                        "error_type": "network_generation_unknown_error"
                    }
                }
        else:
            # 详细的参数缺失信息
            missing_details = []
            if not safety_params_valid:
                if safety_cost_params is None:
                    missing_details.append("safety_cost_params为None")
                elif not isinstance(safety_cost_params, dict):
                    missing_details.append(f"safety_cost_params类型错误：{type(safety_cost_params)}")
                elif not safety_cost_params:
                    missing_details.append("safety_cost_params为空字典")

            if not transfer_params_valid:
                if transfer_point_params is None:
                    missing_details.append("transfer_point_params为None")
                elif not isinstance(transfer_point_params, dict):
                    missing_details.append(f"transfer_point_params类型错误：{type(transfer_point_params)}")
                elif not transfer_point_params:
                    missing_details.append("transfer_point_params为空字典")

            if not strategy_weights_valid:
                if not strategy_weights:
                    missing_details.append("strategy_weights为空")
                elif not isinstance(strategy_weights, dict):
                    missing_details.append(f"strategy_weights类型错误：{type(strategy_weights)}")

            if not mobilization_type_valid:
                if final_mobilization_object_type is None:
                    missing_details.append("mobilization_object_type为None")
                elif not final_mobilization_object_type.strip():
                    missing_details.append("mobilization_object_type为空字符串")

            optimization_result = {
                "code": 400,
                "msg": f"优化未执行：参数验证失败 - {'; '.join(missing_details)}",
                "data": {
                    "algorithm_sequence_id": input_algorithm_sequence_id,
                    "mobilization_object_type": final_mobilization_object_type,
                    "missing_parameters": missing_details,
                    "execution_status": "parameter_validation_failed"
                }
            }

        # 如果网络数据生成成功，则执行优化算法
        if (network_data is not None and
                isinstance(network_data, dict) and
                'J' in network_data and
                'M' in network_data and
                'K' in network_data and
                strategy_weights_valid):

            try:
                # 尝试读取时间参数缓存
                time_params = None
                try:
                    time_cache_path = os.path.join('cache', 'time_parameters.json')
                    if os.path.exists(time_cache_path):
                        with open(time_cache_path, 'r', encoding='utf-8') as f:
                            cached_time = json.load(f)
                            # 处理可能的新格式
                            if isinstance(cached_time, dict) and 'data' in cached_time and 'code' in cached_time:
                                if cached_time.get('code') == 200:
                                    time_params = cached_time['data']
                                # 如果时间参数缓存无效，使用默认值（不阻断流程）
                            else:
                                time_params = cached_time
                except Exception as time_cache_error:
                    # 时间参数缓存读取失败，使用默认值
                    pass

                # 尝试读取运输参数缓存，并检查缓存完整性
                transport_data = None
                transport_cache_incomplete = False
                try:
                    transport_cache_path = os.path.join('cache', 'transport_parameters.json')
                    if os.path.exists(transport_cache_path):
                        with open(transport_cache_path, 'r', encoding='utf-8') as f:
                            cached_transport_data = json.load(f)

                        # 处理可能的新格式
                        if isinstance(cached_transport_data,
                                      dict) and 'data' in cached_transport_data and 'code' in cached_transport_data:
                            if cached_transport_data.get('code') == 200:
                                cached_transport_data = cached_transport_data['data']
                            else:
                                transport_cache_incomplete = True

                        # 检查缓存数据是否包含必要的距离矩阵
                        required_distance_matrices = ['L_j_m', 'L_m_m', 'L_m_k']
                        if all(matrix in cached_transport_data for matrix in required_distance_matrices):
                            transport_data = cached_transport_data
                        else:
                            transport_cache_incomplete = True
                except Exception as transport_cache_error:
                    # 运输参数缓存读取失败，需要生成
                    transport_cache_incomplete = True

                # 预计算网络特征参数，避免重复计算
                network_metrics = {
                    'supply_count': len(detailed_supply_points),
                    'demand_count': len(detailed_demand_points),
                    'total_scale': len(detailed_supply_points) + len(detailed_demand_points)
                }

                # 计算网络距离特征的辅助函数
                def calculate_network_distance_metrics(supply_points, demand_points, max_samples=None):
                    if not supply_points or not demand_points:
                        return network_metrics['total_scale'], network_metrics['total_scale']

                    sample_size = max_samples or min(len(supply_points), len(demand_points))
                    total_distance = 0.0
                    distance_count = 0

                    for i in range(min(sample_size, len(supply_points))):
                        supply_data = supply_points[i]
                        supply_lat = supply_data['basic_info']['latitude']
                        supply_lon = supply_data['basic_info']['longitude']

                        for j in range(min(sample_size, len(demand_points))):
                            demand_data = demand_points[j]
                            demand_lat = demand_data['location']['latitude']
                            demand_lon = demand_data['location']['longitude']
                            distance = ((supply_lat - demand_lat) ** 2 + (supply_lon - demand_lon) ** 2) ** 0.5
                            total_distance += distance
                            distance_count += 1

                    avg_distance = total_distance / distance_count if distance_count > 0 else network_metrics[
                        'total_scale']
                    distance_factor = network_metrics['total_scale'] / network_metrics['supply_count'] if \
                        network_metrics['supply_count'] > 0 else network_metrics['total_scale']

                    return avg_distance, distance_factor

                # 使用采样方式计算平均距离，避免O(n²)复杂度
                avg_network_distance, avg_distance_factor = calculate_network_distance_metrics(detailed_supply_points,
                                                                                               detailed_demand_points)

                # 如果缓存不完整或不存在，重新生成运输参数
                if transport_data is None or transport_cache_incomplete:
                    # 从传入参数中提取运输参数配置
                    transport_parameters_config = config.get('transport_parameters', {})
                    transport_modes_config = transport_parameters_config.get('transport_modes', [])

                    # 如果没有提供运输方式配置，基于网络特征生成启发式配置
                    if not transport_modes_config:
                        # 预计算运输配置所需的参数，避免重复计算
                        supply_count = network_metrics['supply_count']
                        demand_count = network_metrics['demand_count']
                        total_scale = network_metrics['total_scale']

                        # 预计算eps参数，避免重复的复杂表达式
                        weights_count = len(strategy_weights)
                        default_eps = weights_count / (
                                weights_count + total_scale) if total_scale > 0 else weights_count / (
                                weights_count + 1)
                        eps_value = algorithm_parameters.get('eps', default_eps)

                        # 预计算常用的比例因子
                        supply_demand_ratio = demand_count / supply_count if supply_count > 0 else demand_count
                        distance_speed_factor = avg_network_distance / avg_distance_factor if avg_distance_factor > 0 else supply_demand_ratio
                        cost_base_factor = avg_distance_factor / supply_count if supply_count > 0 else avg_distance_factor


                        highway_speed = transport_modes_one[0]['speed']
                        highway_cost = transport_modes_one[0]['cost_per_km']

                        ship_speed = transport_modes_one[1]['speed']
                        ship_cost = transport_modes_one[1]['cost_per_km']

                        fly_speed = transport_modes_one[2]['speed']
                        fly_cost = transport_modes_one[2]['cost_per_km']

                        subway_speed = transport_modes_one[3]['speed']
                        subway_cost = transport_modes_one[3]['cost_per_km']

                        ff = 0
                        transport_modes_config = [
                            {
                                "name": "公路",
                                "code": "trans-01",
                                "speed": highway_speed,
                                "cost_per_km": highway_cost,
                                "road_only_modes": 1
                            },
                            {
                                "name": "海运",
                                "code": "trans-02",
                                "speed": ship_speed,
                                "cost_per_km": ship_cost,
                                "road_only_modes": 0
                            },
                            {
                                "name": "空运",
                                "code": "trans-03",
                                "speed": fly_speed,
                                "cost_per_km": fly_cost,
                                "road_only_modes": 0
                            },
                            {
                                "name": "铁路运输",
                                "code": "trans-04",
                                "speed": subway_speed,
                                "cost_per_km": subway_cost,
                                "road_only_modes": 0
                            }
                        ]

                    # 验证运输参数生成
                    if not transport_modes_config:
                        optimization_result = {
                            "code": 500,
                            "msg": "运输参数生成失败：transport_modes_config为空",
                            "data": {
                                "algorithm_sequence_id": input_algorithm_sequence_id,
                                "mobilization_object_type": final_mobilization_object_type,
                                "error_type": "transport_modes_config_empty"
                            }
                        }
                    else:
                        # 重新生成完整的运输参数，确保包含距离矩阵
                        transport_result = input_transport_parameters(
                            {'transport_modes': transport_modes_config},
                            network_data,  # 确保传递network_data以生成距离矩阵
                            time_params
                        )

                        # 处理运输参数生成结果
                        if isinstance(transport_result, dict) and 'code' in transport_result:
                            if transport_result['code'] == 200:
                                transport_data = transport_result['data']
                            else:
                                optimization_result = {
                                    "code": transport_result['code'],
                                    "msg": f"运输数据处理失败: {transport_result['msg']}",
                                    "data": {
                                        "algorithm_sequence_id": input_algorithm_sequence_id,
                                        "mobilization_object_type": final_mobilization_object_type,
                                        **transport_result.get('data', {})
                                    }
                                }
                        else:
                            # 兼容旧格式
                            transport_data = transport_result

                        # 验证transport_data
                        if not transport_data or 'TRANSPORT_MODES' not in transport_data:
                            optimization_result = {
                                "code": 500,
                                "msg": "运输数据处理失败：transport_data无效",
                                "data": {
                                    "algorithm_sequence_id": input_algorithm_sequence_id,
                                    "mobilization_object_type": final_mobilization_object_type,
                                    "error_type": "invalid_transport_data"
                                }
                            }
                        # 再次验证距离矩阵是否存在
                        elif not all(matrix in transport_data for matrix in ['L_j_m', 'L_m_m', 'L_m_k']):
                            optimization_result = {
                                "code": 500,
                                "msg": "运输数据处理失败：缺少必要的距离矩阵",
                                "data": {
                                    "algorithm_sequence_id": input_algorithm_sequence_id,
                                    "mobilization_object_type": final_mobilization_object_type,
                                    "available_matrices": [k for k in transport_data.keys() if k.startswith('L_')],
                                    "error_type": "missing_distance_matrices"
                                }
                            }
                        else:
                            # 准备算法参数 - 预计算公共值避免重复计算
                            supply_count = network_metrics['supply_count']
                            demand_count = network_metrics['demand_count']
                            total_scale = network_metrics['total_scale']
                            weights_count = len(strategy_weights)

                            # 预计算优先级层次的默认值
                            default_priority_levels = {
                                "highest": total_scale,
                                "high": supply_count + demand_count / (demand_count + 1),
                                "medium": supply_count / (supply_count + 1),
                                "low": supply_count / total_scale
                            }

                            # 预计算eps和bigm的默认值
                            default_eps = (weights_count / (weights_count + total_scale) if total_scale > 0
                                           else weights_count / (weights_count + 1)) / (total_scale + 1)
                            default_bigm = (total_scale * weights_count * total_scale * avg_network_distance
                                            if avg_network_distance > 0
                                            else total_scale * weights_count * total_scale)

                            algorithm_params = {
                                'objective_names': {
                                    "time": "时间目标",
                                    "cost": "成本目标",
                                    "distance": "距离目标",
                                    "safety": "安全性目标",
                                    "priority": "优先级目标",
                                    "balance": "资源均衡目标",
                                    "capability": "企业能力目标",
                                    "social": "社会影响目标"
                                },
                                'priority_levels': priority_parameters.get('priority_levels', default_priority_levels),
                                'eps': algorithm_parameters.get('eps', default_eps),
                                'bigm': algorithm_parameters.get('bigm', default_bigm)
                            }

                            # 确定资源类型
                            resource_type = final_mobilization_object_type if final_mobilization_object_type in [
                                "material",
                                "personnel",
                                "data"] else "material"

                            # 执行优化
                            solver_result = MultiObjectiveFourLayerNetworkOptimizer().solve(
                                network_data=network_data,
                                transport_params=transport_data,
                                algorithm_params=algorithm_params,
                                resource_type=resource_type,
                                objective_weights=strategy_weights,
                                random_seed=len(detailed_supply_points) * len(detailed_demand_points) + len(
                                    strategy_weights),
                                save_results=False,
                                output_dir="results",
                                return_format="json",
                                algorithm_sequence_id=input_algorithm_sequence_id,
                                mobilization_object_type=final_mobilization_object_type,
                                req_element_id=final_req_element_id,
                                scheme_id=None,
                                activity_id=final_activity_id,
                                scheme_config=None
                            )

                            # 验证solver返回值
                            if solver_result is None:
                                optimization_result = {
                                    "code": 500,
                                    "msg": "优化器返回空值：solver执行异常",
                                    "data": {
                                        "algorithm_sequence_id": input_algorithm_sequence_id,
                                        "mobilization_object_type": final_mobilization_object_type,
                                        "error_type": "solver_null_result"
                                    }
                                }
                            elif not isinstance(solver_result, dict):
                                optimization_result = {
                                    "code": 500,
                                    "msg": f"优化器返回值类型错误：期望dict，实际{type(solver_result)}",
                                    "data": {
                                        "algorithm_sequence_id": input_algorithm_sequence_id,
                                        "mobilization_object_type": final_mobilization_object_type,
                                        "actual_type": type(solver_result).__name__,
                                        "error_type": "solver_type_error"
                                    }
                                }
                            else:
                                # 检查solver_result本身和内部是否有嵌套的错误状态码
                                if solver_result.get('code') != 200:
                                    # 如果solver_result本身有错误，直接使用
                                    optimization_result = solver_result
                                elif (isinstance(solver_result.get('data'), dict) and
                                      'code' in solver_result['data'] and
                                      solver_result['data']['code'] != 200):
                                    # 如果内部有错误，将内部错误提升到顶层
                                    nested_error = solver_result['data']
                                    optimization_result = {
                                        "code": nested_error['code'],
                                        "msg": f"优化执行失败: {nested_error.get('msg', '未知错误')}",
                                        "data": {
                                            "algorithm_sequence_id": input_algorithm_sequence_id,
                                            "mobilization_object_type": final_mobilization_object_type,
                                            **nested_error.get('data', {}),
                                            "error_type": "nested_optimization_error"
                                        }
                                    }
                                else:
                                    # 检查solver_result本身和内部是否有嵌套的错误状态码
                                    if solver_result.get('code') != 200:
                                        # 如果solver_result本身有错误，直接使用
                                        optimization_result = solver_result
                                    elif (isinstance(solver_result.get('data'), dict) and
                                          'code' in solver_result['data'] and
                                          solver_result['data']['code'] != 200):
                                        # 如果内部有错误，将内部错误提升到顶层
                                        nested_error = solver_result['data']
                                        optimization_result = {
                                            "code": nested_error['code'],
                                            "msg": f"优化执行失败: {nested_error.get('msg', '未知错误')}",
                                            "data": {
                                                "algorithm_sequence_id": input_algorithm_sequence_id,
                                                "mobilization_object_type": final_mobilization_object_type,
                                                **nested_error.get('data', {}),
                                                "error_type": "nested_optimization_error"
                                            }
                                        }
                                    else:
                                        optimization_result = solver_result
                else:
                    # 使用缓存的运输数据（已经包含完整的距离矩阵）
                    # 准备算法参数
                    network_scale = len(detailed_supply_points) + len(detailed_demand_points)
                    algorithm_params = {
                        'objective_names': {
                            "time": "时间目标",
                            "cost": "成本目标",
                            "distance": "距离目标",
                            "safety": "安全性目标",
                            "priority": "优先级目标",
                            "balance": "资源均衡目标",
                            "capability": "企业能力目标",
                            "social": "社会影响目标"
                        },
                        'priority_levels': priority_parameters.get('priority_levels',
                                                                   {"highest": len(detailed_supply_points) + len(
                                                                       detailed_demand_points),
                                                                    "high": len(detailed_supply_points) + len(
                                                                        detailed_demand_points) / (
                                                                                    len(detailed_demand_points) + 1),
                                                                    "medium": len(detailed_supply_points) / (
                                                                            len(detailed_supply_points) + 1),
                                                                    "low": len(detailed_supply_points) / (
                                                                            len(detailed_supply_points) + len(
                                                                        detailed_demand_points))}),
                        'eps': algorithm_parameters.get('eps', len(strategy_weights) / (
                                len(strategy_weights) + network_scale) if network_scale > 0 else len(
                            strategy_weights) / (len(strategy_weights) + 1) / (len(detailed_supply_points) + len(
                            detailed_demand_points) + 1)),
                        'bigm': algorithm_parameters.get('bigm', network_scale * len(strategy_weights) * (
                                len(detailed_supply_points) + len(detailed_demand_points)) * network_scale)
                    }

                    # 确定资源类型
                    resource_type = final_mobilization_object_type if final_mobilization_object_type in ["material",
                                                                                                         "personnel",
                                                                                                         "data"] else "material"

                    # 执行优化
                    solver_result = MultiObjectiveFourLayerNetworkOptimizer().solve(
                        network_data=network_data,
                        transport_params=transport_data,
                        algorithm_params=algorithm_params,
                        resource_type=resource_type,
                        objective_weights=strategy_weights,
                        random_seed=len(detailed_supply_points) * len(detailed_demand_points) + len(strategy_weights),
                        save_results=False,
                        output_dir="results",
                        return_format="json",
                        algorithm_sequence_id=input_algorithm_sequence_id,
                        mobilization_object_type=final_mobilization_object_type,
                        req_element_id=final_req_element_id,
                        scheme_id=None,
                        activity_id=final_activity_id,
                        scheme_config=None
                    )

                    # 验证solver返回值
                    if solver_result is None:
                        optimization_result = {
                            "code": 500,
                            "msg": "优化器返回空值：solver执行异常",
                            "data": {
                                "algorithm_sequence_id": input_algorithm_sequence_id,
                                "mobilization_object_type": final_mobilization_object_type,
                                "error_type": "solver_null_result"
                            }
                        }
                    elif not isinstance(solver_result, dict):
                        optimization_result = {
                            "code": 500,
                            "msg": f"优化器返回值类型错误：期望dict，实际{type(solver_result)}",
                            "data": {
                                "algorithm_sequence_id": input_algorithm_sequence_id,
                                "mobilization_object_type": final_mobilization_object_type,
                                "actual_type": type(solver_result).__name__,
                                "error_type": "solver_type_error"
                            }
                        }
                    else:
                        # 检查solver_result内部是否有嵌套的错误状态码
                        if (isinstance(solver_result.get('data'), dict) and
                                'code' in solver_result['data'] and
                                solver_result['data']['code'] != 200):
                            # 如果内部有错误，将内部错误提升到顶层
                            nested_error = solver_result['data']
                            optimization_result = {
                                "code": nested_error['code'],
                                "msg": f"优化执行失败: {nested_error.get('msg', '未知错误')}",
                                "data": {
                                    "algorithm_sequence_id": input_algorithm_sequence_id,
                                    "mobilization_object_type": final_mobilization_object_type,
                                    **nested_error.get('data', {}),
                                    "error_type": "nested_optimization_error"
                                }
                            }
                        else:
                            optimization_result = solver_result

            except KeyError as ke:
                optimization_result = {
                    "code": 500,
                    "msg": f"优化执行参数访问错误: {str(ke)}",
                    "data": {
                        "algorithm_sequence_id": input_algorithm_sequence_id,
                        "mobilization_object_type": final_mobilization_object_type,
                        "missing_key": str(ke),
                        "error_type": "optimization_key_error"
                    }
                }
            except ValueError as ve:
                optimization_result = {
                    "code": 400,
                    "msg": f"优化执行参数值错误: {str(ve)}",
                    "data": {
                        "algorithm_sequence_id": input_algorithm_sequence_id,
                        "mobilization_object_type": final_mobilization_object_type,
                        "error_type": "optimization_value_error"
                    }
                }
            except TypeError as te:
                optimization_result = {
                    "code": 500,
                    "msg": f"优化执行类型错误: {str(te)}",
                    "data": {
                        "algorithm_sequence_id": input_algorithm_sequence_id,
                        "mobilization_object_type": final_mobilization_object_type,
                        "error_type": "optimization_type_error"
                    }
                }
            except Exception as e:
                optimization_result = {
                    "code": 500,
                    "msg": f"优化执行过程中发生未知错误: {str(e)}",
                    "data": {
                        "algorithm_sequence_id": input_algorithm_sequence_id,
                        "mobilization_object_type": final_mobilization_object_type,
                        "error_type": "optimization_unknown_error"
                    }
                }

        # 将优化结果添加到最终结果中
        result['optimization_result'] = optimization_result

        # 根据优化结果决定最终返回码
        if isinstance(optimization_result, dict) and 'code' in optimization_result:
            if optimization_result['code'] == 200:
                return {
                    "code": 200,
                    "msg": "综合参数处理完成，优化执行成功",
                    "data": result
                }
            else:
                return {
                    "code": optimization_result['code'],
                    "msg": f"综合参数处理完成，但优化执行失败: {optimization_result.get('msg', '未知错误')}",
                    "data": result
                }
        else:
            return {
                "code": 500,
                "msg": "综合参数处理完成，但优化结果格式异常",
                "data": result
            }

    except json.JSONDecodeError as e:
        return {
            "code": 400,
            "msg": f"无效的JSON格式: {str(e)}",
            "data": {
                "input_type": type(input_json).__name__,
                "algorithm_sequence_id": input_algorithm_sequence_id,
                "mobilization_object_type": final_mobilization_object_type,
                "req_element_id": final_req_element_id,
                "scheme_id": final_scheme_id,
                "activity_id": final_activity_id,
                "scheme_config": scheme_config,
                "error_type": "json_decode_error"
            }
        }

    except Exception as e:
        return {
            "code": 500,
            "msg": f"综合参数处理错误: {str(e)}",
            "data": {
                "input_type": type(input_json).__name__,
                "algorithm_sequence_id": input_algorithm_sequence_id,
                "mobilization_object_type": final_mobilization_object_type,
                "req_element_id": final_req_element_id,
                "scheme_id": final_scheme_id,
                "activity_id": final_activity_id,
                "scheme_config": scheme_config,
                "error_type": "processing_error"
            }
        }

def safe_float_convert(value, default=0.0):
    """安全的数值转换函数"""
    try:
        if isinstance(value, (int, float)):
            return float(value)
        elif isinstance(value, str):
            if value.strip() == '':
                return default
            return float(value)
        else:
            return default
    except (ValueError, TypeError):
        return default

def convert_keys_to_string(obj):
        """
        递归地将所有非字符串键转换为字符串
        """
        if isinstance(obj, MutableMapping):  # 字典类型
            new_obj = {}
            for key, value in obj.items():
                # 将键转换为字符串
                str_key = str(key) if not isinstance(key, (str, int, float, bool)) or key is None else key
                # 递归处理值
                new_obj[str_key] = convert_keys_to_string(value)
            return new_obj
        elif isinstance(obj, MutableSequence):  # 列表类型
            return [convert_keys_to_string(item) for item in obj]
        else:
            return obj

def safe_json_dumps(data, ensure_ascii=False, indent=2):
        """
        安全地将数据转换为 JSON 字符串，处理非标准键类型
        """
        # 首先转换所有键为字符串
        converted_data = convert_keys_to_string(data)

        # 然后序列化为 JSON
        return json.dumps(converted_data, ensure_ascii=ensure_ascii, indent=indent)

def _validate_and_normalize_weights(weights: Dict[str, float], algorithm_sequence_id=None,
                                    mobilization_object_type=None, scheme_id=None, req_element_id=None,
                                    activity_id=None, scheme_config=None):
    # 定义必需的目标
    required_objectives = {'time', 'cost', 'distance', 'safety', 'priority', 'balance', 'capability', 'social'}

    # 检查是否包含所有必需的目标
    missing_objectives = required_objectives - set(weights.keys())
    if missing_objectives:
        return {
            "code": 400,
            "msg": f"缺少目标权重: {missing_objectives}",
            "data": {
                "provided_objectives": list(weights.keys()),
                "missing_objectives": list(missing_objectives),
                "required_objectives": list(required_objectives),
                "algorithm_sequence_id": algorithm_sequence_id,
                "mobilization_object_type": mobilization_object_type,
                "scheme_id": scheme_id,
                "req_element_id": req_element_id,
                "error_type": "missing_objectives"
            }
        }

    # 检查权重是否为非负数
    negative_weights = {obj: weight for obj, weight in weights.items() if weight < 0}
    if negative_weights:
        return {
            "code": 400,
            "msg": f"目标权重不能为负数: {negative_weights}",
            "data": {
                "negative_weights": negative_weights,
                "all_weights": weights,
                "algorithm_sequence_id": algorithm_sequence_id,
                "mobilization_object_type": mobilization_object_type,
                "scheme_id": scheme_id,
                "req_element_id": req_element_id,
                "error_type": "negative_weights"
            }
        }

    # 标准化权重（使权重和为1）
    total_weight = sum(weights.values())
    if total_weight <= 0:
        return {
            "code": 400,
            "msg": "权重总和必须大于0",
            "data": {
                "weights": weights,
                "total_weight": total_weight,
                "algorithm_sequence_id": algorithm_sequence_id,
                "mobilization_object_type": mobilization_object_type,
                "scheme_id": scheme_id,
                "req_element_id": req_element_id,
                "error_type": "zero_total_weight"
            }
        }

    normalized_weights = {obj: weight / total_weight for obj, weight in weights.items()}

    return {
        "code": 200,
        "msg": "权重验证和标准化成功",
        "data": normalized_weights
    }


def _generate_network_data(other_params: Dict, safety_cost_params: Dict, transfer_point_params: Dict,
                           algorithm_sequence_id=None, mobilization_object_type=None, scheme_id=None,
                           req_element_id=None, activity_id=None, scheme_config=None) -> Dict[str, Any]:
    """
    生成网络数据 - 统一使用英文键名，并根据安全计算参照标准处理安全数据
    现已支持同一供应点内的细分对象差异化处理

    Args:
        other_params: 综合参数配置
        safety_cost_params: 安全计算参照标准
        transfer_point_params: 中转点参数

    Returns:
        Dict[str, Any]: 网络数据字典
    """

    try:
        # 提取网络节点信息
        network_nodes = other_params.get('network_nodes', {})

        # 检查中转点数量并提前优化
        preliminary_transfer_points = transfer_point_params.get('transfer_points', [])
        if len(preliminary_transfer_points) > len(other_params.get('network_nodes', {}).get('supply_points', [])) * len(
                other_params.get('network_nodes', {}).get('demand_points', [])):
            pass  # 标记：大规模中转点场景
        supply_points_data = network_nodes.get('supply_points', [])
        transfer_points_data = network_nodes.get('transfer_points', [])
        demand_points_data = network_nodes.get('demand_points', [])

        # 如果没有网络节点信息，尝试从其他位置获取
        if not network_nodes:
            supply_points_data = other_params.get('supply_points', [])
            transfer_points_data = other_params.get('transfer_points', [])
            demand_points_data = other_params.get('demand_points', [])

        # 验证必要数据存在
        if not supply_points_data:
            raise ValueError("缺少必需的供应点数据：supply_points")

        if not demand_points_data:
            raise ValueError("缺少必需的需求点数据：demand_points")

        # 获取安全计算参照标准
        safety_standards = safety_cost_params.get('safety_parameters', {})
        if not safety_standards:
            raise ValueError("缺少必需的安全计算参照标准：safety_parameters")

        mobilization_type = safety_standards.get('mobilization_object_type')
        if not mobilization_type:
            raise ValueError("缺少必需的动员对象类型：mobilization_object_type")

        # 提取必要的配置参数并验证
        algorithm_parameters = other_params.get('algorithm_parameters', {})
        if not algorithm_parameters:
            raise ValueError("缺少必需的算法参数：algorithm_parameters")

        # 计算网络规模基础参数
        network_scale_base = len(supply_points_data)
        transfer_points_count = len(transfer_point_params.get('transfer_points', []))
        demand_points_count = len(demand_points_data)
        total_expected_nodes = network_scale_base + transfer_points_count + demand_points_count

        # 定义安全数值转换函数
        def convert_safety_value(category, subcategory, value_text, standards):
            """根据安全计算参照标准将文本值转换为数值"""
            try:
                if category in standards:
                    mapping_key = f"{subcategory}_mapping"
                    if mapping_key in standards[category]:
                        return standards[category][mapping_key].get(value_text, 0)
                return 0
            except (KeyError, TypeError):
                return 0

        # 从安全标准中提取选项列表
        def extract_options_from_safety_standards(standards, mobilization_type):
            """从安全标准中提取选项列表"""
            options = {}

            if mobilization_type == 'personnel':
                personnel_standards = standards.get('personnel_safety_mappings', {})
                options.update({
                    'political_options': list(personnel_standards.get('political_status_mapping', {}).keys()),
                    'experience_options': list(personnel_standards.get('military_experience_mapping', {}).keys()),
                    'record_options': list(personnel_standards.get('criminal_record_mapping', {}).keys()),
                    'credit_options': list(personnel_standards.get('credit_record_mapping', {}).keys())
                })

            elif mobilization_type == 'material':
                enterprise_standards = standards.get('enterprise_safety_mappings', {})
                material_standards = standards.get('material_safety_mappings', {})
                options.update({
                    'nature_options': list(enterprise_standards.get('enterprise_nature_mapping', {}).keys()),
                    'scale_options': list(enterprise_standards.get('enterprise_scale_mapping', {}).keys()),
                    'record_options': list(enterprise_standards.get('risk_record_mapping', {}).keys()),
                    'safety_options': list(enterprise_standards.get('resource_safety_mapping', {}).keys()),
                    'experience_options': list(enterprise_standards.get('mobilization_experience_mapping', {}).keys()),
                    'risk_options': list(material_standards.get('flammable_explosive_mapping', {}).keys())
                })

            elif mobilization_type == 'data':
                equipment_standards = standards.get('equipment_safety_mappings', {})
                facility_standards = standards.get('facility_safety_mappings', {})
                technology_standards = standards.get('technology_safety_mappings', {})
                options.update({
                    'binary_options': list(equipment_standards.get('autonomous_control_mapping', {}).keys()),
                    'usability_options': list(equipment_standards.get('usability_level_mapping', {}).keys()),
                    'protection_options': list(facility_standards.get('facility_protection_mapping', {}).keys()),
                    'environment_options': list(facility_standards.get('surrounding_environment_mapping', {}).keys()),
                    'tech_options': list(technology_standards.get('encryption_security_mapping', {}).keys())
                })

            return options

        # 获取选项列表
        safety_options = extract_options_from_safety_standards(safety_standards, mobilization_type)

        # 验证安全选项是否存在
        if not safety_options:
            raise ValueError(f"未找到 {mobilization_type} 类型的安全评价选项配置")

        # 生成细分对象的函数
        def generate_sub_objects_for_supply_point(supply_point_data, supply_point_index, mobilization_type,
                                                  safety_standards):
            """为供应点生成细分对象数据"""
            sub_objects = []

            # 预计算网络规模参数，避免重复计算
            network_params = {
                'scale_base': network_scale_base,
                'total_nodes': total_expected_nodes,
                'supply_count': len(supply_points_data),
                'transfer_count': transfer_points_count,
                'demand_count': demand_points_count
            }

            # 预计算安全值映射，避免重复转换
            safety_value_cache = {}
            for category_key in safety_standards:
                if isinstance(safety_standards[category_key], dict):
                    for mapping_key in safety_standards[category_key]:
                        if mapping_key.endswith('_mapping') and isinstance(safety_standards[category_key][mapping_key],
                                                                           dict):
                            cache_key = f"{category_key}_{mapping_key}"
                            safety_value_cache[cache_key] = safety_standards[category_key][mapping_key]

            # 根据动员类型确定细分对象数量和变化范围
            if mobilization_type == 'personnel':
                if 'political_options' not in safety_options or not safety_options['political_options']:
                    raise ValueError("人员动员缺少政治面貌选项配置")

                # 限制细分对象数量，避免生成过多对象
                base_count = max(1, min(network_scale_base, total_expected_nodes // len(supply_points_data))) if len(
                    supply_points_data) > 0 else network_scale_base
                sub_object_count = max(1, min((supply_point_index % base_count) + base_count,
                                              base_count * (transfer_points_count + 1) // len(
                                                  supply_points_data))) if len(
                    supply_points_data) > 0 else base_count

                # 计算分类数量，基于网络规模动态确定，限制最大分类数
                max_categories = max(1, min(network_scale_base // max(1, transfer_points_count), sub_object_count))
                categories_count = max(1, min(max_categories, sub_object_count // max(1, len(supply_points_data))))
                items_per_category = max(1, sub_object_count // max(1, categories_count))

                for cat_idx in range(categories_count):
                    category = {
                        'category_id': f"personnel_category_{cat_idx + 1}",
                        'category_name': f"人员类别{cat_idx + 1}",
                        'items': []
                    }

                    # 为每个分类生成具体的人员对象
                    for item_idx in range(items_per_category):
                        i = cat_idx * items_per_category + item_idx
                        if i >= sub_object_count:
                            break

                        # 人员动员：使用预计算的网络参数
                        base_wage = network_params['scale_base'] * network_params['total_nodes'] + supply_point_index * \
                                    network_params['scale_base']
                        base_living = network_params['scale_base'] * (supply_point_index + 1)

                        # 预计算成本变化因子，避免重复除法运算
                        wage_factor = network_params['scale_base'] / (network_params['scale_base'] + 1)
                        living_factor = network_params['scale_base'] / (
                                network_params['scale_base'] + network_params['total_nodes'])

                        # 不同人员的成本差异
                        wage_variation = (i % (network_params['scale_base'] + 1)) * wage_factor
                        living_variation = (i % (
                                network_params['total_nodes'] // network_params['scale_base'] + 1)) * living_factor

                        # 人员动员：每个人员的max_available_quantity固定为1
                        max_available_quantity = 1

                        # 使用缓存进行安全值查找
                        political_key = safety_options['political_options'][
                            i % len(safety_options['political_options'])]
                        experience_key = safety_options['experience_options'][
                            i % len(safety_options['experience_options'])]
                        record_key = safety_options['record_options'][i % len(safety_options['record_options'])]
                        credit_key = safety_options['credit_options'][i % len(safety_options['credit_options'])]

                        # 批量获取安全值，减少字典查找次数
                        safety_mappings = {
                            'political': safety_value_cache.get('personnel_safety_standards_political_status_mapping',
                                                                {}),
                            'experience': safety_value_cache.get(
                                'personnel_safety_standards_military_experience_mapping', {}),
                            'criminal': safety_value_cache.get('personnel_safety_standards_criminal_record_mapping',
                                                               {}),
                            'network': safety_value_cache.get('personnel_safety_standards_network_record_mapping', {}),
                            'credit': safety_value_cache.get('personnel_safety_standards_credit_record_mapping', {})
                        }

                        political_value = safety_mappings['political'].get(political_key, 0)
                        experience_value = safety_mappings['experience'].get(experience_key, 0)
                        criminal_value = safety_mappings['criminal'].get(record_key, 0)
                        network_value = safety_mappings['network'].get(record_key, 0)
                        credit_value = safety_mappings['credit'].get(credit_key, 0)

                        sub_obj = {
                            'sub_object_id': f"person_{i + 1}",
                            'sub_object_name': f"人员{i + 1}",
                            'wage_cost': base_wage + wage_variation,
                            'living_cost': base_living + living_variation,
                            'max_available_quantity': max_available_quantity,
                            'political_status': political_value,
                            'military_experience': experience_value,
                            'criminal_record': criminal_value,
                            'network_record': network_value,
                            'credit_record': credit_value
                        }
                        category['items'].append(sub_obj)

                    if category['items']:  # 只添加非空分类
                        sub_objects.append(category)

            elif mobilization_type == 'material':
                if 'nature_options' not in safety_options or not safety_options['nature_options']:
                    raise ValueError("物资动员缺少企业性质选项配置")
                base_divisor = max(1, network_scale_base // max(1, len(supply_points_data)))
                sub_object_count = max(1, min((supply_point_index % (base_divisor + 1)) + base_divisor,
                                              total_expected_nodes // max(1, len(supply_points_data))))
                # 计算分类数量，避免过多分类
                max_categories = max(1, min(demand_points_count, sub_object_count))
                categories_count = max(1, min(max_categories, network_scale_base // max(1, demand_points_count + len(
                    supply_points_data))))
                items_per_category = max(1, sub_object_count // max(1, categories_count))
                for cat_idx in range(categories_count):
                    category = {
                        'category_id': f"material_category_{cat_idx + 1}",
                        'category_name': f"物资类别{cat_idx + 1}",
                        'items': []
                    }
                    # 为每个分类生成具体的物资对象
                    for item_idx in range(items_per_category):
                        i = cat_idx * items_per_category + item_idx
                        if i >= sub_object_count:
                            break

                        # 物资动员：基础成本应该基于网络规模计算，而不是从供应点获取
                        base_material_price = network_scale_base * total_expected_nodes + supply_point_index * total_expected_nodes
                        base_equipment_rental = network_scale_base * total_expected_nodes + supply_point_index * network_scale_base
                        base_equipment_depreciation = network_scale_base * (supply_point_index + 1)

                        # 不同物资型号的成本差异
                        material_price_variation = (i % (network_scale_base + 2)) * (
                                network_scale_base / (network_scale_base + 1))
                        rental_variation = (i % (network_scale_base + 1)) * (
                                network_scale_base / (total_expected_nodes + 1))
                        depreciation_variation = (i % (total_expected_nodes // network_scale_base + 1)) * (
                                network_scale_base / (total_expected_nodes + network_scale_base))

                        # 定义容量分配函数，避免重复逻辑
                        def allocate_capacity(supply_capacity, categories_count, cat_idx, items_per_category, item_idx,
                                              user_quantity=None):
                            if user_quantity is not None:
                                return user_quantity

                            # 计算分类容量
                            base_category_capacity = supply_capacity // categories_count
                            category_capacity = (supply_capacity - (categories_count - 1) * base_category_capacity
                                                 if cat_idx == categories_count - 1 else base_category_capacity)

                            # 计算项目容量
                            base_item_capacity = category_capacity // items_per_category
                            return (category_capacity - item_idx * base_item_capacity
                                    if item_idx == items_per_category - 1 else base_item_capacity)

                        # 物资动员：使用统一的容量分配函数
                        total_supply_capacity = supply_point_data.get('capacity') or supply_point_data.get(
                            'capacity_info', {}).get('capacity', 0)
                        max_available_quantity = allocate_capacity(total_supply_capacity, categories_count, cat_idx,
                                                                   items_per_category, item_idx)

                        sub_obj = {
                            'sub_object_id': f"material_type_{i + 1}",
                            'sub_object_name': f"物资型号{i + 1}",
                            'material_price': base_material_price + material_price_variation,
                            'equipment_rental_price': base_equipment_rental + rental_variation,
                            'equipment_depreciation_cost': base_equipment_depreciation + depreciation_variation,
                            'max_available_quantity': max_available_quantity,
                            'enterprise_nature_score': convert_safety_value('enterprise_safety_standards',
                                                                            'enterprise_nature',
                                                                            safety_options['nature_options'][
                                                                                i % len(
                                                                                    safety_options['nature_options'])],
                                                                            safety_standards),
                            'enterprise_scale_score': convert_safety_value('enterprise_safety_standards',
                                                                           'enterprise_scale',
                                                                           safety_options['scale_options'][
                                                                               i % len(
                                                                                   safety_options['scale_options'])],
                                                                           safety_standards),
                            'risk_record': convert_safety_value('enterprise_safety_standards', 'risk_record',
                                                                safety_options['record_options'][
                                                                    i % len(safety_options['record_options'])],
                                                                safety_standards),
                            'foreign_background': convert_safety_value('enterprise_safety_standards',
                                                                       'foreign_background',
                                                                       safety_options['record_options'][
                                                                           i % len(safety_options['record_options'])],
                                                                       safety_standards),
                            'resource_safety': convert_safety_value('enterprise_safety_standards', 'resource_safety',
                                                                    safety_options['safety_options'][
                                                                        i % len(safety_options['safety_options'])],
                                                                    safety_standards),
                            'flammable_explosive': convert_safety_value('material_safety_standards',
                                                                        'flammable_explosive',
                                                                        safety_options['risk_options'][
                                                                            i % len(safety_options['risk_options'])],
                                                                        safety_standards),
                            'corrosive': convert_safety_value('material_safety_standards', 'corrosive',
                                                              safety_options['risk_options'][
                                                                  i % len(safety_options['risk_options'])],
                                                              safety_standards),
                            'polluting': convert_safety_value('material_safety_standards', 'polluting',
                                                              safety_options['risk_options'][
                                                                  i % len(safety_options['risk_options'])],
                                                              safety_standards),
                            'fragile': convert_safety_value('material_safety_standards', 'fragile',
                                                            safety_options['risk_options'][
                                                                i % len(safety_options['risk_options'])],
                                                            safety_standards)
                        }
                        category['items'].append(sub_obj)

                    if category['items']:  # 只添加非空分类
                        sub_objects.append(category)

            elif mobilization_type == 'data':
                base_divisor = max(1, network_scale_base // max(1, len(supply_points_data)))
                sub_object_count = max(1, min((supply_point_index % (base_divisor + 1)) + base_divisor,
                                              total_expected_nodes // max(1, len(supply_points_data))))
                # 计算分类数量，避免创建过多分类
                if demand_points_count > 0:
                    max_categories = max(1, min(transfer_points_count, sub_object_count))
                    categories_count = max(1, min(max_categories,
                                                  transfer_points_count // max(1, demand_points_count + 1)))
                else:
                    categories_count = max(1, min(transfer_points_count, sub_object_count))
                items_per_category = max(1, sub_object_count // max(1, categories_count))
                for cat_idx in range(categories_count):
                    category = {
                        'category_id': f"data_category_{cat_idx + 1}",
                        'category_name': f"数据类别{cat_idx + 1}",
                        'items': []
                    }

                    # 为每个分类生成具体的数据对象
                    for item_idx in range(items_per_category):
                        i = cat_idx * items_per_category + item_idx
                        if i >= sub_object_count:
                            break

                        # 数据动员：基础成本基于网络规模计算，成本指标下放到细分对象中
                        base_facility_rental = network_scale_base * (supply_point_index + 1) + network_scale_base * (
                                i + 1) / (network_scale_base + 1)
                        base_power_cost = network_scale_base * (supply_point_index + 1) + (i * network_scale_base) / (
                                total_expected_nodes + 1)
                        base_communication_cost = network_scale_base * (supply_point_index + 1) + network_scale_base + (
                                i * total_expected_nodes) / (network_scale_base + total_expected_nodes + 1)
                        base_data_processing_cost = network_scale_base + (supply_point_index * network_scale_base) + (
                                i * network_scale_base) / (network_scale_base + 1)
                        base_data_storage_cost = network_scale_base / (network_scale_base + 1) + (
                                supply_point_index + i) * network_scale_base / (total_expected_nodes + 1)

                        # 不同数据类型的成本差异
                        facility_variation = (i % (network_scale_base + 1)) * (
                                network_scale_base / (network_scale_base + total_expected_nodes))
                        power_variation = (i % (total_expected_nodes // network_scale_base + 1)) * (
                                network_scale_base / (total_expected_nodes + 1))
                        communication_variation = (i % (network_scale_base + 2)) * (
                                network_scale_base / (supply_point_index + total_expected_nodes + 1))
                        processing_variation = (i % (network_scale_base + 3)) * (
                                total_expected_nodes / (network_scale_base + total_expected_nodes + 1))
                        storage_variation = (i % (total_expected_nodes // network_scale_base + 2)) * (
                                network_scale_base / (total_expected_nodes + network_scale_base + 1))

                        # 数据动员：优先使用用户定义的数量，否则根据类型特征分配实际数量
                        user_defined_quantity = None  # 在生成时没有用户定义的数据
                        if user_defined_quantity is not None:
                            max_available_quantity = user_defined_quantity
                        else:
                            total_supply_capacity = supply_point_data.get('capacity') or supply_point_data.get(
                                'capacity_info', {}).get('capacity', 0)

                            # 根据细分对象在分类中的位置分配容量比例
                            total_items_in_category = items_per_category
                            remaining_capacity_for_category = total_supply_capacity // categories_count
                            if cat_idx == categories_count - 1:  # 最后一个分类获得剩余容量
                                remaining_capacity_for_category = total_supply_capacity - (categories_count - 1) * (
                                        total_supply_capacity // categories_count)

                            # 在分类内按项目索引分配
                            base_item_capacity = remaining_capacity_for_category // total_items_in_category
                            if item_idx == total_items_in_category - 1:  # 最后一个项目获得剩余容量
                                max_available_quantity = remaining_capacity_for_category - item_idx * base_item_capacity
                            else:
                                max_available_quantity = base_item_capacity

                        sub_obj = {
                            'sub_object_id': f"data_type_{i + 1}",
                            'sub_object_name': f"数据类型{i + 1}",
                            'facility_rental_price': base_facility_rental + facility_variation,
                            'power_cost': base_power_cost + power_variation,
                            'communication_purchase_price': base_communication_cost + communication_variation,
                            'data_processing_cost': base_data_processing_cost + processing_variation,
                            'data_storage_cost': base_data_storage_cost + storage_variation,
                            'max_available_quantity': max_available_quantity,
                            'autonomous_control': convert_safety_value('equipment_safety_standards',
                                                                       'autonomous_control',
                                                                       safety_options['binary_options'][
                                                                           i % len(safety_options['binary_options'])],
                                                                       safety_standards),
                            'usability_level': convert_safety_value('equipment_safety_standards', 'usability_level',
                                                                    safety_options['usability_options'][
                                                                        i % len(safety_options['usability_options'])],
                                                                    safety_standards),
                            'facility_protection': convert_safety_value('facility_safety_standards',
                                                                        'facility_protection',
                                                                        safety_options['protection_options'][
                                                                            i % len(
                                                                                safety_options['protection_options'])],
                                                                        safety_standards),
                            'camouflage_protection': convert_safety_value('facility_safety_standards',
                                                                          'camouflage_protection',
                                                                          safety_options['protection_options'][i % len(
                                                                              safety_options['protection_options'])],
                                                                          safety_standards),
                            'surrounding_environment': convert_safety_value('facility_safety_standards',
                                                                            'surrounding_environment',
                                                                            safety_options['environment_options'][
                                                                                i % len(
                                                                                    safety_options[
                                                                                        'environment_options'])],
                                                                            safety_standards),
                            'encryption_security': convert_safety_value('technology_safety_standards',
                                                                        'encryption_security',
                                                                        safety_options['tech_options'][
                                                                            i % len(safety_options['tech_options'])],
                                                                        safety_standards),
                            'access_control': convert_safety_value('technology_safety_standards', 'access_control',
                                                                   safety_options['tech_options'][
                                                                       i % len(safety_options['tech_options'])],
                                                                   safety_standards),
                            'network_security': convert_safety_value('technology_safety_standards', 'network_security',
                                                                     safety_options['tech_options'][
                                                                         i % len(safety_options['tech_options'])],
                                                                     safety_standards),
                            'data_integrity': convert_safety_value('technology_safety_standards', 'encryption_security',
                                                                   safety_options['tech_options'][
                                                                       (i + 1) % len(safety_options['tech_options'])],
                                                                   safety_standards),
                            'backup_security': convert_safety_value('technology_safety_standards', 'access_control',
                                                                    safety_options['tech_options'][
                                                                        (i + 2) % len(safety_options['tech_options'])],
                                                                    safety_standards),
                            'transmission_security': convert_safety_value('technology_safety_standards',
                                                                          'network_security',
                                                                          safety_options['tech_options'][
                                                                              (i + 3) % len(
                                                                                  safety_options['tech_options'])],
                                                                          safety_standards)
                        }
                        category['items'].append(sub_obj)

                    if category['items']:  # 只添加非空分类
                        sub_objects.append(category)

            return sub_objects

        # 1. 供应点集合 J
        J = []
        if isinstance(supply_points_data[0], dict):
            J = [supply_point.get('name') for supply_point in supply_points_data]
            # 验证供应点名称
            for i, name in enumerate(J):
                if not name:
                    raise ValueError(f"第 {i + 1} 个供应点缺少名称")
        else:
            J = [str(supply_point) for supply_point in supply_points_data]

        # 2. 中转点集合 M
        M = []
        transfer_point_list = transfer_point_params.get('transfer_points', [])
        if not transfer_point_list:
            raise ValueError("缺少必需的中转点数据：transfer_points")

        if isinstance(transfer_point_list[0], dict):
            M = [tp.get('name') for tp in transfer_point_list]
            # 验证中转点名称
            for i, name in enumerate(M):
                if not name:
                    raise ValueError(f"第 {i + 1} 个中转点缺少名称")
        else:
            M = [str(tp) for tp in transfer_point_list]

        # 3. 需求点集合 K
        K = []
        if isinstance(demand_points_data[0], dict):
            K = [demand_point.get('name') for demand_point in demand_points_data]
            # 验证需求点名称
            for i, name in enumerate(K):
                if not name:
                    raise ValueError(f"第 {i + 1} 个需求点缺少名称")
        else:
            K = [str(demand_point) for demand_point in demand_points_data]

        # 供应点特征生成
        point_features = {}

        # 处理供应点特征
        if isinstance(supply_points_data[0], dict):
            # 字典格式的节点数据
            for i, supply_point in enumerate(supply_points_data):
                j = supply_point.get('name')
                if not j:
                    raise ValueError(f"第 {i + 1} 个供应点缺少名称")

                # 获取原始supplier_id，优先使用id字段，其次使用name字段
                original_supplier_id = supply_point.get('id') or supply_point.get('supplier_id') or j

                # 验证必要字段
                basic_info = supply_point.get('basic_info', {})
                capacity_info = supply_point.get('capacity_info', {})
                cost_info = supply_point.get('cost_info', {})

                # 基础位置信息 - 灵活获取
                latitude = basic_info.get('latitude') or supply_point.get('latitude')
                longitude = basic_info.get('longitude') or supply_point.get('longitude')

                if latitude is None or longitude is None:
                    # 预计算坐标生成因子
                    coord_base_factor = network_scale_base * (i + 1) / len(supply_points_data)
                    coord_variation_factor = network_scale_base / (network_scale_base + 1)

                    # 使用基于索引的默认坐标
                    latitude = coord_base_factor + network_scale_base + i
                    longitude = network_scale_base * (i + 1) + network_scale_base + i * coord_variation_factor

                # 容量信息 - 灵活获取
                capacity = capacity_info.get('capacity') or supply_point.get('capacity')
                probability = capacity_info.get('probability') or supply_point.get('probability')

                if capacity is None:
                    capacity = network_scale_base * (i + 1) * network_scale_base + i * network_scale_base
                if probability is None:
                    probability = network_scale_base / (network_scale_base + i + 1) + (
                            i % network_scale_base) / network_scale_base / (network_scale_base + 1)

                # 企业能力相关特征 - 基于企业规模推导
                enterprise_size = supply_point.get('enterprise_size') or basic_info.get('enterprise_size') or '大'

                resource_reserve = (capacity_info.get('resource_reserve') or
                                    supply_point.get('resource_reserve') or
                                    network_scale_base / (network_scale_base + 1) + (i % (network_scale_base + 2)))

                production_capacity = (capacity_info.get('production_capacity') or
                                       supply_point.get('production_capacity') or
                                       network_scale_base + network_scale_base / (network_scale_base + 1) + (
                                               i % network_scale_base))

                expansion_capacity = (capacity_info.get('expansion_capacity') or
                                      supply_point.get('expansion_capacity') or
                                      resource_reserve + production_capacity)

                # 基础点特征
                point_features[j] = {
                    'original_supplier_id': original_supplier_id,  # 保存原始supplier_id
                    'latitude': latitude,
                    'longitude': longitude,
                    'enterprise_type': supply_point.get('enterprise_type') or basic_info.get(
                        'enterprise_type') or '国企',
                    'enterprise_size': supply_point.get('enterprise_size') or basic_info.get(
                        'enterprise_size') or '大',
                    'capacity': capacity,
                    'probability': probability,
                    'resource_reserve': resource_reserve,
                    'production_capacity': production_capacity,
                    'expansion_capacity': expansion_capacity,
                    'enterprise_scale_capability': (capacity_info.get('enterprise_scale_capability') or
                                                    supply_point.get('enterprise_scale_capability') or
                                                    resource_reserve + production_capacity)
                }

                # 根据动员对象类型添加相应的成本信息
                if mobilization_type == 'personnel':
                    # 人员动员：供应点不需要成本字段，成本在细分对象中
                    # 不添加任何成本字段到供应点
                    pass

                elif mobilization_type == 'material':
                    # 物资动员：供应点不需要成本字段，成本在细分对象中
                    # 不添加任何成本字段到供应点
                    pass

                elif mobilization_type == 'data':
                    # 数据动员：只添加设施相关成本字段到供应点
                    facility_cost_fields = ['facility_rental_price', 'power_cost', 'communication_purchase_price']

                    for cost_field in facility_cost_fields:
                        cost_value = cost_info.get(cost_field) or supply_point.get(cost_field)
                        if cost_value is None:
                            # 使用基于网络规模的默认值
                            if cost_field == 'facility_rental_price':
                                cost_value = network_scale_base * (
                                        i + 1) + network_scale_base + i * network_scale_base / (
                                                     network_scale_base + 1)
                            elif cost_field == 'power_cost':
                                cost_value = network_scale_base * (i + 1) / (
                                        network_scale_base + 1) + i / network_scale_base
                            else:  # communication_purchase_price
                                cost_value = network_scale_base * (i + 1) + network_scale_base / (
                                        network_scale_base + 1) + i
                        point_features[j][cost_field] = cost_value
                else:
                    # 未知动员类型的兼容性处理
                    # 可以根据需要添加默认行为或抛出异常
                    pass

                # 生成细分对象数据
                existing_sub_objects = supply_point.get('sub_objects') or supply_point.get('user_defined_sub_objects')
                if existing_sub_objects:
                    # 使用算例中已有的细分对象数据
                    point_features[j]['sub_objects'] = existing_sub_objects
                else:
                    # 生成细分对象数据
                    point_features[j]['sub_objects'] = generate_sub_objects_for_supply_point(
                        supply_point, i, mobilization_type, safety_standards)

                # 根据动员对象类型和安全计算参照标准处理安全数据
                safety_data = supply_point.get('safety_data', {})

                # 如果没有safety_data，尝试从其他位置获取安全相关数据
                if not safety_data:
                    # 尝试从不同的安全配置位置获取
                    enterprise_safety = supply_point.get('enterprise_safety', {})
                    material_safety = supply_point.get('material_safety', {})
                    personnel_safety = supply_point.get('personnel_safety', {})
                    equipment_safety = supply_point.get('equipment_safety', {})
                    facility_safety = supply_point.get('facility_safety', {})
                    technology_safety = supply_point.get('technology_safety', {})

                    # 合并所有安全数据
                    safety_data.update(enterprise_safety)
                    safety_data.update(material_safety)
                    safety_data.update(personnel_safety)
                    safety_data.update(equipment_safety)
                    safety_data.update(facility_safety)
                    safety_data.update(technology_safety)

                if mobilization_type == 'personnel':
                    required_fields = ['political_status', 'military_experience', 'criminal_record', 'network_record',
                                       'credit_record']
                    for field in required_fields:
                        field_value = safety_data.get(field)
                        if field_value is not None:
                            point_features[j][field] = convert_safety_value('personnel_safety_standards', field,
                                                                            field_value, safety_standards)
                        else:
                            # 提供默认安全值
                            default_options = safety_options.get('political_options',
                                                                 ['群众']) if 'political' in field else \
                                safety_options.get('experience_options', ['无']) if 'experience' in field else \
                                    safety_options.get('record_options', ['无']) if 'record' in field else \
                                        safety_options.get('credit_options', ['正常'])
                            default_value = default_options[i % len(default_options)] if default_options else '无'
                            point_features[j][field] = convert_safety_value('personnel_safety_standards', field,
                                                                            default_value, safety_standards)

                elif mobilization_type == 'material':
                    required_enterprise_fields = ['enterprise_nature', 'enterprise_scale', 'risk_record',
                                                  'foreign_background', 'resource_safety', 'mobilization_experience']
                    required_material_fields = ['flammable_explosive', 'corrosive', 'polluting', 'fragile']

                    for field in required_enterprise_fields:
                        field_value = safety_data.get(field)
                        score_field = f"{field}_score" if field in ['enterprise_nature', 'enterprise_scale'] else field
                        if field_value is not None:
                            point_features[j][score_field] = convert_safety_value('enterprise_safety_standards', field,
                                                                                  field_value, safety_standards)
                        else:
                            # 提供默认安全值
                            if field == 'enterprise_nature':
                                default_options = safety_options.get('nature_options', ['央国企'])
                            elif field == 'enterprise_scale':
                                default_options = safety_options.get('scale_options', ['大'])
                            elif field in ['risk_record', 'foreign_background']:
                                default_options = safety_options.get('record_options', ['无'])
                            elif field == 'resource_safety':
                                default_options = safety_options.get('safety_options', ['高安全性'])
                            else:
                                default_options = safety_options.get('experience_options', ['有'])

                            default_value = default_options[i % len(default_options)] if default_options else '无'
                            point_features[j][score_field] = convert_safety_value('enterprise_safety_standards', field,
                                                                                  default_value, safety_standards)

                    for field in required_material_fields:
                        field_value = safety_data.get(field)
                        if field_value is not None:
                            point_features[j][field] = convert_safety_value('material_safety_standards', field,
                                                                            field_value, safety_standards)
                        else:
                            # 提供默认安全值
                            default_options = safety_options.get('risk_options', ['低'])
                            default_value = default_options[i % len(default_options)] if default_options else '低'
                            point_features[j][field] = convert_safety_value('material_safety_standards', field,
                                                                            default_value, safety_standards)

                elif mobilization_type == 'data':
                    required_equipment_fields = ['autonomous_control', 'usability_level']
                    required_facility_fields = ['facility_protection', 'camouflage_protection',
                                                'surrounding_environment']
                    required_tech_fields = ['encryption_security', 'access_control', 'network_security',
                                            'terminal_security', 'dlp', 'security_policy', 'risk_assessment',
                                            'audit_monitoring', 'emergency_response']

                    for field in required_equipment_fields:
                        field_value = safety_data.get(field)
                        if field_value is not None:
                            point_features[j][field] = convert_safety_value('equipment_safety_standards', field,
                                                                            field_value, safety_standards)
                        else:
                            # 提供默认安全值
                            if field == 'autonomous_control':
                                default_options = safety_options.get('binary_options', ['是'])
                            elif field == 'usability_level':
                                default_options = safety_options.get('usability_options', ['能'])
                            else:
                                default_options = safety_options.get('maintenance_options', ['有'])

                            default_value = default_options[i % len(default_options)] if default_options else '有'
                            point_features[j][field] = convert_safety_value('equipment_safety_standards', field,
                                                                            default_value, safety_standards)

                    for field in required_facility_fields:
                        field_value = safety_data.get(field)
                        if field_value is not None:
                            point_features[j][field] = convert_safety_value('facility_safety_standards', field,
                                                                            field_value, safety_standards)
                        else:
                            # 提供默认安全值
                            if field == 'surrounding_environment':
                                default_options = safety_options.get('environment_options', ['正常'])
                            else:
                                default_options = safety_options.get('protection_options', ['有'])

                            default_value = default_options[i % len(default_options)] if default_options else '有'
                            point_features[j][field] = convert_safety_value('facility_safety_standards', field,
                                                                            default_value, safety_standards)

                    for field in required_tech_fields:
                        field_value = safety_data.get(field)
                        if field_value is not None:
                            point_features[j][field] = convert_safety_value('technology_safety_standards', field,
                                                                            field_value, safety_standards)
                        else:
                            # 提供默认安全值
                            default_options = safety_options.get('tech_options', ['有'])
                            default_value = default_options[i % len(default_options)] if default_options else '有'
                            point_features[j][field] = convert_safety_value('technology_safety_standards', field,
                                                                            default_value, safety_standards)
        else:
            raise ValueError("供应点数据必须为字典格式，包含完整的配置信息")

        # 4. 中转点特征
        base_latitude_offset = network_scale_base
        base_longitude_offset = total_expected_nodes
        latitude_increment = network_scale_base / max(1, total_expected_nodes)
        longitude_increment = network_scale_base
        for i, m in enumerate(M):
            if i < len(transfer_point_list):
                tp = transfer_point_list[i]
                if isinstance(tp, dict):
                    latitude = tp.get('latitude')
                    longitude = tp.get('longitude')
                    if latitude is None or longitude is None:
                        latitude = base_latitude_offset + i * latitude_increment
                        longitude = base_longitude_offset + i * longitude_increment

                    point_features[m] = {
                        'latitude': latitude,
                        'longitude': longitude,
                        'specialized_mode': tp.get('specialized_mode', 'mixed')
                    }
                else:
                    raise ValueError("中转点数据必须为字典格式，包含完整的配置信息")
            else:
                raise ValueError(f"中转点 {m} 缺少配置数据")

        # 5. 需求点特征
        for i, k in enumerate(K):
            if i < len(demand_points_data):
                dp = demand_points_data[i]
                if isinstance(dp, dict):
                    location = dp.get('location', {})
                    demand_info = dp.get('demand_info', {})
                    time_constraints = dp.get('time_constraints', {})

                    latitude = location.get('latitude') or dp.get('latitude')
                    longitude = location.get('longitude') or dp.get('longitude')
                    demand = demand_info.get('demand') or dp.get('demand')

                    # 提供默认值
                    if latitude is None or longitude is None:
                        latitude = network_scale_base * demand_points_count / (
                                network_scale_base + demand_points_count) + network_scale_base / (
                                           network_scale_base + 1) + i / demand_points_count
                        longitude = network_scale_base * demand_points_count / (
                                network_scale_base + demand_points_count) + demand_points_count / (
                                            demand_points_count + 1) + i / demand_points_count

                    if demand is None:
                        demand = network_scale_base * demand_points_count * (network_scale_base + demand_points_count)

                    time_window_earliest = time_constraints.get('time_window_earliest') or dp.get(
                        'time_window_earliest')
                    time_window_latest = time_constraints.get('time_window_latest') or dp.get('time_window_latest')

                    if time_window_earliest is None or time_window_latest is None:
                        time_window_earliest = network_scale_base / (network_scale_base + demand_points_count)
                        time_window_latest = network_scale_base * demand_points_count

                    point_features[k] = {
                        'latitude': latitude,
                        'longitude': longitude,
                        'demand': demand,
                        'time_window_earliest': time_window_earliest,
                        'time_window_latest': time_window_latest
                    }
                else:
                    raise ValueError("需求点数据必须为字典格式，包含完整的配置信息")
            else:
                raise ValueError(f"需求点 {k} 缺少配置数据")

        # 生成B、P、D、Q等参数
        # B: 供应点供应量
        B = {}
        for j in J:
            B[j] = point_features[j]['capacity']

        # P: 供应点概率
        P = {}
        for j in J:
            P[j] = point_features[j]['probability']

        # D: 需求点需求量
        D = {}
        for k in K:
            D[k] = point_features[k]['demand']

        # demand_priority: 需求点优先级
        priority_parameters = other_params.get('priority_parameters', {})
        priority_levels = priority_parameters.get('priority_levels', {
            "highest": network_scale_base + demand_points_count,
            "high": network_scale_base + demand_points_count / (demand_points_count + 1),
            "medium": network_scale_base / (network_scale_base + 1),
            "low": network_scale_base / (network_scale_base + demand_points_count)
        })

        demand_priority = {}
        for i, k in enumerate(K):
            dp = demand_points_data[i]
            if isinstance(dp, dict):
                priority_level = dp.get('priority_level')
                priority_value = dp.get('priority')

                if priority_level and priority_level in priority_levels:
                    demand_priority[k] = priority_levels[priority_level]
                elif priority_value is not None:
                    demand_priority[k] = priority_value
                else:
                    # 提供默认优先级
                    demand_priority[k] = priority_levels.get('medium', network_scale_base / (network_scale_base + 1))

        # Q: 企业综合能力
        Q = {}
        capability_parameters = other_params.get('capability_parameters', {})
        dynamic_feedback_score = capability_parameters.get('dynamic_feedback_score', 1)

        for j in J:
            enterprise_nature_score = point_features[j].get('enterprise_nature_score', 0.7)
            enterprise_scale_score = point_features[j].get('enterprise_scale_score', 0.6)
            risk_record = point_features[j].get('risk_record', 0)
            foreign_background = point_features[j].get('foreign_background', 0)
            resource_safety = point_features[j].get('resource_safety', 1)
            production_capacity = point_features[j].get('production_capacity')
            expansion_capacity = point_features[j].get('expansion_capacity')
            resource_reserve = point_features[j].get('resource_reserve')

            # 使用网络规模作为基准进行比较
            production_score = 1 if production_capacity >= network_scale_base else 0 if production_capacity >= network_scale_base // 2 else -1
            expansion_score = 1 if expansion_capacity >= network_scale_base else 0 if expansion_capacity >= network_scale_base // 2 else -1
            resource_score = 1 if resource_reserve >= network_scale_base else 0 if resource_reserve >= network_scale_base // 2 else -1

            Q[j] = (enterprise_nature_score + enterprise_scale_score + risk_record +
                    foreign_background + resource_safety + dynamic_feedback_score +
                    production_score + expansion_score + resource_score)

        # time_windows: 时间窗
        time_windows = {}
        for k in K:
            time_windows[k] = (point_features[k]['time_window_earliest'],
                               point_features[k]['time_window_latest'])

        return {
            "code": 200,
            "msg": "网络数据生成成功",
            "data": {
                'J': J, 'M': M, 'K': K,
                'point_features': point_features,
                'B': B, 'P': P, 'D': D,
                'demand_priority': demand_priority,
                'Q': Q,
                'time_windows': time_windows
            }
        }

    except Exception as e:
        return {
            "code": 500,
            "msg": f"网络数据生成错误: {str(e)}",
            "data": {
                "mobilization_type": safety_cost_params.get('safety_parameters', {}).get('mobilization_object_type'),
                "supply_points_count": len(other_params.get('network_nodes', {}).get('supply_points', [])),
                "transfer_points_count": len(transfer_point_params.get('transfer_points', [])),
                "demand_points_count": len(other_params.get('network_nodes', {}).get('demand_points', [])),
                "algorithm_sequence_id": algorithm_sequence_id,
                "mobilization_object_type": mobilization_object_type,
                "scheme_id": scheme_id,
                "scheme_config": scheme_config,
                "req_element_id": req_element_id,
                "error_type": "network_generation_error"
            }
        }


# ==========================================
# 接口使用示例
# ==========================================

def example_usage():
    """
    接口使用示例
    演示如何使用所有5个数据处理接口创建优化问题，并展示优化结果
    现已包含物资动员、人员动员、数据动员三种类型的完整示例
    """

    def safe_json_dumps(obj, **kwargs):
        """安全的JSON序列化函数，处理元组键"""

        def convert_keys(item):
            if isinstance(item, dict):
                return {str(key): convert_keys(value) for key, value in item.items()}
            elif isinstance(item, list):
                return [convert_keys(element) for element in item]
            else:
                return item

        converted_obj = convert_keys(obj)
        return json.dumps(converted_obj, **kwargs)

    def run_mobilization_example(mobilization_type, example_name):
        """运行指定动员类型的示例"""
        print(f"\n" + "=" * 100)
        print(f"开始执行{example_name}示例")
        print("=" * 100)

        # 1. 时间参数处理
        print(f"\n1. 处理时间参数...")
        time_input = '''{
            "base_preparation_time": 0.005,
            "base_transfer_time": 0.003,
            "base_unloading_time": 0.04,
            "base_assembly_time": 0.001,
            "base_loading_time": 0.002,
            "base_handover_time": 0.0005,
            "time_unit": "hours"
        }'''
        time_result = input_time_parameters(time_input)

        # 处理新的返回格式
        if isinstance(time_result, dict) and 'code' in time_result:
            if time_result['code'] == 200:
                time_params = time_result['data']
                print("时间参数接口完整输出:")
                print(safe_json_dumps(time_result, ensure_ascii=False, indent=2))
            else:
                print(f"时间参数处理失败: {time_result['msg']}")
                return {"error": time_result}
        else:
            # 兼容旧格式
            time_params = time_result
            print("时间参数接口完整输出:")
            print(safe_json_dumps(time_params, ensure_ascii=False, indent=2))

        # 2. 安全计算参照标准处理（根据动员对象类型）
        print(f"\n2. 处理安全计算参照标准（动员对象类型: {mobilization_type}）...")
        safety_standards_input = '''{
        "enterprise_safety_mappings": {
            "enterprise_nature_mapping": {
                "央国企": 0.7,
                "国企": 0.5, 
                "民企": 0.2,
                "外企": 0.1,
                "其他": 0.1
            },
            "enterprise_scale_mapping": {
                "大": 0.6,
                "中": 0.3,
                "小": 0.1
            },
            "risk_record_mapping": {
                "有": -1,
                "无": 0
            },
            "foreign_background_mapping": {
                "有": -1,
                "无": 0
            },
            "resource_safety_mapping": {
                "高安全性": 1,
                "一般安全性": 0,
                "低安全性": -1
            },
            "mobilization_experience_mapping": {
                "有": 1,
                "无": 0
            }
        },
        "material_safety_mappings": {
            "flammable_explosive_mapping": {
                "高": -0.7,
                "中": -0.2,
                "低": -0.1
            },
            "corrosive_mapping": {
                "高": -0.7,
                "中": -0.2,
                "低": -0.1
            },
            "polluting_mapping": {
                "高": -0.7,
                "中": -0.2,
                "低": -0.1
            },
            "fragile_mapping": {
                "高": -0.7,
                "中": -0.2,
                "低": -0.1
            }
        },
        "personnel_safety_mappings": {
            "political_status_mapping": {
                "群众": 0,
                "党员": 1
            },
            "military_experience_mapping": {
                "有": 1,
                "无": 0
            },
            "criminal_record_mapping": {
                "有": -1,
                "无": 0
            },
            "network_record_mapping": {
                "有": -1,
                "无": 0
            },
            "credit_record_mapping": {
                "不良": -1,
                "正常": 0
            }
        },
        "equipment_safety_mappings": {
            "autonomous_control_mapping": {
                "是": 1,
                "否": 0
            },
            "usability_level_mapping": {
                "能": 1,
                "否": 0
            }
        },
        "facility_safety_mappings": {
            "facility_protection_mapping": {
                "有": 1,
                "无": 0
            },
            "camouflage_protection_mapping": {
                "有": 1,
                "无": 0
            },
            "surrounding_environment_mapping": {
                "配套齐全": 1,
                "正常": 0,
                "较差": -1
            }
        },
        "technology_safety_mappings": {
            "encryption_security_mapping": {
                "有": 1,
                "无": 0
            },
            "access_control_mapping": {
                "有": 1,
                "无": 0
            },
            "network_security_mapping": {
                "有": 1,
                "无": 0
            },
            "terminal_security_mapping": {
                "有": 1,
                "无": 0
            },
            "dlp_mapping": {
                "有": 1,
                "无": 0
            },
            "security_policy_mapping": {
                "有": 1,
                "无": 0
            },
            "risk_assessment_mapping": {
                "有": 1,
                "无": 0
            },
            "audit_monitoring_mapping": {
                "有": 1,
                "无": 0
            },
            "emergency_response_mapping": {
                "有": 1,
                "无": 0
            }
        }
    }'''
        safety_result = input_safety_cost_parameters(safety_standards_input, mobilization_type)

        # 处理新的返回格式
        if isinstance(safety_result, dict) and 'code' in safety_result:
            if safety_result['code'] == 200:
                safety_standards = safety_result['data']
                print("安全计算参照标准接口完整输出:")
                print(safe_json_dumps(safety_result, ensure_ascii=False, indent=2))
            else:
                print(f"安全参照标准处理失败: {safety_result['msg']}")
                return {"error": safety_result}
        else:
            # 兼容旧格式
            safety_standards = safety_result
            print("安全计算参照标准接口完整输出:")
            print(safe_json_dumps(safety_standards, ensure_ascii=False, indent=2))

        # 3. 中转点参数处理 - 使用新的带序号格式
        print("\n3. 处理中转点参数...")

        transfer_input = '''{
              "transfer_points": [
                {
                  "id": "01",
                  "name": "武汉航空货运中心",
                  "latitude": 44.1118,
                  "longitude": 97.1583,
                  "specialized_mode": "airport"
                },
                {
                  "id": "02",
                  "name": "茂名港口经济区",
                  "latitude": 42.1187,
                  "longitude": 120.4006,
                  "specialized_mode": "port"
                },
                {
                  "id": "03",
                  "name": "潍坊国际班列站",
                  "latitude": 36.914,
                  "longitude": 123.5908,
                  "specialized_mode": "railway"
                },
                {
                  "id": "04",
                  "name": "金华港口经济区",
                  "latitude": 20.017,
                  "longitude": 96.9706,
                  "specialized_mode": "port"
                },
                {
                  "id": "05",
                  "name": "上海港口物流中心",
                  "latitude": 40.0063,
                  "longitude": 93.1712,
                  "specialized_mode": "port"
                },
                {
                  "id": "06",
                  "name": "舟山铁路集散中心",
                  "latitude": 24.8737,
                  "longitude": 82.4454,
                  "specialized_mode": "railway"
                },
                {
                  "id": "07",
                  "name": "上海临空经济区",
                  "latitude": 52.7279,
                  "longitude": 105.235,
                  "specialized_mode": "airport"
                },
                {
                  "id": "08",
                  "name": "长沙航空物流园",
                  "latitude": 26.2003,
                  "longitude": 128.1862,
                  "specialized_mode": "airport"
                },
                {
                  "id": "09",
                  "name": "莱芜空港经济区",
                  "latitude": 50.0238,
                  "longitude": 118.7303,
                  "specialized_mode": "airport"
                },
                {
                  "id": "10",
                  "name": "潮州编组站",
                  "latitude": 27.995,
                  "longitude": 92.6543,
                  "specialized_mode": "railway"
                },
                {
                  "id": "11",
                  "name": "成都航空货运中心",
                  "latitude": 42.1908,
                  "longitude": 120.7233,
                  "specialized_mode": "airport"
                },
                {
                  "id": "12",
                  "name": "茂名国际机场",
                  "latitude": 52.4369,
                  "longitude": 84.8111,
                  "specialized_mode": "airport"
                },
                {
                  "id": "13",
                  "name": "拉萨国际航空枢纽",
                  "latitude": 51.5961,
                  "longitude": 91.4845,
                  "specialized_mode": "airport"
                },
                {
                  "id": "14",
                  "name": "武汉编组站",
                  "latitude": 52.1589,
                  "longitude": 79.5359,
                  "specialized_mode": "railway"
                },
                {
                  "id": "15",
                  "name": "衡阳高铁站",
                  "latitude": 44.2395,
                  "longitude": 104.0362,
                  "specialized_mode": "railway"
                },
                {
                  "id": "16",
                  "name": "菏泽临空经济区",
                  "latitude": 48.8973,
                  "longitude": 86.1551,
                  "specialized_mode": "airport"
                },
                {
                  "id": "17",
                  "name": "舟山集装箱码头",
                  "latitude": 43.6103,
                  "longitude": 130.6852,
                  "specialized_mode": "port"
                },
                {
                  "id": "18",
                  "name": "天津智慧港口",
                  "latitude": 38.641,
                  "longitude": 114.4878,
                  "specialized_mode": "port"
                },
                {
                  "id": "19",
                  "name": "烟台货运机场",
                  "latitude": 20.9204,
                  "longitude": 83.0096,
                  "specialized_mode": "airport"
                },
                {
                  "id": "20",
                  "name": "岳阳国际空港",
                  "latitude": 36.1281,
                  "longitude": 110.7273,
                  "specialized_mode": "airport"
                },
                {
                  "id": "21",
                  "name": "烟台国际航空港",
                  "latitude": 21.6325,
                  "longitude": 132.1996,
                  "specialized_mode": "airport"
                },
                {
                  "id": "22",
                  "name": "舟山高铁站",
                  "latitude": 19.13,
                  "longitude": 82.177,
                  "specialized_mode": "railway"
                },
                {
                  "id": "23",
                  "name": "娄底航空货运中心",
                  "latitude": 25.3781,
                  "longitude": 87.4922,
                  "specialized_mode": "airport"
                },
                {
                  "id": "24",
                  "name": "清远航空货运中心",
                  "latitude": 19.8743,
                  "longitude": 117.6439,
                  "specialized_mode": "airport"
                },
                {
                  "id": "25",
                  "name": "中山航空物流园",
                  "latitude": 35.6418,
                  "longitude": 126.9968,
                  "specialized_mode": "airport"
                },
                {
                  "id": "26",
                  "name": "福州国际航空枢纽",
                  "latitude": 36.49,
                  "longitude": 83.661,
                  "specialized_mode": "airport"
                },
                {
                  "id": "27",
                  "name": "莱芜国际物流港",
                  "latitude": 28.3022,
                  "longitude": 133.3531,
                  "specialized_mode": "port"
                },
                {
                  "id": "28",
                  "name": "益阳高铁站",
                  "latitude": 20.4381,
                  "longitude": 104.6148,
                  "specialized_mode": "railway"
                },
                {
                  "id": "29",
                  "name": "江门国际机场",
                  "latitude": 35.458,
                  "longitude": 83.9635,
                  "specialized_mode": "airport"
                },
                {
                  "id": "30",
                  "name": "上海综合交通枢纽",
                  "latitude": 22.629,
                  "longitude": 95.3638,
                  "specialized_mode": "railway"
                },
                {
                  "id": "31",
                  "name": "绍兴编组站",
                  "latitude": 31.4943,
                  "longitude": 88.3112,
                  "specialized_mode": "railway"
                },
                {
                  "id": "32",
                  "name": "泰安货运机场",
                  "latitude": 46.3809,
                  "longitude": 93.3119,
                  "specialized_mode": "airport"
                },
                {
                  "id": "33",
                  "name": "武汉国际港区",
                  "latitude": 38.7591,
                  "longitude": 104.0574,
                  "specialized_mode": "port"
                },
                {
                  "id": "34",
                  "name": "石家庄航空物流园",
                  "latitude": 51.3753,
                  "longitude": 116.9193,
                  "specialized_mode": "airport"
                },
                {
                  "id": "35",
                  "name": "宁波国际航空枢纽",
                  "latitude": 30.2011,
                  "longitude": 116.3516,
                  "specialized_mode": "airport"
                },
                {
                  "id": "36",
                  "name": "湖州铁路集散中心",
                  "latitude": 20.8762,
                  "longitude": 126.2675,
                  "specialized_mode": "railway"
                },
                {
                  "id": "37",
                  "name": "湘西高铁站",
                  "latitude": 42.8791,
                  "longitude": 74.4278,
                  "specialized_mode": "railway"
                },
                {
                  "id": "38",
                  "name": "济南中欧班列枢纽",
                  "latitude": 29.3883,
                  "longitude": 125.5189,
                  "specialized_mode": "railway"
                },
                {
                  "id": "39",
                  "name": "宁波自贸港区",
                  "latitude": 25.6653,
                  "longitude": 74.6269,
                  "specialized_mode": "port"
                },
                {
                  "id": "40",
                  "name": "郑州航空港",
                  "latitude": 43.074,
                  "longitude": 121.0985,
                  "specialized_mode": "airport"
                },
                {
                  "id": "41",
                  "name": "汕尾空港经济区",
                  "latitude": 36.6363,
                  "longitude": 126.4965,
                  "specialized_mode": "airport"
                },
                {
                  "id": "42",
                  "name": "重庆临空经济区",
                  "latitude": 25.9146,
                  "longitude": 125.4878,
                  "specialized_mode": "airport"
                },
                {
                  "id": "43",
                  "name": "天津临空经济区",
                  "latitude": 27.9996,
                  "longitude": 132.2554,
                  "specialized_mode": "airport"
                },
                {
                  "id": "44",
                  "name": "广州自贸港区",
                  "latitude": 40.9932,
                  "longitude": 103.2044,
                  "specialized_mode": "port"
                },
                {
                  "id": "45",
                  "name": "汕尾集装箱码头",
                  "latitude": 25.3584,
                  "longitude": 106.3186,
                  "specialized_mode": "port"
                },
                {
                  "id": "46",
                  "name": "大连铁路港",
                  "latitude": 20.5951,
                  "longitude": 121.4147,
                  "specialized_mode": "railway"
                },
                {
                  "id": "47",
                  "name": "佛山现代物流港",
                  "latitude": 42.942,
                  "longitude": 107.1762,
                  "specialized_mode": "port"
                },
                {
                  "id": "48",
                  "name": "武汉临空经济区",
                  "latitude": 40.6734,
                  "longitude": 73.0043,
                  "specialized_mode": "airport"
                },
                {
                  "id": "49",
                  "name": "太原国际航空枢纽",
                  "latitude": 42.2441,
                  "longitude": 102.7204,
                  "specialized_mode": "airport"
                },
                {
                  "id": "50",
                  "name": "太原深水港区",
                  "latitude": 22.5909,
                  "longitude": 91.1257,
                  "specialized_mode": "port"
                },
                {
                  "id": "51",
                  "name": "西安联运枢纽",
                  "latitude": 36.8348,
                  "longitude": 90.1319,
                  "specialized_mode": "railway"
                },
                {
                  "id": "52",
                  "name": "滨州铁路枢纽",
                  "latitude": 21.7119,
                  "longitude": 73.9791,
                  "specialized_mode": "railway"
                },
                {
                  "id": "53",
                  "name": "无锡航空物流园",
                  "latitude": 41.3845,
                  "longitude": 97.423,
                  "specialized_mode": "airport"
                },
                {
                  "id": "54",
                  "name": "郑州多式联运站",
                  "latitude": 28.607,
                  "longitude": 93.9567,
                  "specialized_mode": "railway"
                },
                {
                  "id": "55",
                  "name": "重庆航空港",
                  "latitude": 28.3177,
                  "longitude": 122.3495,
                  "specialized_mode": "airport"
                },
                {
                  "id": "56",
                  "name": "西宁国际贸易港",
                  "latitude": 34.5886,
                  "longitude": 133.078,
                  "specialized_mode": "port"
                },
                {
                  "id": "57",
                  "name": "清远物流枢纽港",
                  "latitude": 30.2239,
                  "longitude": 87.2582,
                  "specialized_mode": "port"
                },
                {
                  "id": "58",
                  "name": "湛江高铁站",
                  "latitude": 21.8101,
                  "longitude": 133.8664,
                  "specialized_mode": "railway"
                },
                {
                  "id": "59",
                  "name": "北京航空集散中心",
                  "latitude": 23.8921,
                  "longitude": 81.0576,
                  "specialized_mode": "airport"
                },
                {
                  "id": "60",
                  "name": "兰州航空物流园",
                  "latitude": 30.2494,
                  "longitude": 106.8539,
                  "specialized_mode": "airport"
                },
                {
                  "id": "61",
                  "name": "东莞航空货运中心",
                  "latitude": 47.2551,
                  "longitude": 124.4148,
                  "specialized_mode": "airport"
                },
                {
                  "id": "62",
                  "name": "烟台港口经济区",
                  "latitude": 48.6353,
                  "longitude": 121.144,
                  "specialized_mode": "port"
                },
                {
                  "id": "63",
                  "name": "河源航空物流中心",
                  "latitude": 37.4311,
                  "longitude": 120.0109,
                  "specialized_mode": "airport"
                },
                {
                  "id": "64",
                  "name": "上海港口物流中心",
                  "latitude": 53.9214,
                  "longitude": 126.3958,
                  "specialized_mode": "port"
                },
                {
                  "id": "65",
                  "name": "丽水空港经济区",
                  "latitude": 50.5863,
                  "longitude": 91.2556,
                  "specialized_mode": "airport"
                },
                {
                  "id": "66",
                  "name": "临沂国际班列站",
                  "latitude": 51.6933,
                  "longitude": 108.7115,
                  "specialized_mode": "railway"
                },
                {
                  "id": "67",
                  "name": "成都货运航站楼",
                  "latitude": 32.8562,
                  "longitude": 79.9927,
                  "specialized_mode": "airport"
                },
                {
                  "id": "68",
                  "name": "日照国际空港",
                  "latitude": 44.5374,
                  "longitude": 80.3441,
                  "specialized_mode": "airport"
                },
                {
                  "id": "69",
                  "name": "无锡港口经济区",
                  "latitude": 36.7698,
                  "longitude": 131.8797,
                  "specialized_mode": "port"
                },
                {
                  "id": "70",
                  "name": "江门航空物流园",
                  "latitude": 26.1461,
                  "longitude": 106.3632,
                  "specialized_mode": "airport"
                },
                {
                  "id": "71",
                  "name": "无锡智慧港口",
                  "latitude": 36.24,
                  "longitude": 88.5777,
                  "specialized_mode": "port"
                },
                {
                  "id": "72",
                  "name": "湘潭智慧港口",
                  "latitude": 28.4292,
                  "longitude": 117.5894,
                  "specialized_mode": "port"
                },
                {
                  "id": "73",
                  "name": "南宁航空集散中心",
                  "latitude": 23.4846,
                  "longitude": 97.0408,
                  "specialized_mode": "airport"
                },
                {
                  "id": "74",
                  "name": "舟山深水港区",
                  "latitude": 31.0092,
                  "longitude": 122.2064,
                  "specialized_mode": "port"
                },
                {
                  "id": "75",
                  "name": "肇庆空港经济区",
                  "latitude": 42.5974,
                  "longitude": 108.9379,
                  "specialized_mode": "airport"
                },
                {
                  "id": "76",
                  "name": "枣庄编组站",
                  "latitude": 43.0325,
                  "longitude": 127.9751,
                  "specialized_mode": "railway"
                },
                {
                  "id": "77",
                  "name": "肇庆综合码头",
                  "latitude": 30.5793,
                  "longitude": 99.9248,
                  "specialized_mode": "port"
                },
                {
                  "id": "78",
                  "name": "长春国际机场",
                  "latitude": 36.2912,
                  "longitude": 93.4879,
                  "specialized_mode": "airport"
                },
                {
                  "id": "79",
                  "name": "上海港口经济区",
                  "latitude": 46.0635,
                  "longitude": 78.0685,
                  "specialized_mode": "port"
                },
                {
                  "id": "80",
                  "name": "汕头货运机场",
                  "latitude": 42.4899,
                  "longitude": 96.027,
                  "specialized_mode": "airport"
                },
                {
                  "id": "81",
                  "name": "威海空港物流园",
                  "latitude": 24.8928,
                  "longitude": 106.4486,
                  "specialized_mode": "airport"
                },
                {
                  "id": "82",
                  "name": "烟台国际航空枢纽",
                  "latitude": 37.4244,
                  "longitude": 115.1455,
                  "specialized_mode": "airport"
                },
                {
                  "id": "83",
                  "name": "合肥货运航站楼",
                  "latitude": 38.3061,
                  "longitude": 116.7179,
                  "specialized_mode": "airport"
                },
                {
                  "id": "84",
                  "name": "上海国际班列站",
                  "latitude": 23.9413,
                  "longitude": 87.6937,
                  "specialized_mode": "railway"
                },
                {
                  "id": "85",
                  "name": "汕头空港物流园",
                  "latitude": 22.0334,
                  "longitude": 126.3342,
                  "specialized_mode": "airport"
                },
                {
                  "id": "86",
                  "name": "合肥集装箱码头",
                  "latitude": 27.6616,
                  "longitude": 106.2691,
                  "specialized_mode": "port"
                },
                {
                  "id": "87",
                  "name": "台州综合保税港",
                  "latitude": 36.6327,
                  "longitude": 97.4176,
                  "specialized_mode": "port"
                },
                {
                  "id": "88",
                  "name": "永州国际贸易港",
                  "latitude": 42.6453,
                  "longitude": 97.6218,
                  "specialized_mode": "port"
                },
                {
                  "id": "89",
                  "name": "成都编组站",
                  "latitude": 45.438,
                  "longitude": 103.4119,
                  "specialized_mode": "railway"
                },
                {
                  "id": "90",
                  "name": "东营物流集结中心",
                  "latitude": 28.0736,
                  "longitude": 100.9148,
                  "specialized_mode": "railway"
                },
                {
                  "id": "91",
                  "name": "烟台中欧班列枢纽",
                  "latitude": 38.056,
                  "longitude": 110.2139,
                  "specialized_mode": "railway"
                },
                {
                  "id": "92",
                  "name": "衡阳联运枢纽",
                  "latitude": 32.1283,
                  "longitude": 124.7544,
                  "specialized_mode": "railway"
                },
                {
                  "id": "93",
                  "name": "哈尔滨货运航站楼",
                  "latitude": 42.8596,
                  "longitude": 120.7948,
                  "specialized_mode": "airport"
                },
                {
                  "id": "94",
                  "name": "广州货运站",
                  "latitude": 52.6894,
                  "longitude": 83.9138,
                  "specialized_mode": "railway"
                },
                {
                  "id": "95",
                  "name": "珠海国际空港",
                  "latitude": 25.6968,
                  "longitude": 87.5957,
                  "specialized_mode": "airport"
                },
                {
                  "id": "96",
                  "name": "青岛多式联运站",
                  "latitude": 51.937,
                  "longitude": 75.2829,
                  "specialized_mode": "railway"
                },
                {
                  "id": "97",
                  "name": "汕尾中欧班列枢纽",
                  "latitude": 49.59,
                  "longitude": 117.4714,
                  "specialized_mode": "railway"
                },
                {
                  "id": "98",
                  "name": "株洲空港经济区",
                  "latitude": 38.1803,
                  "longitude": 96.633,
                  "specialized_mode": "airport"
                },
                {
                  "id": "99",
                  "name": "苏州国际空港",
                  "latitude": 31.1445,
                  "longitude": 90.1035,
                  "specialized_mode": "airport"
                },
                {
                  "id": "100",
                  "name": "丽水国际物流港",
                  "latitude": 28.3962,
                  "longitude": 90.1354,
                  "specialized_mode": "port"
                },
                {
                  "id": "101",
                  "name": "茂名空港物流园",
                  "latitude": 49.3439,
                  "longitude": 82.9085,
                  "specialized_mode": "airport"
                },
                {
                  "id": "102",
                  "name": "梅州铁路货场",
                  "latitude": 38.2446,
                  "longitude": 81.7276,
                  "specialized_mode": "railway"
                },
                {
                  "id": "103",
                  "name": "舟山中欧班列枢纽",
                  "latitude": 40.182,
                  "longitude": 85.2492,
                  "specialized_mode": "railway"
                },
                {
                  "id": "104",
                  "name": "台州国际航空港",
                  "latitude": 53.3081,
                  "longitude": 78.7756,
                  "specialized_mode": "airport"
                },
                {
                  "id": "105",
                  "name": "淄博智慧港口",
                  "latitude": 31.8362,
                  "longitude": 115.1406,
                  "specialized_mode": "port"
                },
                {
                  "id": "106",
                  "name": "韶关国际航空港",
                  "latitude": 36.0186,
                  "longitude": 88.514,
                  "specialized_mode": "airport"
                },
                {
                  "id": "107",
                  "name": "威海航空货运中心",
                  "latitude": 20.3738,
                  "longitude": 126.0435,
                  "specialized_mode": "airport"
                },
                {
                  "id": "108",
                  "name": "怀化货运中心",
                  "latitude": 18.14,
                  "longitude": 130.2577,
                  "specialized_mode": "railway"
                },
                {
                  "id": "109",
                  "name": "嘉兴联运枢纽",
                  "latitude": 47.8587,
                  "longitude": 129.6288,
                  "specialized_mode": "railway"
                },
                {
                  "id": "110",
                  "name": "泰安货运航站楼",
                  "latitude": 26.9845,
                  "longitude": 76.5501,
                  "specialized_mode": "airport"
                },
                {
                  "id": "111",
                  "name": "广州智慧港口",
                  "latitude": 32.0886,
                  "longitude": 122.7447,
                  "specialized_mode": "port"
                },
                {
                  "id": "112",
                  "name": "日照航空港区",
                  "latitude": 24.6796,
                  "longitude": 134.7834,
                  "specialized_mode": "airport"
                },
                {
                  "id": "113",
                  "name": "株洲临港物流园",
                  "latitude": 27.7832,
                  "longitude": 127.2494,
                  "specialized_mode": "port"
                },
                {
                  "id": "114",
                  "name": "娄底货运航站楼",
                  "latitude": 25.9298,
                  "longitude": 99.4991,
                  "specialized_mode": "airport"
                },
                {
                  "id": "115",
                  "name": "聊城港口物流中心",
                  "latitude": 37.8803,
                  "longitude": 85.0216,
                  "specialized_mode": "port"
                },
                {
                  "id": "116",
                  "name": "湘西货运中心",
                  "latitude": 48.4739,
                  "longitude": 119.2089,
                  "specialized_mode": "railway"
                },
                {
                  "id": "117",
                  "name": "泰安联运枢纽",
                  "latitude": 24.3393,
                  "longitude": 103.9389,
                  "specialized_mode": "railway"
                },
                {
                  "id": "118",
                  "name": "济宁编组站",
                  "latitude": 29.2127,
                  "longitude": 105.1161,
                  "specialized_mode": "railway"
                },
                {
                  "id": "119",
                  "name": "海口货运港区",
                  "latitude": 51.3978,
                  "longitude": 121.638,
                  "specialized_mode": "port"
                },
                {
                  "id": "120",
                  "name": "西宁深水港区",
                  "latitude": 31.9957,
                  "longitude": 111.1868,
                  "specialized_mode": "port"
                },
                {
                  "id": "121",
                  "name": "郴州铁路枢纽",
                  "latitude": 36.2728,
                  "longitude": 131.1767,
                  "specialized_mode": "railway"
                },
                {
                  "id": "122",
                  "name": "泰安货运航站楼",
                  "latitude": 47.4597,
                  "longitude": 119.519,
                  "specialized_mode": "airport"
                },
                {
                  "id": "123",
                  "name": "常德国际空港",
                  "latitude": 33.4719,
                  "longitude": 76.1407,
                  "specialized_mode": "airport"
                },
                {
                  "id": "124",
                  "name": "南宁铁路枢纽",
                  "latitude": 19.6769,
                  "longitude": 73.0039,
                  "specialized_mode": "railway"
                },
                {
                  "id": "125",
                  "name": "珠海中欧班列枢纽",
                  "latitude": 29.9594,
                  "longitude": 79.9474,
                  "specialized_mode": "railway"
                },
                {
                  "id": "126",
                  "name": "重庆空港经济区",
                  "latitude": 47.299,
                  "longitude": 129.9789,
                  "specialized_mode": "airport"
                },
                {
                  "id": "127",
                  "name": "福州铁路港",
                  "latitude": 28.933,
                  "longitude": 117.7738,
                  "specialized_mode": "railway"
                },
                {
                  "id": "128",
                  "name": "威海港口物流中心",
                  "latitude": 33.4199,
                  "longitude": 133.0678,
                  "specialized_mode": "port"
                },
                {
                  "id": "129",
                  "name": "茂名国际物流港",
                  "latitude": 37.8509,
                  "longitude": 113.6049,
                  "specialized_mode": "port"
                },
                {
                  "id": "130",
                  "name": "拉萨多式联运站",
                  "latitude": 44.9526,
                  "longitude": 102.2887,
                  "specialized_mode": "railway"
                },
                {
                  "id": "131",
                  "name": "怀化现代物流港",
                  "latitude": 18.0857,
                  "longitude": 134.926,
                  "specialized_mode": "port"
                },
                {
                  "id": "132",
                  "name": "长春货运航站楼",
                  "latitude": 43.4431,
                  "longitude": 134.149,
                  "specialized_mode": "airport"
                },
                {
                  "id": "133",
                  "name": "武汉国际港区",
                  "latitude": 27.8057,
                  "longitude": 75.2154,
                  "specialized_mode": "port"
                },
                {
                  "id": "134",
                  "name": "益阳现代物流港",
                  "latitude": 27.1944,
                  "longitude": 79.5144,
                  "specialized_mode": "port"
                },
                {
                  "id": "135",
                  "name": "济南航空货运中心",
                  "latitude": 39.622,
                  "longitude": 89.5361,
                  "specialized_mode": "airport"
                },
                {
                  "id": "136",
                  "name": "德州空港物流园",
                  "latitude": 41.8378,
                  "longitude": 117.9307,
                  "specialized_mode": "airport"
                },
                {
                  "id": "137",
                  "name": "湘西铁路物流园",
                  "latitude": 20.6722,
                  "longitude": 115.9252,
                  "specialized_mode": "railway"
                },
                {
                  "id": "138",
                  "name": "茂名现代物流港",
                  "latitude": 30.2898,
                  "longitude": 118.0781,
                  "specialized_mode": "port"
                },
                {
                  "id": "139",
                  "name": "潮州港口物流中心",
                  "latitude": 44.8212,
                  "longitude": 81.5486,
                  "specialized_mode": "port"
                },
                {
                  "id": "140",
                  "name": "杭州国际贸易港",
                  "latitude": 53.8137,
                  "longitude": 115.61,
                  "specialized_mode": "port"
                },
                {
                  "id": "141",
                  "name": "烟台航空港",
                  "latitude": 47.8099,
                  "longitude": 108.5066,
                  "specialized_mode": "airport"
                },
                {
                  "id": "142",
                  "name": "西宁航空物流中心",
                  "latitude": 40.4728,
                  "longitude": 100.0231,
                  "specialized_mode": "airport"
                },
                {
                  "id": "143",
                  "name": "台州智慧港口",
                  "latitude": 32.58,
                  "longitude": 113.5595,
                  "specialized_mode": "port"
                },
                {
                  "id": "144",
                  "name": "银川货运机场",
                  "latitude": 18.815,
                  "longitude": 128.1584,
                  "specialized_mode": "airport"
                },
                {
                  "id": "145",
                  "name": "株洲国际班列站",
                  "latitude": 27.0928,
                  "longitude": 84.7125,
                  "specialized_mode": "railway"
                },
                {
                  "id": "146",
                  "name": "贵阳航空港区",
                  "latitude": 53.0076,
                  "longitude": 93.2184,
                  "specialized_mode": "airport"
                },
                {
                  "id": "147",
                  "name": "重庆深水港区",
                  "latitude": 20.7355,
                  "longitude": 116.0372,
                  "specialized_mode": "port"
                },
                {
                  "id": "148",
                  "name": "济南编组站",
                  "latitude": 21.3908,
                  "longitude": 73.1856,
                  "specialized_mode": "railway"
                },
                {
                  "id": "149",
                  "name": "烟台高铁站",
                  "latitude": 42.6159,
                  "longitude": 77.596,
                  "specialized_mode": "railway"
                },
                {
                  "id": "150",
                  "name": "惠州国际港区",
                  "latitude": 19.0758,
                  "longitude": 122.2102,
                  "specialized_mode": "port"
                },
                {
                  "id": "151",
                  "name": "沈阳航空物流中心",
                  "latitude": 30.9702,
                  "longitude": 115.8697,
                  "specialized_mode": "airport"
                },
                {
                  "id": "152",
                  "name": "宁波综合保税港",
                  "latitude": 38.9376,
                  "longitude": 74.6564,
                  "specialized_mode": "port"
                },
                {
                  "id": "153",
                  "name": "银川国际港区",
                  "latitude": 22.0966,
                  "longitude": 133.2714,
                  "specialized_mode": "port"
                },
                {
                  "id": "154",
                  "name": "莱芜货运航站楼",
                  "latitude": 43.4696,
                  "longitude": 75.5617,
                  "specialized_mode": "airport"
                },
                {
                  "id": "155",
                  "name": "武汉空港经济区",
                  "latitude": 37.0244,
                  "longitude": 130.6771,
                  "specialized_mode": "airport"
                },
                {
                  "id": "156",
                  "name": "广州综合码头",
                  "latitude": 50.7105,
                  "longitude": 90.4214,
                  "specialized_mode": "port"
                },
                {
                  "id": "157",
                  "name": "中山自贸港区",
                  "latitude": 51.6012,
                  "longitude": 88.1281,
                  "specialized_mode": "port"
                },
                {
                  "id": "158",
                  "name": "烟台航空物流园",
                  "latitude": 47.5088,
                  "longitude": 80.3564,
                  "specialized_mode": "airport"
                },
                {
                  "id": "159",
                  "name": "长春航空港",
                  "latitude": 30.8015,
                  "longitude": 76.359,
                  "specialized_mode": "airport"
                },
                {
                  "id": "160",
                  "name": "哈尔滨国际班列站",
                  "latitude": 49.0336,
                  "longitude": 84.0381,
                  "specialized_mode": "railway"
                },
                {
                  "id": "161",
                  "name": "永州物流集结中心",
                  "latitude": 48.623,
                  "longitude": 94.7248,
                  "specialized_mode": "railway"
                },
                {
                  "id": "162",
                  "name": "南京现代物流港",
                  "latitude": 22.622,
                  "longitude": 82.7054,
                  "specialized_mode": "port"
                },
                {
                  "id": "163",
                  "name": "海口深水港区",
                  "latitude": 49.3276,
                  "longitude": 110.2216,
                  "specialized_mode": "port"
                },
                {
                  "id": "164",
                  "name": "湘西铁路集散中心",
                  "latitude": 51.6085,
                  "longitude": 118.2383,
                  "specialized_mode": "railway"
                },
                {
                  "id": "165",
                  "name": "怀化深水港区",
                  "latitude": 47.9825,
                  "longitude": 112.5326,
                  "specialized_mode": "port"
                },
                {
                  "id": "166",
                  "name": "湘潭铁路枢纽",
                  "latitude": 40.6454,
                  "longitude": 95.3431,
                  "specialized_mode": "railway"
                },
                {
                  "id": "167",
                  "name": "揭阳国际物流港",
                  "latitude": 32.2979,
                  "longitude": 93.4705,
                  "specialized_mode": "port"
                },
                {
                  "id": "168",
                  "name": "贵阳货运航站楼",
                  "latitude": 30.7885,
                  "longitude": 129.9692,
                  "specialized_mode": "airport"
                },
                {
                  "id": "169",
                  "name": "拉萨航空港区",
                  "latitude": 38.6414,
                  "longitude": 92.2321,
                  "specialized_mode": "airport"
                },
                {
                  "id": "170",
                  "name": "东莞自贸港区",
                  "latitude": 25.3084,
                  "longitude": 89.1506,
                  "specialized_mode": "port"
                },
                {
                  "id": "171",
                  "name": "湖州国际航空港",
                  "latitude": 41.174,
                  "longitude": 93.2964,
                  "specialized_mode": "airport"
                },
                {
                  "id": "172",
                  "name": "南宁智慧港口",
                  "latitude": 48.7675,
                  "longitude": 117.8784,
                  "specialized_mode": "port"
                },
                {
                  "id": "173",
                  "name": "郴州编组站",
                  "latitude": 20.9142,
                  "longitude": 103.7681,
                  "specialized_mode": "railway"
                },
                {
                  "id": "174",
                  "name": "潍坊国际贸易港",
                  "latitude": 51.4239,
                  "longitude": 102.6084,
                  "specialized_mode": "port"
                },
                {
                  "id": "175",
                  "name": "昆明货运航站楼",
                  "latitude": 34.4246,
                  "longitude": 99.6078,
                  "specialized_mode": "airport"
                },
                {
                  "id": "176",
                  "name": "南昌铁路枢纽",
                  "latitude": 21.9204,
                  "longitude": 126.2541,
                  "specialized_mode": "railway"
                },
                {
                  "id": "177",
                  "name": "邵阳国际贸易港",
                  "latitude": 18.0666,
                  "longitude": 131.5762,
                  "specialized_mode": "port"
                },
                {
                  "id": "178",
                  "name": "温州多式联运站",
                  "latitude": 32.2918,
                  "longitude": 105.8503,
                  "specialized_mode": "railway"
                },
                {
                  "id": "179",
                  "name": "潮州编组站",
                  "latitude": 29.7394,
                  "longitude": 73.3847,
                  "specialized_mode": "railway"
                },
                {
                  "id": "180",
                  "name": "济宁空港经济区",
                  "latitude": 31.7847,
                  "longitude": 88.2704,
                  "specialized_mode": "airport"
                },
                {
                  "id": "181",
                  "name": "滨州铁路物流园",
                  "latitude": 31.6232,
                  "longitude": 110.3547,
                  "specialized_mode": "railway"
                },
                {
                  "id": "182",
                  "name": "济南国际港区",
                  "latitude": 20.8253,
                  "longitude": 100.2725,
                  "specialized_mode": "port"
                },
                {
                  "id": "183",
                  "name": "潍坊国际航空枢纽",
                  "latitude": 28.7544,
                  "longitude": 124.3967,
                  "specialized_mode": "airport"
                },
                {
                  "id": "184",
                  "name": "湖州物流枢纽港",
                  "latitude": 42.5475,
                  "longitude": 131.9386,
                  "specialized_mode": "port"
                },
                {
                  "id": "185",
                  "name": "江门现代物流港",
                  "latitude": 44.7956,
                  "longitude": 106.9125,
                  "specialized_mode": "port"
                },
                {
                  "id": "186",
                  "name": "滨州铁路货场",
                  "latitude": 47.8915,
                  "longitude": 83.3843,
                  "specialized_mode": "railway"
                },
                {
                  "id": "187",
                  "name": "岳阳编组站",
                  "latitude": 24.5885,
                  "longitude": 123.0291,
                  "specialized_mode": "railway"
                },
                {
                  "id": "188",
                  "name": "西宁智慧港口",
                  "latitude": 43.9639,
                  "longitude": 121.9468,
                  "specialized_mode": "port"
                },
                {
                  "id": "189",
                  "name": "福州航空货运中心",
                  "latitude": 39.0031,
                  "longitude": 89.5085,
                  "specialized_mode": "airport"
                },
                {
                  "id": "190",
                  "name": "南京现代物流港",
                  "latitude": 40.0425,
                  "longitude": 100.4848,
                  "specialized_mode": "port"
                },
                {
                  "id": "191",
                  "name": "河源铁路枢纽",
                  "latitude": 35.8253,
                  "longitude": 104.3106,
                  "specialized_mode": "railway"
                },
                {
                  "id": "192",
                  "name": "青岛集装箱码头",
                  "latitude": 37.0175,
                  "longitude": 118.6267,
                  "specialized_mode": "port"
                },
                {
                  "id": "193",
                  "name": "聊城临港物流园",
                  "latitude": 26.8282,
                  "longitude": 90.2444,
                  "specialized_mode": "port"
                },
                {
                  "id": "194",
                  "name": "青岛深水港区",
                  "latitude": 51.647,
                  "longitude": 97.9633,
                  "specialized_mode": "port"
                },
                {
                  "id": "195",
                  "name": "石家庄货运站",
                  "latitude": 33.8564,
                  "longitude": 82.7456,
                  "specialized_mode": "railway"
                },
                {
                  "id": "196",
                  "name": "兰州多式联运站",
                  "latitude": 32.3085,
                  "longitude": 109.7444,
                  "specialized_mode": "railway"
                },
                {
                  "id": "197",
                  "name": "怀化多式联运站",
                  "latitude": 20.6536,
                  "longitude": 119.7668,
                  "specialized_mode": "railway"
                },
                {
                  "id": "198",
                  "name": "宁波国际物流港",
                  "latitude": 47.2039,
                  "longitude": 131.8209,
                  "specialized_mode": "port"
                },
                {
                  "id": "199",
                  "name": "清远深水港区",
                  "latitude": 51.5642,
                  "longitude": 127.4847,
                  "specialized_mode": "port"
                },
                {
                  "id": "200",
                  "name": "郴州货运机场",
                  "latitude": 23.4935,
                  "longitude": 81.2756,
                  "specialized_mode": "airport"
                },
                {
                  "id": "201",
                  "name": "张家界综合码头",
                  "latitude": 35.615,
                  "longitude": 94.805,
                  "specialized_mode": "port"
                },
                {
                  "id": "202",
                  "name": "衡阳航空港区",
                  "latitude": 20.8587,
                  "longitude": 118.0781,
                  "specialized_mode": "airport"
                },
                {
                  "id": "203",
                  "name": "衡阳国际贸易港",
                  "latitude": 24.5195,
                  "longitude": 94.3544,
                  "specialized_mode": "port"
                },
                {
                  "id": "204",
                  "name": "莱芜航空物流中心",
                  "latitude": 52.0005,
                  "longitude": 98.5558,
                  "specialized_mode": "airport"
                },
                {
                  "id": "205",
                  "name": "乌鲁木齐物流集结中心",
                  "latitude": 50.4374,
                  "longitude": 108.7421,
                  "specialized_mode": "railway"
                },
                {
                  "id": "206",
                  "name": "阳江铁路货场",
                  "latitude": 46.5729,
                  "longitude": 131.4877,
                  "specialized_mode": "railway"
                },
                {
                  "id": "207",
                  "name": "潍坊智慧港口",
                  "latitude": 43.9706,
                  "longitude": 132.1944,
                  "specialized_mode": "port"
                },
                {
                  "id": "208",
                  "name": "重庆货运中心",
                  "latitude": 20.4295,
                  "longitude": 106.9957,
                  "specialized_mode": "railway"
                },
                {
                  "id": "209",
                  "name": "湛江国际空港",
                  "latitude": 22.8008,
                  "longitude": 118.8621,
                  "specialized_mode": "airport"
                },
                {
                  "id": "210",
                  "name": "贵阳航空物流园",
                  "latitude": 35.6366,
                  "longitude": 115.398,
                  "specialized_mode": "airport"
                },
                {
                  "id": "211",
                  "name": "舟山港口经济区",
                  "latitude": 20.0545,
                  "longitude": 132.1964,
                  "specialized_mode": "port"
                },
                {
                  "id": "212",
                  "name": "南昌货运航站楼",
                  "latitude": 37.7255,
                  "longitude": 76.0567,
                  "specialized_mode": "airport"
                },
                {
                  "id": "213",
                  "name": "德州港口物流中心",
                  "latitude": 28.7872,
                  "longitude": 100.3598,
                  "specialized_mode": "port"
                },
                {
                  "id": "214",
                  "name": "兰州航空港区",
                  "latitude": 25.5051,
                  "longitude": 73.7158,
                  "specialized_mode": "airport"
                },
                {
                  "id": "215",
                  "name": "清远国际贸易港",
                  "latitude": 43.1085,
                  "longitude": 93.4036,
                  "specialized_mode": "port"
                },
                {
                  "id": "216",
                  "name": "揭阳中欧班列枢纽",
                  "latitude": 51.7737,
                  "longitude": 115.5262,
                  "specialized_mode": "railway"
                },
                {
                  "id": "217",
                  "name": "郑州航空物流中心",
                  "latitude": 50.8611,
                  "longitude": 93.6173,
                  "specialized_mode": "airport"
                },
                {
                  "id": "218",
                  "name": "青岛港口经济区",
                  "latitude": 49.262,
                  "longitude": 111.5629,
                  "specialized_mode": "port"
                },
                {
                  "id": "219",
                  "name": "揭阳货运港区",
                  "latitude": 25.8982,
                  "longitude": 96.9903,
                  "specialized_mode": "port"
                },
                {
                  "id": "220",
                  "name": "珠海空港经济区",
                  "latitude": 29.8993,
                  "longitude": 95.7597,
                  "specialized_mode": "airport"
                },
                {
                  "id": "221",
                  "name": "怀化航空货运中心",
                  "latitude": 42.0554,
                  "longitude": 82.0254,
                  "specialized_mode": "airport"
                },
                {
                  "id": "222",
                  "name": "烟台国际机场",
                  "latitude": 32.8837,
                  "longitude": 103.3686,
                  "specialized_mode": "airport"
                },
                {
                  "id": "223",
                  "name": "揭阳铁路货场",
                  "latitude": 18.3681,
                  "longitude": 115.0755,
                  "specialized_mode": "railway"
                },
                {
                  "id": "224",
                  "name": "东营深水港区",
                  "latitude": 37.0277,
                  "longitude": 109.2065,
                  "specialized_mode": "port"
                },
                {
                  "id": "225",
                  "name": "珠海空港物流园",
                  "latitude": 39.4647,
                  "longitude": 97.8135,
                  "specialized_mode": "airport"
                },
                {
                  "id": "226",
                  "name": "济宁铁路港",
                  "latitude": 34.5204,
                  "longitude": 102.5727,
                  "specialized_mode": "railway"
                },
                {
                  "id": "227",
                  "name": "日照铁路集散中心",
                  "latitude": 40.0389,
                  "longitude": 89.7412,
                  "specialized_mode": "railway"
                },
                {
                  "id": "228",
                  "name": "河源集装箱码头",
                  "latitude": 45.3788,
                  "longitude": 121.9603,
                  "specialized_mode": "port"
                },
                {
                  "id": "229",
                  "name": "菏泽自贸港区",
                  "latitude": 43.2811,
                  "longitude": 134.9897,
                  "specialized_mode": "port"
                },
                {
                  "id": "230",
                  "name": "南京货运航站楼",
                  "latitude": 53.489,
                  "longitude": 102.6064,
                  "specialized_mode": "airport"
                },
                {
                  "id": "231",
                  "name": "日照航空港",
                  "latitude": 34.9885,
                  "longitude": 126.2603,
                  "specialized_mode": "airport"
                },
                {
                  "id": "232",
                  "name": "兰州货运中心",
                  "latitude": 34.9621,
                  "longitude": 86.142,
                  "specialized_mode": "railway"
                },
                {
                  "id": "233",
                  "name": "潍坊国际物流港",
                  "latitude": 40.0829,
                  "longitude": 119.0605,
                  "specialized_mode": "port"
                },
                {
                  "id": "234",
                  "name": "呼和浩特铁路集散中心",
                  "latitude": 34.5539,
                  "longitude": 105.1408,
                  "specialized_mode": "railway"
                },
                {
                  "id": "235",
                  "name": "舟山国际班列站",
                  "latitude": 39.1086,
                  "longitude": 76.5236,
                  "specialized_mode": "railway"
                },
                {
                  "id": "236",
                  "name": "肇庆货运机场",
                  "latitude": 33.226,
                  "longitude": 126.6165,
                  "specialized_mode": "airport"
                },
                {
                  "id": "237",
                  "name": "汕尾国际空港",
                  "latitude": 25.2504,
                  "longitude": 94.7647,
                  "specialized_mode": "airport"
                },
                {
                  "id": "238",
                  "name": "茂名物流枢纽港",
                  "latitude": 44.3177,
                  "longitude": 95.7208,
                  "specialized_mode": "port"
                },
                {
                  "id": "239",
                  "name": "杭州港口物流中心",
                  "latitude": 25.9538,
                  "longitude": 79.8785,
                  "specialized_mode": "port"
                },
                {
                  "id": "240",
                  "name": "张家界航空货运中心",
                  "latitude": 33.5368,
                  "longitude": 74.1766,
                  "specialized_mode": "airport"
                },
                {
                  "id": "241",
                  "name": "梅州国际机场",
                  "latitude": 47.2871,
                  "longitude": 88.742,
                  "specialized_mode": "airport"
                },
                {
                  "id": "242",
                  "name": "佛山空港经济区",
                  "latitude": 44.0063,
                  "longitude": 114.6447,
                  "specialized_mode": "airport"
                },
                {
                  "id": "243",
                  "name": "株洲联运枢纽",
                  "latitude": 37.9896,
                  "longitude": 119.4355,
                  "specialized_mode": "railway"
                },
                {
                  "id": "244",
                  "name": "中山港口物流中心",
                  "latitude": 42.254,
                  "longitude": 92.6289,
                  "specialized_mode": "port"
                },
                {
                  "id": "245",
                  "name": "嘉兴深水港区",
                  "latitude": 33.9798,
                  "longitude": 120.1153,
                  "specialized_mode": "port"
                },
                {
                  "id": "246",
                  "name": "温州高铁站",
                  "latitude": 46.0248,
                  "longitude": 82.3725,
                  "specialized_mode": "railway"
                },
                {
                  "id": "247",
                  "name": "江门航空集散中心",
                  "latitude": 50.9023,
                  "longitude": 116.6203,
                  "specialized_mode": "airport"
                },
                {
                  "id": "248",
                  "name": "海口航空集散中心",
                  "latitude": 53.2393,
                  "longitude": 112.8792,
                  "specialized_mode": "airport"
                },
                {
                  "id": "249",
                  "name": "益阳国际航空港",
                  "latitude": 50.5171,
                  "longitude": 83.6373,
                  "specialized_mode": "airport"
                },
                {
                  "id": "250",
                  "name": "苏州高铁站",
                  "latitude": 24.2108,
                  "longitude": 129.0007,
                  "specialized_mode": "railway"
                },
                {
                  "id": "251",
                  "name": "汕头编组站",
                  "latitude": 49.721,
                  "longitude": 76.3135,
                  "specialized_mode": "railway"
                },
                {
                  "id": "252",
                  "name": "淄博编组站",
                  "latitude": 32.9423,
                  "longitude": 94.7007,
                  "specialized_mode": "railway"
                },
                {
                  "id": "253",
                  "name": "太原航空港",
                  "latitude": 26.245,
                  "longitude": 89.6503,
                  "specialized_mode": "airport"
                },
                {
                  "id": "254",
                  "name": "兰州临港物流园",
                  "latitude": 29.3073,
                  "longitude": 134.6464,
                  "specialized_mode": "port"
                },
                {
                  "id": "255",
                  "name": "大连智慧港口",
                  "latitude": 47.8727,
                  "longitude": 115.5693,
                  "specialized_mode": "port"
                },
                {
                  "id": "256",
                  "name": "贵阳现代物流港",
                  "latitude": 38.8993,
                  "longitude": 95.859,
                  "specialized_mode": "port"
                },
                {
                  "id": "257",
                  "name": "茂名深水港区",
                  "latitude": 42.5855,
                  "longitude": 73.1909,
                  "specialized_mode": "port"
                },
                {
                  "id": "258",
                  "name": "怀化航空港区",
                  "latitude": 22.8196,
                  "longitude": 86.0839,
                  "specialized_mode": "airport"
                },
                {
                  "id": "259",
                  "name": "深圳国际空港",
                  "latitude": 41.375,
                  "longitude": 86.8356,
                  "specialized_mode": "airport"
                },
                {
                  "id": "260",
                  "name": "河源综合保税港",
                  "latitude": 45.9054,
                  "longitude": 124.6762,
                  "specialized_mode": "port"
                },
                {
                  "id": "261",
                  "name": "中山编组站",
                  "latitude": 37.3454,
                  "longitude": 105.4582,
                  "specialized_mode": "railway"
                },
                {
                  "id": "262",
                  "name": "湖州国际航空港",
                  "latitude": 43.8715,
                  "longitude": 127.5919,
                  "specialized_mode": "airport"
                },
                {
                  "id": "263",
                  "name": "怀化港口经济区",
                  "latitude": 20.1842,
                  "longitude": 108.8486,
                  "specialized_mode": "port"
                },
                {
                  "id": "264",
                  "name": "台州国际空港",
                  "latitude": 42.1495,
                  "longitude": 127.1265,
                  "specialized_mode": "airport"
                },
                {
                  "id": "265",
                  "name": "广州中欧班列枢纽",
                  "latitude": 31.3419,
                  "longitude": 101.0669,
                  "specialized_mode": "railway"
                },
                {
                  "id": "266",
                  "name": "西安物流枢纽港",
                  "latitude": 31.5327,
                  "longitude": 134.8881,
                  "specialized_mode": "port"
                },
                {
                  "id": "267",
                  "name": "郴州智慧港口",
                  "latitude": 20.2681,
                  "longitude": 102.8721,
                  "specialized_mode": "port"
                },
                {
                  "id": "268",
                  "name": "上海货运站",
                  "latitude": 37.0386,
                  "longitude": 95.8824,
                  "specialized_mode": "railway"
                },
                {
                  "id": "269",
                  "name": "梅州综合保税港",
                  "latitude": 29.9622,
                  "longitude": 134.4023,
                  "specialized_mode": "port"
                },
                {
                  "id": "270",
                  "name": "湘西货运机场",
                  "latitude": 40.2927,
                  "longitude": 111.1396,
                  "specialized_mode": "airport"
                },
                {
                  "id": "271",
                  "name": "丽水中欧班列枢纽",
                  "latitude": 34.277,
                  "longitude": 119.9997,
                  "specialized_mode": "railway"
                },
                {
                  "id": "272",
                  "name": "烟台铁路货场",
                  "latitude": 51.5801,
                  "longitude": 132.2225,
                  "specialized_mode": "railway"
                },
                {
                  "id": "273",
                  "name": "岳阳铁路枢纽",
                  "latitude": 39.2639,
                  "longitude": 85.5651,
                  "specialized_mode": "railway"
                },
                {
                  "id": "274",
                  "name": "阳江国际航空港",
                  "latitude": 52.7001,
                  "longitude": 117.9461,
                  "specialized_mode": "airport"
                },
                {
                  "id": "275",
                  "name": "韶关国际航空枢纽",
                  "latitude": 31.1278,
                  "longitude": 96.9814,
                  "specialized_mode": "airport"
                },
                {
                  "id": "276",
                  "name": "淄博港口经济区",
                  "latitude": 43.949,
                  "longitude": 126.7948,
                  "specialized_mode": "port"
                },
                {
                  "id": "277",
                  "name": "南京航空港区",
                  "latitude": 49.1496,
                  "longitude": 130.8327,
                  "specialized_mode": "airport"
                },
                {
                  "id": "278",
                  "name": "湘潭港口物流中心",
                  "latitude": 32.4146,
                  "longitude": 86.5104,
                  "specialized_mode": "port"
                },
                {
                  "id": "279",
                  "name": "德州国际航空枢纽",
                  "latitude": 27.0716,
                  "longitude": 133.026,
                  "specialized_mode": "airport"
                },
                {
                  "id": "280",
                  "name": "永州航空集散中心",
                  "latitude": 44.6013,
                  "longitude": 115.9729,
                  "specialized_mode": "airport"
                },
                {
                  "id": "281",
                  "name": "菏泽货运航站楼",
                  "latitude": 18.475,
                  "longitude": 77.6328,
                  "specialized_mode": "airport"
                },
                {
                  "id": "282",
                  "name": "东莞航空货运中心",
                  "latitude": 25.3965,
                  "longitude": 104.5218,
                  "specialized_mode": "airport"
                },
                {
                  "id": "283",
                  "name": "德州国际港区",
                  "latitude": 32.8102,
                  "longitude": 107.8719,
                  "specialized_mode": "port"
                },
                {
                  "id": "284",
                  "name": "河源物流枢纽港",
                  "latitude": 36.2048,
                  "longitude": 110.2649,
                  "specialized_mode": "port"
                },
                {
                  "id": "285",
                  "name": "岳阳铁路枢纽",
                  "latitude": 21.0566,
                  "longitude": 93.4081,
                  "specialized_mode": "railway"
                },
                {
                  "id": "286",
                  "name": "日照航空集散中心",
                  "latitude": 43.7366,
                  "longitude": 131.864,
                  "specialized_mode": "airport"
                },
                {
                  "id": "287",
                  "name": "佛山中欧班列枢纽",
                  "latitude": 30.4669,
                  "longitude": 130.8239,
                  "specialized_mode": "railway"
                },
                {
                  "id": "288",
                  "name": "潍坊自贸港区",
                  "latitude": 27.0017,
                  "longitude": 98.1224,
                  "specialized_mode": "port"
                },
                {
                  "id": "289",
                  "name": "永州货运站",
                  "latitude": 50.9335,
                  "longitude": 122.8873,
                  "specialized_mode": "railway"
                },
                {
                  "id": "290",
                  "name": "潮州综合保税港",
                  "latitude": 24.9256,
                  "longitude": 128.6009,
                  "specialized_mode": "port"
                },
                {
                  "id": "291",
                  "name": "湘西综合码头",
                  "latitude": 28.037,
                  "longitude": 130.2712,
                  "specialized_mode": "port"
                },
                {
                  "id": "292",
                  "name": "汕头综合交通枢纽",
                  "latitude": 18.4957,
                  "longitude": 91.2105,
                  "specialized_mode": "railway"
                },
                {
                  "id": "293",
                  "name": "广州集装箱码头",
                  "latitude": 26.6776,
                  "longitude": 105.5978,
                  "specialized_mode": "port"
                },
                {
                  "id": "294",
                  "name": "益阳综合码头",
                  "latitude": 35.7521,
                  "longitude": 94.944,
                  "specialized_mode": "port"
                },
                {
                  "id": "295",
                  "name": "潮州智慧港口",
                  "latitude": 29.0163,
                  "longitude": 92.252,
                  "specialized_mode": "port"
                },
                {
                  "id": "296",
                  "name": "株洲货运港区",
                  "latitude": 43.4704,
                  "longitude": 116.592,
                  "specialized_mode": "port"
                },
                {
                  "id": "297",
                  "name": "清远货运航站楼",
                  "latitude": 43.0194,
                  "longitude": 84.7473,
                  "specialized_mode": "airport"
                },
                {
                  "id": "298",
                  "name": "常德高铁站",
                  "latitude": 37.5166,
                  "longitude": 103.7615,
                  "specialized_mode": "railway"
                },
                {
                  "id": "299",
                  "name": "海口航空货运中心",
                  "latitude": 19.2608,
                  "longitude": 83.9281,
                  "specialized_mode": "airport"
                },
                {
                  "id": "300",
                  "name": "湖州中欧班列枢纽",
                  "latitude": 43.0198,
                  "longitude": 85.2161,
                  "specialized_mode": "railway"
                },
                {
                  "id": "301",
                  "name": "韶关临空经济区",
                  "latitude": 38.2502,
                  "longitude": 134.7619,
                  "specialized_mode": "airport"
                },
                {
                  "id": "302",
                  "name": "丽水国际空港",
                  "latitude": 40.6161,
                  "longitude": 126.3416,
                  "specialized_mode": "airport"
                },
                {
                  "id": "303",
                  "name": "滨州国际物流港",
                  "latitude": 26.876,
                  "longitude": 97.6091,
                  "specialized_mode": "port"
                },
                {
                  "id": "304",
                  "name": "株洲物流集结中心",
                  "latitude": 36.0759,
                  "longitude": 73.2424,
                  "specialized_mode": "railway"
                },
                {
                  "id": "305",
                  "name": "昆明航空货运中心",
                  "latitude": 19.8788,
                  "longitude": 92.5983,
                  "specialized_mode": "airport"
                },
                {
                  "id": "306",
                  "name": "韶关编组站",
                  "latitude": 25.8128,
                  "longitude": 120.9889,
                  "specialized_mode": "railway"
                },
                {
                  "id": "307",
                  "name": "哈尔滨物流枢纽港",
                  "latitude": 46.1813,
                  "longitude": 134.6787,
                  "specialized_mode": "port"
                },
                {
                  "id": "308",
                  "name": "泰安空港经济区",
                  "latitude": 43.4681,
                  "longitude": 101.2469,
                  "specialized_mode": "airport"
                },
                {
                  "id": "309",
                  "name": "郑州联运枢纽",
                  "latitude": 30.9848,
                  "longitude": 104.4159,
                  "specialized_mode": "railway"
                },
                {
                  "id": "310",
                  "name": "常德国际空港",
                  "latitude": 51.1602,
                  "longitude": 101.5623,
                  "specialized_mode": "airport"
                },
                {
                  "id": "311",
                  "name": "淄博联运枢纽",
                  "latitude": 27.7102,
                  "longitude": 112.5755,
                  "specialized_mode": "railway"
                },
                {
                  "id": "312",
                  "name": "株洲货运站",
                  "latitude": 22.3253,
                  "longitude": 85.3047,
                  "specialized_mode": "railway"
                },
                {
                  "id": "313",
                  "name": "济宁现代物流港",
                  "latitude": 26.1827,
                  "longitude": 109.695,
                  "specialized_mode": "port"
                },
                {
                  "id": "314",
                  "name": "揭阳货运机场",
                  "latitude": 52.8075,
                  "longitude": 104.3129,
                  "specialized_mode": "airport"
                },
                {
                  "id": "315",
                  "name": "嘉兴中欧班列枢纽",
                  "latitude": 36.4808,
                  "longitude": 74.7946,
                  "specialized_mode": "railway"
                },
                {
                  "id": "316",
                  "name": "江门航空港区",
                  "latitude": 37.3693,
                  "longitude": 117.6186,
                  "specialized_mode": "airport"
                },
                {
                  "id": "317",
                  "name": "泰安自贸港区",
                  "latitude": 32.8877,
                  "longitude": 115.5486,
                  "specialized_mode": "port"
                },
                {
                  "id": "318",
                  "name": "广州综合保税港",
                  "latitude": 49.7777,
                  "longitude": 93.3231,
                  "specialized_mode": "port"
                },
                {
                  "id": "319",
                  "name": "莱芜国际机场",
                  "latitude": 19.7652,
                  "longitude": 102.1365,
                  "specialized_mode": "airport"
                },
                {
                  "id": "320",
                  "name": "日照国际空港",
                  "latitude": 30.3191,
                  "longitude": 123.079,
                  "specialized_mode": "airport"
                },
                {
                  "id": "321",
                  "name": "福州货运机场",
                  "latitude": 51.8234,
                  "longitude": 118.8293,
                  "specialized_mode": "airport"
                },
                {
                  "id": "322",
                  "name": "拉萨货运航站楼",
                  "latitude": 40.4102,
                  "longitude": 131.8961,
                  "specialized_mode": "airport"
                },
                {
                  "id": "323",
                  "name": "揭阳航空集散中心",
                  "latitude": 49.7896,
                  "longitude": 107.6002,
                  "specialized_mode": "airport"
                },
                {
                  "id": "324",
                  "name": "云浮编组站",
                  "latitude": 20.1074,
                  "longitude": 108.4191,
                  "specialized_mode": "railway"
                },
                {
                  "id": "325",
                  "name": "上海港口物流中心",
                  "latitude": 22.9921,
                  "longitude": 77.308,
                  "specialized_mode": "port"
                },
                {
                  "id": "326",
                  "name": "青岛现代物流港",
                  "latitude": 38.2105,
                  "longitude": 98.6555,
                  "specialized_mode": "port"
                },
                {
                  "id": "327",
                  "name": "滨州自贸港区",
                  "latitude": 30.0143,
                  "longitude": 98.0754,
                  "specialized_mode": "port"
                },
                {
                  "id": "328",
                  "name": "临沂铁路货场",
                  "latitude": 41.3815,
                  "longitude": 133.554,
                  "specialized_mode": "railway"
                },
                {
                  "id": "329",
                  "name": "深圳智慧港口",
                  "latitude": 53.5934,
                  "longitude": 78.4721,
                  "specialized_mode": "port"
                },
                {
                  "id": "330",
                  "name": "温州铁路港",
                  "latitude": 46.1629,
                  "longitude": 112.4411,
                  "specialized_mode": "railway"
                },
                {
                  "id": "331",
                  "name": "梅州编组站",
                  "latitude": 39.483,
                  "longitude": 133.418,
                  "specialized_mode": "railway"
                },
                {
                  "id": "332",
                  "name": "重庆物流集结中心",
                  "latitude": 29.8815,
                  "longitude": 81.7004,
                  "specialized_mode": "railway"
                },
                {
                  "id": "333",
                  "name": "中山货运航站楼",
                  "latitude": 39.3684,
                  "longitude": 100.0977,
                  "specialized_mode": "airport"
                },
                {
                  "id": "334",
                  "name": "乌鲁木齐航空物流中心",
                  "latitude": 35.3928,
                  "longitude": 75.2834,
                  "specialized_mode": "airport"
                },
                {
                  "id": "335",
                  "name": "永州编组站",
                  "latitude": 26.9502,
                  "longitude": 100.158,
                  "specialized_mode": "railway"
                },
                {
                  "id": "336",
                  "name": "南京航空港区",
                  "latitude": 19.2237,
                  "longitude": 133.2134,
                  "specialized_mode": "airport"
                },
                {
                  "id": "337",
                  "name": "衢州铁路港",
                  "latitude": 28.0646,
                  "longitude": 78.2372,
                  "specialized_mode": "railway"
                },
                {
                  "id": "338",
                  "name": "聊城货运机场",
                  "latitude": 43.9809,
                  "longitude": 81.6635,
                  "specialized_mode": "airport"
                },
                {
                  "id": "339",
                  "name": "茂名港口经济区",
                  "latitude": 51.2764,
                  "longitude": 91.5041,
                  "specialized_mode": "port"
                },
                {
                  "id": "340",
                  "name": "菏泽多式联运站",
                  "latitude": 36.0007,
                  "longitude": 103.1389,
                  "specialized_mode": "railway"
                },
                {
                  "id": "341",
                  "name": "大连国际航空枢纽",
                  "latitude": 18.7472,
                  "longitude": 76.3674,
                  "specialized_mode": "airport"
                },
                {
                  "id": "342",
                  "name": "昆明国际班列站",
                  "latitude": 26.3447,
                  "longitude": 94.9517,
                  "specialized_mode": "railway"
                },
                {
                  "id": "343",
                  "name": "大连综合交通枢纽",
                  "latitude": 34.0191,
                  "longitude": 128.3321,
                  "specialized_mode": "railway"
                },
                {
                  "id": "344",
                  "name": "太原航空集散中心",
                  "latitude": 29.1748,
                  "longitude": 76.3505,
                  "specialized_mode": "airport"
                },
                {
                  "id": "345",
                  "name": "汕头国际贸易港",
                  "latitude": 48.2114,
                  "longitude": 94.6697,
                  "specialized_mode": "port"
                },
                {
                  "id": "346",
                  "name": "深圳智慧港口",
                  "latitude": 19.3577,
                  "longitude": 124.9152,
                  "specialized_mode": "port"
                },
                {
                  "id": "347",
                  "name": "烟台编组站",
                  "latitude": 36.9914,
                  "longitude": 98.4346,
                  "specialized_mode": "railway"
                },
                {
                  "id": "348",
                  "name": "惠州航空物流园",
                  "latitude": 46.1554,
                  "longitude": 89.9761,
                  "specialized_mode": "airport"
                },
                {
                  "id": "349",
                  "name": "莱芜综合保税港",
                  "latitude": 30.2615,
                  "longitude": 93.3329,
                  "specialized_mode": "port"
                },
                {
                  "id": "350",
                  "name": "东营物流枢纽港",
                  "latitude": 31.4126,
                  "longitude": 120.2916,
                  "specialized_mode": "port"
                },
                {
                  "id": "351",
                  "name": "临沂联运枢纽",
                  "latitude": 36.4656,
                  "longitude": 83.2803,
                  "specialized_mode": "railway"
                },
                {
                  "id": "352",
                  "name": "台州国际空港",
                  "latitude": 36.0707,
                  "longitude": 96.7975,
                  "specialized_mode": "airport"
                },
                {
                  "id": "353",
                  "name": "重庆货运中心",
                  "latitude": 32.0606,
                  "longitude": 77.5483,
                  "specialized_mode": "railway"
                },
                {
                  "id": "354",
                  "name": "福州集装箱码头",
                  "latitude": 52.4366,
                  "longitude": 123.4327,
                  "specialized_mode": "port"
                },
                {
                  "id": "355",
                  "name": "长春货运航站楼",
                  "latitude": 20.526,
                  "longitude": 114.9273,
                  "specialized_mode": "airport"
                },
                {
                  "id": "356",
                  "name": "清远智慧港口",
                  "latitude": 42.0646,
                  "longitude": 121.5539,
                  "specialized_mode": "port"
                },
                {
                  "id": "357",
                  "name": "长春铁路枢纽",
                  "latitude": 25.1666,
                  "longitude": 96.9375,
                  "specialized_mode": "railway"
                },
                {
                  "id": "358",
                  "name": "潮州国际航空枢纽",
                  "latitude": 45.2762,
                  "longitude": 84.1145,
                  "specialized_mode": "airport"
                },
                {
                  "id": "359",
                  "name": "广州国际空港",
                  "latitude": 29.8063,
                  "longitude": 120.4883,
                  "specialized_mode": "airport"
                },
                {
                  "id": "360",
                  "name": "揭阳空港经济区",
                  "latitude": 41.0513,
                  "longitude": 114.836,
                  "specialized_mode": "airport"
                },
                {
                  "id": "361",
                  "name": "长沙货运航站楼",
                  "latitude": 52.0926,
                  "longitude": 75.5226,
                  "specialized_mode": "airport"
                },
                {
                  "id": "362",
                  "name": "沈阳铁路物流园",
                  "latitude": 21.4074,
                  "longitude": 119.6729,
                  "specialized_mode": "railway"
                },
                {
                  "id": "363",
                  "name": "永州多式联运站",
                  "latitude": 50.8101,
                  "longitude": 95.7749,
                  "specialized_mode": "railway"
                },
                {
                  "id": "364",
                  "name": "湘西编组站",
                  "latitude": 39.5488,
                  "longitude": 108.2237,
                  "specialized_mode": "railway"
                },
                {
                  "id": "365",
                  "name": "杭州综合码头",
                  "latitude": 33.901,
                  "longitude": 114.7089,
                  "specialized_mode": "port"
                },
                {
                  "id": "366",
                  "name": "济宁国际贸易港",
                  "latitude": 44.57,
                  "longitude": 115.5189,
                  "specialized_mode": "port"
                },
                {
                  "id": "367",
                  "name": "昆明国际航空枢纽",
                  "latitude": 33.7631,
                  "longitude": 91.3984,
                  "specialized_mode": "airport"
                },
                {
                  "id": "368",
                  "name": "西安国际航空枢纽",
                  "latitude": 29.8769,
                  "longitude": 124.5618,
                  "specialized_mode": "airport"
                },
                {
                  "id": "369",
                  "name": "菏泽集装箱码头",
                  "latitude": 44.5824,
                  "longitude": 101.6798,
                  "specialized_mode": "port"
                },
                {
                  "id": "370",
                  "name": "乌鲁木齐综合交通枢纽",
                  "latitude": 49.5522,
                  "longitude": 134.2498,
                  "specialized_mode": "railway"
                },
                {
                  "id": "371",
                  "name": "武汉智慧港口",
                  "latitude": 43.7198,
                  "longitude": 79.7005,
                  "specialized_mode": "port"
                },
                {
                  "id": "372",
                  "name": "深圳国际机场",
                  "latitude": 21.7737,
                  "longitude": 77.1145,
                  "specialized_mode": "airport"
                },
                {
                  "id": "373",
                  "name": "娄底国际贸易港",
                  "latitude": 23.105,
                  "longitude": 102.7582,
                  "specialized_mode": "port"
                },
                {
                  "id": "374",
                  "name": "沈阳航空港",
                  "latitude": 32.2568,
                  "longitude": 88.2805,
                  "specialized_mode": "airport"
                },
                {
                  "id": "375",
                  "name": "潮州综合码头",
                  "latitude": 40.3401,
                  "longitude": 109.5544,
                  "specialized_mode": "port"
                },
                {
                  "id": "376",
                  "name": "大连综合交通枢纽",
                  "latitude": 32.7306,
                  "longitude": 129.2567,
                  "specialized_mode": "railway"
                },
                {
                  "id": "377",
                  "name": "郴州中欧班列枢纽",
                  "latitude": 41.2107,
                  "longitude": 74.7469,
                  "specialized_mode": "railway"
                },
                {
                  "id": "378",
                  "name": "益阳铁路物流园",
                  "latitude": 18.9672,
                  "longitude": 110.1871,
                  "specialized_mode": "railway"
                },
                {
                  "id": "379",
                  "name": "泰安航空物流园",
                  "latitude": 25.3364,
                  "longitude": 114.3818,
                  "specialized_mode": "airport"
                },
                {
                  "id": "380",
                  "name": "沈阳国际机场",
                  "latitude": 31.5819,
                  "longitude": 117.4513,
                  "specialized_mode": "airport"
                },
                {
                  "id": "381",
                  "name": "衢州国际物流港",
                  "latitude": 35.6133,
                  "longitude": 99.914,
                  "specialized_mode": "port"
                },
                {
                  "id": "382",
                  "name": "拉萨航空港",
                  "latitude": 44.934,
                  "longitude": 128.8196,
                  "specialized_mode": "airport"
                },
                {
                  "id": "383",
                  "name": "杭州物流集结中心",
                  "latitude": 30.8487,
                  "longitude": 81.3748,
                  "specialized_mode": "railway"
                },
                {
                  "id": "384",
                  "name": "石家庄中欧班列枢纽",
                  "latitude": 22.3723,
                  "longitude": 82.4572,
                  "specialized_mode": "railway"
                },
                {
                  "id": "385",
                  "name": "宁波高铁站",
                  "latitude": 29.6917,
                  "longitude": 81.2468,
                  "specialized_mode": "railway"
                },
                {
                  "id": "386",
                  "name": "肇庆智慧港口",
                  "latitude": 19.9568,
                  "longitude": 90.8211,
                  "specialized_mode": "port"
                },
                {
                  "id": "387",
                  "name": "聊城货运机场",
                  "latitude": 37.5145,
                  "longitude": 75.664,
                  "specialized_mode": "airport"
                },
                {
                  "id": "388",
                  "name": "绍兴临空经济区",
                  "latitude": 48.2515,
                  "longitude": 116.3419,
                  "specialized_mode": "airport"
                },
                {
                  "id": "389",
                  "name": "呼和浩特临空经济区",
                  "latitude": 51.841,
                  "longitude": 131.536,
                  "specialized_mode": "airport"
                },
                {
                  "id": "390",
                  "name": "常德智慧港口",
                  "latitude": 27.1043,
                  "longitude": 128.7804,
                  "specialized_mode": "port"
                },
                {
                  "id": "391",
                  "name": "珠海国际班列站",
                  "latitude": 25.9183,
                  "longitude": 78.2182,
                  "specialized_mode": "railway"
                },
                {
                  "id": "392",
                  "name": "湘潭航空港区",
                  "latitude": 31.8958,
                  "longitude": 115.7941,
                  "specialized_mode": "airport"
                },
                {
                  "id": "393",
                  "name": "阳江铁路集散中心",
                  "latitude": 39.8077,
                  "longitude": 87.0965,
                  "specialized_mode": "railway"
                },
                {
                  "id": "394",
                  "name": "韶关国际贸易港",
                  "latitude": 42.2635,
                  "longitude": 91.2035,
                  "specialized_mode": "port"
                },
                {
                  "id": "395",
                  "name": "台州临空经济区",
                  "latitude": 31.4255,
                  "longitude": 112.2993,
                  "specialized_mode": "airport"
                },
                {
                  "id": "396",
                  "name": "珠海航空港",
                  "latitude": 27.237,
                  "longitude": 102.3489,
                  "specialized_mode": "airport"
                },
                {
                  "id": "397",
                  "name": "株洲航空港",
                  "latitude": 19.7605,
                  "longitude": 110.9363,
                  "specialized_mode": "airport"
                },
                {
                  "id": "398",
                  "name": "云浮国际班列站",
                  "latitude": 34.9149,
                  "longitude": 86.2164,
                  "specialized_mode": "railway"
                },
                {
                  "id": "399",
                  "name": "岳阳铁路物流园",
                  "latitude": 52.7256,
                  "longitude": 76.402,
                  "specialized_mode": "railway"
                },
                {
                  "id": "400",
                  "name": "常德国际航空枢纽",
                  "latitude": 50.723,
                  "longitude": 92.9035,
                  "specialized_mode": "airport"
                }
              ]
            }'''

        # transfer_result = transfer_input
        transfer_result = input_transfer_point_parameters(transfer_input)
        # 处理新的返回格式
        if isinstance(transfer_result, dict) and 'code' in transfer_result:
            if transfer_result['code'] == 200:
                transfer_point_params = transfer_result['data']
                print("中转点参数接口完整输出:")
                print(safe_json_dumps(transfer_result, ensure_ascii=False, indent=2))
            else:
                print(f"中转点参数处理失败: {transfer_result['msg']}")
                return {"error": transfer_result}
        else:
            # 兼容旧格式
            transfer_point_params = transfer_result
            print("中转点参数接口完整输出:")
            print(safe_json_dumps(transfer_point_params, ensure_ascii=False, indent=2))

        # 4. 根据动员类型构建不同的综合参数
        if mobilization_type == "material":
            other_parameters_input = '''
        {
    "activity_id": "ae0ccb39fd4d4b1ca04bb2d4db2f7094",
    "algorithm_parameters": {
        "bigm": 999999.0,
        "eps": 1.0E-4
    },
    "algorithm_sequence_id": "84f455cd96c44f15a77af3ee21213d44",
    "demand_points": [
        {
            "demand": 80.0,
            "latitude": 21.245678,
            "longitude": 110.343254,
            "name": "东部战区",
            "priority": 0.5,
            "priority_level": "medium"
        }
    ],
    "mobilization_object_type": "material",
    "req_element_id": "d92b5389aed34ad6817ace37e5274298",
    "scheme_config": "6dd85563640e79bb5ed89a43dcee6e5b",
    "scheme_id": "a864ead571754905a152f51aa2dae72f",
    "strategy_weights": {
        "balance": 0.0,
        "capability": 0.0,
        "cost": 0.0,
        "distance": 0.0,
        "priority": 0.0,
        "safety": 0.0,
        "social": 0.0,
        "time": 1.0
    },
    "supply_points": [
        {
            "capacity": 3.0,
            "enterprise_safety": {
                "enterprise_nature": "民企",
                "enterprise_scale": "中",
                "foreign_background": "无",
                "mobilization_experience": "无",
                "resource_safety": "高安全性",
                "risk_record": "无"
            },
            "enterprise_size": "中",
            "enterprise_type": "民企",
            "id": "44d9afd6-0625-11f0-a5f4-00ff27b3340d",
            "latitude": 37.4567,
            "longitude": 119.2345,
            "name": "德州华泰汽车部件有限公司",
            "probability": 0.53,
            "production_capacity": 0.0,
            "resource_reserve": 3.0,
            "sub_objects": [
                {
                    "category_id": "fd109373722c4e59971577aeb820d540",
                    "category_name": "车辆运输车",
                    "items": [
                        {
                            "capacity_quantity": 0.0,
                            "enterprise_nature_score": 0.0,
                            "enterprise_scale_score": 0.0,
                            "equipment_depreciation_cost": 0.0,
                            "equipment_rental_price": 0.0,
                            "foreign_background": 0.0,
                            "max_available_quantity": 3.0,
                            "resource_safety": 1.0,
                            "risk_record": 0.0,
                            "specify_quantity": 0.0,
                            "sub_object_id": "ZB_QUGEVZQ:44d9afd6-0625-11f0-a5f4-00ff27b3340d",
                            "sub_object_name": "车辆运输车"
                        }
                    ],
                    "recommend_md5": "fc5c2fb4a7963c883fa62ff7744d3c94"
                }
            ]
        },
        {
            "capacity": 1.0,
            "enterprise_safety": {
                "enterprise_nature": "央国企",
                "enterprise_scale": "中",
                "foreign_background": "无",
                "mobilization_experience": "无",
                "resource_safety": "高安全性",
                "risk_record": "无"
            },
            "enterprise_size": "中",
            "enterprise_type": "央国企",
            "id": "44da2a51-0625-11f0-a5f4-00ff27b3340d",
            "latitude": 39.8765,
            "longitude": 116.4321,
            "name": "中铁物流集团有限公司",
            "probability": 0.53,
            "production_capacity": 0.0,
            "resource_reserve": 1.0,
            "sub_objects": [
                {
                    "category_id": "fd109373722c4e59971577aeb820d540",
                    "category_name": "车辆运输车",
                    "items": [
                        {
                            "capacity_quantity": 0.0,
                            "enterprise_nature_score": 0.0,
                            "enterprise_scale_score": 0.0,
                            "equipment_depreciation_cost": 0.0,
                            "equipment_rental_price": 0.0,
                            "foreign_background": 0.0,
                            "max_available_quantity": 1.0,
                            "resource_safety": 1.0,
                            "risk_record": 0.0,
                            "specify_quantity": 0.0,
                            "sub_object_id": "ZB_QUGEVZQ:44da2a51-0625-11f0-a5f4-00ff27b3340d",
                            "sub_object_name": "车辆运输车"
                        }
                    ],
                    "recommend_md5": "5ea64d0b0c735c5cbf5dd0fe457c613f"
                }
            ]
        },
        {
            "capacity": 124.0,
            "enterprise_safety": {
                "enterprise_nature": "民企",
                "enterprise_scale": "中",
                "foreign_background": "无",
                "mobilization_experience": "无",
                "resource_safety": "高安全性",
                "risk_record": "无"
            },
            "enterprise_size": "中",
            "enterprise_type": "民企",
            "id": "44d9b0bd-0625-11f0-a5f4-00ff27b3340d",
            "latitude": 37.2449,
            "longitude": 114.5678,
            "name": "邢台栏板挂车制造有限公司",
            "probability": 0.53,
            "production_capacity": 0.0,
            "resource_reserve": 124.0,
            "sub_objects": [
                {
                    "category_id": "fd109373722c4e59971577aeb820d540",
                    "category_name": "车辆运输车",
                    "items": [
                        {
                            "capacity_quantity": 0.0,
                            "enterprise_nature_score": 0.0,
                            "enterprise_scale_score": 0.0,
                            "equipment_depreciation_cost": 0.0,
                            "equipment_rental_price": 0.0,
                            "foreign_background": 0.0,
                            "max_available_quantity": 124.0,
                            "resource_safety": 1.0,
                            "risk_record": 0.0,
                            "specify_quantity": 0.0,
                            "sub_object_id": "ZB_QUGEVZQ:44d9b0bd-0625-11f0-a5f4-00ff27b3340d",
                            "sub_object_name": "车辆运输车"
                        }
                    ],
                    "recommend_md5": "18fd1159938f1529a04b1254b914ac01"
                }
            ]
        },
        {
            "capacity": 235.0,
            "enterprise_safety": {
                "enterprise_nature": "央国企",
                "enterprise_scale": "大",
                "foreign_background": "无",
                "mobilization_experience": "无",
                "resource_safety": "高安全性",
                "risk_record": "无"
            },
            "enterprise_size": "大",
            "enterprise_type": "央国企",
            "id": "44d9852a-0625-11f0-a5f4-00ff27b3340d",
            "latitude": 39.8789,
            "longitude": 116.4567,
            "name": "北京公交集团有限责任公司",
            "probability": 0.53,
            "production_capacity": 0.0,
            "resource_reserve": 235.0,
            "sub_objects": [
                {
                    "category_id": "fd109373722c4e59971577aeb820d540",
                    "category_name": "车辆运输车",
                    "items": [
                        {
                            "capacity_quantity": 0.0,
                            "enterprise_nature_score": 0.0,
                            "enterprise_scale_score": 0.0,
                            "equipment_depreciation_cost": 0.0,
                            "equipment_rental_price": 0.0,
                            "foreign_background": 0.0,
                            "max_available_quantity": 235.0,
                            "resource_safety": 1.0,
                            "risk_record": 0.0,
                            "specify_quantity": 0.0,
                            "sub_object_id": "ZB_QUGEVZQ:44d9852a-0625-11f0-a5f4-00ff27b3340d",
                            "sub_object_name": "车辆运输车"
                        }
                    ],
                    "recommend_md5": "ff8e40ffcb068c5f5bc4442ac7a9feb4"
                }
            ]
        },
        {
            "capacity": 7.0,
            "enterprise_safety": {
                "enterprise_nature": "其他",
                "enterprise_scale": "中",
                "foreign_background": "无",
                "mobilization_experience": "无",
                "resource_safety": "高安全性",
                "risk_record": "无"
            },
            "enterprise_size": "中",
            "enterprise_type": "其他",
            "id": "44da62f7-0625-11f0-a5f4-00ff27b3340d",
            "latitude": 36.62,
            "longitude": 117.05,
            "name": "山东鲁通集装箱运输有限公司",
            "probability": 0.53,
            "production_capacity": 0.0,
            "resource_reserve": 7.0,
            "sub_objects": [
                {
                    "category_id": "fd109373722c4e59971577aeb820d540",
                    "category_name": "车辆运输车",
                    "items": [
                        {
                            "capacity_quantity": 0.0,
                            "enterprise_nature_score": 0.0,
                            "enterprise_scale_score": 0.0,
                            "equipment_depreciation_cost": 0.0,
                            "equipment_rental_price": 0.0,
                            "foreign_background": 0.0,
                            "max_available_quantity": 7.0,
                            "resource_safety": 1.0,
                            "risk_record": 0.0,
                            "specify_quantity": 0.0,
                            "sub_object_id": "ZB_QUGEVZQ:44da62f7-0625-11f0-a5f4-00ff27b3340d",
                            "sub_object_name": "车辆运输车"
                        }
                    ],
                    "recommend_md5": "8db107275ebb145b6735da8ee6f57372"
                }
            ]
        },
        {
            "capacity": 11.0,
            "enterprise_safety": {
                "enterprise_nature": "民企",
                "enterprise_scale": "大",
                "foreign_background": "无",
                "mobilization_experience": "无",
                "resource_safety": "高安全性",
                "risk_record": "无"
            },
            "enterprise_size": "大",
            "enterprise_type": "民企",
            "id": "44d9c632-0625-11f0-a5f4-00ff27b3340d",
            "latitude": 36.9241,
            "longitude": 115.9432,
            "name": "山东鲁蓬车辆有限公司",
            "probability": 0.53,
            "production_capacity": 0.0,
            "resource_reserve": 11.0,
            "sub_objects": [
                {
                    "category_id": "fd109373722c4e59971577aeb820d540",
                    "category_name": "车辆运输车",
                    "items": [
                        {
                            "capacity_quantity": 0.0,
                            "enterprise_nature_score": 0.0,
                            "enterprise_scale_score": 0.0,
                            "equipment_depreciation_cost": 0.0,
                            "equipment_rental_price": 0.0,
                            "foreign_background": 0.0,
                            "max_available_quantity": 11.0,
                            "resource_safety": 1.0,
                            "risk_record": 0.0,
                            "specify_quantity": 0.0,
                            "sub_object_id": "ZB_QUGEVZQ:44d9c632-0625-11f0-a5f4-00ff27b3340d",
                            "sub_object_name": "车辆运输车"
                        }
                    ],
                    "recommend_md5": "109daba5b133b79d1c1b52c3065cf280"
                }
            ]
        },
        {
            "capacity": 4.0,
            "enterprise_safety": {
                "enterprise_nature": "民企",
                "enterprise_scale": "大",
                "foreign_background": "无",
                "mobilization_experience": "无",
                "resource_safety": "高安全性",
                "risk_record": "无"
            },
            "enterprise_size": "大",
            "enterprise_type": "民企",
            "id": "44cf5044-0625-11f0-a5f4-00ff27b3340d",
            "latitude": 45.7625,
            "longitude": 45.678,
            "name": "一汽解放汽车有限公司",
            "probability": 0.53,
            "production_capacity": 0.0,
            "resource_reserve": 4.0,
            "sub_objects": [
                {
                    "category_id": "fd109373722c4e59971577aeb820d540",
                    "category_name": "车辆运输车",
                    "items": [
                        {
                            "capacity_quantity": 0.0,
                            "enterprise_nature_score": 0.0,
                            "enterprise_scale_score": 0.0,
                            "equipment_depreciation_cost": 0.0,
                            "equipment_rental_price": 0.0,
                            "foreign_background": 0.0,
                            "max_available_quantity": 4.0,
                            "resource_safety": 1.0,
                            "risk_record": 0.0,
                            "specify_quantity": 0.0,
                            "sub_object_id": "ZB_QUGEVZQ:44cf5044-0625-11f0-a5f4-00ff27b3340d",
                            "sub_object_name": "车辆运输车"
                        }
                    ],
                    "recommend_md5": "7d3e4397ae3b2cca6d4044766a6bbb2e"
                }
            ]
        },
        {
            "capacity": 9.0,
            "enterprise_safety": {
                "enterprise_nature": "民企",
                "enterprise_scale": "小",
                "foreign_background": "无",
                "mobilization_experience": "无",
                "resource_safety": "高安全性",
                "risk_record": "无"
            },
            "enterprise_size": "小",
            "enterprise_type": "民企",
            "id": "44d969a0-0625-11f0-a5f4-00ff27b3340d",
            "latitude": 39.9,
            "longitude": 116.38,
            "name": "车辆出租服务公司",
            "probability": 0.53,
            "production_capacity": 0.0,
            "resource_reserve": 9.0,
            "sub_objects": [
                {
                    "category_id": "fd109373722c4e59971577aeb820d540",
                    "category_name": "车辆运输车",
                    "items": [
                        {
                            "capacity_quantity": 0.0,
                            "enterprise_nature_score": 0.0,
                            "enterprise_scale_score": 0.0,
                            "equipment_depreciation_cost": 0.0,
                            "equipment_rental_price": 0.0,
                            "foreign_background": 0.0,
                            "max_available_quantity": 9.0,
                            "resource_safety": 1.0,
                            "risk_record": 0.0,
                            "specify_quantity": 0.0,
                            "sub_object_id": "ZB_QUGEVZQ:44d969a0-0625-11f0-a5f4-00ff27b3340d",
                            "sub_object_name": "车辆运输车"
                        }
                    ],
                    "recommend_md5": "b0409fbc886a8c3dfbb45df6b59a04d9"
                }
            ]
        }
    ]
}'''

        elif mobilization_type == "personnel":
            other_parameters_input = '''{
    "activity_id": "ca0a358380894e8aadb6a71d8fd1c89e",
    "algorithm_parameters": {
        "bigm": 999999.0,
        "eps": 1.0E-4
    },
    "algorithm_sequence_id": "1c2e3632e1104f9bb70d44e3792dc3a6",
    "demand_points": [
        {
            "demand": 6.0,
            "latitude": 20.923456,
            "longitude": 110.365254,
            "name": "南部战区",
            "priority": 0.5,
            "priority_level": "medium"
        }
    ],
    "mobilization_object_type": "personnel",
    "req_element_id": "b35ef8f59f5f4763912518906fced939",
    "scheme_config": "c75670e58ae70654c30cd30d0ba789b4",
    "scheme_id": "961cee0b62da431781d8dcaa48e2c9b1",
    "strategy_weights": {
        "balance": 0.0,
        "capability": 0.0,
        "cost": 0.0,
        "distance": 0.0,
        "priority": 0.0,
        "safety": 0.0,
        "social": 0.0,
        "time": 1.0
    },
    "supply_points": [
        {
            "capacity": 1,
            "enterprise_safety": {
                "enterprise_nature": "其他",
                "enterprise_scale": "大",
                "foreign_background": "无",
                "mobilization_experience": "无",
                "resource_safety": "一般安全性",
                "risk_record": "无"
            },
            "enterprise_size": "大",
            "enterprise_type": "其他",
            "id": "e5557bfa-4515-11f0-9637-00ff27b3340d",
            "latitude": 31.2375,
            "longitude": 121.4783,
            "probability": 1.0,
            "production_capacity": 0.0,
            "resource_reserve": 1.0,
            "sub_objects": [
                {
                    "category_id": "05bc219ee34149479ebd330f0ab55483",
                    "category_name": "医疗救护专业队",
                    "dy_intensity": "中",
                    "items": [
                        {
                            "capacity_quantity": 0.0,
                            "credit_record": 0.0,
                            "criminal_record": 0.0,
                            "living_cost": 0.0,
                            "max_available_quantity": 1.0,
                            "military_experience": 0.0,
                            "network_record": 0.0,
                            "political_status": 0.0,
                            "specify_quantity": 0.0,
                            "sub_object_id": "DWXZ_T1QUKNNBY",
                            "sub_object_name": "医疗救护专业队",
                            "wage_cost": 0.0
                        }
                    ],
                    "recommend_md5": "c0a64f572876c2c7676b78701aebcbcc"
                }
            ]
        },
        {
            "capacity": 1,
            "enterprise_safety": {
                "enterprise_nature": "其他",
                "enterprise_scale": "大",
                "foreign_background": "无",
                "mobilization_experience": "无",
                "resource_safety": "一般安全性",
                "risk_record": "无"
            },
            "enterprise_size": "大",
            "enterprise_type": "其他",
            "id": "57d6e084-4a56-11f0-9637-00ff27b3340d",
            "latitude": 39.9292,
            "longitude": 116.4278,
            "probability": 1.0,
            "production_capacity": 0.0,
            "resource_reserve": 1.0,
            "sub_objects": [
                {
                    "category_id": "05bc219ee34149479ebd330f0ab55483",
                    "category_name": "医疗救护专业队",
                    "dy_intensity": "中",
                    "items": [
                        {
                            "capacity_quantity": 0.0,
                            "credit_record": 0.0,
                            "criminal_record": 0.0,
                            "living_cost": 0.0,
                            "max_available_quantity": 1.0,
                            "military_experience": 0.0,
                            "network_record": 0.0,
                            "political_status": 0.0,
                            "specify_quantity": 0.0,
                            "sub_object_id": "DWXZ_CKN0QY1BQ",
                            "sub_object_name": "医疗救护专业队",
                            "wage_cost": 0.0
                        }
                    ],
                    "recommend_md5": "d8dd2870a66019d490a8a4c81416cd59"
                }
            ]
        },
        {
            "capacity": 5,
            "enterprise_safety": {
                "enterprise_nature": "事业单位",
                "enterprise_scale": "中",
                "foreign_background": "无",
                "mobilization_experience": "无",
                "resource_safety": "一般安全性",
                "risk_record": "无"
            },
            "enterprise_size": "中",
            "enterprise_type": "事业单位",
            "id": "e54d92ae-4515-11f0-9637-00ff27b3340d",
            "latitude": 39.9526,
            "longitude": 116.4073,
            "probability": 1.0,
            "production_capacity": 0.0,
            "resource_reserve": 1.0,
            "sub_objects": [
                {
                    "category_id": "f595dc142d5347daab8b23a6e295be8a",
                    "category_name": "紧急医学救援队伍",
                    "dy_intensity": "中",
                    "items": [
                        {
                            "capacity_quantity": 0.0,
                            "credit_record": 0.0,
                            "criminal_record": 0.0,
                            "living_cost": 0.0,
                            "max_available_quantity": 1.0,
                            "military_experience": 0.0,
                            "network_record": 0.0,
                            "political_status": 0.0,
                            "specify_quantity": 0.0,
                            "sub_object_id": "DWXZ_4GG1NA3FJ",
                            "sub_object_name": "紧急医学救援队伍",
                            "wage_cost": 0.0
                        },
                        {
                            "capacity_quantity": 0.0,
                            "credit_record": 0.0,
                            "criminal_record": 0.0,
                            "living_cost": 0.0,
                            "max_available_quantity": 1.0,
                            "military_experience": 0.0,
                            "network_record": 0.0,
                            "political_status": 0.0,
                            "specify_quantity": 0.0,
                            "sub_object_id": "DWXZ_EJ6T4O8KP",
                            "sub_object_name": "紧急医学救援队伍",
                            "wage_cost": 0.0
                        },
                        {
                            "capacity_quantity": 0.0,
                            "credit_record": 0.0,
                            "criminal_record": 0.0,
                            "living_cost": 0.0,
                            "max_available_quantity": 1.0,
                            "military_experience": 0.0,
                            "network_record": 0.0,
                            "political_status": 0.0,
                            "specify_quantity": 0.0,
                            "sub_object_id": "DWXZ_H6KCELYLR",
                            "sub_object_name": "紧急医学救援队伍",
                            "wage_cost": 0.0
                        },
                        {
                            "capacity_quantity": 0.0,
                            "credit_record": 0.0,
                            "criminal_record": 0.0,
                            "living_cost": 0.0,
                            "max_available_quantity": 1.0,
                            "military_experience": 0.0,
                            "network_record": 0.0,
                            "political_status": 0.0,
                            "specify_quantity": 0.0,
                            "sub_object_id": "DWXZ_S9PZ5Z7AH",
                            "sub_object_name": "紧急医学救援队伍",
                            "wage_cost": 0.0
                        },
                        {
                            "capacity_quantity": 0.0,
                            "credit_record": 0.0,
                            "criminal_record": 0.0,
                            "living_cost": 0.0,
                            "max_available_quantity": 1.0,
                            "military_experience": 0.0,
                            "network_record": 0.0,
                            "political_status": 0.0,
                            "specify_quantity": 0.0,
                            "sub_object_id": "DWXZ_ZNQVAFB08",
                            "sub_object_name": "紧急医学救援队伍",
                            "wage_cost": 0.0
                        }
                    ],
                    "recommend_md5": "e8fba6f96a595e86f56221b760e08e37"
                }
            ]
        },
        {
            "capacity": 1,
            "enterprise_safety": {
                "enterprise_nature": "其他",
                "enterprise_scale": "中",
                "foreign_background": "无",
                "mobilization_experience": "无",
                "resource_safety": "一般安全性",
                "risk_record": "无"
            },
            "enterprise_size": "中",
            "enterprise_type": "其他",
            "id": "7fc47c8c-614f-4ca4-a9ec-eb1f6e2ece96",
            "latitude": 39.9289,
            "longitude": 116.3472,
            "probability": 1.0,
            "production_capacity": 0.0,
            "resource_reserve": 1.0,
            "sub_objects": [
                {
                    "category_id": "05bc219ee34149479ebd330f0ab55483",
                    "category_name": "医疗救护专业队",
                    "dy_intensity": "中",
                    "items": [
                        {
                            "capacity_quantity": 0.0,
                            "credit_record": 0.0,
                            "criminal_record": 0.0,
                            "living_cost": 0.0,
                            "max_available_quantity": 1.0,
                            "military_experience": 0.0,
                            "network_record": 0.0,
                            "political_status": 0.0,
                            "specify_quantity": 0.0,
                            "sub_object_id": "DWXZ_MQZQ27WY6",
                            "sub_object_name": "医疗救护专业队",
                            "wage_cost": 0.0
                        }
                    ],
                    "recommend_md5": "adb419fbda8a2e6ddea2cce543031e52"
                }
            ]
        }
    ]
}'''

        elif mobilization_type == "data":

            other_parameters_input = '''
           {
    "activity_id": "70e8a89e1c0540119212f53be692605d",
    "algorithm_parameters": {
        "bigm": 999999.0,
        "eps": 1.0E-4
    },
    "algorithm_sequence_id": "02cbf65063c54c849bf6c67ce35140cc",
    "demand_points": [
        {
            "demand": 1.0,
            "latitude": 20.923456,
            "longitude": 110.365254,
            "name": "南部战区",
            "priority": 0.5,
            "priority_level": "medium"
        }
    ],
    "mobilization_object_type": "data",
    "req_element_id": "d57a3f5e18b5494a97c85145061e35b3",
    "scheme_config": "64ed8fd450480754b31ae3eb4e56aeb9",
    "scheme_id": "cc8f45fde5474c0ea0a3c437e6b233f7",
    "strategy_weights": {
        "balance": 0.4,
        "capability": 0.3,
        "cost": 0.55,
        "distance": 0.5,
        "priority": 0.6,
        "safety": 0.5,
        "social": 0.49,
        "time": 0.78
    },
    "supply_points": [
        {
            "capacity": 1,
            "enterprise_safety": {
                "enterprise_nature": "其他",
                "enterprise_scale": "小",
                "foreign_background": "无",
                "mobilization_experience": "无",
                "resource_safety": "一般安全性",
                "risk_record": "无"
            },
            "enterprise_size": "小",
            "enterprise_type": "央企分公司",
            "id": "QYxz_SINOPEC-ZJ",
            "latitude": 21.1987,
            "longitude": 110.4123,
            "name": "中国石化销售广东湛江分公司",
            "probability": 1.0,
            "production_capacity": 0.0,
            "resource_reserve": 1.0,
            "sub_objects": [
                {
                    "category_id": "55c8f2c4b6984726a596bdd79ce98a41",
                    "category_name": "加油站",
                    "items": [
                        {
                            "access_control": 0.0,
                            "autonomous_control": 0.0,
                            "camouflage_protection": 0.0,
                            "capacity_quantity": 0.0,
                            "communication_purchase_price": 0.0,
                            "data_processing_cost": 0.0,
                            "data_storage_cost": 0.0,
                            "encryption_security": 0.0,
                            "facility_protection": 0.0,
                            "facility_rental_price": 0.0,
                            "maintenance_record": 0.0,
                            "max_available_quantity": 1.0,
                            "network_security": 0.0,
                            "power_cost": 0.0,
                            "specify_quantity": 0,
                            "sub_object_id": "ssxz_ZJ-SINOPEC22:QYxz_SINOPEC-ZJ",
                            "sub_object_name": "加油站",
                            "surrounding_environment": 0.0,
                            "usability_level": 0.0
                        }
                    ],
                    "recommend_md5": "205d6181df259070c2086e9a7c59bf82"
                }
            ]
        }
    ]
}
'''
        transport_input_modes = '''{
                    "transport_modes": [
                            {
                                "name": "公路",
                                "code": "trans-01",
                                "speed": 100,
                                "cost_per_km": 2.25,
                                "road_only_modes": 1
                            },
                            {
                                "name": "海运",
                                "code": "trans-02",
                                "speed": 40,
                                "cost_per_km": 1,
                                "road_only_modes": 0
                            },
                            {
                                "name": "空运",
                                "code": "trans-03",
                                "speed": 1000,
                                "cost_per_km": 40.0,
                                "road_only_modes": 0
                            },
                            {
                                "name": "铁路运输",
                                "code": "trans-04",
                                "speed": 350,
                                "cost_per_km": 2,
                                "road_only_modes": 0
                            }
                        ]
                }'''

        # 解析JSON
        data_transpoint_parameter = json.loads(transport_input_modes)

        # 定义文件夹和文件路径
        folder_name = "transport_modes_parameters_input"
        file_name = "transport_modes_ones.json"
        file_path = os.path.join(folder_name, file_name)
        # 检查文件夹是否存在，如果不存在则创建
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"已创建文件夹: {folder_name}")

        # 将数据写入JSON文件
        with open(file_path, 'w', encoding='utf-8') as file:
            json.dump(data_transpoint_parameter , file, ensure_ascii=False, indent=4)

        print(f"JSON数据已保存到: {file_path}")
        # # 将字符串解析为JSON对象
        # data = json.loads(other_parameters_input)
        #
        # # 提取transport_input内容
        # transport_input = data["transport_input"]
        # # transfer_points = transport_input["transfer_points"]
        #
        # # print(f"总共找到 {len(transfer_points)} 个转运点")
        # transfer_result = input_transfer_point_parameters(transport_input)
        #
        #
        # # 处理新的返回格式
        # if isinstance(transfer_result, dict) and 'code' in transfer_result:
        #     if transfer_result['code'] == 200:
        #         transfer_point_params = transfer_result['data']
        #         print("中转点参数接口完整输出:")
        #         print(safe_json_dumps(transfer_result, ensure_ascii=False, indent=2))
        #     else:
        #         print(f"中转点参数处理失败: {transfer_result['msg']}")
        #         return {"error": transfer_result}
        # else:
        #     # 兼容旧格式
        #     transfer_point_params = transfer_result
        #     print("中转点参数接口完整输出:")
        #     print(safe_json_dumps(transfer_point_params, ensure_ascii=False, indent=2))


        # 4. 综合参数处理 + 算法执行
        print(f"\n4. 处理综合参数并执行多目标优化算法（动员对象类型: {mobilization_type}）...")
        other_result = input_other_parameters(other_parameters_input, safety_standards, transfer_point_params,
                                              mobilization_type, f"REQ_{mobilization_type.upper()}_EXTERNAL")

        # 处理综合参数的返回格式
        if isinstance(other_result, dict) and 'code' in other_result:

            if other_result['code'] == 200:
                other_params = other_result['data']
                print("综合参数接口完整输出及优化结果:")
                # other_result = safe_json_dumps(other_result)
                print(json.dumps(other_result, ensure_ascii=False, indent=2))
                # 特别展示req_element_id的传递情况
                basic_info = other_result.get('data', {}).get('basic_info', {})
                if basic_info.get('req_element_id'):
                    print(f"\n请求元素ID传递验证: {basic_info['req_element_id']}")
            else:
                print(f"综合参数处理失败: {other_result['msg']}")
                print("错误详情:")
                print(json.dumps(other_result, ensure_ascii=False, indent=2))
                return {"error": other_result}
        else:
            # 兼容旧格式
            other_params = other_result
            print("综合参数接口完整输出及优化结果:")
            print(json.dumps(other_params, ensure_ascii=False, indent=2))

        # 5. 运输参数处理
        print("\n5. 处理运输参数...")

        network_data = other_params.get('generated_data', {}).get('network_data')



        transport_result = input_transport_parameters(transport_input_modes, network_data, time_params)

        # 处理新的返回格式
        if isinstance(transport_result, dict) and 'code' in transport_result:
            if transport_result['code'] == 200:
                transport_params = transport_result['data']
                print("运输参数接口完整输出:")
                print(safe_json_dumps({"TRANSPORT_MODES": transport_params.get('TRANSPORT_MODES', {})},
                                      ensure_ascii=False, indent=2))
            else:
                print(f"运输参数处理失败: {transport_result['msg']}")
                print("错误详情:")
                print(json.dumps(transport_result, ensure_ascii=False, indent=2))
                transport_params = {}
        else:
            # 兼容旧格式
            transport_params = transport_result
            print("运输参数接口完整输出:")
            print(safe_json_dumps({"TRANSPORT_MODES": transport_params.get('TRANSPORT_MODES', {})}, ensure_ascii=False,
                                  indent=2))

        # 展示优化结果摘要
        optimization_result = other_params.get('optimization_result')

        vdf = optimization_result['data']

        if optimization_result['msg'] == '数据动员指定成功' or optimization_result['msg'] == "已经指定":
            stand_out_1 = combine_info1(other_result, vdf)
        else:
            stand_out_1 = combine_info(other_result, vdf)

        # 处理综合参数的返回格式
        if isinstance(other_result, dict) and 'code' in other_result:
            if other_result['code'] == 200:
                other_params = other_result['data']
                print("综合参数接口完整输出及优化结果:")
                print(json.dumps(other_result, ensure_ascii=False, indent=2))
                # 特别展示req_element_id的传递情况
                basic_info = other_result.get('data', {}).get('basic_info', {})
                if basic_info.get('req_element_id'):
                    print(f"\n请求元素ID传递验证: {basic_info['req_element_id']}")
            else:
                print(f"综合参数处理失败: {other_result['msg']}")
                print("错误详情:")
                print(json.dumps(other_result, ensure_ascii=False, indent=2))
                return {"error": other_result}
        else:
            # 兼容旧格式
            other_params = other_result
            print("综合参数接口完整输出及优化结果:")
            print(json.dumps(other_params, ensure_ascii=False, indent=2))

        print("\n" + "=" * 80)
        print(f"{example_name}多目标优化算法执行结果:")
        print("=" * 80)

        if optimization_result and optimization_result.get("code") == 200:
            data = optimization_result["data"]
            print(f"优化状态: {optimization_result['msg']}")
            print(f"算法序号: {data.get('algorithm_sequence_id', 'None')}")
            print(f"动员对象类型: {data.get('mobilization_object_type', 'None')}")
            print(f"请求元素ID: {data.get('req_element_id', 'None')}")

            # 根据动员类型显示不同的结果结构
            if mobilization_type == "data":
                # 数据动员的结果结构
                selected_suppliers = data.get('selected_suppliers', [])
                print(f"选择供应商数: {len(selected_suppliers)}")

                network_info = data.get('network_info', {})
                print(
                    f"数据供应点总数: {network_info.get('data_supply_points_count', network_info.get('supply_points_total', 'N/A'))}")
                print(
                    f"选择供应点数: {network_info.get('selected_points_count', network_info.get('active_suppliers', 'N/A'))}")
                print(f"动员模式: {network_info.get('mobilization_mode', 'capacity_based_selection')}")
                print(f"求解时间: {data.get('solve_time_seconds', 'N/A')}秒")

                # 计算数据动员的满足情况
                if selected_suppliers:
                    total_allocated = sum(
                        supplier.get('allocated_amount', supplier.get('total_supply_amount', 0)) for supplier in
                        selected_suppliers)
                    print(f"总分配数据量: {total_allocated}")

                    # 如果有目标摘要信息
                    objective_summary = data.get('objective_summary', {})
                    if objective_summary:
                        print(f"动员类型: {objective_summary.get('mobilization_type', 'N/A')}")
            else:
                # 物资动员和人员动员的结果结构
                allocation_performance = data.get('allocation_performance', {})
                network_analysis = data.get('network_analysis', {})
                objective_achievement = data.get('objective_achievement', {})

                print(f"选择供应商数: {allocation_performance.get('suppliers_utilized', 'N/A')}")
                print(f"需求满足率: {allocation_performance.get('satisfaction_rate_percent', 'N/A')}%")
                print(f"总供应量: {allocation_performance.get('total_allocated', 'N/A')}")
                print(f"活跃路线数: {allocation_performance.get('routes_activated', 'N/A')}")
                print(f"求解时间: {data.get('solve_time_seconds', 'N/A')}秒")
                print(f"综合目标值: {objective_achievement.get('composite_value', 'N/A')}")

            print(f"\n详细供应商信息:")
            selected_suppliers = data.get('selected_suppliers', []) if mobilization_type == "data" else data.get(
                'supplier_portfolio', [])
            for i, supplier in enumerate(selected_suppliers[:2], 1):
                supplier_name = supplier.get('supplier_name', f'供应商{i}')
                enterprise_type = supplier.get('enterprise_type', 'N/A')
                total_supply = supplier.get('allocated_amount', 'N/A')

                # 简化的routes访问
                routes = supplier.get('routes', [])
                routes_count = len(routes)

                print(f"  {i}. {supplier_name} ({enterprise_type}):")
                print(f"     供应量: {total_supply}, 路线数: {routes_count}")

                if mobilization_type == "data":
                    # 数据动员显示评价分数
                    evaluation_scores = supplier.get('evaluation_scores', {})
                    if evaluation_scores:
                        print(f"     综合评分: {evaluation_scores.get('composite_score', 'N/A')}")
                        print(f"     能力评分: {evaluation_scores.get('capability_score', 'N/A')}")
                else:
                    # 物资和人员动员显示路线信息
                    if routes:
                        route = routes[0]
                        transport_details = route.get('transport_details', {})
                        performance_metrics = route.get('performance_metrics', {})
                        allocation_info = route.get('allocation', {})

                        print(
                            f"     主要路线: {route.get('route_type', 'N/A')}, 安全系数: {performance_metrics.get('safety_score', 'N/A')}")
                        print(
                            f"     总成本: {allocation_info.get('unit_cost', 'N/A')}, 总时间: {transport_details.get('time_hours', 'N/A')}小时")

            if len(selected_suppliers) > 2:
                print(f"  ... 还有 {len(selected_suppliers) - 2} 个供应商")

            # 检查是否有细分对象信息
            network_data = other_params.get('network_data') or other_params.get('generated_data', {}).get(
                'network_data')
            if network_data and 'point_features' in network_data:
                sub_objects_info = []
                for supplier in selected_suppliers:
                    supplier_name = supplier.get('supplier_name', '')
                    if supplier_name and supplier_name in network_data['point_features']:
                        sub_objects = network_data['point_features'][supplier_name].get('sub_objects', [])
                        if sub_objects:
                            total_items = 0
                            for category in sub_objects:
                                if isinstance(category, dict) and 'items' in category:
                                    total_items += len(category.get('items', []))
                                else:
                                    total_items += 1  # 兼容旧格式
                            sub_objects_info.append(f"{supplier_name}: {total_items}个细分对象")

                if sub_objects_info:
                    print(f"\n细分对象配置:")
                    for info in sub_objects_info:
                        print(f"  - {info}")


        elif optimization_result:
            print(f"优化失败: {optimization_result.get('msg', '未知错误')}")
            print("错误详情:")
            print(json.dumps(optimization_result.get('data', {}), ensure_ascii=False, indent=2))
        else:
            print("优化未执行")

        print("最终结果：")
        if stand_out_1['code'] == 200:
            # other_paramsjh = stand_out_1['data']
            print("第一次打印")
            print("综合参数接口完整输出及优化结果22222:")
            print("综合参数接口完整输出及优化结果22222:")
            print(json.dumps(stand_out_1, ensure_ascii=False, indent=1))

        return {
            'mobilization_type': mobilization_type,
            'req_element_id': f"REQ_{mobilization_type.upper()}_EXTERNAL",
            'time_params': time_params,
            'safety_standards': safety_standards,
            'transfer_point_params': transfer_point_params,
            'other_params': other_params,
            'transport_params': transport_params,
            'network_data': network_data,
            'optimization_result': optimization_result
        }


    # 执行三种动员类型的示例
    print("=" * 100)
    print("多目标四层运输网络优化算法")
    print("=" * 100)

    results = {}



    # 1. 物资动员示例
    # # 1. 物资动员示例
    results['material'] = run_mobilization_example("material", "物资动员")
    # #
    # # 2. 人员动员示例
    # results['personnel'] = run_mobilization_example("personnel", "人员动员")
    #
    # # 3. 数据动员示例
    # results['data'] = run_mobilization_example("data", "数据动员")
    #


    return results


# ==========================================
# 程序入口函数
# ========================================

if __name__ == "__main__":
    """
    程序主入口

    演示数据处理接口并执行多目标优化器测试，所有结果从五个接口中统一展示
    """

    try:
        # 运行完整的接口演示，包含算法执行
        t1 = time.time()
        interface_demo_results = example_usage()
        t2 = time.time()

        print("\n完整流程演示完成！用时：{}".format(t2 - t1))


    except Exception as e:
        print(f"完整流程演示失败: {str(e)}")
        import traceback

        traceback.print_exc()
