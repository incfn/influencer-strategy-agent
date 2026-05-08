#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快手达人激励策略预测 Agent（v2）
核心升级：
1. 适配动态达人池（流入+流出）
2. 三种阶梯式激励方案完整仿真（方案1名额限制、方案2比例提成、方案3双重叠加）
3. 月度总预算硬约束 10 万元
4. 方案结构可视化（Markdown 阶梯表）
用法：
  python agent.py --baseline              # 仅基线预测
  python agent.py --simulate --top-n 5    # 仿真寻优
  python agent.py --simulate --report strategy.md
"""

import argparse
import io
import os
import sys
import textwrap
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# DeepSeek LLM 可选依赖（失败不阻断主流程）
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Windows 中文显示支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
matplotlib.rcParams['axes.unicode_minus'] = False

np.random.seed(2025)

DATA_PATH = "kuaishou_influencer_data.csv"
TOTAL_BUDGET = 100_000   # 月度总预算上限 10 万元
MIN_BUDGET = 95_000      # 月度总预算下限 9.5 万元（保证激励力度）

LAYERS = ['头部', '腰部', '尾部']

# ============================================================
# 1. 数据加载层
# ============================================================

class DataLoader:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.months: List[str] = []

    def load(self) -> bool:
        try:
            self.df = pd.read_csv(DATA_PATH, encoding='utf-8-sig')
        except FileNotFoundError:
            print(f"[错误] 数据文件未找到: {DATA_PATH}")
            return False
        self.months = sorted(self.df['月度'].unique())
        print(f"[数据加载] 达人记录: {len(self.df)} 条")
        print(f"[数据加载] 时间跨度: {self.months[0]} 至 {self.months[-1]}")
        active_per_month = self.df.groupby('月度')['达人ID'].nunique()
        print(f"[数据加载] 每月活跃达人: {dict(active_per_month)}")
        return True

    def get_layer_summary(self) -> pd.DataFrame:
        summary = []
        for layer in LAYERS:
            sub = self.df[self.df['达人分层'] == layer]
            summary.append({
                '达人分层': layer,
                '月均活跃': round(sub.groupby('月度')['达人ID'].nunique().mean(), 0),
                '人均视频数': round(sub['月度视频条数'].mean(), 0),
                '人均GMV': round(sub['月度总GMV'].mean(), 0),
                '人均激励': round(sub['实际获得激励金额'].mean(), 0),
                '平均投产比': round(sub['GMV投产比'].mean(), 2),
                '平均流失率': f"{sub['达人流失率'].mean():.1%}",
            })
        return pd.DataFrame(summary)

    def get_last_month_active(self) -> pd.DataFrame:
        last_month = self.months[-1]
        last_df = self.df[self.df['月度'] == last_month].copy()
        hist = self.df.groupby('达人ID').agg({
            '月度视频条数': 'mean',
            '月度总GMV': 'mean',
            '实际获得激励金额': 'mean',
        }).rename(columns={
            '月度视频条数': '历史平均视频数',
            '月度总GMV': '历史平均GMV',
            '实际获得激励金额': '历史平均激励',
        })
        last_df = last_df.merge(hist, on='达人ID', how='left')
        return last_df

    def get_monthly_inflow_estimate(self) -> float:
        """估算历史月均新达人流入数"""
        inflows = []
        prev_ids = set()
        for month in self.months:
            curr_ids = set(self.df[self.df['月度'] == month]['达人ID'])
            new_ids = curr_ids - prev_ids
            inflows.append(len(new_ids))
            # 更新 prev_ids 为下月期初（排除本月退出者）
            churned = set(self.df[(self.df['月度'] == month) & (self.df['是否当月退出'])]['达人ID'])
            prev_ids = curr_ids - churned
        # 去掉第一个月（初始池），取后续月份均值
        return np.mean(inflows[1:]) if len(inflows) > 1 else 10


# ============================================================
# 2. 基线预测引擎
# ============================================================

class BaselinePredictor:
    def __init__(self, df: pd.DataFrame, months: List[str]):
        self.df = df
        self.months = months

    def _trend_forecast(self, series: pd.Series) -> float:
        if len(series) < 2:
            return series.iloc[-1] if len(series) > 0 else 0
        x = np.arange(len(series))
        y = series.values
        weights = np.ones_like(y)
        if len(y) >= 3:
            weights[-3:] = [1, 2, 3]
        coef = np.polyfit(x, y, 1, w=weights)
        pred = coef[0] * len(series) + coef[1]
        return max(0, pred)

    def predict(self) -> Dict:
        results = {}
        total_posts = total_gmv = total_incentive = total_churned = total_start = 0

        for layer in LAYERS:
            layer_df = self.df[self.df['达人分层'] == layer]
            monthly = layer_df.groupby('月度').agg({
                '达人ID': 'nunique',
                '月度视频条数': 'sum',
                '月度总GMV': 'sum',
                '实际获得激励金额': 'sum',
                '达人流失率': 'mean',
            }).reset_index()

            active_pred = self._trend_forecast(monthly['达人ID'])
            last_active = monthly['达人ID'].iloc[-1]
            active_pred = max(active_pred, last_active * 0.85)

            avg_posts = monthly['月度视频条数'] / monthly['达人ID']
            avg_gmv = monthly['月度总GMV'] / monthly['达人ID']
            avg_incentive = monthly['实际获得激励金额'] / monthly['达人ID']

            posts_per = self._trend_forecast(avg_posts)
            gmv_per = self._trend_forecast(avg_gmv)
            inc_per = self._trend_forecast(avg_incentive)

            layer_posts = active_pred * posts_per
            layer_gmv = active_pred * gmv_per
            layer_inc = active_pred * inc_per
            churn_rate = monthly['达人流失率'].tail(3).mean()

            results[layer] = {
                '预测活跃人数': round(active_pred, 0),
                '人均视频数预测': round(posts_per, 0),
                '人均GMV预测': round(gmv_per, 0),
                '预测视频数总量': round(layer_posts, 0),
                '预测GMV总量': round(layer_gmv, 0),
                '预测激励总额': round(layer_inc, 0),
                '预测流失率': churn_rate,
            }
            total_posts += layer_posts
            total_gmv += layer_gmv
            total_incentive += layer_inc
            total_churned += active_pred * churn_rate
            total_start += active_pred

        results['整体'] = {
            '预测活跃人数': round(total_start, 0),
            '预测视频数总量': round(total_posts, 0),
            '预测GMV总量': round(total_gmv, 0),
            '预测激励总额': round(total_incentive, 0),
            '预测整体流失率': total_churned / total_start if total_start > 0 else 0,
            '预测投产比': total_gmv / total_incentive if total_incentive > 0 else 0,
        }
        return results

    def format_report(self) -> str:
        pred = self.predict()
        lines = []
        lines.append("## 三、基线预测（维持现有策略不变）")
        lines.append("")
        lines.append("假设下月（第 7 个月）不调整任何激励参数，基于近 6 个月线性趋势外推：")
        lines.append("")
        lines.append("| 分层 | 预测活跃人数 | 预测视频数总量 | 预测GMV总量 | 预测激励总额 | 预测投产比 | 预测流失率 |")
        lines.append("|------|-------------|-------------|------------|-------------|-----------|-----------|")
        for layer in LAYERS + ['整体']:
            p = pred[layer]
            if layer == '整体':
                lines.append(f"| {layer} | {p['预测活跃人数']:,.0f} | {p['预测视频数总量']:,.0f} | {p['预测GMV总量']:,.0f} 元 | {p['预测激励总额']:,.0f} 元 | {p['预测投产比']:.2f} | {p['预测整体流失率']:.1%} |")
            else:
                roi = p['预测GMV总量'] / p['预测激励总额'] if p['预测激励总额'] > 0 else 0
                lines.append(f"| {layer} | {p['预测活跃人数']:,.0f} | {p['预测视频数总量']:,.0f} | {p['预测GMV总量']:,.0f} 元 | {p['预测激励总额']:,.0f} 元 | {roi:.2f} | {p['预测流失率']:.1%} |")
        lines.append("")
        return "\n".join(lines)


# ============================================================
# 3. 激励参数仿真引擎
# ============================================================

class IncentiveSimulator:
    def __init__(self, df: pd.DataFrame, months: List[str], baseline: Dict):
        self.df = df
        self.months = months
        self.baseline = baseline

        # 历史各层实际发放占比
        total_by_layer = df.groupby('达人分层')['实际获得激励金额'].sum()
        total_all = total_by_layer.sum()
        self.historical_incentive_ratio = {
            layer: (total_by_layer.get(layer, 0) / total_all if total_all > 0 else 0.33)
            for layer in LAYERS
        }

        # 参数搜索空间（删除配比参数，扩大其他参数范围，共 750 组）
        self.post_coefs = [0.8, 0.9, 1.0, 1.1, 1.2]
        self.gmv_coefs = [0.6, 0.8, 1.0, 1.2, 1.5]
        self.reward_coefs = [0.6, 0.8, 1.0, 1.2, 1.5]
        self.quota_scales = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

        self.avg_inflow = DataLoader().get_monthly_inflow_estimate()

    # ---- 两种方案计算（与生成器逻辑一致） ----

    def _calc_scheme1(self, creators_data: List[Dict], post_coef: float, reward_coef: float, quota_scale: float = 1.0) -> Dict[str, float]:
        rules = [
            {'post_th': 150 * post_coef, 'reward': 700 * reward_coef, 'quota': max(5, int(20 * quota_scale))},
            {'post_th': 100 * post_coef, 'reward': 500 * reward_coef, 'quota': max(5, int(50 * quota_scale))},
            {'post_th': 50 * post_coef,  'reward': 300 * reward_coef, 'quota': max(5, int(100 * quota_scale))},
        ]
        rewards = {c['id']: 0.0 for c in creators_data}
        assigned = set()
        for rule in rules:
            eligible = [c for c in creators_data if c['posts'] >= rule['post_th'] and c['id'] not in assigned]
            eligible.sort(key=lambda x: x['gmv'], reverse=True)
            for c in eligible[:rule['quota']]:
                rewards[c['id']] = rule['reward']
                assigned.add(c['id'])
        return rewards

    def _calc_scheme2(self, creators_data: List[Dict], gmv_coef: float, reward_coef: float) -> Dict[str, float]:
        rules = [
            {'gmv_th': 80000 * gmv_coef, 'rate': 0.010 * reward_coef},
            {'gmv_th': 20000 * gmv_coef, 'rate': 0.008 * reward_coef},
            {'gmv_th': 5000 * gmv_coef,  'rate': 0.005 * reward_coef},
        ]
        rewards = {}
        for c in creators_data:
            gmv = c['gmv']
            reward = 0.0
            for rule in rules:
                if gmv > rule['gmv_th']:
                    reward = gmv * rule['rate']
                    break
                elif gmv >= rule['gmv_th']:
                    reward = gmv * rule['rate']
                    break
            rewards[c['id']] = reward
        return rewards

    def _calc_all_incentives(self, creators_data: List[Dict], post_coef: float, gmv_coef: float, reward_coef: float, quota_scale: float = 1.0) -> Dict[str, Tuple[str, float]]:
        s1 = self._calc_scheme1(creators_data, post_coef, reward_coef, quota_scale)
        s2 = self._calc_scheme2(creators_data, gmv_coef, reward_coef)

        result = {}
        for c in creators_data:
            cid = c['id']
            r1, r2 = s1[cid], s2[cid]
            total = r1 + r2
            if r1 > 0 and r2 > 0:
                result[cid] = ('条数+GMV叠加', total)
            elif r1 > 0:
                result[cid] = ('纯条数激励', total)
            elif r2 > 0:
                result[cid] = ('纯GMV激励', total)
            else:
                result[cid] = ('未达标', 0.0)
        return result

    # ---- 单组参数仿真 ----

    def _simulate_single(self, params: Dict, last_active_df: pd.DataFrame) -> Optional[Dict]:
        post_c = params['条数门槛系数']
        gmv_c = params['GMV门槛系数']
        reward_c = params['奖励系数']
        quota_s = params.get('名额系数', 1.0)

        total_posts = total_gmv = total_incentive = 0
        churn_count = {layer: 0 for layer in LAYERS}
        start_count = {layer: 0 for layer in LAYERS}

        # 处理存量达人
        creators_data = []
        for _, row in last_active_df.iterrows():
            layer = row['达人分层']
            avg_posts = row['历史平均视频数']
            avg_gmv = row['历史平均GMV']
            avg_incentive = row['历史平均激励']

            # 视频数预测
            if post_c * 150 > avg_posts * 0.9:
                posts_pred = avg_posts * 0.92
            elif post_c * 50 < avg_posts * 0.4:
                posts_pred = avg_posts * 0.95
            else:
                posts_pred = avg_posts
            posts_pred = max(0, posts_pred * (1 + np.random.normal(0, 0.03)))

            # GMV预测
            efficiency = avg_gmv / avg_posts if avg_posts > 0 else 0
            gmv_pred = posts_pred * efficiency * (1 + np.random.normal(0, 0.05))
            gmv_pred = max(0, gmv_pred)

            creators_data.append({
                'id': row['达人ID'],
                'layer': layer,
                'posts': posts_pred,
                'gmv': gmv_pred,
                'avg_incentive': avg_incentive,
            })

        # 计算两种方案（可叠加）
        incentives = self._calc_all_incentives(creators_data, post_c, gmv_c, reward_c, quota_s)

        # 退出决策 + 汇总
        for c in creators_data:
            layer = c['layer']
            cid = c['id']
            avg_incentive = c['avg_incentive']
            scheme, new_incentive = incentives[cid]

            start_count[layer] += 1

            base_churn = {'头部': 0.015, '腰部': 0.04, '尾部': 0.07}[layer]
            expected = {'头部': 1000, '腰部': 600, '尾部': 350}[layer]
            shortfall = max(0, (expected - new_incentive) / expected) if expected > 0 else 0
            churn_prob = base_churn + 0.15 * shortfall
            churn_prob = min(churn_prob, 0.25)

            if np.random.random() < churn_prob:
                churn_count[layer] += 1
                continue

            total_posts += c['posts']
            total_gmv += c['gmv']
            total_incentive += new_incentive

        # 新达人流入
        inflow = max(3, int(self.avg_inflow + np.random.normal(0, 3)))
        inflow_posts = 0
        inflow_gmv = 0
        inflow_incentive = 0
        for _ in range(inflow):
            layer = np.random.choice(LAYERS, p=[0.10, 0.30, 0.60])
            if layer == '头部':
                posts = np.random.normal(280, 40)
                gmv = np.random.normal(60000, 15000)
            elif layer == '腰部':
                posts = np.random.normal(180, 30)
                gmv = np.random.normal(3500, 800)
            else:
                posts = np.random.normal(140, 25)
                gmv = np.random.normal(400, 100)
            posts = max(20, posts)
            gmv = max(0, gmv)

            # 新达人激励计算
            newbie_data = [{'id': 'tmp', 'layer': layer, 'posts': posts, 'gmv': gmv, 'avg_incentive': 0}]
            newbie_inc = self._calc_all_incentives(newbie_data, post_c, gmv_c, reward_c, quota_s)
            scheme, raw = newbie_inc['tmp']
            actual = raw

            inflow_posts += posts
            inflow_gmv += gmv
            inflow_incentive += actual

        total_posts += inflow_posts
        total_gmv += inflow_gmv
        total_incentive += inflow_incentive

        total_start = sum(start_count.values())
        tail_churn = churn_count['尾部'] / start_count['尾部'] if start_count['尾部'] > 0 else 0
        overall_churn = sum(churn_count.values()) / total_start if total_start > 0 else 0
        roi = total_gmv / total_incentive if total_incentive > 0 else 0

        result = {
            '参数': params,
            '预测视频数总量': total_posts,
            '预测GMV总量': total_gmv,
            '预测激励总额': total_incentive,
            '预测投产比': roi,
            '整体流失率': overall_churn,
            '尾部流失率': tail_churn,
            '预测活跃人数': total_start - sum(churn_count.values()) + inflow,
        }

        # 硬约束
        baseline_posts = self.baseline['整体']['预测视频数总量']
        if total_posts < baseline_posts * 0.90:
            return None
        if tail_churn > 0.15:
            return None
        if total_incentive < MIN_BUDGET or total_incentive > TOTAL_BUDGET:
            return None

        return result

    def run(self, n_runs: int = 3, top_n: int = 3) -> Tuple[List[Dict], pd.DataFrame]:
        print("\n[仿真引擎] 构建参数网格...")
        param_grid = []
        for p_c in self.post_coefs:
            for g_c in self.gmv_coefs:
                for r_c in self.reward_coefs:
                    for q_s in self.quota_scales:
                        param_grid.append({
                            '条数门槛系数': p_c,
                            'GMV门槛系数': g_c,
                            '奖励系数': r_c,
                            '名额系数': round(q_s, 2),
                        })

        print(f"[仿真引擎] 有效参数组合: {len(param_grid)} 组")
        print(f"[仿真引擎] 蒙特卡洛次数: {n_runs} 次/组")

        last_active_df = self.df[self.df['月度'] == self.months[-1]].merge(
            self.df.groupby('达人ID').agg({
                '月度视频条数': 'mean',
                '月度总GMV': 'mean',
                '实际获得激励金额': 'mean',
            }).rename(columns={
                '月度视频条数': '历史平均视频数',
                '月度总GMV': '历史平均GMV',
                '实际获得激励金额': '历史平均激励',
            }),
            on='达人ID', how='left'
        )

        valid_results = []
        for i, params in enumerate(param_grid):
            if (i + 1) % 50 == 0 or i == len(param_grid) - 1:
                print(f"  进度: {i+1}/{len(param_grid)}")

            agg = {k: [] for k in ['预测视频数总量', '预测GMV总量', '预测激励总额', '预测投产比',
                                     '整体流失率', '尾部流失率', '预测活跃人数']}
            valid_count = 0

            for _ in range(n_runs):
                res = self._simulate_single(params, last_active_df)
                if res is not None:
                    valid_count += 1
                    for k in agg:
                        agg[k].append(res[k])

            if valid_count < n_runs * 0.5:
                continue

            valid_results.append({
                **params,
                **{k: np.mean(v) for k, v in agg.items()},
                '约束通过率': valid_count / n_runs,
            })

        if not valid_results:
            print("[警告] 无参数组合满足硬约束")
            return [], pd.DataFrame()

        results_df = pd.DataFrame(valid_results)
        results_df = results_df.sort_values('预测投产比', ascending=False).reset_index(drop=True)

        top_strategies = []
        for i in range(min(top_n, len(results_df))):
            row = results_df.iloc[i]
            top_strategies.append({
                '排名': i + 1,
                '条数门槛系数': row['条数门槛系数'],
                'GMV门槛系数': row['GMV门槛系数'],
                '奖励系数': row['奖励系数'],
                '预测视频数总量': row['预测视频数总量'],
                '预测GMV总量': row['预测GMV总量'],
                '预测激励总额': row['预测激励总额'],
                '预测投产比': row['预测投产比'],
                '整体流失率': row['整体流失率'],
                '尾部流失率': row['尾部流失率'],
                '预测活跃人数': row['预测活跃人数'],
            })

        print(f"[仿真引擎] 完成。满足约束: {len(results_df)} 组")
        print(f"[仿真引擎] 最优投产比: {results_df['预测投产比'].iloc[0]:.2f}")
        return top_strategies, results_df

    def sensitivity_analysis(self, results_df: pd.DataFrame) -> Dict[str, float]:
        sens = {}
        for col in ['条数门槛系数', 'GMV门槛系数', '奖励系数']:
            if col in results_df.columns:
                grouped = results_df.groupby(col)['预测投产比'].mean()
                sens[col] = round(grouped.std(), 3)
        return sens


# ============================================================
# 4. DeepSeek LLM 建议节点（数据安全：只传相对变化，不传绝对数值）
# ============================================================

class LLMAdvisor:
    """调用 DeepSeek API 分析策略对比结果，生成自然语言建议"""

    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY", "")
        self.enabled = bool(self.api_key) and OpenAI is not None

    def _build_prompt(self, baseline: Dict, strategies: List[Dict]) -> str:
        """构造安全 Prompt：只给策略特征和优劣势，不给逐条指标数值"""

        base = baseline.get('整体', baseline)
        base_posts = base.get('预测视频数总量', 1)
        base_gmv = base.get('预测GMV总量', 1)
        base_tail_churn = base.get('预测整体流失率', base.get('尾部流失率', 0))
        base_active = base.get('预测活跃人数', 1)
        base_incentive = base.get('预测激励总额', 1)
        base_roi = base.get('预测投产比', 1)

        lines = [
            "你正在查看一张「六维相对提升对比图」，对比了基线与 Top 3 优化策略。",
            "图中六个维度：视频数、GMV、激励总额、投产比、尾部留存健康度、活跃人数。",
            "",
            "=== 各策略核心特征（不要复述数字，据此分析优缺点即可） ===",
        ]

        for s in strategies:
            rank = s.get('排名', '?')
            pc = s.get('条数门槛系数', 1.0)
            gc = s.get('GMV门槛系数', 1.0)
            rc = s.get('奖励系数', 1.0)
            qs = s.get('名额系数', 1.0)

            posts_chg = (s['预测视频数总量'] - base_posts) / base_posts * 100 if base_posts else 0
            gmv_chg = (s['预测GMV总量'] - base_gmv) / base_gmv * 100 if base_gmv else 0
            incentive_chg = (s['预测激励总额'] - base_incentive) / base_incentive * 100 if base_incentive else 0
            roi_chg = s['预测投产比'] - base_roi
            tail_churn_chg = (s['尾部流失率'] - base_tail_churn) * 100
            active_chg = (s['预测活跃人数'] - base_active) / base_active * 100 if base_active else 0

            traits = []
            if abs(posts_chg) < 1 and abs(gmv_chg) < 2:
                traits.append("内容基本盘无损")
            elif abs(posts_chg) < 1 and gmv_chg < -2:
                traits.append("视频数稳住但GMV有所牺牲")
            else:
                traits.append("内容基本盘略有波动")

            if roi_chg > -1.5:
                traits.append("资金效率降幅最小")
            elif roi_chg > -2.5:
                traits.append("资金效率降幅居中")
            else:
                traits.append("资金效率降幅最大")

            if tail_churn_chg < -3.5:
                traits.append("尾部生态改善最显著")
            elif tail_churn_chg < -1.5:
                traits.append("尾部生态改善明显")
            else:
                traits.append("尾部生态改善有限")

            if active_chg > 8:
                traits.append("达人池扩张最快")
            elif active_chg > 5:
                traits.append("达人池稳步扩张")
            else:
                traits.append("达人池扩张一般")

            traits_str = "；".join(traits)
            lines.append(f"策略{rank}（条数系数{pc}, GMV系数{gc}, 奖励系数{rc}, 名额系数{qs}）：")
            lines.append(f"  特征：{traits_str}。")
            lines.append("")

        lines.extend([
            "=== 硬约束 ===",
            "三个策略均满足：视频数跌幅未超10%、尾部流失率未超15%、预算在9.5-10万之间。",
            "",
            "=== 你的任务 ===",
            "基于上述策略特征，直接对比各策略的优缺点（不要复述任何百分比数字），然后综合考虑给出明确推荐。",
            "",
            "要求：",
            "1. 不要列举每个策略的指标数值，直接分析优劣势和trade-off",
            "2. 明确指出推荐哪一个策略，并说明核心依据",
            "3. 指出该策略的最大潜在风险",
            "",
            "控制在 180 字以内，语言干练，一段式输出，适合向业务负责人汇报。",
        ])
        return "\n".join(lines)

    def analyze(self, baseline: Dict, strategies: List[Dict]) -> Optional[str]:
        if not self.enabled:
            print("[LLM建议] 未配置 DEEPSEEK_API_KEY 或未安装 openai 库，跳过")
            return None

        prompt = self._build_prompt(baseline, strategies)
        print("[LLM建议] 正在请求 DeepSeek 分析...")

        try:
            client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com/v1")
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": (
                        "你是一个严格的文本复述助手。你的唯一职责是根据用户给出的精确数字进行描述和比较。\n"
                        "绝对禁止：引入外部知识、行业常识、预训练数据；自行计算、推导、估算、放大或缩小数字。\n"
                        "用户给出的百分比就是最终结论，例如 +0.3% 仅表示增长千分之三，绝非 30% 或 3 倍。"
                    )},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=350,
                timeout=30,
            )
            advice = response.choices[0].message.content.strip()
            print("[LLM建议] 分析完成")
            return advice
        except Exception as e:
            print(f"[LLM建议] API 调用失败: {e}")
            return None


# ============================================================
# 5. 报告生成器
# ============================================================

class ReportGenerator:
    def __init__(self, loader: DataLoader, baseline: BaselinePredictor,
                 simulator: Optional[IncentiveSimulator] = None,
                 top_strategies: Optional[List[Dict]] = None,
                 results_df: Optional[pd.DataFrame] = None,
                 llm_advice: Optional[str] = None):
        self.loader = loader
        self.baseline = baseline
        self.simulator = simulator
        self.top_strategies = top_strategies or []
        self.results_df = results_df
        self.llm_advice = llm_advice

    def generate(self) -> str:
        lines = []
        lines.append("# 快手达人激励策略预测报告")
        lines.append(f"\n生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        lines.append("\n---\n")
        lines.append(self._build_status())
        lines.append(self._build_scheme_structure())
        lines.append(self.baseline.format_report())
        if self.top_strategies:
            lines.append(self._build_simulation())
        if self.llm_advice:
            lines.append(self._build_llm_advice())
        if self.results_df is not None and not self.results_df.empty:
            lines.append(self._build_sensitivity())
        if self.top_strategies:
            lines.append(self._build_risks())
        return "\n".join(lines)

    def _build_status(self) -> str:
        lines = []
        lines.append("## 一、业务现状摘要")
        lines.append("")
        lines.append(self.loader.get_layer_summary().to_markdown(index=False))
        lines.append("")

        total_by_layer = self.loader.df.groupby('达人分层')['实际获得激励金额'].sum()
        total_all = total_by_layer.sum()
        gmv_by_layer = self.loader.df.groupby('达人分层')['月度总GMV'].sum()
        scheme_dist = self.loader.df['参与激励方案类型'].value_counts(normalize=True)

        lines.append("**核心洞察：**")
        lines.append(f"- 头部达人（约10%）贡献了 **{gmv_by_layer['头部']/gmv_by_layer.sum():.1%}** 的 GMV，却只拿到 **{total_by_layer['头部']/total_all:.1%}** 的激励资金，预算利用不充分。")
        lines.append(f"- 腰尾部达人合计占资金 **{(total_by_layer['腰部']+total_by_layer['尾部'])/total_all:.1%}**，但 GMV 贡献仅 **{(gmv_by_layer['腰部']+gmv_by_layer['尾部'])/gmv_by_layer.sum():.1%}**，投产效率存在显著优化空间。")
        lines.append(f"- 两种方案覆盖：纯条数激励 **{scheme_dist.get('纯条数激励', 0):.1%}**、纯 GMV 激励 **{scheme_dist.get('纯GMV激励', 0):.1%}**、条数+GMV叠加 **{scheme_dist.get('条数+GMV叠加', 0):.1%}**。")
        lines.append(f"- 尾部达人月均投产比仅 **{self.loader.df[self.loader.df['达人分层']=='尾部']['GMV投产比'].mean():.2f}**，是资金效率最大的拖累项。")
        lines.append("")
        return "\n".join(lines)

    def _build_scheme_structure(self) -> str:
        lines = []
        lines.append("## 二、激励方案阶梯结构")
        lines.append("")
        lines.append("### 方案一：纯条数阶梯激励（保视频数量，侧腰尾部）")
        lines.append("")
        lines.append("| 档位 | 条数门槛 | 奖励金额 | 名额限制 | 排序规则 |")
        lines.append("|------|---------|---------|---------|")
        lines.append("| 高挡 | ≥150 条 | 700 元 | 前 20 人 | 同档按 GMV 倒序 |")
        lines.append("| 中挡 | ≥100 条 | 500 元 | 前 50 人 | 同档按 GMV 倒序 |")
        lines.append("| 基础挡 | ≥50 条 | 300 元 | 前 100 人 | 同档按 GMV 倒序 |")
        lines.append("")
        lines.append("*处理逻辑：高挡优先，已获高挡者不再参与低档竞争。*")
        lines.append("*名额限制：高挡前 20 人、中挡前 50 人、基础挡前 100 人，同档内按 GMV 倒序取前 N 名。*")
        lines.append("")

        lines.append("### 方案二：纯 GMV 分层提成（抓头部质量，按最高档不累进）")
        lines.append("")
        lines.append("| 档位 | GMV 门槛 | 提成比例 | 作用定位 |")
        lines.append("|------|---------|---------|")
        lines.append("| 超高额 | >80,000 元 | 1.0% | 激励头部爆款带货 |")
        lines.append("| 高额 | >20,000 元 | 0.8% | 稳定腰部转化效率 |")
        lines.append("| 达标额 | ≥5,000 元 | 0.5% | 保底基础转化激励 |")
        lines.append("")
        lines.append("*处理逻辑：取最高达标档计算，不累进叠加。*")
        lines.append("*名额限制：无，凡达标者均可获得对应档位的 GMV 提成。*")
        lines.append("")

        lines.append("### 结算规则")
        lines.append("- 两种方案同时开放报名，所有达人可同时参与；")
        lines.append("- 月末结算时，两种方案**可叠加计算**，保留预算硬约束；")
        lines.append("- 月度总预算硬约束 **9.5-10 万元**，仿真中仅保留激励总额落在该区间的策略，超支或不足均直接淘汰。")
        lines.append("")
        return "\n".join(lines)

    def _build_simulation(self) -> str:
        lines = []
        lines.append("## 四、最优激励策略推荐")
        lines.append("")
        lines.append("在满足以下硬约束的前提下筛选：")
        lines.append("- 预测下月视频数总量 ≥ 基线预测 × 90%（保平台 KPI）")
        lines.append("- 尾部达人月流失率 ≤ 15%（保生态体量）")
        lines.append("- 月度总激励支出控制在 9.5-10 万元（预算硬约束）")
        lines.append("")

        base_roi = self.baseline.predict()['整体']['预测投产比']

        for s in self.top_strategies:
            delta_roi = s['预测投产比'] - base_roi
            pc = s['条数门槛系数']
            gc = s['GMV门槛系数']
            rc = s['奖励系数']

            lines.append(f"### Top {s['排名']} 策略参数明细")
            lines.append("")

            # 方案一明细
            lines.append("**方案一：纯条数阶梯激励**")
            lines.append("| 档位 | 条数门槛 | 奖励金额 | 名额限制 |")
            lines.append("|------|---------|---------|---------|")
            qs = s.get('名额系数', 1.0)
            lines.append(f"| 高挡 | ≥{int(150*pc)} 条 | {int(700*rc)} 元 | {max(5, int(20*qs))} 人 |")
            lines.append(f"| 中挡 | ≥{int(100*pc)} 条 | {int(500*rc)} 元 | {max(5, int(50*qs))} 人 |")
            lines.append(f"| 基础挡 | ≥{int(50*pc)} 条 | {int(300*rc)} 元 | {max(5, int(100*qs))} 人 |")
            lines.append("")
            lines.append("*名额限制：同档内按 GMV 倒序取前 N 名，已获高挡者不再参与低档竞争。*")
            lines.append("")

            # 方案二明细
            lines.append("**方案二：纯 GMV 分层提成**")
            lines.append("| 档位 | GMV 门槛 | 提成比例 |")
            lines.append("|------|---------|---------|")
            lines.append(f"| 超高额 | >{int(80000*gc):,} 元 | {0.01*rc:.1%} |")
            lines.append(f"| 高额 | >{int(20000*gc):,} 元 | {0.008*rc:.1%} |")
            lines.append(f"| 达标额 | ≥{int(5000*gc):,} 元 | {0.005*rc:.1%} |")
            lines.append("")
            lines.append("*名额限制：无，凡 GMV 达标者均可按对应档位获得提成。*")
            lines.append("")

            # 名额系数
            lines.append("**名额系数**")
            lines.append(f"- 名额系数：{qs:.1f}（方案一名额 = 基准 × {qs:.1f}）")
            lines.append("")

            lines.append("**预期效果：**")
            lines.append("| 指标 | 基线预测 | 本策略 | 变化 |")
            lines.append("|------|---------|--------|------|")
            base_posts = self.baseline.predict()['整体']['预测视频数总量']
            base_gmv = self.baseline.predict()['整体']['预测GMV总量']
            base_inc = self.baseline.predict()['整体']['预测激励总额']
            lines.append(f"| 视频数总量 | {base_posts:,.0f} | {s['预测视频数总量']:,.0f} | {s['预测视频数总量']/base_posts-1:+.1%} |")
            lines.append(f"| GMV 总量 | {base_gmv:,.0f} 元 | {s['预测GMV总量']:,.0f} 元 | {s['预测GMV总量']/base_gmv-1:+.1%} |")
            lines.append(f"| 激励总额 | {base_inc:,.0f} 元 | {s['预测激励总额']:,.0f} 元 | {s['预测激励总额']/base_inc-1:+.1%} |")
            lines.append(f"| 投产比 | {base_roi:.2f} | **{s['预测投产比']:.2f}** | **{delta_roi:+.2f}** |")
            lines.append(f"| 尾部流失率 | {self.baseline.predict()['尾部']['预测流失率']:.1%} | {s['尾部流失率']:.1%} | {s['尾部流失率']-self.baseline.predict()['尾部']['预测流失率']:+.1%} |")
            base_active = self.baseline.predict()['整体']['预测活跃人数']
            lines.append(f"| 预测活跃人数 | {base_active:,.0f} | {s['预测活跃人数']:,.0f} | {s['预测活跃人数']/base_active-1:+.1%} |")
            lines.append("")
        return "\n".join(lines)

    def _build_sensitivity(self) -> str:
        lines = []
        lines.append("## 五、敏感性分析")
        lines.append("")
        sens = self.simulator.sensitivity_analysis(self.results_df)
        if not sens:
            return ""
        lines.append("参数变动对投产比的影响程度（标准差越大，越敏感）：")
        lines.append("")
        lines.append("| 参数 | 投产比标准差 | 解读 |")
        lines.append("|------|-------------|------|")
        for param, std in sorted(sens.items(), key=lambda x: x[1], reverse=True):
            if std > 0.3:
                interpret = "高敏感，小幅调整即显著影响资金效率"
            elif std > 0.15:
                interpret = "中等敏感，需结合业务节奏微调"
            else:
                interpret = "低敏感，可优先固定此参数"
            lines.append(f"| {param} | {std:.3f} | {interpret} |")
        lines.append("")
        return "\n".join(lines)

    def _build_llm_advice(self) -> str:
        lines = []
        lines.append("## 五、策略选择建议")
        lines.append("")
        lines.append("> 以下建议由 LLM 基于策略参数与相对变化指标生成，未接触原始业务数据。")
        lines.append("")
        lines.append(self.llm_advice)
        lines.append("")
        return "\n".join(lines)

    def _build_risks(self) -> str:
        lines = []
        lines.append("## 六、风险提示与落地建议")
        lines.append("")
        best = self.top_strategies[0] if self.top_strategies else None
        if best is None:
            return ""

        base = self.baseline.predict()
        post_change = best['预测视频数总量'] / base['整体']['预测视频数总量'] - 1
        churn_change = best['尾部流失率'] - base['尾部']['预测流失率']

        lines.append("**主要 trade-off：**")
        lines.append("")
        if post_change < -0.05:
            lines.append(f"- 视频数风险：最优策略下视频数预计下降 {abs(post_change):.1%}，需与平台 KPI 考核口径确认是否可接受。")
        else:
            lines.append(f"- 视频数可控：最优策略下视频数预计变化 {post_change:+.1%}，基本稳定。")

        if churn_change > 0.05:
            lines.append(f"- 尾部流失风险：尾部流失率预计上升 {churn_change:.1%}，建议配套推出尾部达人帮扶计划。")
        else:
            lines.append(f"- 尾部流失可控：尾部流失率预计变化 {churn_change:+.1%}，处于安全区间。")

        lines.append("")
        lines.append("**落地建议：**")
        lines.append("1. **灰度实验**：先在小范围城市（如杭州或南京）试点 Top 1 策略，跑 1 个月观察真实达人反馈。")
        lines.append("2. **动态调参**：不要一次性把奖励系数压到最低，建议分 2-3 个月逐步下调，给达人缓冲期。")
        lines.append("3. **方案定位**：纯条数激励保基本盘视频数、纯 GMV 提成抓头部内容质量，两者可叠加形成互补，共同激励腰部向上突破。")
        lines.append("4. **监控预警**：策略上线后，每日跟踪「尾部活跃人数」「整体视频数」「预算消耗进度」，若连续 7 日跌破基线 90% 或预算超支，立即回滚。")
        lines.append("")
        return "\n".join(lines)


# ============================================================
# 5. 雷达图生成器
# ============================================================

class ImprovementBarChartGenerator:
    """生成基线预测 vs Top 策略的六维相对提升条形图"""

    CATEGORIES = ['视频数总量', 'GMV总量', '投产比', '尾部留存率', '活跃人数', '激励总额']

    def _calc_improvements(self, baseline: Dict, strategies: List[Dict]) -> List[Dict]:
        """
        计算各策略相对于基线的提升百分比
        返回: [{维度名: 提升百分比, ...}, ...]
        """
        b = baseline['整体']
        base_tail_churn = baseline['尾部']['预测流失率']

        base_raw = {
            '视频数总量': b['预测视频数总量'],
            'GMV总量': b['预测GMV总量'],
            '投产比': b['预测投产比'],
            '尾部留存率': 1.0 - base_tail_churn,
            '活跃人数': b['预测活跃人数'],
            '激励总额': b['预测激励总额'],
        }

        improvements = []
        for s in strategies:
            raw = {
                '视频数总量': s['预测视频数总量'],
                'GMV总量': s['预测GMV总量'],
                '投产比': s['预测投产比'],
                '尾部留存率': 1.0 - s['尾部流失率'],
                '活跃人数': s['预测活跃人数'],
                '激励总额': s['预测激励总额'],
            }
            imp = {}
            for cat in self.CATEGORIES:
                base_val = base_raw[cat]
                if base_val and base_val != 0:
                    imp[cat] = (raw[cat] - base_val) / base_val * 100
                else:
                    imp[cat] = 0.0
            improvements.append(imp)
        return improvements

    def generate(self, baseline: Dict, strategies: List[Dict], output_path: str = 'improvement_chart.png'):
        improvements = self._calc_improvements(baseline, strategies)

        x = np.arange(len(self.CATEGORIES))
        width = 0.22
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        fig, ax = plt.subplots(figsize=(12, 7))

        for i, (s, imp) in enumerate(zip(strategies, improvements)):
            vals = [imp[cat] for cat in self.CATEGORIES]
            offset = width * (i - 1)
            bars = ax.bar(
                x + offset, vals, width,
                label=f"Top {s['排名']}",
                color=colors[i],
                edgecolor='white', linewidth=0.5,
            )

            # 在柱子上方或下方标注百分比
            for bar, val in zip(bars, vals):
                height = bar.get_height()
                ax.annotate(
                    f'{val:+.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4 if height >= 0 else -11),
                    textcoords="offset points",
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=9, fontweight='bold',
                    color=colors[i],
                )

        # 基线 0% 参考线
        ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(self.CATEGORIES, fontsize=12)
        ax.set_ylabel('相对基线的变化 (%)', fontsize=13)
        ax.set_title(
            '基线预测 vs 最优策略 — 六维相对提升对比\n（基线 = 0%，正值表示优于基线）',
            fontsize=15, pad=20, fontweight='bold',
        )
        ax.legend(loc='upper left', frameon=True, fontsize=11)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)

        # 自动调整 Y 轴边界，给标注留空间
        all_vals = []
        for imp in improvements:
            all_vals.extend([imp[cat] for cat in self.CATEGORIES])
        ymin, ymax = min(all_vals, default=0), max(all_vals, default=0)
        margin = max(abs(ymin), abs(ymax)) * 0.15 + 2
        ax.set_ylim(ymin - margin, ymax + margin)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n[提升对比图] 已保存至: {output_path}")


class ComparisonTableGenerator:
    """生成基线 vs Top3 策略的六维绝对数值对比表格图"""

    CATEGORIES = ['视频数总量', 'GMV总量', '激励总额', '投产比', '尾部流失率', '预测活跃人数']

    def generate(self, baseline: Dict, strategies: List[Dict], output_path: str = 'comparison_table.png'):
        b = baseline['整体']
        base_tail_churn = baseline['尾部']['预测流失率']

        # 准备表格数据（每行对应一个策略）
        rows = []

        # 基线行
        rows.append([
            '基线预测',
            f"{b['预测视频数总量']:,.0f}",
            f"{b['预测GMV总量']:,.0f} 元",
            f"{b['预测激励总额']:,.0f} 元",
            f"{b['预测投产比']:.2f}",
            f"{base_tail_churn:.1%}",
            f"{b['预测活跃人数']:,.0f}",
        ])

        for s in strategies:
            rows.append([
                f"Top {s['排名']}",
                f"{s['预测视频数总量']:,.0f}",
                f"{s['预测GMV总量']:,.0f} 元",
                f"{s['预测激励总额']:,.0f} 元",
                f"{s['预测投产比']:.2f}",
                f"{s['尾部流失率']:.1%}",
                f"{s['预测活跃人数']:,.0f}",
            ])

        fig, ax = plt.subplots(figsize=(14, 4.2))
        ax.axis('off')
        ax.axis('tight')

        table = ax.table(
            cellText=rows,
            colLabels=['策略'] + self.CATEGORIES,
            loc='center',
            cellLoc='center',
        )

        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2.5)

        # 配色
        header_color = '#4A4A4A'
        base_color = '#F0F0F0'
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        light_colors = ['#FFEBEB', '#E8F8F5', '#E8F4F8']

        n_cols = len(self.CATEGORIES) + 1

        # 表头样式
        for j in range(n_cols):
            cell = table[(0, j)]
            cell.set_facecolor(header_color)
            cell.set_text_props(color='white', fontweight='bold', fontsize=11)
            cell.set_edgecolor('white')
            cell.set_linewidth(1.5)

        # 数据行样式
        row_bg = [base_color] + light_colors[:len(strategies)]
        for i, bg in enumerate(row_bg):
            for j in range(n_cols):
                cell = table[(i + 1, j)]
                cell.set_facecolor(bg)
                cell.set_edgecolor('white')
                cell.set_linewidth(1.5)
                if j == 0:
                    cell.set_text_props(fontweight='bold', fontsize=11)
                    # 策略名颜色
                    if i == 0:
                        cell.set_text_props(color='#666666')
                    else:
                        cell.set_text_props(color=colors[i - 1])
                else:
                    cell.set_text_props(fontsize=10.5)

        plt.title(
            '基线预测 vs 最优策略 — 六维核心指标绝对值对比',
            fontsize=14, fontweight='bold', pad=20,
        )
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"\n[对比表格图] 已保存至: {output_path}")


class ReportDashboardGenerator:
    """生成整合汇报看板：条形图 + 表格 + Top1策略 + 落地建议"""

    BAR_CATEGORIES = ['视频数总量', 'GMV总量', '激励总额', '投产比', '尾部留存率', '活跃人数']
    TABLE_CATEGORIES = ['视频数总量', 'GMV总量', '激励总额', '投产比', '尾部流失率', '预测活跃人数']

    def _calc_bar_improvements(self, baseline: Dict, strategies: List[Dict]) -> List[Dict]:
        b = baseline['整体']
        base_tail_churn = baseline['尾部']['预测流失率']
        base_raw = {
            '视频数总量': b['预测视频数总量'],
            'GMV总量': b['预测GMV总量'],
            '激励总额': b['预测激励总额'],
            '投产比': b['预测投产比'],
            '尾部留存率': 1.0 - base_tail_churn,
            '活跃人数': b['预测活跃人数'],
        }
        improvements = []
        for s in strategies:
            raw = {
                '视频数总量': s['预测视频数总量'],
                'GMV总量': s['预测GMV总量'],
                '激励总额': s['预测激励总额'],
                '投产比': s['预测投产比'],
                '尾部留存率': 1.0 - s['尾部流失率'],
                '活跃人数': s['预测活跃人数'],
            }
            imp = {}
            for cat in self.BAR_CATEGORIES:
                bv = base_raw[cat]
                imp[cat] = (raw[cat] - bv) / bv * 100 if bv and bv != 0 else 0.0
            improvements.append(imp)
        return improvements

    def _draw_bar_chart(self, ax, baseline: Dict, strategies: List[Dict]):
        improvements = self._calc_bar_improvements(baseline, strategies)
        x = np.arange(len(self.BAR_CATEGORIES))
        width = 0.22
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

        for i, (s, imp) in enumerate(zip(strategies, improvements)):
            vals = [imp[cat] for cat in self.BAR_CATEGORIES]
            offset = width * (i - 1)
            bars = ax.bar(x + offset, vals, width, label=f"Top {s['排名']}", color=colors[i], edgecolor='white', linewidth=0.5)
            for bar, val in zip(bars, vals):
                height = bar.get_height()
                ax.annotate(f'{val:+.1f}%', xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4 if height >= 0 else -11), textcoords="offset points",
                            ha='center', va='bottom' if height >= 0 else 'top', fontsize=8, fontweight='bold', color=colors[i])

        ax.axhline(y=0, color='#333333', linestyle='-', linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(self.BAR_CATEGORIES, fontsize=10)
        ax.set_ylabel('相对基线的变化 (%)', fontsize=11)
        ax.set_title('六维相对提升对比', fontsize=13, fontweight='bold', pad=10)
        ax.legend(loc='upper left', frameon=True, fontsize=9)
        ax.yaxis.grid(True, linestyle='--', alpha=0.6)
        ax.set_axisbelow(True)
        all_vals = []
        for imp in improvements:
            all_vals.extend([imp[cat] for cat in self.BAR_CATEGORIES])
        ymin, ymax = min(all_vals, default=0), max(all_vals, default=0)
        margin = max(abs(ymin), abs(ymax)) * 0.15 + 2
        ax.set_ylim(ymin - margin, ymax + margin)

    def _draw_table(self, ax, baseline: Dict, strategies: List[Dict]):
        b = baseline['整体']
        base_tail_churn = baseline['尾部']['预测流失率']
        rows = []
        rows.append([
            '基线预测',
            f"{b['预测视频数总量']:,.0f}",
            f"{b['预测GMV总量']:,.0f} 元",
            f"{b['预测激励总额']:,.0f} 元",
            f"{b['预测投产比']:.2f}",
            f"{base_tail_churn:.1%}",
            f"{b['预测活跃人数']:,.0f}",
        ])
        for s in strategies:
            rows.append([
                f"Top {s['排名']}",
                f"{s['预测视频数总量']:,.0f}",
                f"{s['预测GMV总量']:,.0f} 元",
                f"{s['预测激励总额']:,.0f} 元",
                f"{s['预测投产比']:.2f}",
                f"{s['尾部流失率']:.1%}",
                f"{s['预测活跃人数']:,.0f}",
            ])

        ax.axis('off')
        ax.axis('tight')
        table = ax.table(cellText=rows, colLabels=['策略'] + self.TABLE_CATEGORIES, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9.5)
        table.scale(1, 2.0)

        header_color = '#4A4A4A'
        base_color = '#F0F0F0'
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        light_colors = ['#FFEBEB', '#E8F8F5', '#E8F4F8']
        n_cols = len(self.TABLE_CATEGORIES) + 1

        for j in range(n_cols):
            cell = table[(0, j)]
            cell.set_facecolor(header_color)
            cell.set_text_props(color='white', fontweight='bold', fontsize=9.5)
            cell.set_edgecolor('white')
            cell.set_linewidth(1.2)

        row_bg = [base_color] + light_colors[:len(strategies)]
        for i, bg in enumerate(row_bg):
            for j in range(n_cols):
                cell = table[(i + 1, j)]
                cell.set_facecolor(bg)
                cell.set_edgecolor('white')
                cell.set_linewidth(1.2)
                if j == 0:
                    cell.set_text_props(fontweight='bold', fontsize=9.5)
                    if i == 0:
                        cell.set_text_props(color='#666666')
                    else:
                        cell.set_text_props(color=colors[i - 1])

    def _draw_strategy_card(self, ax, strategy: Dict, color: str, light_color: str, rank_label: str):
        ax.axis('off')
        if not strategy:
            return

        pc = strategy['条数门槛系数']
        rc = strategy['奖励系数']
        gc = strategy['GMV门槛系数']

        # 背景卡片
        rect = plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                              facecolor=light_color, edgecolor=color, linewidth=2.5,
                              zorder=0, joinstyle='round')
        rect.set_capstyle('round')
        ax.add_patch(rect)

        # 标题栏背景
        title_rect = plt.Rectangle((0.02, 0.88), 0.96, 0.10, transform=ax.transAxes,
                                    facecolor=color, edgecolor='none', zorder=1, alpha=0.9)
        ax.add_patch(title_rect)

        # 标题文字
        ax.text(0.50, 0.93, f"Top {strategy['排名']} {rank_label}", transform=ax.transAxes,
                fontsize=13, fontweight='bold', color='white', ha='center', va='center', zorder=2)

        qs = strategy.get('名额系数', 1.0)

        # 内容文字
        content = (
            f"方案一 · 纯条数阶梯\n"
            f"  高挡  ≥{150*pc:.0f}条  |  {700*rc:.0f}元\n"
            f"  中挡  ≥{100*pc:.0f}条  |  {500*rc:.0f}元\n"
            f"  基础挡 ≥{50*pc:.0f}条   |  {300*rc:.0f}元\n"
            f"  名额限制：高挡前{max(5, int(20*qs))}人 / 中挡前{max(5, int(50*qs))}人 / 基础挡前{max(5, int(100*qs))}人\n"
            f"\n"
            f"方案二 · GMV提成\n"
            f"  超高额 >{80000*gc:,.0f}元 | {0.01*rc:.1%}\n"
            f"  高额   >{20000*gc:,.0f}元 | {0.008*rc:.1%}\n"
            f"  达标额 ≥{5000*gc:,.0f}元  | {0.005*rc:.1%}\n"
            f"  名额限制：无，凡达标者均可获得提成\n"
        )

        ax.text(0.06, 0.84, content, transform=ax.transAxes, fontsize=9.5,
                verticalalignment='top', linespacing=1.35, color='#333333')

    def _draw_llm_advice(self, ax, advice_text: Optional[str]):
        ax.axis('off')

        # 卡片整体缩窄居中，与落地建议模块标题等高、同色
        card_x, card_w = 0.00, 1.00
        rect = plt.Rectangle((card_x, 0.02), card_w, 0.96, transform=ax.transAxes,
                              facecolor='#F0F7FF', edgecolor='#2E6EB5', linewidth=2,
                              zorder=0, joinstyle='round')
        ax.add_patch(rect)

        title_rect = plt.Rectangle((card_x, 0.82), card_w, 0.16, transform=ax.transAxes,
                                    facecolor='#2E6EB5', edgecolor='none', zorder=1)
        ax.add_patch(title_rect)

        ax.text(0.50, 0.90, "策略选择建议", transform=ax.transAxes,
                fontsize=14, fontweight='bold', color='white', ha='center', va='center', zorder=2)

        if advice_text:
            display_text = advice_text.replace('## ', '').replace('### ', '').replace('**', '')
            wrapped_text = textwrap.fill(display_text, width=81)
            ax.text(0.03, 0.74, wrapped_text, transform=ax.transAxes,
                    fontsize=12, verticalalignment='top', linespacing=1.4,
                    color='#1a1a1a')
        else:
            ax.text(0.50, 0.45, "未配置 DEEPSEEK_API_KEY 或调用失败，LLM 建议不可用",
                    transform=ax.transAxes, fontsize=11, ha='center', va='center',
                    color='#888888', style='italic')

    def _draw_suggestions(self, ax):
        ax.axis('off')

        # 背景卡片
        rect = plt.Rectangle((0.01, 0.02), 0.98, 0.96, transform=ax.transAxes,
                              facecolor='#FAFAFA', edgecolor='#4A4A4A', linewidth=2,
                              zorder=0, joinstyle='round')
        ax.add_patch(rect)

        # 标题栏背景
        title_rect = plt.Rectangle((0.01, 0.82), 0.98, 0.16, transform=ax.transAxes,
                                    facecolor='#4A4A4A', edgecolor='none', zorder=1)
        ax.add_patch(title_rect)

        ax.text(0.50, 0.90, "落地建议与风险提示", transform=ax.transAxes,
                fontsize=14, fontweight='bold', color='white', ha='center', va='center', zorder=2)

        # 四条建议，分栏展示 (2x2)
        suggestions = [
            ("1  灰度实验", "先在1-2个城市试点1个月，观察真实达人反馈后再全量推广。", '#FFEBEB', '#FF6B6B'),
            ("2  动态调参", "分2-3个月逐步下调奖励系数，给达人缓冲期，避免断崖式流失。", '#E8F8F5', '#4ECDC4'),
            ("3  方案定位", "条数激励保基本盘、GMV提成抓头部质量，两者可叠加互补。", '#E8F4F8', '#45B7D1'),
            ("4  监控预警", "每日跟踪尾部活跃人数、整体视频数、预算消耗，若连续7日跌破基线90%或预算超支，立即回滚。", '#F5F5F5', '#888888'),
        ]

        positions = [(0.06, 0.72), (0.52, 0.72), (0.06, 0.38), (0.52, 0.38)]
        for (title, desc, bg, border), (x, y) in zip(suggestions, positions):
            # 小卡片背景
            card = plt.Rectangle((x, y-0.28), 0.42, 0.32, transform=ax.transAxes,
                                  facecolor=bg, edgecolor=border, linewidth=1.8, zorder=1)
            ax.add_patch(card)

            ax.text(x+0.02, y-0.02, title, transform=ax.transAxes, fontsize=10.5,
                    fontweight='bold', color=border, verticalalignment='top', zorder=2)
            ax.text(x+0.02, y-0.08, desc, transform=ax.transAxes, fontsize=9.5,
                    color='#444444', verticalalignment='top', linespacing=1.4, zorder=2)

    def generate(self, baseline: Dict, strategies: List[Dict],
                 output_path: str = 'dashboard.png',
                 llm_advice: Optional[str] = None):
        import matplotlib.gridspec as gridspec

        # LLM 建议区域高度调高到 0.85，使标题栏实际像素高度与落地建议一致
        llm_height = 0.85

        base_fig_height = 24
        extra_height = (llm_height - 0.9) * 2.5
        fig_height = base_fig_height + extra_height
        fig = plt.figure(figsize=(18, fig_height))
        gs = gridspec.GridSpec(5, 3, height_ratios=[2.2, llm_height, 1.3, 1.8, 1.3], hspace=0.30, wspace=0.18)

        ax1 = fig.add_subplot(gs[0, :])
        self._draw_bar_chart(ax1, baseline, strategies)

        ax_llm = fig.add_subplot(gs[1, :])
        self._draw_llm_advice(ax_llm, llm_advice)

        ax2 = fig.add_subplot(gs[2, :])
        self._draw_table(ax2, baseline, strategies)

        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        light_colors = ['#FFF0F0', '#F0FAF8', '#F0F7FA']
        rank_labels = ['最优策略', '次优策略', '备选策略']

        for i in range(min(3, len(strategies))):
            ax = fig.add_subplot(gs[3, i])
            self._draw_strategy_card(ax, strategies[i], colors[i], light_colors[i], rank_labels[i])

        ax_sug = fig.add_subplot(gs[4, :])
        self._draw_suggestions(ax_sug)

        fig.suptitle('快手达人激励策略优化方案汇报', fontsize=18, fontweight='bold', y=0.98)

        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"\n[汇报看板] 已保存至: {output_path}")


# ============================================================
# 6. 主控 Agent
# ============================================================

class KuaishouIncentiveAgent:
    def __init__(self):
        self.loader = DataLoader()
        self.baseline: Optional[BaselinePredictor] = None
        self.simulator: Optional[IncentiveSimulator] = None

    def run(self, baseline_only: bool = False, simulate: bool = False,
            top_n: int = 3, output_file: Optional[str] = None,
            chart_file: Optional[str] = None,
            table_file: Optional[str] = None,
            dashboard_file: Optional[str] = None) -> str:

        if not self.loader.load():
            return "[错误] 数据加载失败"

        print("\n" + "="*60)
        print("  快手达人激励策略预测 Agent（v2）")
        print("="*60)

        self.baseline = BaselinePredictor(self.loader.df, self.loader.months)
        baseline_pred = self.baseline.predict()
        print("\n[基线预测] 完成")
        for layer in LAYERS + ['整体']:
            p = baseline_pred[layer]
            if layer == '整体':
                print(f"  {layer}: 视频数 {p['预测视频数总量']:,.0f}, GMV {p['预测GMV总量']:,.0f}, 投产比 {p['预测投产比']:.2f}")
            else:
                print(f"  {layer}: 活跃 {p['预测活跃人数']:,.0f} 人, 流失率 {p['预测流失率']:.1%}")

        top_strategies = None
        results_df = None
        llm_advice = None

        if simulate:
            self.simulator = IncentiveSimulator(self.loader.df, self.loader.months, baseline_pred)
            top_strategies, results_df = self.simulator.run(n_runs=3, top_n=top_n)

            if top_strategies:
                print(f"\n[最优策略 Top {top_n}]")
                for s in top_strategies:
                    print(f"  Rank {s['排名']}: 条数系数={s['条数门槛系数']}, GMV系数={s['GMV门槛系数']}, 奖励系数={s['奖励系数']}, "
                          f"投产比={s['预测投产比']:.2f}, 尾部流失={s['尾部流失率']:.1%}")

                # DeepSeek LLM 分析
                advisor = LLMAdvisor()
                llm_advice = advisor.analyze(baseline_pred, top_strategies)
            else:
                print("\n[警告] 未找到满足约束的最优策略")

        # 生成相对提升条形图
        if chart_file and simulate and top_strategies:
            ImprovementBarChartGenerator().generate(
                baseline_pred, top_strategies, output_path=chart_file
            )

        # 生成绝对数值对比表格图
        if table_file and simulate and top_strategies:
            ComparisonTableGenerator().generate(
                baseline_pred, top_strategies, output_path=table_file
            )

        # 生成整合汇报看板
        if dashboard_file and simulate and top_strategies:
            ReportDashboardGenerator().generate(
                baseline_pred, top_strategies, output_path=dashboard_file,
                llm_advice=llm_advice
            )

        report_gen = ReportGenerator(
            self.loader, self.baseline,
            self.simulator, top_strategies, results_df,
            llm_advice=llm_advice
        )
        report = report_gen.generate()

        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"\n[完成] 报告已保存至: {output_file}")

        if baseline_only:
            lines = report.split('\n')
            baseline_lines = []
            capture = False
            for line in lines:
                if '## 一、' in line:
                    capture = True
                if capture:
                    baseline_lines.append(line)
                if '## 四、' in line:
                    break
            report = '\n'.join(baseline_lines)

        print("\n" + "="*60)
        print("  分析完成")
        print("="*60)
        return report


# ============================================================
# 6. CLI 入口
# ============================================================

def main():
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    parser = argparse.ArgumentParser(description="快手达人激励策略预测 Agent（v2）")
    parser.add_argument("--baseline", action="store_true", help="仅做基线预测")
    parser.add_argument("--simulate", action="store_true", help="运行激励参数仿真寻优")
    parser.add_argument("--report", type=str, default=None, help="输出报告到指定文件")
    parser.add_argument("--top-n", type=int, default=3, help="推荐 Top N 策略（默认 3）")
    parser.add_argument("--chart", type=str, default=None, help="生成相对提升条形图 PNG（如 chart.png）")
    parser.add_argument("--table", type=str, default=None, help="生成绝对数值对比表格图 PNG（如 table.png）")
    parser.add_argument("--dashboard", type=str, default=None, help="生成整合汇报看板 PNG（如 dashboard.png）")

    args = parser.parse_args()

    if not args.baseline and not args.simulate:
        print("请指定模式: --baseline 或 --simulate")
        sys.exit(1)

    agent = KuaishouIncentiveAgent()
    report = agent.run(
        baseline_only=args.baseline,
        simulate=args.simulate,
        top_n=args.top_n,
        output_file=args.report,
        chart_file=args.chart,
        table_file=args.table,
        dashboard_file=args.dashboard,
    )
    print("\n" + report)


if __name__ == "__main__":
    main()
