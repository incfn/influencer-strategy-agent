#!/usr/bin/env python3
# -*- coding: utf-8 -*"
"""
快手本地生活达人激励数据合成器（v3）
核心升级：
1. 达人池动态流动（每月自然流失 + 外部流入）
2. 两种阶梯式激励方案（方案1带名额限制、方案2比例提成）
3. 月度总预算硬约束 10 万元
4. 最终结算两种方案可叠加计算
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

np.random.seed(42)

# ============================================================
# 1. 全局参数与业务规则
# ============================================================

MONTHS = ['2025-10', '2025-11', '2025-12', '2026-01', '2026-02', '2026-03']
TOTAL_BUDGET = 100_000   # 月度总预算上限 10 万元
MIN_BUDGET = 95_000      # 月度总预算下限 9.5 万元（保证激励力度）

BUDGET_RATIO = {'头部': 0.40, '腰部': 0.30, '尾部': 0.30}

# 方案1：纯条数阶梯激励（带名额限制，同条数按GMV倒序）
# 规则：高挡优先，已获高挡者不再参与低档竞争
SCHEME1_RULES = [
    {'name': '150条档', 'post_th': 150, 'reward': 700, 'quota': 20},
    {'name': '100条档', 'post_th': 100, 'reward': 500, 'quota': 50},
    {'name': '50条档',  'post_th': 50,  'reward': 300, 'quota': 100},
]

# 方案2：纯GMV分层提成（取最高档，不累进）
# 规则：GMV > 80000 -> 1%; > 20000 -> 0.8%; >= 5000 -> 0.5%
SCHEME2_RULES = [
    {'name': '超高额', 'gmv_th': 80000, 'rate': 0.010},
    {'name': '高额',   'gmv_th': 20000, 'rate': 0.008},
    {'name': '达标额', 'gmv_th': 5000,  'rate': 0.005},
]

# ============================================================
# 2. 达人对象与池管理
# ============================================================

class Creator:
    _id_counter = 0

    def __init__(self, layer: str, is_newcomer: bool = False):
        Creator._id_counter += 1
        self.id = f"KOL_{Creator._id_counter:03d}"
        self.layer = layer
        self.diligence = np.random.beta(2, 2)
        self.conversion = np.random.beta(2, 2)
        self.is_newcomer = is_newcomer
        self.age_months = 0  # 在平台活跃月数

        # 退出敏感度
        if layer == '头部':
            self.churn_sensitivity = np.random.uniform(0.3, 0.6)
        elif layer == '腰部':
            self.churn_sensitivity = np.random.uniform(0.5, 0.8)
        else:
            self.churn_sensitivity = np.random.uniform(0.7, 1.0)

    def generate_posts(self) -> int:
        """生成本月视频条数"""
        noise = np.random.normal(0, 0.05)
        newbie_factor = 0.85 if (self.is_newcomer and self.age_months == 0) else 1.0
        if self.layer == '头部':
            base = 320
            posts = int(base * (0.7 + 0.6 * self.diligence) * newbie_factor * (1 + noise))
        elif self.layer == '腰部':
            base = 210
            posts = int(base * (0.75 + 0.5 * self.diligence) * newbie_factor * (1 + noise))
        else:
            base = 160
            posts = int(base * (0.75 + 0.5 * self.diligence) * newbie_factor * (1 + noise))
        return max(20, min(posts, 450))

    def generate_gmv(self, posts: int) -> float:
        """生成本月GMV"""
        noise = np.random.normal(0, 0.08)
        newbie_factor = 0.70 if (self.is_newcomer and self.age_months == 0) else 1.0
        if self.layer == '头部':
            base = 80000
            gmv = base * (0.5 + 1.0 * self.conversion) * newbie_factor * (posts / 300) ** 0.7 * (1 + noise)
        elif self.layer == '腰部':
            base = 5000
            gmv = base * (0.5 + 1.0 * self.conversion) * newbie_factor * (posts / 200) ** 0.6 * (1 + noise)
        else:
            base = 500
            gmv = base * (0.5 + 1.0 * self.conversion) * newbie_factor * (posts / 150) ** 0.5 * (1 + noise)
        return max(0, gmv)


# ============================================================
# 3. 激励计算引擎
# ============================================================

def calc_scheme1(creators_data: List[Dict]) -> Dict[str, float]:
    """
    方案1：纯条数阶梯，带名额限制
    高挡优先，同挡内按GMV倒序取前N名
    返回: {creator_id: reward}
    """
    rewards = {c['id']: 0.0 for c in creators_data}
    assigned = set()

    for rule in SCHEME1_RULES:
        eligible = [
            c for c in creators_data
            if c['posts'] >= rule['post_th'] and c['id'] not in assigned
        ]
        # 同条数按GMV倒序（实际直接按GMV倒序即可，高GMV优先）
        eligible.sort(key=lambda x: x['gmv'], reverse=True)
        for c in eligible[:rule['quota']]:
            rewards[c['id']] = rule['reward']
            assigned.add(c['id'])

    return rewards


def calc_scheme2(creators_data: List[Dict]) -> Dict[str, float]:
    """
    方案2：纯GMV分层提成，取最高档不累进
    返回: {creator_id: reward}
    """
    rewards = {}
    for c in creators_data:
        gmv = c['gmv']
        reward = 0.0
        for rule in SCHEME2_RULES:
            if gmv > rule['gmv_th']:
                reward = gmv * rule['rate']
                break  # 高挡优先
            elif gmv >= rule['gmv_th']:
                reward = gmv * rule['rate']
                break
        rewards[c['id']] = reward
    return rewards


def calc_all_incentives(creators_data: List[Dict]) -> Dict[str, Tuple[str, float]]:
    """
    计算两种方案，可叠加
    返回: {creator_id: (方案类型, 最终激励金额)}
    """
    s1 = calc_scheme1(creators_data)
    s2 = calc_scheme2(creators_data)

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


# ============================================================
# 4. 月度仿真主循环
# ============================================================

def simulate_months() -> pd.DataFrame:
    # 初始达人池
    initial_counts = {'头部': 30, '腰部': 90, '尾部': 180}
    active_creators: List[Creator] = []
    for layer, count in initial_counts.items():
        for _ in range(count):
            active_creators.append(Creator(layer))

    records: List[Dict] = []

    for month in MONTHS:
        # 4.1 生成本月行为数据
        month_data = []
        for creator in active_creators:
            posts = creator.generate_posts()
            gmv = creator.generate_gmv(posts)
            month_data.append({
                'id': creator.id,
                'layer': creator.layer,
                'posts': posts,
                'gmv': gmv,
                'creator_obj': creator,
            })

        # 4.2 计算激励（两种方案可叠加）
        incentives = calc_all_incentives(month_data)

        # 4.3 预算硬约束：控制在 9.5-10 万元区间，超支等比例压缩
        total_raw = sum(v[1] for v in incentives.values())
        compress_factor = 1.0
        if total_raw > TOTAL_BUDGET:
            compress_factor = TOTAL_BUDGET / total_raw
        elif total_raw < MIN_BUDGET:
            # 若低于下限，适度抬升（模拟平台追加预算或降低压缩）
            compress_factor = min(1.2, MIN_BUDGET / total_raw)

        # 4.4 退出决策 + 记录
        new_active = []
        layer_stats = {layer: {'count': 0, 'churned': 0, 'gmv': 0, 'incentive': 0} for layer in ['头部', '腰部', '尾部']}

        for entry in month_data:
            creator = entry['creator_obj']
            layer = entry['layer']
            posts = entry['posts']
            gmv = entry['gmv']
            scheme, raw_reward = incentives[entry['id']]
            actual_reward = raw_reward * compress_factor

            layer_stats[layer]['count'] += 1
            layer_stats[layer]['gmv'] += gmv
            layer_stats[layer]['incentive'] += actual_reward

            # 退出决策
            base_churn = {'头部': 0.015, '腰部': 0.04, '尾部': 0.07}[layer]
            expected = {'头部': 1000, '腰部': 600, '尾部': 350}[layer]
            shortfall = max(0, (expected - actual_reward) / expected) if expected > 0 else 0
            churn_prob = base_churn + creator.churn_sensitivity * shortfall * 0.12
            churn_prob = min(churn_prob, 0.25)

            # 新手保护期
            if creator.age_months == 0 and creator.is_newcomer:
                churn_prob *= 0.5

            will_churn = np.random.random() < churn_prob
            if will_churn:
                layer_stats[layer]['churned'] += 1

            records.append({
                '月度': month,
                '达人分层': layer,
                '达人ID': entry['id'],
                '月度视频条数': posts,
                '月度总GMV': round(gmv, 2),
                '参与激励方案类型': scheme,
                '实际获得激励金额': round(actual_reward, 2),
                '是否当月退出': will_churn,
            })

            if not will_churn:
                creator.age_months += 1
                new_active.append(creator)

        # 4.5 新达人流入（每月 5-15 人）
        inflow = np.random.randint(5, 16)
        for _ in range(inflow):
            layer = np.random.choice(['头部', '腰部', '尾部'], p=[0.10, 0.30, 0.60])
            new_creator = Creator(layer, is_newcomer=True)
            new_active.append(new_creator)

        active_creators = new_active

        # 4.6 补充层级别统计指标
        for layer in ['头部', '腰部', '尾部']:
            stats = layer_stats[layer]
            count = stats['count']
            if count == 0:
                continue
            churn_rate = stats['churned'] / count
            layer_budget = TOTAL_BUDGET * BUDGET_RATIO[layer]
            spend_rate = stats['incentive'] / layer_budget if layer_budget > 0 else 0
            roi = stats['gmv'] / stats['incentive'] if stats['incentive'] > 0 else 0

            # 回填到本层本月所有记录
            for r in records:
                if r['月度'] == month and r['达人分层'] == layer:
                    r['达人流失率'] = round(churn_rate, 4)
                    r['资金消耗率'] = round(spend_rate, 4)
                    r['GMV投产比'] = round(roi, 2)

    df = pd.DataFrame(records)
    df = df[[
        '月度', '达人分层', '达人ID', '月度视频条数', '月度总GMV',
        '参与激励方案类型', '实际获得激励金额', '是否当月退出',
        '达人流失率', '资金消耗率', 'GMV投产比'
    ]]
    return df


# ============================================================
# 5. 主入口
# ============================================================

def main():
    print("[快手达人数据合成器 v2] 开始生成...")
    print(f"[规则] 月度总预算区间: {MIN_BUDGET:,}-{TOTAL_BUDGET:,} 元")
    print(f"[规则] 两种阶梯激励方案，可叠加计算")
    print(f"[规则] 达人池动态流动（自然流失 + 外部流入）")

    df = simulate_months()
    print(f"\n[信息] 生成记录: {len(df)} 条")

    # 达人动态统计
    monthly_active = df.groupby('月度')['达人ID'].nunique()
    monthly_inflow = []
    prev_ids = set()
    for month in MONTHS:
        curr_ids = set(df[df['月度'] == month]['达人ID'])
        new_ids = curr_ids - prev_ids
        monthly_inflow.append(len(new_ids))
        prev_ids = curr_ids - set(df[(df['月度'] == month) & (df['是否当月退出'])]['达人ID'])

    print(f"\n[动态达人池] 每月活跃人数: {monthly_active.to_dict()}")
    print(f"[动态达人池] 每月大致流入: {dict(zip(MONTHS, monthly_inflow))}")

    # 业务校验
    print("\n[校验] 业务规则检查")
    for layer in ['头部', '腰部', '尾部']:
        sub = df[df['达人分层'] == layer]
        avg_posts = sub['月度视频条数'].mean()
        avg_gmv = sub['月度总GMV'].mean()
        print(f"  {layer}: 人均视频数 {avg_posts:.0f} 条, 人均 GMV {avg_gmv:,.0f} 元")

    gmv_by_layer = df.groupby('达人分层')['月度总GMV'].mean()
    print(f"\n[校验] GMV 倍数（头部/腰部）: {gmv_by_layer['头部']/gmv_by_layer['腰部']:.1f}x")
    print(f"[校验] GMV 倍数（腰部/尾部）: {gmv_by_layer['腰部']/gmv_by_layer['尾部']:.1f}x")

    # 两种方案覆盖情况
    scheme_dist = df['参与激励方案类型'].value_counts(normalize=True)
    print(f"\n[校验] 方案覆盖分布:")
    for scheme, pct in scheme_dist.items():
        print(f"  {scheme}: {pct:.1%}")

    # 月度总发放
    monthly_spend = df.groupby('月度')['实际获得激励金额'].sum()
    print(f"\n[校验] 月度实际发放（应接近 {TOTAL_BUDGET:,} 元）:")
    for month, spend in monthly_spend.items():
        print(f"  {month}: {spend:,.0f} 元")

    output_path = "kuaishou_influencer_data.csv"
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n[完成] 数据已保存至: {output_path}")


if __name__ == "__main__":
    main()
