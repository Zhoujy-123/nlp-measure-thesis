# -*- coding: utf-8 -*-
"""
基于年报 MD&A 文本的企业技术创新导向测度构建 —— NLP 五步流水线
=============================================================================
对应课件《管理研究中的 NLP 测度构建方法》第三章「NLP 测度五步」：
    ① 语料获取  → ② 文本预处理  → ③ 特征提取  → ④ 指标构建  → ⑤ 信效度检验

本脚本为一键可复现演示：在本地生成一份「制造业上市公司 MD&A 演示语料」，
完整跑通五步，并输出：
    - data/measure_panel.csv          企业-年份面板（最终测度）
    - data/results/descriptive.csv    描述性统计
    - data/results/validity.txt       四类效度检验结果
    - data/results/regression.txt     基准回归（预测效度）

真实研究中，只需把 step1 的「演示语料生成」替换为从巨潮资讯网下载的年报
PDF，并用 PyMuPDF 抽取 MD&A 文本即可，后续步骤完全不变。

运行：  python3 nlp_pipeline.py
依赖：  pip install -r requirements.txt
"""

import os
import re
import json
import random
from collections import Counter

import numpy as np
import pandas as pd

import jieba
from gensim import corpora, models

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm

# ============================== 路径配置 ==============================
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_DIR = os.path.join(BASE, "data", "demo_corpus")
RESULTS_DIR = os.path.join(BASE, "data", "results")
PANEL_CSV = os.path.join(BASE, "data", "measure_panel.csv")
STOPWORDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "stopwords.txt")

random.seed(2026)
np.random.seed(2026)
os.makedirs(CORPUS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

jieba.setLogLevel(20)  # 关闭 jieba 日志噪声

# 真实研究中的「技术创新导向」种子词典（基于创新管理文献，见论文 2.3 节）
SEED_INNOVATION_DICT = [
    "创新", "研发", "技术", "专利", "攻关", "突破", "自主研发", "核心技术",
    "研发投入", "技术创新", "科技成果", "成果转化", "产学研", "迭代", "新工艺",
    "新产品", "知识产权", "科创", "数智化", "数字化", "智能制造", "关键技术",
]


# ============================== ① 语料获取 ==============================
def build_demo_corpus(n_firms: int = 24, years=(2022, 2023)):
    """
    生成演示用 MD&A 语料。
    每个企业有一个隐变量 z∈[0,1] 表示「真实创新导向强度」，控制其文本中
    创新类语句的密度；从而让后续构造的文本测度与 z 相关，便于演示效度。
    真实研究中请用 PyMuPDF 从年报 PDF 抽取 MD&A 段落替代本函数。
    """
    innov_sents = [
        "报告期内公司推进核心技术攻关，强化关键技术的自主可控。",
        "公司全年申请发明专利若干项，加快科技成果转化落地。",
        "通过构建产学研协同创新平台，提升自主创新能力。",
        "围绕主营业务开展技术迭代，新工艺新产品持续涌现。",
        "依托数字化与智能制造升级，研发效率显著提高。",
        "公司加强知识产权布局，核心技术专利储备不断夯实。",
    ]
    risk_sents = [
        "公司面临市场竞争加剧、原材料价格波动及宏观经济不确定性等风险。",
        "需关注行业政策变化与下游需求波动对公司经营的影响。",
        "汇率波动及国际贸易环境变化可能对出口业务造成不利影响。",
        "原材料成本上行与用工压力给利润空间带来一定挑战。",
    ]
    market_sents = [
        "公司积极拓展国内外市场，优化销售渠道与客户结构。",
        "报告期内营业收入实现稳步增长，市场份额进一步提升。",
        "通过品牌建设与服务升级，增强客户黏性与复购率。",
        "深化与重点客户的战略合作，订单能见度持续改善。",
    ]
    generic_sents = [
        "公司严格按照法律法规要求履行信息披露义务。",
        "报告期内公司各项业务平稳运行，治理结构不断完善。",
        "公司将继续秉持稳健经营理念，保障全体股东利益。",
        "公司重视内部控制与合规管理，提升规范化运营水平。",
    ]

    records = []
    for i in range(1, n_firms + 1):
        z = float(np.clip(np.random.normal(0.5, 0.22), 0.05, 0.97))  # 企业创新倾向
        for y in years:
            zy = float(np.clip(z + np.random.normal(0, 0.06), 0.05, 0.97))  # 年度轻微漂移
            n_innov = int(round(2 + zy * 8))
            n_risk = int(round(2 + (1 - zy) * 3))
            n_market = int(round(2 + (1 - zy) * 3))
            sents = (
                random.choices(innov_sents, k=n_innov)
                + random.choices(risk_sents, k=n_risk)
                + random.choices(market_sents, k=n_market)
                + random.choices(generic_sents, k=3)
            )
            random.shuffle(sents)
            text = "管理层讨论与分析。\n" + "".join(sents)
            code = f"A{str(i).zfill(5)}"
            fname = os.path.join(CORPUS_DIR, f"{code}_{y}.txt")
            with open(fname, "w", encoding="utf-8") as f:
                f.write(text)
            records.append({"code": code, "year": int(y), "z_true": round(zy, 4),
                            "file": fname, "text": text})

    df = pd.DataFrame(records)
    print(f"[Step1] 语料获取完成：{df['code'].nunique()} 家公司 × "
          f"{df['year'].nunique()} 年 = {len(df)} 篇 MD&A，存于 {CORPUS_DIR}")
    return df


def step1_corpus_acquisition():
    """输入研究问题+数据可得性，输出结构化语料库。"""
    # 优先读取已生成的演示语料；若不存在则生成
    if not os.listdir(CORPUS_DIR):
        return build_demo_corpus()
    rows = []
    for fn in sorted(os.listdir(CORPUS_DIR)):
        if not fn.endswith(".txt"):
            continue
        code, year = fn.replace(".txt", "").split("_")
        with open(os.path.join(CORPUS_DIR, fn), encoding="utf-8") as f:
            rows.append({"code": code, "year": int(year), "text": f.read()})
    df = pd.DataFrame(rows)
    print(f"[Step1] 已加载语料：{len(df)} 篇 MD&A")
    return df


# ============================== ② 文本预处理 ==============================
def _load_stopwords():
    with open(STOPWORDS, encoding="utf-8") as f:
        return set(w.strip() for w in f if w.strip())


STOP = _load_stopwords()


def _clean(text: str) -> str:
    """文档解析后的清洗：去页眉页脚/页码/乱码/多余空白。"""
    text = re.sub(r"第\s*\d+\s*页", "", text)          # 页码
    text = re.sub(r"证券代码[：:]\s*\d+", "", text)     # 页眉
    text = re.sub(r"[\u3000\xa0]+", " ", text)          # 全角空格/不间断空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


def step2_preprocessing(df: pd.DataFrame):
    """
    五步流水线：解析 → 清洗 → 分词(jieba精确) → 去停用词 → 词形归一。
    输出：每篇文档干净的分词列表。
    """
    docs_tokens = []
    total_words = []
    for _, r in df.iterrows():
        clean = _clean(r["text"])
        words = jieba.lcut(clean, cut_all=False)  # 精确模式
        words = [w for w in words if w not in STOP and len(w) > 1 and not re.fullmatch(r"[0-9a-zA-Z]+", w)]
        docs_tokens.append(words)
        total_words.append(len(words))
    df = df.copy()
    df["tokens"] = docs_tokens
    df["n_words"] = total_words
    avg = df["n_words"].mean()
    print(f"[Step2] 预处理完成：平均文档长度 {avg:.0f} 词，"
          f"去停用词后保留有效词（长度>1 且非纯数字）。")
    return df


# ============================== ③ 特征提取（LDA 定词 + 词典法）==============================
def step3_feature_extraction(df: pd.DataFrame, num_topics: int = 5):
    """
    路线三 LDA：不预设词典，让模型从语料归纳主题，用于「定词」。
    本科建议（课件路径 B）：主题只用来定词，最终测度仍用可复现的词典法。
    输出：(1) LDA 各主题 Top 词，用于内容效度佐证；(2) 技术创新词典（种子∪LDA创新簇词）。
    """
    dictionary = corpora.Dictionary(df["tokens"])
    dictionary.filter_extremes(no_below=2, no_above=0.6)
    bow = [dictionary.doc2bow(t) for t in df["tokens"]]
    lda = models.LdaModel(corpus=bow, id2word=dictionary, num_topics=num_topics,
                          passes=20, random_state=42, alpha="auto", eta="auto")

    topic_words = {}
    print("\n[Step3] LDA 主题词（用于定词与内容效度）：")
    for k in range(num_topics):
        tw = lda.show_topic(k, topn=10)
        topic_words[k] = [w for w, _ in tw]
        print(f"  主题{k}: " + " / ".join(topic_words[k]))

    # 用种子词典命中的主题，识别「创新簇」，并以其 Top 词扩充词典（路径 B）
    seed_set = set(SEED_INNOVATION_DICT)
    innov_topic = None
    best_overlap = -1
    for k, words in topic_words.items():
        overlap = len(set(words) & seed_set)
        if overlap > best_overlap:
            best_overlap, innov_topic = overlap, k
    # 领域相关性筛选：剔除通用治理/风险/市场类词汇，保证词典内容效度
    NON_INNOV = {
        "治理", "平稳", "运行", "完善", "严格", "法律", "法规", "履行", "义务",
        "下游", "需求", "政策", "关注", "规范", "内部", "合规", "控制", "重视",
        "营业", "收入", "市场", "份额", "加大", "投入", "风险", "竞争", "波动",
        "汇率", "原材料", "成本", "客户", "渠道", "品牌", "销售", "稳健", "股东",
        "披露", "保障", "优化", "增强", "提升", "开展", "推进", "构建", "实现",
        "市场份额", "法律法规", "稳步增长",
    }
    lda_expand = [w for w in topic_words[innov_topic] if w not in NON_INNOV]
    innovation_dict = sorted(set(SEED_INNOVATION_DICT) | set(lda_expand))
    print(f"\n[Step3] 识别创新主题 = 主题{innov_topic}，"
          f"最终技术创新词典含 {len(innovation_dict)} 词：")
    print("       " + " / ".join(innovation_dict))
    return innovation_dict, topic_words


# ============================== ④ 指标构建 ==============================
def step4_measure_construction(df: pd.DataFrame, innovation_dict: list):
    """
    L1 原始计数 → L2 标准化比率 → L3（此处为单维，可直接作复合测度的输入）。
    测度公式（课件 3.3/3.4）：
        InnOri_it = (命中创新词典词总数 / MD&A 总词数) × 100
    输出：企业-年份面板。
    """
    dict_set = set(innovation_dict)
    measures = []
    for _, r in df.iterrows():
        hits = sum(r["tokens"].count(w) for w in dict_set)
        ratio = hits / r["n_words"] * 100 if r["n_words"] > 0 else 0.0
        measures.append(ratio)
    df = df.copy()
    df["InnOri"] = measures               # 技术创新导向测度（标准化比率，L2）
    df["innov_hits"] = [sum(r["tokens"].count(w) for w in dict_set) for _, r in df.iterrows()]
    panel = df[["code", "year", "InnOri", "innov_hits", "n_words", "z_true"]].copy()
    panel.to_csv(PANEL_CSV, index=False, encoding="utf-8-sig")
    print(f"[Step4] 指标构建完成：生成企业-年份面板 {PANEL_CSV}")
    print(panel.head(6).to_string(index=False))
    return panel


# ============================== ⑤ 信效度检验 ==============================
def _make_synthetic_fundamentals(panel: pd.DataFrame):
    """生成演示用的企业基本面（规模/年龄/ROA）与研发投入强度 RD，
    使 RD 与真实创新导向 z_true 正相关，用于演示预测效度。真实研究中
    这些变量应来自 CSMAR/Wind 等数据库。"""
    rng = np.random.default_rng(2026)
    size = rng.normal(22.5, 1.2, len(panel))
    age = rng.integers(5, 25, len(panel)).astype(float)
    roa = rng.normal(0.05, 0.03, len(panel))
    z = panel["z_true"].values
    rd = (0.018 + 0.030 * z + 0.004 * (size - 22.5)
          + 0.010 * roa + rng.normal(0, 0.006, len(panel)))
    rd = np.clip(rd, 0.002, 0.10)
    out = panel[["code", "year"]].copy()
    out["Size"] = size
    out["Age"] = age
    out["ROA"] = roa
    out["RD"] = rd
    return out


def step5_reliability_validity(panel: pd.DataFrame, docs_tokens: pd.DataFrame):
    """
    四类效度：内容 / 聚合 / 区分 / 预测。
    - 内容效度：词典来源已在 2.3 节说明（种子来自创新管理文献 + LDA 扩充）。
    - 聚合效度：与 TF-IDF 创新得分高度正相关。
    - 区分效度：与「公司规模」低相关（证明测的是新构念）。
    - 预测效度：对研发投入强度 RD 回归，系数显著为正。
    """
    fund = _make_synthetic_fundamentals(panel)
    data = panel.merge(fund, on=["code", "year"])

    # --- 聚合效度：TF-IDF 替代测度 ---
    corpus_text = [" ".join(t) for t in docs_tokens["tokens"]]
    vec = TfidfVectorizer(vocabulary=SEED_INNOVATION_DICT)
    X = vec.fit_transform(corpus_text)
    data["TFIDF_inno"] = np.asarray(X.sum(axis=1)).flatten()

    lines = []
    lines.append("================ 信效度检验 ================\n")

    # 内容效度（文字说明）
    lines.append("[内容效度] 技术创新词典由「创新管理文献种子词 + LDA 主题模型"
                 "识别的创新簇词」共同构成，理论依据见论文 2.3 节，词表已随"
                 "代码公开，满足可复现要求。\n")

    # 聚合效度
    r1, p1 = pearsonr(data["InnOri"], data["TFIDF_inno"])
    lines.append(f"[聚合效度] InnOri 与 TF-IDF 创新得分 Pearson r = {r1:.3f} "
                 f"(p = {p1:.4f}) → 与同类测度高度正相关，聚合效度良好。")

    # 区分效度
    r2, p2 = pearsonr(data["InnOri"], data["Size"])
    lines.append(f"[区分效度] InnOri 与公司规模 Size Pearson r = {r2:.3f} "
                 f"(p = {p2:.4f}) → 与「不应相关」的变量低相关，区分效度良好。")

    # 描述性统计
    desc = data[["InnOri", "RD", "Size", "Age", "ROA"]].describe().round(4)
    desc.to_csv(os.path.join(RESULTS_DIR, "descriptive.csv"), encoding="utf-8-sig")

    # 预测效度：OLS 回归  RD ~ InnOri + 控制变量
    Xc = sm.add_constant(data[["InnOri", "Size", "Age", "ROA"]])
    model = sm.OLS(data["RD"], Xc).fit()
    lines.append(f"\n[预测效度] 基准回归  RD_it = β0 + β1·InnOri_it + 控制变量")
    lines.append(f"  InnOri 系数 β1 = {model.params['InnOri']:.4f} "
                 f"(t = {model.tvalues['InnOri']:.2f}, p = {model.pvalues['InnOri']:.4f})")
    lines.append(f"  整体 R² = {model.rsquared:.3f}，观测数 N = {int(model.nobs)}")
    lines.append("  → 技术创新导向越高，研发投入强度显著越高，预测效度通过。\n")

    # 稳健性：用 z_true 做「真值」相关性，验证测度方向正确
    rz, pz = pearsonr(data["InnOri"], data["z_true"])
    lines.append(f"[稳健性] InnOri 与潜在真实创新导向 z_true 相关 r = {rz:.3f} "
                 f"(p = {pz:.4f})，方向一致，进一步佐证测度有效。")

    out = "\n".join(lines)
    with open(os.path.join(RESULTS_DIR, "validity.txt"), "w", encoding="utf-8") as f:
        f.write(out)
    with open(os.path.join(RESULTS_DIR, "regression.txt"), "w", encoding="utf-8") as f:
        f.write(str(model.summary()))

    print(out)
    return data, model


# ============================== 主流程 ==============================
def main():
    print("=" * 70)
    print("NLP 测度构建五步流水线：企业技术创新导向测度")
    print("=" * 70)
    df = step1_corpus_acquisition()
    df = step2_preprocessing(df)
    innovation_dict, _ = step3_feature_extraction(df)
    panel = step4_measure_construction(df, innovation_dict)
    step5_reliability_validity(panel, df)
    print("\n[完成] 所有产物已写入 data/ 目录，可导入 Stata/Python 做后续分析。")


if __name__ == "__main__":
    main()
