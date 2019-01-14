from sklearn.feature_extraction.text import TfidfTransformer
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer as TFIDF


text = ['兰州市 城关区 人民检察院 指控 ， 2015 年 6 月 13 日 0 时许 ， 被告人 郑 某某 饮酒 后 驾驶 甘 A ＊ ＊ ＊ 17 号 “ 大众 ” 牌 小型 轿车 ， 在 兰州市 城关区 火车站 东路 华联 宾馆 门前 ， 与 同 向 行驶 的 由 崔某 驾驶 的 甘 A8 ＊ ＊ ＊ 6 号 出租车 发生 碰撞 。 经 鉴定 ， 被告人 郑 某某 血液 中 酒精 含量 为 164.97 毫克 ／ 100 毫升 。',
        '经 审理 查明 ： 2014 年 7 月间 ， 被告人 王某 为 替 其 患有 脑瘫 的 儿子 、 女儿 办理 户口 登记 ， 向 他人 提供 儿子 、 女儿 的 相关 信息 ， 以 人民币 480 元 的 价格 向 他人 购买 编号 分别 为 L350398498 和 L350398499 的 出生 医学 证明 。 2014 年 8 月 11 日 ， 被告人 王 某持 上述 出生 医学 证明 到 思明 公安分局 办证 大厅 欲 办理 户口 登记 时 被 该处 工作人员 发现 。 2014 年 8 月 20 日 ， 被告人 王 某经 公安机关 电话 联系 主动 到 公安机关 投案 ， 并 如实 交代 上述 伪造证件 事实 。 经 鉴定 ， 上述 二本 出生 医学 证明 均系 伪造 。 以上 事实 ， 被告人 王某 在 开庭审理 过程 中 亦 不 持异议 ， 并 有 被告人 庭前 供述 和 辩解 、 伪造 的 出生 医学 证明 、 厦门市 中医院 分娩 记录 、 出生 申报 登记表 的 提取 笔录 、 扣押 物品 文件 清单 、 照片 、 厦门市 公安局 《 文检 鉴定书 》 、 被告人 的 户籍 资料 及 公安机关 出具 的 抓获 经过 说明 等 证据 在案 证实 ， 足以认定 。',
        '东莞市 第三 市区 人民检察院 指控 称 ， 被告人 罗 2 某 从 飞哥 （ 另案处理 ） 处 购买 甲基苯丙胺 （ 俗称 冰毒 ） 用于 吸食 。 2015 年 12 月 25 日 21 时 40 分 ， 民警 在 东莞市 东坑镇 塔岗村 东兴 东路 368 号 旁边 抓获 罗 2 某 ， 当场 在 其 所 驾驶 的 摩托车 车头 左 把手 手套 内 查获 一包 甲基苯丙胺 （ 净重 10.57 克 ） 。 公诉 机关 提供 了 现场 勘验 材料 ， 鉴定 意见 检验 鉴定 报告 ， 扣押 物品 清单 等 书证 ， 视听资料 审讯 录像 ， 被告人 罗 2 某 的 供述 与 辩解 等 证据 ， 并 据此 认为 被告人 罗 2 某 无视国法 ， 非法 持有 甲基苯丙胺 十克 以上 不满 五十克 ， 其 行为 已触犯 《 中华人民共和国 刑法 》 × × 之 规定 ， 犯罪事实 清楚 ， 证据 确实 充分 ， 应当 以 × × 罪 追究其 刑事责任 。 且 被告人 罗 2 某 此前 因 故意犯罪 被 判处 × × ， 刑罚 执行 完毕 以后 ， 五年 以内 再犯 应当 判处 × × 以上 刑罚 之罪 ， 是 累犯 ， 根据 《 中华人民共和国 刑法 》 × × × × 之 规定 ， 应当 从重 处罚 。 建议 对 其 判处 六个月 至 一年 六个月 × × ， 并 处罚金 ， 提请 本院 依法 判处 。'
        ]


def train_tfidf(train_data):
    tfidf = TFIDF(
        min_df=1e-9,
        max_features=1000,
        ngram_range=(1, 3),
        use_idf=1,
        smooth_idf=1
    )
    tfidf.fit(train_data)

    return tfidf


def cal_TF_IDF(texts_sequences):
    x = texts_sequences
    transformer = TfidfTransformer(smooth_idf=False)
    tfidf = transformer.fit_transform(x)
    tfidf = tfidf.toarray()
    #print(tfidf[0:5])
    print(tfidf)
    return tfidf

def cal_TF_IDF_2(train_data):
    tfidf = train_tfidf(train_data)
    vec = tfidf.transform(train_data)
    vec = vec.toarray()
    # print(vec[0:5])
    # print(type(vec))
    return vec

#test(text)

#cal_TF_IDF(text)