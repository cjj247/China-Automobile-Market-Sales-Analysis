# 中国汽车市场销量分析

## 项目概述

基于 2018 年 1 月至 2024 年 4 月的中国汽车车型月度销量数据，使用 Python 完成数据清洗、指标计算与探索分析，并将结果接入 Tableau，制作由 12 张工作表组成的单页交互看板。分析内容包括市场总量、新能源渗透、中国品牌份额、品牌与车型竞争格局、价格带及增长表现。

## 数据来源

- 数据集：[China Automobile Monthly Sales Data (2018–2024.4)](https://www.kaggle.com/datasets/felixzhao/china-automobile-monthly-sales-data-2018-2024-4)
- 下载镜像：[GitHub 原始 CSV](https://github.com/hoangphuc-ngo/Forecasting-China-EV-Sales/raw/refs/heads/main/Data/Raw/China%20Automobile%20Sales%20Data.csv)
- 许可证：MIT
- 原始数据共 38,806 条记录，属于真实公开整理数据，并非模拟生成

## 分析流程

1. 使用 `re`、pandas 和 NumPy 清理文本、日期、销量与价格字段，删除 33 条完全重复记录，并修正 327 条异常能源标签。
2. 计算月度销量、同比、环比、新能源占比、中国品牌占比、品牌与车型份额，以及滚动 12 个月增长指标。
3. 输出 38,773 行、38 列的 Tableau 专用数据，制作 1 个仪表盘和 12 张工作表。
4. 根据可视化结果提炼市场结构、品牌竞争和车型增长方面的业务结论。

## 核心结论

- 最近 12 个月样本销量同比下降 2.3%，市场竞争正在由行业增量转向结构性竞争。
- 新能源销量占比由 28.2% 上升至 32.8%，是最明确的结构性增长来源。
- 中国品牌份额由 50.5% 上升至 53.7%，整体竞争优势继续扩大。
- BYD 为最近 12 个月销量最高品牌，前五品牌合计占 40.4%，头部品牌集中度较高。
- 秦 PLUS 为最近 12 个月销量最高车型；车型资源配置应结合销量规模、增长率和历史基数综合判断。

## 项目文件

| 文件 | 说明 |
| --- | --- |
| `data/China Automobile Sales Data.csv` | 下载的原始公开数据 |
| `data/china_auto_sales_clean.csv` | Python 清洗后的标准明细数据 |
| `data/china_auto_sales_tableau.csv` | 包含分析指标的 Tableau 专用数据 |
| `中国汽车市场销量分析.ipynb` | 已运行的 Notebook，展示读取、清洗、分析、图表和结果 |
| `pipeline.py` | 可一键重新执行数据清洗与指标计算的 Python 程序 |
| `中国汽车市场销量分析.twbx` | 最终 Tableau 打包工作簿 |
| `requirements.txt` | Python 依赖版本 |

## 运行方法

```bash
pip install -r requirements.txt
python pipeline.py
```

脚本运行后会更新 `data/china_auto_sales_clean.csv` 和 `data/china_auto_sales_tableau.csv`。本数据不包含成本、利润、库存和订单信息，因此分析结论只用于解释汽车市场销量结构。
