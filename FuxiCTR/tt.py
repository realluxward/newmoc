import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats

set_matplotlib_formats("svg")

def plot_transport_tarrif(self):
    """
    绘制 Factory-DC 运输成本网络图
    """
    # 创建一个空的无向图
    G = nx.Graph()
    # 向图中添加所有的工厂和配送中心
    G.add_nodes_from(self.factory_list + self.dc_list)
    # 向图中添加所有的运输费用记录，其中边的两端分别为配送中心
    G.add_edges_from(
        list(
            self.transport_tariff[["factory_id", "dc_id"]]
            .drop_duplicates()
            .itertuples(index=False, name=None)
        )
    )
    # 定义边的宽度，根据运输费用进行设置
    width = {}
    for idx, row in self.transport_tariff.iterrows():
        width[(row["factory_id"], row["dc_id"])] = row["unit_cost"] / 500
    # 获取一个二分图的布局
    pos = nx.bipartite_layout(G, self.factory_list)
    # 绘制图形，并设置边的宽度和节点的大小
    nx.draw(G, pos, width=[width[e] for e in G.edges], node_size=30)
    # 在左边添加“Factory”标注
    plt.text(-1.1, 0, "Factory", fontsize=12)
    # 在右边添加“DC”标注
    plt.text(0.15, 0, "DC", fontsize=12)
    # 添加标题
    plt.title("Transport Tariff")
    # 导出图像到本地
    plt.savefig("./image/transport_tariff.png", dpi=300, bbox_inches="tight")
    # 显示图像
    plt.show()
