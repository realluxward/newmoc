import torch
import numpy as np
from IPython import embed
from collections import defaultdict
from matplotlib import pyplot as plt
methods = ["moe","rq"]
data_dict = {}
for i in range(1,8):
    data_dict[f"moe_cid_{i}"] = {}
data_dict["join"] = {}
for method in methods:
    path = f"data/toys-split-{method}-7/test.csv"

    moe_cid_dict = {}
    for i in range(1,8):
        moe_cid_dict[f"moe_cid_{i}"] = defaultdict(int)
    moe_cid_dict["join"] = {}
    moe_cid_dict["join"][method] = defaultdict(int)

    vis = defaultdict(int)
    with open(path, 'r') as read_f:
        lines = read_f.read().splitlines()
        for line in lines[1:]:
            user_id,item_id,label,sales_type,brand,categories,moe_cid_1,moe_cid_2,moe_cid_3,moe_cid_4,moe_cid_5,moe_cid_6,moe_cid_7 = line.split(',')
            if vis[item_id]==1:
                continue
            vis[item_id]=1
            moe_cid_dict["moe_cid_1"][(moe_cid_1, moe_cid_1)] += 1
            moe_cid_dict["moe_cid_2"][(moe_cid_1, moe_cid_2)] += 1
            moe_cid_dict["moe_cid_3"][(moe_cid_1, moe_cid_3)] += 1
            moe_cid_dict["moe_cid_4"][(moe_cid_1, moe_cid_4)] += 1
            moe_cid_dict["moe_cid_5"][(moe_cid_1, moe_cid_5)] += 1
            moe_cid_dict["moe_cid_6"][(moe_cid_1, moe_cid_6)] += 1
            moe_cid_dict["moe_cid_7"][(moe_cid_1, moe_cid_7)] += 1
            # moe_cid_dict["moe_cid_1"][(moe_cid_1)] += 1
            # moe_cid_dict["moe_cid_2"][(moe_cid_2)] += 1
            # moe_cid_dict["moe_cid_3"][(moe_cid_3)] += 1
            # moe_cid_dict["moe_cid_4"][(moe_cid_4)] += 1
            # moe_cid_dict["moe_cid_5"][(moe_cid_5)] += 1
            # moe_cid_dict["moe_cid_6"][(moe_cid_6)] += 1
            # moe_cid_dict["moe_cid_7"][(moe_cid_7)] += 1
            moe_cid_dict["join"][method][(moe_cid_1, "_"+moe_cid_2)] += 1

    data = sorted(moe_cid_dict[f"moe_cid_{1}"].items(),key=lambda x:-x[1])
    # import random
    # random.shuffle(data)
    top_num = 2
    largest_value = [v[0][0] for v in data[:top_num]]

    for i in range(1,8):
        moe_cid_dict[f"moe_cid_{i}"] = {}

    with open(path, 'r') as read_f:
        lines = read_f.read().splitlines()
        for line in lines[1:]:
            user_id,item_id,label,sales_type,brand,categories,moe_cid_1,moe_cid_2,moe_cid_3,moe_cid_4,moe_cid_5,moe_cid_6,moe_cid_7 = line.split(',')
            if vis[item_id]==1:
                continue
            vis[item_id]=1
            if moe_cid_1 not in largest_value:
                continue
            moe_cid_dict["moe_cid_1"][(moe_cid_1)] += 1
            moe_cid_dict["moe_cid_2"][(moe_cid_2)] += 1
            moe_cid_dict["moe_cid_3"][(moe_cid_3)] += 1
            moe_cid_dict["moe_cid_4"][(moe_cid_4)] += 1
            moe_cid_dict["moe_cid_5"][(moe_cid_5)] += 1
            moe_cid_dict["moe_cid_6"][(moe_cid_6)] += 1
            moe_cid_dict["moe_cid_7"][(moe_cid_7)] += 1

    for i in range(1,8):
        data = sorted(moe_cid_dict[f"moe_cid_{i}"].items(),key=lambda x:-x[1])
        # data = [v for v in data if v[0][0] in largest_value]
        data = [v[1] for v in data]
        # data = [v[1]/sum(moe_cid_dict[f"moe_cid_{i}"].values()) for v in data]
        data = np.array(data)
        data_dict[f"moe_cid_{i}"][method] = data

    # data = sorted(moe_cid_dict["join"].items(),key=lambda x:-x[1])
    # data = [v for v in data if v[0][0] in largest_value]
    # data = [v[1] for v in data]
    # data = sorted(data, key=lambda x:-x)
    # data = np.array(data)
    # data_dict["join"][method] = data

# embed()
    data = moe_cid_dict["join"][method].keys()
    data = [v for v in data if v[0] in largest_value]
    # data = data[:100]
    cnt = moe_cid_dict["join"][method]

    import networkx as nx
    import matplotlib.pyplot as plt

    plt.figure(figsize=(6, 6))
    # G = nx.Graph()
    # for pair in data:
    #     G.add_edge(pair[0], pair[1])

    # # 设置节点布局，这里使用二分图布局
    # pos = nx.bipartite_layout(G, [n for n in G.nodes if "_" not in n])

    # # 计算边的宽度
    # edge_widths = [cnt[edge]/100 for edge in G.edges()]

    # # 绘制节点
    # nx.draw_networkx_nodes(G, pos, node_color='r', node_size=0.01)

    # # 绘制边，使用不同宽度代表边的权重
    # nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='black')
    G = nx.Graph()

    # 添加节点
    left_nodes = [node[0] for node in data]
    right_nodes = [f"_{i}" for i in range(256)]
    G.add_nodes_from(left_nodes, bipartite=0)
    G.add_nodes_from(right_nodes, bipartite=1)

    # 添加边
    for node1, node2 in data:
        G.add_edge(node1, node2, weight=cnt[(node1, node2)], width=cnt[(node1, node2)])

    # 绘制二分图
    pos = nx.bipartite_layout(G, left_nodes)
    nx.draw_networkx_nodes(G, pos, nodelist=left_nodes, node_color='r', node_size=0.001)
    nx.draw_networkx_nodes(G, pos, nodelist=right_nodes, node_color='r', node_size=0.001)
    edge_widths = [cnt[edge]/10 for edge in G.edges()]
    nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='black')
    plt.axis('off')

    plt.savefig(f"vis/{method}_bi.pdf", bbox_inches='tight')
for i in range(1,8):
    plt.figure(figsize=(6, 6))
    plt.plot(data_dict[f"moe_cid_{i}"]["moe"], label="moe")
    plt.plot(data_dict[f"moe_cid_{i}"]["rq"], label="rq")
    # plt.grid()
    plt.legend()
    # plt.xticks([0,1,2,3,4,5,6,7], [f'{i}' for i in range(1,8)], fontsize=16)
    plt.savefig(f"./vis/stat_{i}.pdf",  bbox_inches='tight')
# plt.figure(figsize=(6, 6))
# plt.plot(data_dict[f"join"]["moe"], label="moe")
# plt.plot(data_dict[f"join"]["rq"], label="rq")
# # plt.grid()
# plt.legend()
# # plt.xticks([0,1,2,3,4,5,6,7], [f'{i}' for i in range(1,8)], fontsize=16)
# plt.savefig(f"./vis/stat_join.pdf",  bbox_inches='tight')