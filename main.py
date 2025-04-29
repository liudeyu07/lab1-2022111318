import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext, simpledialog
import time
import os
import random
import threading
from collections import defaultdict
import heapq
import networkx as nx
import matplotlib.pyplot as plt
from PIL import ImageTk, Image
from tkinter import scrolledtext, Toplevel
import re
import math  # 需要用于k值计算
import numpy as np
from math import log
# ==== WordGraph类完整实现 ====
class WordGraph:
    def __init__(self):
        self.graph = defaultdict(lambda: defaultdict(int))
        self.random = random.Random()
        self._prepare_nx_graph()  # 初始化networkx图结构

    def build_graph(self, file_path):
        self.graph.clear()
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read().lower()

        # 预处理文本
        text = re.sub(r'[^a-z\s]', ' ', text)
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        words = [w for w in re.sub(r'\s+', ' ', text).split() if w]

        # 构建词图
        for i in range(len(words) - 1):
            self.graph[words[i]][words[i + 1]] += 1

        # 构建句子列表用于TF-IDF计算
        self.sentences = sentences
        self._prepare_nx_graph()

    def show_directed_graph(self):
        if not self.graph:
            return False
        try:
            G = nx.DiGraph()
            for src in self.graph:
                for dst, weight in self.graph[src].items():
                    G.add_edge(src, dst, weight=weight)

            # 优化图的大小和布局
            plt.figure(figsize=(10, 8), dpi=100)  # 比之前稍微小一些
            plt.axis('off')

            # 使用层次布局或圆形布局
            if len(G.nodes()) < 15:  # 节点较少时用圆形布局
                pos = nx.circular_layout(G, scale=0.8)
            else:  # 节点较多时用层次布局
                pos = nx.shell_layout(G)

            # 绘制曲线边 (使用connectionstyle参数)
            edge_options = {
                'arrowsize': 15,
                'width': 1,
                'connectionstyle': 'arc3,rad=0.1',  # 使边弯曲
                'edge_color': 'gray',
                'alpha': 0.7
            }

            # 绘制节点和边
            nx.draw_networkx_nodes(
                G, pos,
                node_size=800,
                node_color='lightblue',
                alpha=0.9
            )

            nx.draw_networkx_edges(
                G, pos,
                arrows=True,
                **edge_options
            )

            # 添加节点标签（调整位置避免与边重叠）
            nx.draw_networkx_labels(
                G, pos,
                font_size=9,
                font_weight='bold',
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )

            # 添加边权重标签
            edge_labels = {(u, v): d['weight']
                           for u, v, d in G.edges(data=True) if d['weight'] > 1}
            nx.draw_networkx_edge_labels(
                G, pos,
                edge_labels=edge_labels,
                font_size=8,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.7)
            )

            plt.tight_layout()
            plt.savefig("graph.png", dpi=120, bbox_inches='tight')
            plt.close()
            return True
        except Exception as e:
            print(f"Graph Generation Error: {str(e)}")
            return False

    def query_bridge_words(self, word1, word2):
        word1 = word1.lower()
        word2 = word2.lower()
        bridges = []
        if word1 not in self.graph or word2 not in self.graph:
            return f"单词{word1}或{word2}不存在！"
        for bridge in self.graph[word1]:
            if word2 in self.graph.get(bridge, {}):
                bridges.append(bridge)
        return "，".join(bridges) if bridges else "无桥接词"

    def generate_new_text(self, input_text):
        text = input_text.lower()
        words = [w.strip(",.!?") for w in text.split() if w.strip()]
        if len(words) < 2:
            return input_text

        result = [words[0]]
        for i in range(len(words) - 1):
            current = words[i]
            next_word = words[i + 1]
            if current in self.graph:
                bridges = [w for w in self.graph[current] if next_word in self.graph.get(w, {})]
                if bridges:
                    result.append(random.choice(bridges))
            result.append(next_word)
        return ' '.join(result)

    def calc_shortest_path(self, start, end=None):
        """
        计算最短路径，如果end为None则计算到所有节点的最短路径
        返回: (消息字符串, 路径字典) 路径字典为 {目标节点: 路径列表}
        """
        start = start.lower()
        if start not in self.graph:
            return "起始单词不存在", {}

        if end is not None:
            end = end.lower()
            if end not in self.graph:
                return "目标单词不存在", {}
            return self._calc_single_path(start, end)
        else:
            return self._calc_all_paths(start)

    def _calc_single_path(self, start, end):
        """计算单条最短路径"""
        heap = [(0, start, [])]
        visited = set()
        while heap:
            cost, node, path = heapq.heappop(heap)
            if node == end:
                return f"最短路径长度：{cost}", {end: path + [node]}
            if node in visited:
                continue
            visited.add(node)
            for neighbor in self.graph[node]:
                heapq.heappush(heap,
                               (cost + self.graph[node][neighbor], neighbor, path + [node]))
        return "路径不存在", {}

    def _calc_all_paths(self, start):
        """计算到所有节点的最短路径"""
        heap = [(0, start, [])]
        visited = {}
        paths = {}

        while heap:
            cost, node, path = heapq.heappop(heap)
            if node in visited:
                continue
            visited[node] = cost
            paths[node] = path + [node]

            for neighbor in self.graph[node]:
                if neighbor not in visited:
                    heapq.heappush(heap,
                                   (cost + self.graph[node][neighbor], neighbor, path + [node]))

        # 移除起始节点自身
        del paths[start]

        if not paths:
            return "没有找到其他可达节点", {}

        return f"找到从'{start}'到{len(paths)}个节点的最短路径", paths

    def highlight_path(self, path):
        if not path: return
        G = nx.DiGraph()
        edge_colors = []
        for src in self.graph:
            for dst in self.graph[src]:
                G.add_edge(src, dst)
                edge_colors.append('gray')

        # 生成高亮路径数据
        highlight_edges = []
        for i in range(len(path) - 1):
            src, dst = path[i], path[i + 1]
            if G.has_edge(src, dst):
                index = list(G.edges()).index((src, dst))
                edge_colors[index] = 'red'
                highlight_edges.append((src, dst))

        pos = nx.spring_layout(G)
        plt.figure()
        nx.draw(G, pos,
                edge_color=edge_colors,
                with_labels=True,
                node_color='lightblue',
                arrows=True)
        plt.savefig("highlight_path.png", dpi=150)
        plt.close()

    def _compute_word_importance(self, words):
        """基于TF-IDF计算单词重要性"""
        tf = defaultdict(int)
        for w in words:
            tf[w] += 1

        # 假设每个句子是一个文档(简单实现)
        docs = ' '.join(words).split('.')
        idf = defaultdict(int)
        total_docs = len(docs)
        for doc in docs:
            words_in_doc = set(doc.strip().split())
            for w in words_in_doc:
                idf[w] += 1

        word_scores = {}
        for w in words:
            tf_score = tf[w] / len(words)
            idf_score = math.log(total_docs / (idf.get(w, 1) + 1)) + 1
            word_scores[w] = tf_score * idf_score * 10  # 放大系数

        return word_scores

    def _prepare_nx_graph(self):
        """将内部图结构转换为networkx格式"""
        self.nx_graph = nx.DiGraph()
        for src in self.graph:
            for dst, weight in self.graph[src].items():
                self.nx_graph.add_edge(src, dst, weight=weight)



    def _compute_tfidf(self):
        """计算单词的TF-IDF权重"""
        word_docs = defaultdict(int)
        tf = defaultdict(dict)

        for i, sent in enumerate(self.sentences):
            words = set(sent.split())
            for word in words:
                tf[i][word] = tf[i].get(word, 0) + 1
                word_docs[word] += 1
        tfidf = defaultdict(dict)
        for doc_id, word_counts in tf.items():
            max_tf = max(word_counts.values()) if word_counts else 1
            for word, count in word_counts.items():
                tf_val = 0.5 + 0.5 * (count / max_tf)
                idf = log(len(self.sentences) / (1 + word_docs[word]))
                tfidf[doc_id][word] = tf_val * idf
        global_weights = defaultdict(float)
        for doc_weights in tfidf.values():
            for word, weight in doc_weights.items():
                global_weights[word] += weight

        return global_weights

    def calculate_pagerank(self, damping=0.85, max_iter=100, tol=1e-6):
        """
        专业的PageRank实现
        返回: {单词: PR值} 的字典
        """
        if not hasattr(self, 'nx_graph'):
            return {}

        nodes = sorted(self.nx_graph.nodes())
        n = len(nodes)
        node_index = {node: i for i, node in enumerate(nodes)}

        # 构建转移矩阵
        M = np.zeros((n, n))
        for src, dest, data in self.nx_graph.edges(data=True):
            M[node_index[dest], node_index[src]] = data.get('weight', 1)

        # 列归一化
        col_sums = M.sum(axis=0)
        M = M / np.where(col_sums > 0, col_sums, 1)

        # 处理悬挂节点(全为0的列)
        dangling = np.where(M.sum(axis=0) == 0)[0]
        M[:, dangling] = 1.0 / n

        # 初始化PR值(使用TF-IDF权重)
        initial_weights = self._compute_tfidf()
        pr = np.array([initial_weights.get(node, 1.0 / n) for node in nodes])
        pr = pr / pr.sum()  # 归一化

        # 迭代计算
        for _ in range(max_iter):
            new_pr = damping * M @ pr + (1 - damping) / n
            delta = np.abs(new_pr - pr).sum()
            pr = new_pr
            if delta < tol:
                break

        return {node: float(pr[i]) for i, node in enumerate(nodes)}

    def cal_page_rank(self, word, **kwargs):
            """
            兼容原接口的PR查询
            """
            pr_dict = self.calculate_pagerank(**kwargs)
            return round(pr_dict.get(word.lower(), 0), 4)

    def random_walk(self, walk_callback=None, delay=0.5):
        """
        改进的随机游走功能
        walk_callback: 回调函数，用于检查是否应停止遍历
        delay: 每次移动之间的延迟时间(秒)
        返回: (结果描述, 路径列表)
        """
        if not self.graph:
            return "无法执行随机游走（图为空）", []

        current = random.choice(list(self.graph.keys()))
        path = [current]
        visited_edges = set()

        try:
            while True:
                # 检查是否应该停止
                if walk_callback and walk_callback():
                    break

                # 添加延迟以降低速度
                time.sleep(delay)

                # 检查是否有出边
                if current not in self.graph or not self.graph[current]:
                    break

                # 随机选择下一个节点
                next_node = random.choice(list(self.graph[current].keys()))
                edge = (current, next_node)

                # 检查是否重复边
                if edge in visited_edges:
                    break

                visited_edges.add(edge)
                path.append(next_node)
                current = next_node

        except KeyboardInterrupt:
            pass

        # 保存到文件(不带箭头)
        try:
            with open("random_walk.txt", "w", encoding='utf-8') as f:
                f.write(" ".join(path))  # 改用空格分隔
        except Exception as e:
            print(f"保存随机游走结果失败: {e}")

        return "随机游走完成: " + " ".join(path), path


# ==== UI界面完整实现 ====
class GraphUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.wg = WordGraph()
        self.file_loaded = False
        self._setup_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _setup_ui(self):
        self.title("Word Graph Analyzer")
        self.geometry("1200x800")

        # 主操作区
        control_frame = ttk.Frame(self)
        control_frame.pack(fill=tk.X, padx=5, pady=5)

        # 文件选择区
        self.file_path = tk.StringVar()
        file_entry = ttk.Entry(control_frame, textvariable=self.file_path, width=50)
        file_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="浏览", command=self._browse_file).pack(side=tk.LEFT)
        ttk.Button(control_frame, text="加载", command=self._load_file).pack(side=tk.LEFT, padx=5)

        # 功能按钮区
        func_frame = ttk.Frame(control_frame)
        func_frame.pack(side=tk.LEFT, padx=15)
        self._create_button(func_frame, "显示图谱", self._show_graph)
        self._create_button(func_frame, "桥接词查询", self._bridge_dialog)
        self._create_button(func_frame, "生成文本", self._gen_text_dialog)
        self._create_button(func_frame, "PR值查询", self._pr_dialog)
        self._create_button(func_frame, "随机游走", self._do_random_walk)
        self._create_button(func_frame, "最短路径", self._path_dialog)
        self._create_button(func_frame, "保存结果", self._save_result)

        # 输出展示区
        self.output = scrolledtext.ScrolledText(self, wrap=tk.WORD, height=20)
        self.output.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 状态栏
        self.status = ttk.Label(self, text="就绪", relief=tk.SUNKEN)
        self.status.pack(fill=tk.X, side=tk.BOTTOM)

    def _create_button(self, parent, text, command):
        btn = ttk.Button(parent, text=text, command=command)
        btn.pack(side=tk.LEFT, padx=2)
        return btn

    def _browse_file(self):
        path = filedialog.askopenfilename(filetypes=[("文本文件", "*.txt")])
        if path:
            self.file_path.set(path)
            self.status.config(text=f"已选定文件：{os.path.basename(path)}")

    def _load_file(self):
        def load_task():
            path = self.file_path.get()
            if not path:
                self._show_error("请先选择文件")
                return
            self._update_status("正在加载并构建词图...")
            try:
                self.wg.build_graph(path)
                self.file_loaded = True
                self._output(f"成功加载文件：{os.path.basename(path)}")
                self._update_status("文件加载完成")
            except Exception as e:
                self._show_error(f"加载失败：{str(e)}")

        threading.Thread(target=load_task, daemon=True).start()

    def _show_graph(self):
        if not self._check_loaded(): return
        self._update_status("正在生成图谱...")

        def task():
            try:
                if self.wg.show_directed_graph():
                    self.after(0, self._display_image)
                else:
                    self._show_error("生成图谱失败")
            except Exception as e:
                self._show_error(f"图表错误：{str(e)}")

        threading.Thread(target=task, daemon=True).start()

    def _bridge_dialog(self):
        d = BridgeDialog(self)
        self.wait_window(d)
        if d.result:
            self._output(f"桥接词结果：\n{d.result}")

    def _path_dialog(self):
        d = PathDialog(self)
        self.wait_window(d)
        if d.result:
            if len(d.words) == 1:
                # 只有一个单词的情况
                msg, paths = self.wg.calc_shortest_path(d.words[0])
                self._output(msg)
                for target, path in paths.items():
                    self._output(f"到 '{target}' 的最短路径: {' → '.join(path)}")
            else:
                # 两个单词的情况
                msg, path_dict = self.wg.calc_shortest_path(d.words[0], d.words[1])
                self._output(f"{msg}\n路径：{' → '.join(list(path_dict.values())[0])}")
                if path_dict:
                    self._display_highlight(list(path_dict.values())[0])

    def _pr_dialog(self):
        if not self._check_loaded(): return

        def show_results():
            d = PRDialog(self)
            self.wait_window(d)
            if d.word.get():
                word = d.word.get().lower()
                pr = self.wg.cal_page_rank(word)
                self._output(f"'{word}'的PR值: {pr}")

                # 显示PR分布图(可选)
                if messagebox.askyesno("可视化", "显示PR值分布图吗？"):
                    self._show_pr_distribution()

        threading.Thread(target=show_results, daemon=True).start()

    def _show_pr_distribution(self):
        """显示PR值分布的可视化"""
        pr_dict = self.wg.calculate_pagerank()
        top_words = sorted(pr_dict.items(), key=lambda x: -x[1])[:20]

        plt.figure(figsize=(10, 6))
        plt.barh([w[0] for w in top_words], [w[1] for w in top_words])
        plt.title("Top 20 PageRank值分布")
        plt.xlabel("PR值")
        plt.tight_layout()

        img_path = "pr_distribution.png"
        plt.savefig(img_path, dpi=120)
        plt.close()

        # 显示图片
        img = ImageTk.PhotoImage(Image.open(img_path))
        win = Toplevel(self)
        win.title("PR值分布")
        label = ttk.Label(win, image=img)
        label.image = img
        label.pack()

    def _display_highlight(self, path):
        self.wg.highlight_path(path)
        img = ImageTk.PhotoImage(Image.open("highlight_path.png"))
        win = tk.Toplevel(self)
        win.title("路径高亮显示")
        label = ttk.Label(win, image=img)
        label.image = img
        label.pack()

    def _gen_text_dialog(self):
        d = GenTextDialog(self)
        self.wait_window(d)
        if d.text:
            result = self.wg.generate_new_text(d.text)
            self._output(f"生成文本：\n{result}")

    def _do_random_walk(self):
        """改进的随机游走功能，带停止按钮和实时显示"""
        if not self._check_loaded():
            return

        self._output("开始随机游走...(点击停止按钮可随时终止)")

        # 创建全局变量控制游走
        self.stop_walk_flag = False
        self.walk_path = []

        # 创建控制窗口
        walk_window = tk.Toplevel(self)
        walk_window.title("随机游走控制")
        walk_window.protocol("WM_DELETE_WINDOW", lambda: setattr(self, 'stop_walk_flag', True))

        # 当前路径显示区
        path_frame = ttk.Frame(walk_window)
        path_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        ttk.Label(path_frame, text="当前路径:").pack(anchor=tk.W)
        self.path_text = scrolledtext.ScrolledText(path_frame, height=10, width=50)
        self.path_text.pack(fill=tk.BOTH, expand=True)

        # 控制按钮区
        btn_frame = ttk.Frame(walk_window)
        btn_frame.pack(pady=5)

        stop_btn = ttk.Button(btn_frame, text="停止并保存",
                              command=lambda: self._stop_walk(walk_window))
        stop_btn.pack(side=tk.LEFT, padx=5)

        # 开始游走
        threading.Thread(target=self._walk_task, daemon=True).start()

    def _stop_walk(self, window):
        """停止游走并关闭窗口"""
        self.stop_walk_flag = True
        window.destroy()

        # 显示最终结果
        if self.walk_path:
            self._output("已停止随机游走，最终路径:")
            self._output(" ".join(self.walk_path))
            self._display_highlight(self.walk_path)

    def _walk_task(self):
        """执行随机游走的线程任务"""
        # 设置延迟时间(默认为0.5秒，可根据需要调整)
        delay = 0.5

        def update_callback():
            """更新UI的回调函数"""
            self.path_text.delete(1.0, tk.END)
            self.path_text.insert(tk.END, " ".join(self.walk_path))
            return self.stop_walk_flag

        res_str, path = self.wg.random_walk(update_callback, delay)
        self.walk_path = path

        # 在主界面显示结果
        self.after(0, lambda: self._output(res_str))

    def _save_result(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("文本文件", "*.txt")]
        )
        if path:
            try:
                with open(path, 'w') as f:
                    f.write(self.output.get("1.0", tk.END))
                self._output(f"结果已保存到：{os.path.basename(path)}")
            except Exception as e:
                self._show_error(f"保存失败：{str(e)}")

    def _display_image(self):
        try:
            img = ImageTk.PhotoImage(Image.open("graph.png"))
            win = tk.Toplevel(self)
            win.title("文本图谱可视化")
            label = ttk.Label(win, image=img)
            label.image = img
            label.pack()
        except Exception as e:
            self._show_error(f"显示图像错误：{str(e)}")

    def _update_status(self, msg):
        self.status.config(text=msg)

    def _show_error(self, msg):
        messagebox.showerror("错误", msg)
        self.status.config(text=msg)

    def _output(self, text):
        self.output.insert(tk.END, text + "\n")
        self.output.see(tk.END)

    def _check_loaded(self):
        if not self.file_loaded:
            self._show_error("请先加载文本文件！")
            return False
        return True

    def _on_close(self):
        plt.close('all')
        self.destroy()


# ==== 对话框类的完整实现 ====
class BridgeDialog(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.result = None
        self.word1 = tk.StringVar()
        self.word2 = tk.StringVar()
        self._setup_ui()

    def _setup_ui(self):
        self.title("桥接词查询")
        ttk.Label(self, text="起始词：").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(self, textvariable=self.word1).grid(row=0, column=1)
        ttk.Label(self, text="目标词：").grid(row=1, column=0)
        ttk.Entry(self, textvariable=self.word2).grid(row=1, column=1)
        btn_frame = ttk.Frame(self)
        btn_frame.grid(row=2, columnspan=2, pady=10)
        ttk.Button(btn_frame, text="确定", command=self._submit).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(side=tk.LEFT)

    def _submit(self):
        w1 = self.word1.get().strip()
        w2 = self.word2.get().strip()
        if w1 and w2:
            self.result = self.master.wg.query_bridge_words(w1, w2)
            self.destroy()


class PathDialog(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.words = []
        self.result = None
        self.start = tk.StringVar()
        self.end = tk.StringVar()
        self._setup_ui()

    def _setup_ui(self):
        self.title("最短路径查询")
        self.geometry("400x200")

        # 说明文本
        ttk.Label(self,
                  text="输入1个单词查询到所有节点的最短路径\n输入2个单词查询特定路径").pack(pady=5)

        # 输入框框架
        input_frame = ttk.Frame(self)
        input_frame.pack(pady=10)

        ttk.Label(input_frame, text="起始词：").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(input_frame, textvariable=self.start).grid(row=0, column=1)
        ttk.Label(input_frame, text="目标词(可选)：").grid(row=1, column=0)
        ttk.Entry(input_frame, textvariable=self.end).grid(row=1, column=1)

        # 按钮框架
        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="查询", command=self._submit).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(side=tk.LEFT)

    def _submit(self):
        s = self.start.get().strip()
        e = self.end.get().strip()
        if s:
            self.words = [s]
            if e:
                self.words.append(e)
            self.result = True
            self.destroy()


class PRDialog(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.result = None
        self.word = tk.StringVar()
        self._setup_ui()

    def _setup_ui(self):
        self.title("PageRank值查询")
        self.geometry("300x150")

        ttk.Label(self, text="查询单词:").pack(pady=10)
        ttk.Entry(self, textvariable=self.word, width=20).pack()

        btn_frame = ttk.Frame(self)
        btn_frame.pack(pady=15)
        ttk.Button(btn_frame, text="计算", command=self._calculate).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack(side=tk.LEFT)

    def _calculate(self):
        word = self.word.get().strip()
        if word:
            self.result = self.master.wg.cal_page_rank(word)
            self.destroy()


class GenTextDialog(tk.Toplevel):
    def __init__(self, master):
        super().__init__(master)
        self.text = None
        self._create_widgets()

    def _create_widgets(self):
        self.title("生成新文本")
        self.geometry("500x350")

        text_frame = ttk.Frame(self)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        ttk.Label(text_frame, text="输入原始文本:").pack(anchor=tk.W)
        self.input_area = scrolledtext.ScrolledText(text_frame, wrap=tk.WORD, height=10)
        self.input_area.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.Frame(text_frame)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="生成", command=self._confirm).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="取消", command=self.destroy).pack()

    def _confirm(self):
        input_text = self.input_area.get("1.0", tk.END).strip()
        if input_text:
            self.text = input_text
            self.destroy()


# ====== 主程序入口 ======
if __name__ == "__main__":
    app = GraphUI()
    app.mainloop()