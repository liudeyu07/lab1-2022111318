import pytest
from collections import defaultdict
import heapq


# 导入WordGraph类 - 使用以下方式之一:
# 方式1: 从现有模块导入
from main import WordGraph


class TestCalcShortestPath:

    @pytest.fixture
    def setup_graph(self):
        """创建一个简单的测试词图"""
        wg = WordGraph()

        # 构建测试词图:
        # the -> quick -> brown -> fox
        #  |      |                |
        #  v      v                v
        # lazy -> dog  <-----------|

        wg.graph["the"]["quick"] = 1
        wg.graph["the"]["lazy"] = 1
        wg.graph["quick"]["brown"] = 1
        wg.graph["quick"]["dog"] = 1
        wg.graph["brown"]["fox"] = 1
        wg.graph["fox"]["dog"] = 1
        wg.graph["lazy"]["dog"] = 1

        return wg

    # 路径1: 起始单词不存在
    def test_start_word_not_exist(self, setup_graph):
        """测试基本路径1: 起始单词不存在"""
        wg = setup_graph
        message, paths = wg.calc_shortest_path("nonexistent")
        assert message == "起始单词不存在"
        assert paths == {}

    # 路径2: 目标单词不存在
    def test_end_word_not_exist(self, setup_graph):
        """测试基本路径2: 目标单词不存在"""
        wg = setup_graph
        message, paths = wg.calc_shortest_path("the", "nonexistent")
        assert message == "目标单词不存在"
        assert paths == {}

    # 路径3: 单点到单点，首次循环找到目标
    def test_single_path_direct(self, setup_graph):
        """测试基本路径3: 单点到单点，直接相连"""
        wg = setup_graph
        message, paths = wg.calc_shortest_path("the", "quick")
        assert "最短路径长度：1" in message
        assert paths == {"quick": ["the", "quick"]}

    # 路径4: 单点到单点，heap为空无法找到路径
    def test_single_path_not_exist(self, setup_graph):
        """测试基本路径4: 单点到单点，路径不存在"""
        wg = setup_graph
        # 创建一个新的独立图来测试路径不存在的情况
        test_wg = WordGraph()
        test_wg.graph["isolated"] = defaultdict(int)  # 孤立的起始节点
        test_wg.graph["target"] = defaultdict(int)  # 孤立的目标节点

        message, paths = test_wg.calc_shortest_path("isolated", "target")
        assert message == "路径不存在"
        assert paths == {}

    # 路径5: 单点到单点，节点已访问过（跳过）
    def test_single_path_visited_skip(self, setup_graph):
        """测试基本路径5: 单点到单点，节点已访问跳过"""
        wg = setup_graph
        # 添加一个环路
        wg.graph["dog"]["lazy"] = 1
        wg.graph["lazy"]["the"] = 1
        message, paths = wg.calc_shortest_path("the", "dog")
        assert "最短路径长度" in message
        # 验证找到的是最短路径，不会被环路影响
        assert len(paths["dog"]) == 3

    # 路径6: 单点到单点，节点未访问继续搜索
    def test_single_path_multiple_steps(self, setup_graph):
        """测试基本路径6: 单点到单点，多步骤搜索"""
        wg = setup_graph
        message, paths = wg.calc_shortest_path("the", "fox")
        assert "最短路径长度" in message
        assert paths["fox"] == ["the", "quick", "brown", "fox"]

    # 路径7: 单点到所有点，成功找到多个路径
    def test_all_paths_success(self, setup_graph):
        """测试基本路径7: 单点到所有点，成功找到多个路径"""
        wg = setup_graph
        message, paths = wg.calc_shortest_path("the")
        assert "找到" in message and "个节点的最短路径" in message
        assert len(paths) == 5  # quick, brown, fox, lazy, dog
        for word in ["quick", "lazy", "dog", "brown", "fox"]:
            assert word in paths

    # 路径8: 单点到所有点，节点已访问（跳过）
    def test_all_paths_visited_skip(self, setup_graph):
        """测试基本路径8: 单点到所有点，节点已访问跳过"""
        wg = setup_graph
        # 添加环路
        wg.graph["dog"]["the"] = 1
        message, paths = wg.calc_shortest_path("the")
        # 即使有环路，仍应找到所有节点的最短路径
        assert "找到" in message
        assert len(paths) == 5
        # dog应该通过更短的路径到达
        assert len(paths["dog"]) < 5

    # 路径9: 单点到所有点，heap为空后正常结束
    def test_all_paths_normal_end(self, setup_graph):
        """测试基本路径9: 单点到所有点，正常结束处理"""
        wg = setup_graph
        # 修改图以确保heap最终为空
        test_graph = WordGraph()
        test_graph.graph["start"]["end"] = 1
        message, paths = test_graph.calc_shortest_path("start")
        assert "找到从'start'到1个节点的最短路径" in message
        assert paths == {"end": ["start", "end"]}

    # 路径10: 单点到所有点，没有可达节点
    def test_all_paths_no_reachable(self, setup_graph):
        """测试基本路径10: 单点到所有点，没有可达节点"""
        wg = WordGraph()  # 创建空图
        wg.graph["isolated"] = defaultdict(int)  # 添加孤立节点
        message, paths = wg.calc_shortest_path("isolated")
        assert message == "没有找到其他可达节点"
        assert paths == {}


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])

