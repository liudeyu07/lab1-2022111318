import pytest
from collections import defaultdict


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
        # 创建一个新的独立图来测试路径不存在的情况
        test_wg = WordGraph()
        test_wg.graph["isolated"] = defaultdict(int)  # 孤立的起始节点
        test_wg.graph["target"] = defaultdict(int)  # 孤立的目标节点

        message, paths = test_wg.calc_shortest_path("isolated", "target")
        assert message == "路径不存在"
        assert paths == {}

    # 路径5: 单点到单点，节点已访问过（跳过continue分支）
    def test_single_path_visited_skip(self, setup_graph):
        """测试基本路径5: 单点到单点，测试visited节点跳过逻辑"""
        # 创建专门的测试图
        wg = WordGraph()

        # 确保所有节点都存在于图中
        wg.graph["start"]["a"] = 1
        wg.graph["start"]["b"] = 2
        wg.graph["a"]["c"] = 1
        wg.graph["b"]["c"] = 1  # c会通过两条路径被加入heap
        wg.graph["c"]["target"] = 1
        wg.graph["target"] = defaultdict(int)  # 确保target节点存在

        message, paths = wg.calc_shortest_path("start", "target")
        assert "最短路径长度：3" in message
        # 最短路径应该是 start -> a -> c -> target
        assert paths["target"] == ["start", "a", "c", "target"]

    # 路径6: 单点到单点，节点未访问继续搜索
    def test_single_path_multiple_steps(self, setup_graph):
        """测试基本路径6: 单点到单点，多步骤搜索"""
        wg = setup_graph
        message, paths = wg.calc_shortest_path("the", "fox")
        assert "最短路径长度：3" in message
        assert paths["fox"] == ["the", "quick", "brown", "fox"]

    # 路径7: 单点到所有点，成功找到多个路径
    def test_all_paths_success(self, setup_graph):
        """测试基本路径7: 单点到所有点，成功找到多个路径"""
        wg = setup_graph
        message, paths = wg.calc_shortest_path("the")
        assert "找到从'the'到5个节点的最短路径" in message
        assert len(paths) == 5  # quick, brown, fox, lazy, dog
        expected_nodes = {"quick", "lazy", "dog", "brown", "fox"}
        assert set(paths.keys()) == expected_nodes

    # 路径8: 单点到所有点，节点已访问（跳过continue分支）
    def test_all_paths_visited_skip(self, setup_graph):
        """测试基本路径8: 单点到所有点，测试_calc_all_paths中的visited跳过逻辑"""
        # 专门构造一个会产生visited跳过的图
        wg = WordGraph()
        wg.graph["start"]["a"] = 1
        wg.graph["start"]["b"] = 2
        wg.graph["a"]["target"] = 1
        wg.graph["b"]["target"] = 1  # 两条路径都能到达target，但通过a的更短

        message, paths = wg.calc_shortest_path("start")
        assert "找到从'start'到3个节点的最短路径" in message
        # 验证target是通过最短路径(start->a->target)到达的
        assert paths["target"] == ["start", "a", "target"]

    # 路径9: 单点到所有点，简单情况正常结束
    def test_all_paths_simple_case(self):
        """测试基本路径9: 单点到所有点，简单图的正常结束情况"""
        # 创建一个简单的线性图，确保测试与其他用例不同
        wg = WordGraph()
        wg.graph["first"]["second"] = 1
        wg.graph["second"]["third"] = 1

        message, paths = wg.calc_shortest_path("first")
        assert "找到从'first'到2个节点的最短路径" in message
        assert len(paths) == 2
        assert paths["second"] == ["first", "second"]
        assert paths["third"] == ["first", "second", "third"]

    # 路径10: 单点到所有点，没有可达节点
    def test_all_paths_no_reachable(self):
        """测试基本路径10: 单点到所有点，没有可达节点"""
        wg = WordGraph()
        wg.graph["isolated"] = defaultdict(int)  # 添加孤立节点
        message, paths = wg.calc_shortest_path("isolated")
        assert message == "没有找到其他可达节点"
        assert paths == {}


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v"])
