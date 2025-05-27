import pytest
import tempfile
import os


class TestQueryBridgeWords:
    """桥接词查询功能的黑盒测试"""

    @pytest.fixture
    def setup_word_graph(self):
        """设置测试用的词图"""
        # 修正测试内容，确保词图结构清晰
        test_content = """
        apple good nice. good nice great.
        cat runs fast. fast means quick. quick helps speed.
        dog likes bone. bone hard stone.
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_file = f.name

        # 需要导入WordGraph类
        from main import WordGraph
        wg = WordGraph()
        wg.build_graph(temp_file)

        os.unlink(temp_file)
        return wg

    def test_E1_single_bridge_word(self, setup_word_graph):
        """E1: 两个单词都存在且有单个桥接词"""
        wg = setup_word_graph
        result = wg.query_bridge_words("apple", "nice")
        expected = 'The bridge words from "apple" to "nice" is: "good"'
        assert result == expected, f"Expected: {expected}, Got: {result}"

    def test_E2_multiple_bridge_words(self, setup_word_graph):
        """E2: 两个单词都存在且有多个桥接词"""
        # 创建专门的测试数据
        test_content = """
        start good end. start nice end. start great end.
        """

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
            f.write(test_content)
            temp_file = f.name

        from main import WordGraph
        wg = WordGraph()
        wg.build_graph(temp_file)
        os.unlink(temp_file)

        result = wg.query_bridge_words("start", "end")
        assert "The bridge words from \"start\" to \"end\" are:" in result
        assert "good" in result and "nice" in result and "great" in result

    def test_I1_no_bridge_words(self, setup_word_graph):
        """I1: 两个单词都存在但无桥接词"""
        wg = setup_word_graph
        result = wg.query_bridge_words("apple", "cat")
        expected = 'No bridge words from "apple" to "cat"!'
        assert result == expected, f"Expected: {expected}, Got: {result}"

    def test_I2_first_word_not_exist(self, setup_word_graph):
        """I2: 第一个单词不存在（第二个单词存在）"""
        wg = setup_word_graph
        result = wg.query_bridge_words("nonexistent", "good")
        expected = 'No "nonexistent" in the graph!'
        assert result == expected, f"Expected: {expected}, Got: {result}"

    def test_I3_second_word_not_exist(self, setup_word_graph):
        """I3: 第二个单词不存在（第一个单词存在）"""
        wg = setup_word_graph
        result = wg.query_bridge_words("apple", "nonexistent")
        expected = 'No "nonexistent" in the graph!'
        assert result == expected, f"Expected: {expected}, Got: {result}"

    def test_I4_both_words_not_exist(self, setup_word_graph):
        """I4: 两个单词都不存在"""
        wg = setup_word_graph
        result = wg.query_bridge_words("nonexistent1", "nonexistent2")
        expected = 'No "nonexistent1" and "nonexistent2" in the graph!'
        assert result == expected, f"Expected: {expected}, Got: {result}"

    def test_I5_empty_string_input(self, setup_word_graph):
        """I5: 输入包含特殊字符（空字符串）"""
        wg = setup_word_graph
        result = wg.query_bridge_words("", "good")
        expected = 'No "" in the graph!'
        assert result == expected, f"Expected: {expected}, Got: {result}"

    def test_I5_whitespace_input(self, setup_word_graph):
        """I5: 输入包含特殊字符（包含空格）"""
        wg = setup_word_graph
        result = wg.query_bridge_words("apple good", "nice")
        expected = 'No "apple good" in the graph!'
        assert result == expected, f"Expected: {expected}, Got: {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

