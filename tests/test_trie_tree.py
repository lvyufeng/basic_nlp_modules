import unittest

from data_structure import WordLinkedList, TernarySearchTree

class TestWordLinkedList(unittest.TestCase):
    def test_trie_tree(self):
        c = WordLinkedList()
        c.add('重庆大学')
        assert c.find('重庆大学') == True
        assert c.find('清华大学') == False

    def test_ternary_trie(self):
        sentence = '重庆大学在重庆市'
        offset = 0
        dic = TernarySearchTree()
        dic.add_word('重庆')
        dic.add_word('重庆市')
        dic.add_word('重庆大学')

        word = dic.match_long(sentence, offset)

        assert str(word) == '重庆大学'
