class Node(object):
    '''
    node in linked list
    '''
    def __init__(self, item:int):
        self.element = item
        self.next = None

    def __str__(self):
        return 'element:' + self.element

class WordLinkedList(object):

    def __init__(self):
        self.root = None

    def add(self, word:str):
        '''
        add new word
        '''
        if self.root == None:
            self.root = Node(ord(word[0]))
        parNode = self.root
        for i in range(1, len(word)):
            currNode = Node(ord(word[i]))
            parNode.next = currNode
            parNode = currNode
        
    def find(self, input:str):
        currNode = self.root
        i = 0

        while currNode is not None and i < len(input):
            if (currNode.element != ord(input[i])):
                return False
            currNode = currNode.next
            i = i + 1
        return True

# ternary search tree

class WordEntry:
    '''
    word entry
    '''

    def __init__(self, word:str, type: str):
        self.word = word
        self.types = {type}

    def add_type(self, type:str):
        self.types.add(type)
    
    def __str__(self):
        return self.word

class TSTNode(object):
    def __init__(self, key: int):
        self.left = None
        self.right = None
        self.mid = None

        self.data = None
        self.splitchar = key

    def __str__(self):
        return 'splitchar:' + chr(self.splitchar)
    
    def add_word(self, word, type):
        if self.data is None:
            self.data = WordEntry(word, type)
        else:
            self.data.add_type(type)
    
class TernarySearchTree(object):

    def __init__(self):
        self.root = None

    def add_word(self, word, args = None):
        node = self._get_or_create_node(word)
        node.add_word(word, args)

    def _get_or_create_node(self, key:str) -> TSTNode:
        charIndex = 0
        currChar = ord(key[charIndex])

        if self.root == None:
            self.root = TSTNode(currChar)
        
        currNode = self.root
        while True:
            charComp = currChar - currNode.splitchar
            if charComp == 0:
                charIndex += 1
                if charIndex == len(key):
                    return currNode
                currChar = ord(key[charIndex])
                if currNode.mid is None:
                    currNode.mid = TSTNode(currChar)
                currNode = currNode.mid
            elif charComp < 0:
                if currNode.left is None:
                    currNode.left = TSTNode(currChar)
                currNode = currNode.left
            else:
                if currNode.right is None:
                    currNode.right = TSTNode(currChar)
                currNode = currNode.right

    def get_node(self, key:str):
        if not key or not self.root:
            return None
        currNode = self.root
        charIndex = 0
        while True:
            if currNode is None:
                return None
            charComp = ord(key[charIndex]) - currNode.splitchar
            if charComp == 0:
                charIndex += 1

                if charIndex == len(key):
                    return currNode
            elif charComp < 0:
                currNode = currNode.left
            else:
                currNode = currNode.right

    def match_long(self, key:str, offset:int) -> str:
        ret = []
        if not key or not self.root or offset >= len(key):
            return ret
        
        currNode = self.root
        charIndex = offset
        while True:
            if currNode is None:
                charIndex += 1
                currNode = self.root
                continue
                # return ret
            charComp = ord(key[charIndex]) - currNode.splitchar
            if charComp == 0:
                charIndex += 1
                if currNode.data is not None:
                    ret.append((str(currNode.data),charIndex))
                if charIndex == len(key):
                    return ret
                currNode = currNode.mid
            elif charComp < 0:
                currNode = currNode.left
            else:
                currNode = currNode.right
