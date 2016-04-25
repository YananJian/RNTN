class Node:
    def __init__(self,label,word=None):
        self.label = label 
        self.word = word
        self.left = None
        self.right = None
        self.isLeaf = False

class Parser:
    def parse(self, token, depth=0):
        idx = token.find(" ", 0, len(token))
        if idx > 0:
            node = Node(int(token[idx-1])-1)
        elif idx == 0:
            node = Node(int(token[idx+1])-1)
        else:
            return
        
        l_ct = 0
        r_ct = 0
        i = 1
        if token[idx+1] == '(':
            l_ct += 1
            i += 1
            while l_ct != r_ct:
                if token[idx + i] == '(':
                    l_ct += 1
                elif token[idx + i] == ')':
                    r_ct += 1
                i += 1
            node.left = self.parse(token[idx+1:idx+i], depth+1)
            node.right = self.parse(token[idx+i+1:-1])
            return node
        else:
            node.word = ''.join(token[idx+1:-1])
            node.isLeaf = True
            return node

def findWords(root, leaves):
    if root.isLeaf:
        leaves.add(root.word)
    if root.left is not None:
        findWords(root.left, leaves)
    if root.right is not None:
        findWords(root.right,leaves)


if __name__ == '__main__':
    file = 'trees/train.txt'
    print "Reading trees.."
    p = Parser()
    trees = []
    words = set()
    with open(file,'r') as fid:
        lines = fid.readlines()
        for l in lines:
            node = p.parse(l)
            trees.append(node)
            findWords(node, words)

    wordMap = dict(zip(iter(words),xrange(len(words))))
    print wordMap
