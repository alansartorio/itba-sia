from cube import \
    Cube, solved_cubes, turns, cube_rotations, \
    apply_action, apply_algorythm, parse_algorythm

class Node:
    def __init__(self, state):
        self.state = state
        self.explored = False
        self.child_nodes = []
        
    def add_child(self, child):
        self.child_nodes.append(child)

    
class Tree:
    def __init__(self, root):
        self.root = root
        self.visited = set([])
        self.queue = []
    

    def bfs(self):
        self.visited.add(self.root)
        self.queue.append(self.root)
        while self.queue:
            s = self.queue.pop(0)
            print(str(s.state))
            print("-----------")
            for n in s.child_nodes:
                if n not in self.visited:
                    self.visited.add(n)
                    self.queue.append(n)
        self.visited.clear()
                     
    def dfs(self, node):
        if node not in self.visited:
            print(str(node.state))
            print()
            self.visited.add(node)
            for child in node.child_nodes:
                self.dfs(child)
