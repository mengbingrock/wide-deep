
class Graph:
    """
    计算图类
    """

    def __init__(self):
        self.nodes = []  # 计算图内的节点的列表
        #self.name_scope = None

    def add_node(self, node):
        """
        添加节点
        """
        self.nodes.append(node)

    def node_count(self):
        return len(self.nodes)

    def clear_jacobian(self):
        
        """
        清除图中全部节点的雅可比矩阵
        """
        for node in self.nodes:
            node.clear_jacobian()


default_graph = Graph()