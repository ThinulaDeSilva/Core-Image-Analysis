# this class is used to hold all the contour info when constructing the dependency tree
class ContourInfo(object):
    # initializes the set of dependencies for the contour
    def __init__(self, index, points):
        self.index = index
        self.points = points
        self.dependencies = set()

    # adds a contour to its set of dependencies
    def add_dependency(self, index):
        self.dependencies.add(index)

    # removes a contour from its set of dependencies
    def remove_dependency(self, index):
        self.dependencies.discard(index)

    # returns whether the contour is a leaf node in the dependency tree
    def is_leaf(self):
        return not bool(self.dependencies)

    def dependencies(self):
        return self.dependencies

    def depth(self):
        # returns the x value of the leftmost column in the contour
        return self.points[:, :, 0].min()
