import sys
import math
import heapq
from collections import defaultdict

COST = {
    "BASIC": {"A": 1, "B": 0, "C": 0, "D": 0},
    "TENTACLE": {"A": 0, "B": 1, "C": 1, "D": 0},
    "HARVESTER": {"A": 0, "B": 0, "C": 1, "D": 1},
    "SPORER": {"A": 0, "B": 1, "C": 0, "D": 1},
    "ROOT": {"A": 1, "B": 1, "C": 1, "D": 1},
}


class Coordinate:
    _type = None

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"{self._type}: ({self.x}, {self.y})"

    def getNeighbours(self, grid):
        x, y = self.x, self.y
        neighbours = []
        if x > 0:
            neighbours.append(grid[y][x - 1])
        if x < len(grid[0]) - 1:
            neighbours.append(grid[y][x + 1])
        if y > 0:
            neighbours.append(grid[y - 1][x])
        if y < len(grid) - 1:
            neighbours.append(grid[y + 1][x])
        return neighbours

    def isOrganFacingThisCell(self, organ):
        if organ.organ_dir == "N":
            return self.y < organ.y
        if organ.organ_dir == "S":
            return self.y > organ.y
        if organ.organ_dir == "E":
            return self.x > organ.x
        if organ.organ_dir == "W":
            return self.x < organ.x
        return False

    def hasOpponentTentacleNeighbour(self, grid):
        neighbours = self.getNeighbours(grid)
        for neighbour in neighbours:
            if (
                isinstance(neighbour, Tentacle)
                and not neighbour.isMine()
                and self.isOrganFacingThisCell(neighbour)
            ):
                return True
        return False


class Entity(Coordinate):
    _type = None

    def __init__(
        self, x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
    ):
        super().__init__(x, y)
        self.owner = owner
        self.organ_id = organ_id
        self.organ_dir = organ_dir
        self.organ_parent_id = organ_parent_id
        self.organ_root_id = organ_root_id


class Wall(Entity):
    _type = "WALL"

    def __init__(
        self, x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
    ):
        super().__init__(
            x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
        )


class Organ(Entity):
    _type = "ORGAN"

    def __init__(
        self, x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
    ):
        super().__init__(
            x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
        )

    def isMine(self):
        return self.owner == 1


class Root(Organ):
    _type = "ROOT"

    def __init__(
        self, x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
    ):
        super().__init__(
            x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
        )


class Basic(Organ):
    _type = "BASIC"

    def __init__(
        self, x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
    ):
        super().__init__(
            x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
        )


class Harvester(Organ):
    _type = "HARVESTER"

    def __init__(
        self, x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
    ):
        super().__init__(
            x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
        )


class Tentacle(Organ):
    _type = "TENTACLE"

    def __init__(
        self, x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
    ):
        super().__init__(
            x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
        )


class Sporer(Organ):
    _type = "SPORER"

    def __init__(
        self, x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
    ):
        super().__init__(
            x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
        )


class Protein(Entity):
    _type = "PROTEIN"

    def __init__(
        self, x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
    ):
        super().__init__(
            x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
        )

    def isBeingHarvested(self, grid):
        neighbours = self.getNeighbours(grid)
        for neighbour in neighbours:
            if (
                isinstance(neighbour, Harvester)
                and neighbour.isMine()
                and self.isOrganFacingThisCell(neighbour)
            ):
                return True
        return False


class A(Protein):
    _type = "A"

    def __init__(
        self, x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
    ):
        super().__init__(
            x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
        )


class B(Protein):
    _type = "B"

    def __init__(
        self, x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
    ):
        super().__init__(
            x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
        )


class C(Protein):
    _type = "C"

    def __init__(
        self, x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
    ):
        super().__init__(
            x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
        )


class D(Protein):
    _type = "D"

    def __init__(
        self, x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
    ):
        super().__init__(
            x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
        )


class Action:
    def __init__(self, action, id, x, y, _type, direction):
        self.action = action
        self.id = id
        self.x = x
        self.y = y
        self._type = _type
        self.direction = direction

    def __str__(self):
        return (
            f"{self.action} {self.id} {self.x} {self.y} {self._type} {self.direction}"
        )

    def cost(self):
        return COST[self._type]

    def canBuy(self, stock):
        if self.action == None:
            return False
        for key in stock:
            if stock[key] < self.cost()[key]:
                return False
        return True


class Node:
    def __init__(self, entity, parent=None, g=0, h=0, spore_time=False):
        self.entity = entity
        self.parent = parent
        self.g = g  # Cost from start to current node
        self.h = h  # Heuristic from current node to goal
        self.f = g + h  # Total cost function f(n) = g(n) + h(n)'
        self.spore_time = spore_time
        self.spore_shot = False

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.entity.x == other.entity.x and self.entity.y == other.entity.y

    def __hash__(self):
        return hash((self.entity.x, self.entity.y))


def heuristic(coord1, coord2):
    # Example heuristic (Manhattan distance)
    return abs(coord1.x - coord2.x) + abs(coord1.y - coord2.y)


def aStarSearch(start, condition, grid, squashProtein=False):

    # Initialize open and close(d lists
    open_list = []
    closed_list = set()

    # Start node initialization
    start_node = Node(start)
    heapq.heappush(open_list, start_node)

    # Dictionary to track g-values
    g_values = defaultdict(lambda: float("inf"))
    g_values[(start.x, start.y)] = 0

    while open_list:
        # Get the node with the smallest f value
        current_node = heapq.heappop(open_list)

        # Add to closed list
        closed_list.add(current_node)

        # Check if this node satisfies the condition
        if condition(current_node.entity):
            path = []
            while current_node:
                path.append(current_node)
                if (
                    isinstance(current_node.entity, Organ)
                    and current_node.entity.isMine()
                ):
                    break
                current_node = current_node.parent

            path.reverse()
            return path

        # Get neighbors, ensuring valid indices
        x, y = current_node.entity.x, current_node.entity.y
        neighbors = []
        if x > 0:  # Left
            neighbors.append(grid[y][x - 1])
        if x < len(grid[0]) - 1:  # Right
            neighbors.append(grid[y][x + 1])
        if y > 0:  # Up
            neighbors.append(grid[y - 1][x])
        if y < len(grid) - 1:  # Down
            neighbors.append(grid[y + 1][x])

        for neighbor in neighbors:
            if (
                neighbor._type == "WALL"
                or (
                    isinstance(neighbor, Protein)
                    and not condition(neighbor)
                    and not squashProtein
                )
                or (
                    isinstance(neighbor, Organ)
                    and not neighbor.isMine()
                    and not condition(neighbor)
                )
                or neighbor.hasOpponentTentacleNeighbour(grid)
            ):
                continue

            movement_cost = (
                0 if isinstance(neighbor, Organ) and neighbor.isMine() else 1
            )

            tentative_g = current_node.g + movement_cost

            neighbor_node = Node(neighbor, current_node)
            # Skip nodes already evaluated
            if neighbor_node in closed_list:
                continue

            # Check if this path to the neighbor is better
            if tentative_g < g_values[(neighbor.x, neighbor.y)]:
                g_values[(neighbor.x, neighbor.y)] = tentative_g
                neighbor_node.g = tentative_g
                neighbor_node.h = heuristic(neighbor, start)
                neighbor_node.f = neighbor_node.g + neighbor_node.h

                heapq.heappush(open_list, neighbor_node)

    # If no cell satisfies the condition
    return []


def bestSporeForPath(path):
    xs = set([node.entity.x for node in path])
    ys = set([node.entity.y for node in path])

    xLines = [{"x": x, "val": sum([x == node.entity.x for node in path])} for x in xs]
    yLines = [{"y": y, "val": sum([y == node.entity.y for node in path])} for y in ys]

    maxXVal = max(xLines, key=lambda x: x["val"])
    maxYVal = max(yLines, key=lambda y: y["val"])

    sporePath = []
    if maxXVal["val"] > maxYVal["val"]:
        for node in path:
            if (
                node.entity.x == maxXVal["x"]
                and node.entity._type == None
                or node.entity._type == "SPORER"
            ):
                sporePath.append(node)
    else:
        for node in path:
            if (
                node.entity.y == maxYVal["y"]
                and node.entity._type == None
                or node.entity._type == "SPORER"
            ):
                sporePath.append(node)

    new_path = []
    for node in path:
        if node == sporePath[0]:
            node.spore_time = True
        elif node in sporePath:
            node.spore_shot = True
        new_path.append(node)
    return new_path


def getDirection(origin, destination):
    if origin.x == destination.x:
        return "N" if origin.y > destination.y else "S"
    return "W" if origin.x > destination.x else "E"


def canBuy(type, stock):
    for key in stock:
        if stock[key] < COST[type][key]:
            return False
    return True


def getPurchasableOrgan(stock):
    for key in COST:
        if canBuy(key, stock):
            return key
    return None


def getRoadToProtein(path, stock):
    start = path[0].entity
    dest = path[1].entity
    if len(path) == 3 and path[0].spore_time == False:
        protein = path[2].entity
        direction = getDirection(dest, protein)
        return Action("GROW", start.organ_id, dest.x, dest.y, "HARVESTER", direction)
        # print(f"GROW {start.organ_id} {dest.x} {dest.y} HARVESTER {direction}")
    else:
        organ = getPurchasableOrgan(stock)

        if organ:
            return Action("GROW", start.organ_id, dest.x, dest.y, organ, "X")
        else:
            return Action(None, None, None, None, None, None)


def getRoadToTentacle(path, stock):
    start = path[0].entity
    dest = path[1].entity
    if len(path) == 3:
        opponent = path[2].entity
        direction = getDirection(dest, opponent)
        return Action("GROW", start.organ_id, dest.x, dest.y, "TENTACLE", direction)
    else:
        organ = getPurchasableOrgan(stock)
        if organ:
            return Action("GROW", start.organ_id, dest.x, dest.y, organ, "X")
        else:
            return Action(None, None, None, None, None, None)


def printFreeCellPath(path):
    start = path[0].entity
    dest = path[1].entity
    print(f"GROW {start.organ_id} {dest.x} {dest.y} BASIC")


# width: columns in the game grid
# height: rows in the game grid
width, height = [int(i) for i in input().split()]

# game loop
while True:
    grid = [[Coordinate(x, y) for x in range(width)] for y in range(height)]

    entity_count = int(input())
    entities = []
    for i in range(entity_count):
        inputs = input().split()
        x = int(inputs[0])
        y = int(inputs[1])  # grid coordinate
        _type = inputs[2]  # WALL, ROOT, BASIC, TENTACLE, HARVESTER, SPORER, A, B, C, D
        owner = int(inputs[3])  # 1 if your organ, 0 if enemy organ, -1 if neither
        organ_id = int(inputs[4])  # id of this entity if it's an organ, 0 otherwise
        organ_dir = inputs[5]  # N,E,S,W or X if not an organ
        organ_parent_id = int(inputs[6])
        organ_root_id = int(inputs[7])

        match _type:
            case "WALL":
                entity = Wall(
                    x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
                )
            case "ROOT":
                entity = Root(
                    x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
                )
            case "BASIC":
                entity = Basic(
                    x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
                )
            case "HARVESTER":
                entity = Harvester(
                    x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
                )
            case "TENTACLE":
                entity = Tentacle(
                    x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
                )
            case "SPORER":
                entity = Sporer(
                    x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
                )
            case "A":
                entity = A(
                    x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
                )
            case "B":
                entity = B(
                    x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
                )
            case "C":
                entity = C(
                    x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
                )
            case "D":
                entity = D(
                    x, y, owner, organ_id, organ_dir, organ_parent_id, organ_root_id
                )
        grid[y][x] = entity
        entities.append(entity)

    # my_d: your protein stock
    my_a, my_b, my_c, my_d = [int(i) for i in input().split()]
    my_stock = {"A": my_a, "B": my_b, "C": my_c, "D": my_d}

    # opp_d: opponent's protein stock
    opp_a, opp_b, opp_c, opp_d = [int(i) for i in input().split()]
    opp_stock = {"A": opp_a, "B": opp_b, "C": opp_c, "D": opp_d}

    required_actions_count = int(
        input()
    )  # your number of organisms, output an action for each one in any order

    myRoots = [
        entity for entity in entities if entity._type == "ROOT" and entity.isMine()
    ]

    for i in range(required_actions_count):
        myRoot = myRoots[i]
        closestProteinPath = aStarSearch(
            myRoot,
            lambda entity: isinstance(entity, Protein)
            and not entity.isBeingHarvested(grid),
            grid,
        )
        closestOpponentPath = aStarSearch(
            myRoot,
            lambda entity: isinstance(entity, Organ) and not entity.isMine(),
            grid,
        )

        closestFreeCellPath = aStarSearch(
            myRoot,
            lambda entity: entity._type == None,
            grid,
        )

        # Write an action using print
        # To debug: print("Debug messages...", file=sys.stderr, flush=True)

        if len(closestProteinPath):
            action = getRoadToProtein(closestProteinPath, my_stock)
            if action.canBuy(my_stock):
                print(action)
                continue

        if len(closestOpponentPath) > 2:
            action = getRoadToTentacle(closestOpponentPath, my_stock)
            if action.canBuy(my_stock):
                print(action)
                continue
        if len(closestFreeCellPath):
            printFreeCellPath(closestFreeCellPath)
            continue

        closestProteinPath = aStarSearch(
            myRoot,
            lambda entity: isinstance(entity, Protein)
            and not entity.isBeingHarvested(grid),
            grid,
            squashProtein=True,
        )
        closestOpponentPath = aStarSearch(
            myRoot,
            lambda entity: isinstance(entity, Organ) and not entity.isMine(),
            grid,
            squashProtein=True,
        )

        closestFreeCellPath = aStarSearch(
            myRoot, lambda entity: entity._type == None, grid, squashProtein=True
        )

        if len(closestProteinPath):
            action = getRoadToProtein(closestProteinPath, my_stock)
            if action.canBuy(my_stock):
                print(action)
                continue

        if len(closestOpponentPath) > 2:
            action = getRoadToTentacle(closestOpponentPath, my_stock)
            if action.canBuy(my_stock):
                print(action)
                continue
        if len(closestFreeCellPath):
            printFreeCellPath(closestFreeCellPath)
            continue

        print("WAIT")
