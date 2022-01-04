class Board:

    def __init__(self, size=11):
        self.tiles = [[Tile([j, i]) for i in range(size)] for j in range(size)]
        self.size = size

    def can_place(self, position):
        return self.tiles[position[0]][position[1]].state == Tile.states['empty']
    # position = position in list / tuple
    # type = "black" or "white"
    def place(self, position, type):
        self.tiles[position[0]][position[1]] = Tile(position, state=type)

    # black connects top bottom, white left and right
    def check_win(self, type):
        if type == 'black':
            for column in range(self.size):
                if self.tiles[0][column].state == Tile.states[type]:
                    # do bfs

                    frontier = [self.tiles[0][column]]
                    visited = []
                    while len(frontier) != 0:

                        curr = frontier.pop()
                        visited.append(curr)

                        if curr.position[0] == self.size - 1:
                            return True
                        # lovely one liner to add to frontier
                        frontier.extend(filter(lambda t: t.state == Tile.states[type] and not t in visited,
                                               self.get_adjacent_tiles(curr.position)))
        elif type == 'white':
            for row in range(self.size):
                if self.tiles[row][0].state == Tile.states[type]:
                    # do bfs

                    frontier = [self.tiles[row][0]]
                    visited = []
                    #print()
                    while len(frontier) != 0:
                        #print([str(x.position) for x in frontier])
                        curr = frontier.pop()
                        visited.append(curr)


                        if curr.position[1] == self.size - 1:
                            return True
                        # lovely one liner to add to frontier
                        frontier.extend(filter(lambda t: t.state == Tile.states[type] and not t in visited,
                                               self.get_adjacent_tiles(curr.position)))
        return False

    # return list of all adjacent locations
    # to ensure hex, things are adjacent along a line from top left to bottom right
    def get_adjacent_positions(self, position):
        out = []
        if position[0] > 0:
            out.append([position[0] - 1, position[1]])
            if position[1] < self.size - 1:
                out.append([position[0] - 1, position[1] + 1])

        if position[0] < self.size - 1:
            out.append([position[0] + 1, position[1]])
            if position[1] > 0:
                out.append([position[0] + 1, position[1] - 1])

        if position[1] > 0:
            out.append([position[0], position[1] - 1])

        if position[1] < self.size - 1:
            out.append([position[0], position[1] + 1])

        return out

    def get_adjacent_tiles(self, position):
        return [self.tiles[x[0]][x[1]] for x in self.get_adjacent_positions(position)]

    def __repr__(self):
        return self.__str__() + "\nsize=" + self.size

    def __str__(self):
        return "\n".join(["".join([str(j) for j in i]) for i in self.tiles])

    def get_integer_representation(self):
        out = []
        for row in self.tiles:
            out_row = []
            for tile in row:
                out_row.append(tile.state)
            out.append(out_row)
        return out

    def clone(self):
        out = Board(size=self.size)
        for i in range(self.size):
            for j in range(self.size):
                out.tiles[i][j].state = self.tiles[i][j].state
        return out

class Tile:
    states = {'empty': 0, 'black': -1, 'white': 1}
    rep = {0: "-", -1: "B", 1: "W"}

    # row column
    def __init__(self, position, state="empty"):
        self.state = Tile.states[state]
        self.position = position

    def __repr__(self):
        str(self.__str__())  # + "@" + str(self.position)

    def __str__(self):
        return "[" + self.rep[self.state] + "]"