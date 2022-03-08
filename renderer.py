import pygame
from board import Tile
from threading import Thread

class Hex_Renderer:
    white_hex = pygame.image.load("white_hex.png")
    black_hex = pygame.image.load("black_hex.png")
    blank_hex = pygame.image.load("hex.png")
    hex_size = blank_hex.get_height()

    def __init__(self, width = 800, height = 600):
        self.display = None
        self.clock = None
        self.width = width
        self.height = height
        self.updates = 0
        self.thread = None
        self.hexes = []
        self.updated = False
        self.done = False
        print("Width: " + str(self.width) + ", Height: " + str(self.height))

        self.start()


    def start(self):
        pygame.init()
        self.display = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.thread = Thread(target=self.update)
        self.done = False
        self.thread.start()


    def update_hexes(self,hexes):
        self.hexes = hexes
        self.updated = False
        if self.done:
            self.start()

    def kill(self):
        self.done = True
        self.thread.join()

    def update(self):
        while not self.done:
            if not self.updated:
                self.display.fill((255,255,255))
                for i in range(len(self.hexes)):
                    for j in range(len(self.hexes)):
                        self.draw_hex(self.hexes[j][i])
                pygame.display.update()
                self.updated = True


    def draw_hex(self,hex):
        if hex.state == Tile.states['black']:
            img = self.black_hex
        elif hex.state == Tile.states['white']:
            img = self.white_hex
        elif hex.state == Tile.states['empty']:
            img = self.blank_hex

        self.display.blit(img,((hex.position[1]*self.hex_size)
                               ,hex.position[0]*self.hex_size + (hex.position[1])/2*self.hex_size))