import pygame
from board import Tile

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
        self.start()
        self.updates = 0
        print("Width: " + str(self.width) + ", Height: " + str(self.height))




    def start(self):
        pygame.init()
        self.display = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()



    def update(self,hexes):
        self.display.fill((255,255,255))
        for i in range(len(hexes)):
            for j in range(len(hexes)):
                self.draw_hex(hexes[j][i])
        pygame.display.update()


    def draw_hex(self,hex):
        if hex.state == Tile.states['black']:
            img = self.black_hex
        elif hex.state == Tile.states['white']:
            img = self.white_hex
        elif hex.state == Tile.states['empty']:
            img = self.blank_hex

        self.display.blit(img,((hex.position[1]*self.hex_size)
                               ,hex.position[0]*self.hex_size + (hex.position[1])/2*self.hex_size))