from random import choice
import numpy as np
from math import log2
class Game2048:
    def __init__(self):
        self.n = 4
        self.grid=[0]*(self.n*self.n)
        self.game_over=False
        self.score=0
        self.random_squares=[2]*90+[4]*10
        self.right=[[x for x in range(y,y+self.n)] for y in range(0,self.n*self.n,self.n)]
        self.left=[x[::-1] for x in self.right]
        self.down=[[x for x in range(y,self.n*self.n,self.n)] for y in range(self.n)]
        self.up=[x[::-1] for x in self.down]
        self.d=[self.up,self.down,self.left,self.right]
        
    def add_line(self, line):
        line=line[::-1]
        slice_1=[x for x in line if x]
        slice_2=[x for x in line if not x]
        for i in range(len(slice_1)-1):
            if slice_1[i]==slice_1[i+1]:
                self.score+=slice_1[i]
                slice_1[i],slice_1[i+1]=slice_1[i]*2,0
        slice_2.extend([x for x in slice_1 if not x])
        line=[x for x in slice_1 if x]+slice_2
        return line[::-1]
    
    def edit_grid(self, direction_grid):
        for line in direction_grid:
            index=0
            added_across=self.add_line([self.grid[square] for square in line])
            for sq in line:
                self.grid[sq]=added_across[index]; index+=1
        if all([x for x in self.grid]):
            self.game_over=True
        else:
            empty_squares=[i for i in range(self.n*self.n) if not self.grid[i]]
            self.grid[choice(empty_squares)]=choice(self.random_squares)
        """self.score += sum(self.grid)"""
        return self.grid
    
    def move(self, pred):
        self.score += 1
        return self.edit_grid(self.d[pred])

    def reset(self):
        self.__init__()
        return self.grid
    
    def get_grid(self):
        return self.grid

    def get_game_over(self):
        return self.game_over

    def get_score(self):
        return self.score

    def step(self, action):
        prev_score = self.score
        self.move(action)
        a = self.grid
        return a, self.score - prev_score, self.game_over

