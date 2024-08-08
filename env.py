
from numpy import array
import torch
from torch.functional import Tensor
from collections import defaultdict

ingredient = {
    "gạo": 1000,
    "cá": 1000,
    "thịt gà":2000,
}
ingredient = defaultdict(lambda: 0, ingredient)
cook_mode = {
    "luộc": 100,
    "nướng": 1000,
    "rán":800,
}
cook_mode = defaultdict(lambda: 0, cook_mode)
non_exist_dish = ["cá luộc"]
special_dish = {
    "gạo luộc": ("cơm",1000),
    "gạo nướng": ("bún",15000),
}

class Env:
    def __init__(self,initial_money:int,max_dish_per_meal:int,history_size:int) -> None:
        self.initial_money=initial_money
        self.max_dish_per_meal=max_dish_per_meal
        self.history_size=history_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.prev_dish = Tensor().to(self.device).new_zeros(history_size)
        self.lookup_table = [(i + ' ' + j,i,j) for i in ingredient for j in cook_mode]
    def size(self):
        return (1 + self.history_size, len(ingredient) * len(cook_mode))

    def reset(self) -> torch.Tensor:
        self.money = self.initial_money
        self.prev_dish = self.prev_dish.new_zeros(self.history_size)
        return torch.cat((torch.Tensor([self.money]),self.prev_dish));

    def cat(self):
        return torch.cat((torch.Tensor([self.money]).to(self.device),self.prev_dish))

    def push_history(self,output):
        self.prev_dish = self.prev_dish.roll(1)
        self.prev_dish[0] = output
    def step(self,output:int) -> tuple[torch.Tensor, float, bool]:
        dish_name = self.lookup_table[output][0]
        if dish_name in non_exist_dish:
            return Tensor(), -20, True

        if dish_name in special_dish:
            self.money -= special_dish[self.lookup_table[output][0]][1]
            dish_name = special_dish[self.lookup_table[output][0]][0]
        else:
            self.money -= ingredient[self.lookup_table[output][1]]
            self.money -= cook_mode[self.lookup_table[output][2]]

        print(dish_name,self.money,flush= True)

        if self.money <= 0:
            return self.reset(), 0, True

        for i in range(self.max_dish_per_meal):
            if self.prev_dish[i] == output:
                self.push_history(output)
                return self.cat(), -200, False

        self.push_history(output)
        return self.cat(), 30, False
 
