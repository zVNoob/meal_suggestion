
import torch
from torch.functional import Tensor

import pandas as pd
import numpy as np

cook_mode = {
    #name : (price,cacbonhidrate,fat,protein)
    "luộc": (10,0,0,0),
    "hấp": (30,0,0,0),
    "nướng": (3000,-0.1,-0.1,-0.1),
    "rán": (200,0,10,0),
    "rang": (150,0,3,0),
    "xào": (158,0,1,0),
    "raw": (0,0,0,0),
}

class Env:
    def __init__(self,initial_money:int,max_dish_per_meal:int,history_size:int) -> None:
        if max_dish_per_meal > history_size:
            raise Exception("You can't have more max_dish_per_meal than history_size")
        self.initial_money=initial_money
        self.max_dish_per_meal=max_dish_per_meal
        self.history_size=history_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else
                                   "xpu" if torch.xpu.is_available() else
                                   "cpu")
        self.prev_dish = Tensor().to(self.device).new_zeros(history_size).float()
        # initalize lookup table
        self.lookup_table = []
        temp = pd.read_csv("data.csv")
        pd.set_option('future.no_silent_downcasting', True)
        temp = temp.replace(r'^\s*$', np.nan, regex=True)
        for _,i in temp.iterrows():
            for j in cook_mode.items():
                t = i[j[0]]
                if np.isnan(t) == False:
                    price = i["price"] + j[1][0]
                    if t != 0:
                        price = t
                    self.lookup_table.append((i["name"] + ' ' + (j[0] if j[0] != "raw" else ""),
                                              int(price),
                                              float(i["protein"] + j[1][2]),
                                              float(i["fat"] + j[1][1]),
                                              not bool(np.isnan(i["terminal"]))))

    def size(self):
        return (1 + self.history_size, len(self.lookup_table))

    def reset(self) -> torch.Tensor:
        self.money = self.initial_money
        self.prev_dish = self.prev_dish.new_zeros(self.history_size).float()
        self.protein = 0
        self.fat = 0
        self.meal = []

        return self.cat()

    def cat(self):
        return torch.cat((torch.Tensor([self.money]).to(self.device).float(),self.prev_dish))

    def push_history(self,output):
        self.meal.append(self.lookup_table[output][0])
        self.prev_dish = self.prev_dish.roll(1)
        self.prev_dish[0] = output

    def step(self,output,show = True) -> tuple[torch.Tensor, float, bool]:
        
        reward = 0
        #Check for duplicate and terminal
        enough = False
        for i in range(self.max_dish_per_meal):
            if self.prev_dish[i] == output:
                reward -= 1000
            if self.lookup_table[output][4]:
                enough = True
                break
        if not enough:
            reward -= 200
        self.push_history(output)

        self.money -= self.lookup_table[output][1]

        if self.money <= 0:
            if show:
                print(self.meal,self.money,self.protein,self.fat)
            return self.cat(), reward, True


        # check for nutrition
        self.protein += self.lookup_table[output][2]
        self.fat += self.lookup_table[output][3]

        # over-nutrient

        if self.protein > 300:
            reward -= (self.protein - 300) / 10

        if self.fat > 102:
            reward -= (self.fat - 102) / 7.5

        if self.lookup_table[output][4]:
            # a meal is done
            # print for each meal
            if show:
                print(self.meal,self.money,self.protein,self.fat)
            self.meal = []
            # check for under-nutrient
            if self.protein < 96:
                reward -= (96 - self.protein) / 5
            if self.fat < 114:
                reward -= (114 - self.fat) / 10
            # reset for each meal
            self.protein = max(0, self.protein - 130)
            self.fat = max(0, self.fat - 101.4)

        return self.cat(), 60 + reward, False
 
