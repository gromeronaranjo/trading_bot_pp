import torch
from dataclasses import dataclass

class Node():
    def __init__(self, action, parent, perc_change, money_in_account, money_to_spend):
        self.parent = parent
        self.action = action
        self.times_visited = 0

        if parent is None:
            self.money_in_account = money_in_account
            self.money_to_spend = money_to_spend
            self.positions = {}
        else:
            self.positions = parent.positions.copy()
            self.money_to_spend = parent.money_to_spend
            self.money_in_account = parent.money_in_account

            generated_percentage = sum(
                percentage * perc_change[stock]
                for stock, percentage in self.positions.items()
            )

            self.money_in_account *= 1 + generated_percentage

        if action.direction == "buy":
            percentage = min(action.percentage_invested, self.money_to_spend)

            self.positions[action.stock] = (
                self.positions.get(action.stock, 0) + percentage
            )

            self.money_to_spend -= percentage

        elif action.direction == "hold":
            pass

        elif action.direction in ("sell", "close"):
            stock = action.stock

            if stock in self.positions:
                self.money_to_spend += self.positions[stock]
                del self.positions[stock]

        self.percentage_invested = sum(self.positions.values())

    def visited(self):
        self.times_visited += 1

class Action():
    def __init__(self, percentage_invested, direction, stock):
        self.percentage_invested = percentage_invested
        self.direction = direction
        self.stock = stock

def generate_action_space(stock_list, percentages):
    action_space = []
    for percentage in percentages:
        for stock in stock_list:
            action_space.append(Action(percentage, "buy", stock))

    for stock in stock_list:
        action_space.append(Action(None, "close", stock))

    action_space.append(Action(None, "hold", None))

