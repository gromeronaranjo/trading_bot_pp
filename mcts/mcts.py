import math
import random
import torch
from tqdm import tqdm
from dataclasses import dataclass
from models.stockformer import *
from models.latentdynamics import *


@dataclass
class LatentDynamicsConfig:
    dense_size: int = 128


@dataclass
class StockFormerConfig:
    n_stocks: int = 23
    n_features: int = 362
    dense_size: int = 128
    n_heads: int = 1
    n_encoder_blocks: int = 2
    dropout: float = 0.2
    time_steps: int = 20
    prediction_steps: int = 2
    kernel_size: int = 2


class Action:
    def __init__(self, percentage_invested, direction, stock):
        self.percentage_invested = percentage_invested
        self.direction = direction
        self.stock = stock


class Node:
    def __init__(self, parent, action, depth, money, cash, positions):
        self.parent = parent
        self.action = action
        self.depth = depth
        self.money = money
        self.cash = cash
        self.positions = positions.copy()
        self.children = []
        self.untried_actions = []
        self.times_visited = 0
        self.total_reward = 0.0
        self.period_reward = 0.0

    def average_reward(self):
        if self.times_visited == 0:
            return 0.0
        return self.total_reward / self.times_visited


def generate_action_space(stock_list, percentages):
    actions = []

    for stock in stock_list:
        for percentage in percentages:
            actions.append(Action(percentage, "buy", stock))

        actions.append(Action(None, "close", stock))

    actions.append(Action(None, "hold", None))

    return actions


def get_valid_actions(node, actions):
    valid_actions = []

    for action in actions:
        if action.direction == "hold":
            valid_actions.append(action)

        elif action.direction == "buy" and node.cash + 1e-9 >= action.percentage_invested:
            valid_actions.append(action)

        elif action.direction == "close" and action.stock in node.positions:
            valid_actions.append(action)

    return valid_actions


def predict_from_history(stockformer, low, high, te, stock_embedding):
    low_out = low
    high_out = high

    for encoder in stockformer.duel_freq_spatio_temporal_encoders:
        low_out, high_out = encoder(low_out, high_out, te, stock_embedding)

    low_future = stockformer.future_pred_low(low_out)
    high_future = stockformer.future_pred_high(high_out)

    out = stockformer.dual_frequency_fusion(low_future, high_future)

    return_pred = stockformer.return_proj(out)

    return return_pred.squeeze(0).squeeze(-1)


def advance_te(te):
    next_1 = te[:, -1:].clone()
    next_1[..., 0] = (next_1[..., 0] + 1) % 5

    next_2 = next_1.clone()
    next_2[..., 0] = (next_2[..., 0] + 1) % 5

    return torch.cat([te[:, 2:], next_1, next_2], dim=1)


def build_market_states(stockformer, latent_model, x, te, stock_embedding, max_depth):
    low_features, high_features = stockformer.decoupling(x)

    low = stockformer.feature_proj(low_features)
    high = stockformer.feature_proj(high_features)

    returns = predict_from_history(stockformer, low, high, te, stock_embedding)

    market_states = [{
        "low": low,
        "high": high,
        "te": te,
        "returns": returns,
        "returns_cpu": returns.cpu()
    }]

    for depth in range(max_depth - 1):
        state = market_states[-1]

        predicted_returns = state["returns"].unsqueeze(-1)
        predicted_directions = (state["returns"] >= 0).float().unsqueeze(-1)

        next_low, next_high = latent_model(predicted_returns, predicted_directions)

        next_low = next_low.unsqueeze(0)
        next_high = next_high.unsqueeze(0)

        low = torch.cat([state["low"][:, 2:], next_low], dim=1)
        high = torch.cat([state["high"][:, 2:], next_high], dim=1)

        te = advance_te(state["te"])

        returns = predict_from_history(stockformer, low, high, te, stock_embedding)

        market_states.append({
            "low": low,
            "high": high,
            "te": te,
            "returns": returns,
            "returns_cpu": returns.cpu()
        })

    return market_states


def apply_action(node, action, market_states, stock_to_idx):
    money = node.money
    cash = node.cash
    positions = node.positions.copy()

    if action.direction == "buy":
        percentage = action.percentage_invested
        positions[action.stock] = positions.get(action.stock, 0.0) + percentage
        cash -= percentage

    elif action.direction == "close":
        if action.stock in positions:
            cash += positions[action.stock]
            del positions[action.stock]

    returns = market_states[node.depth]["returns_cpu"]

    for day in range(2):
        generated_return = 0.0

        for stock, percentage in positions.items():
            index = stock_to_idx[stock]
            generated_return += percentage * returns[day, index].item()

        money *= 1.0 + generated_return

    return money, cash, positions


def transition(node, action, actions, market_states, stock_to_idx):
    old_money = node.money

    money, cash, positions = apply_action(node, action, market_states, stock_to_idx)

    child = Node(node, action, node.depth + 1, money, cash, positions)
    child.period_reward = money - old_money
    child.untried_actions = get_valid_actions(child, actions)

    return child


def ucb(parent, child, exploration=1.4):
    if child.times_visited == 0:
        return float("inf")

    exploitation = child.average_reward()
    exploration_value = exploration * math.sqrt(math.log(parent.times_visited + 1) / child.times_visited)

    return exploitation + exploration_value


def select(node, max_depth):
    while node.depth < max_depth and len(node.untried_actions) == 0 and len(node.children) > 0:
        best_child = node.children[0]
        best_score = ucb(node, best_child)

        for child in node.children[1:]:
            score = ucb(node, child)

            if score > best_score:
                best_child = child
                best_score = score

        node = best_child

    return node


def expand(node, actions, market_states, stock_to_idx):
    action = random.choice(node.untried_actions)
    node.untried_actions.remove(action)

    child = transition(node, action, actions, market_states, stock_to_idx)
    node.children.append(child)

    return child


def rollout(node, actions, market_states, stock_to_idx, max_depth):
    current = node

    while current.depth < max_depth:
        valid_actions = get_valid_actions(current, actions)
        action = random.choice(valid_actions)
        current = transition(current, action, actions, market_states, stock_to_idx)

    return current.money - 1.0


def backpropagate(node, reward):
    while node is not None:
        node.times_visited += 1
        node.total_reward += reward
        node = node.parent


def best_sequence(root):
    sequence = []
    node = root

    while len(node.children) > 0:
        best_child = node.children[0]

        for child in node.children[1:]:
            if child.times_visited > best_child.times_visited:
                best_child = child

        node = best_child
        sequence.append(node)

    return sequence


def print_sequence(sequence):
    day = 0
    to_print = []

    for i in sequence:
        day += 2

        day_info = {
            "day": day,
            "stock": i.action.stock,
            "direction": i.action.direction,

            "percentage_invested": i.action.percentage_invested,
            "cash_perc": i.cash,
            "money_perc": i.money,
        }

        to_print.append(day_info)

    if len(sequence) > 0:
        total_percentage_gained = (sequence[-1].money - 1.0) * 100
    else:
        total_percentage_gained = 0.0

    for i in to_print:
        print(i)

    print("total percentage gained:", total_percentage_gained)

    return to_print, total_percentage_gained
    

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

stockformer_config = StockFormerConfig()
latent_config = LatentDynamicsConfig()

stockformer = StockFormer(stockformer_config).to(device)
latent_model = LatentDynamicsModel(latent_config).to(device)

stockformer.load_state_dict(torch.load("/Users/gromeronaranjo/Desktop/personal_project/logs/checkpoints/best.pt", map_location=device))
latent_model.load_state_dict(torch.load("/Users/gromeronaranjo/Desktop/personal_project/logs/checkpoints/latent_dynamics_best.pt", map_location=device))

stockformer.eval()
latent_model.eval()

stock_list = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "JPM", "BAC", "GS", "XOM", "CVX", "JNJ", "LLY", "UNH", "WMT", "COST", "HD", "CAT", "GE", "NEE", "DUK", "KO", "PEP"]

stock_to_idx = {stock: index for index, stock in enumerate(stock_list)}

percentages = [0.025, 0.05, 0.075]

actions = generate_action_space(stock_list, percentages)

X = torch.load("/Users/gromeronaranjo/Desktop/personal_project/X", map_location="cpu")
TE = torch.load("/Users/gromeronaranjo/Desktop/personal_project/TE", map_location="cpu")

x = X[-1:].float().to(device)
te = TE[-1:].to(device)

simulations = 10000
max_depth = 15

with torch.inference_mode():
    stock_embedding = stockformer.struc2vec.embedding.weight
    market_states = build_market_states(stockformer, latent_model, x, te, stock_embedding, max_depth)

root = Node(None, None, 0, 1.0, 1.0, {})
root.untried_actions = get_valid_actions(root, actions)

for simulation in tqdm(range(simulations), desc="MCTS simulations"):
    node = select(root, max_depth)

    if node.depth < max_depth and len(node.untried_actions) > 0:
        node = expand(node, actions, market_states, stock_to_idx)

    reward = rollout(node, actions, market_states, stock_to_idx, max_depth)

    backpropagate(node, reward)

print_sequence(root, market_states, stock_to_idx)