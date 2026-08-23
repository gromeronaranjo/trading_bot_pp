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
    def __init__(self, amount_invested, direction, stock):
        self.amount_invested = amount_invested
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


def generate_action_space(stock_list, amounts):
    actions = []

    for stock in stock_list:
        for amount in amounts:
            actions.append(Action(amount, "buy", stock))

        actions.append(Action(None, "close", stock))

    actions.append(Action(None, "hold", None))

    return actions


def get_valid_actions(node, actions):
    valid_actions = []

    for action in actions:
        if action.direction == "hold":
            valid_actions.append(action)

        elif action.direction == "buy" and node.cash + 1e-9 >= action.amount_invested:
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
    cash = node.cash
    positions = node.positions.copy()

    if action.direction == "buy":
        amount = action.amount_invested
        positions[action.stock] = positions.get(action.stock, 0.0) + amount
        cash -= amount

    elif action.direction == "close":
        if action.stock in positions:
            cash += positions[action.stock]
            del positions[action.stock]

    returns = market_states[node.depth]["returns_cpu"]

    for day in range(2):
        for stock in positions:
            index = stock_to_idx[stock]
            stock_return = returns[day, index].item()
            positions[stock] *= 1.0 + stock_return

    money = cash + sum(positions.values())

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


def max_children(node):
    return max(3, int(0.75 * (node.times_visited + 1) ** 0.35))


def select(node, max_depth):
    while node.depth < max_depth:
        allowed_children = max_children(node)

        if len(node.untried_actions) > 0 and len(node.children) < allowed_children:
            return node

        if len(node.children) == 0:
            return node

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


def rollout(node, actions, market_states, stock_to_idx, max_depth, starting_money):
    current = node

    while current.depth < max_depth:
        valid_actions = get_valid_actions(current, actions)
        action = random.choice(valid_actions)
        current = transition(current, action, actions, market_states, stock_to_idx)

    return current.money / starting_money - 1.0


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


def print_sequence(sequence, starting_money):
    day = 0
    to_print = []

    for i in sequence:
        day += 2

        previous_money = i.parent.money
        return_percentage = (i.money / previous_money - 1.0) * 100

        day_info = {
            "day": day,
            "stock": i.action.stock,
            "direction": i.action.direction,
            "amount_invested": i.action.amount_invested,
            "cash": i.cash,
            "total_money": i.money,
            "return_percentage": return_percentage,
        }

        to_print.append(day_info)

    if len(sequence) > 0:
        total_money = sequence[-1].money
        total_percentage_gained = (total_money / starting_money - 1.0) * 100
    else:
        total_money = starting_money
        total_percentage_gained = 0.0

    print()
    print("trading sequence")
    print()

    for i in to_print:
        for key, value in i.items():
            print(f"{key}: {value}")

        print("\n")

    print("total money:", total_money)
    print("total percentage gained:", f"{total_percentage_gained:.4f}%")

    return to_print, total_money, total_percentage_gained

def predict(X, TE, starting_money, amounts, simulations, max_depth):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

    X = X.float().to(device)
    TE = TE.to(device)

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
    actions = generate_action_space(stock_list, amounts)

    with torch.inference_mode():
        stock_embedding = stockformer.struc2vec.embedding.weight
        market_states = build_market_states(stockformer, latent_model, X, TE, stock_embedding, max_depth)

    root = Node(None, None, 0, starting_money, starting_money, {})
    root.untried_actions = get_valid_actions(root, actions)

    for simulation in tqdm(range(simulations), desc="MCTS simulations"):
        node = select(root, max_depth)

        if node.depth < max_depth and len(node.untried_actions) > 0 and len(node.children) < max_children(node):
            node = expand(node, actions, market_states, stock_to_idx)

        reward = rollout(node, actions, market_states, stock_to_idx, max_depth, starting_money)

        backpropagate(node, reward)

    sequence = best_sequence(root)

    return print_sequence(sequence, starting_money)