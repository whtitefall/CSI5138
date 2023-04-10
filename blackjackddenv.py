import gym
from gym import spaces
from gym.utils import seeding
# import logging
import random
# logging.basicConfig(filename='/Users/maverick/Desktop/RL/blackjack-master/logs/env.log',level=logging.DEBUG, filemode='w')


def cmp(a, b):
    return float(a > b) - float(a < b)

# 1 = Ace, 2-10 = Number cards, Jack/Queen/King = 10
CARDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10] * 4

def count(card):
    if 2<=card<=6:
        return 1
    elif 7<=card<=9:
        return 0
    return -1

def draw_card(inputdeck):
    # print(inputdeck)
    card = inputdeck.pop()
    return int(card)


def draw_hand(inputdeck):
    return [draw_card(inputdeck), draw_card(inputdeck)]


def usable_ace(hand):  # Does this hand have a usable ace?
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):  # Return current hand total
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):  # Is this hand a bust?
    return sum_hand(hand) > 21


def score(hand):  # What is the score of this hand (0 if bust)
    return 0 if is_bust(hand) else sum_hand(hand)


def is_natural(hand):  # Is this hand a natural blackjack?
    return sorted(hand) == [1, 10]

def can_double_down(hand, actionstaken):
    return len(hand) == 2 and actionstaken == 0

    

    


class BlackjackEnv(gym.Env):
    """Simple blackjack environment
    Blackjack is a card game where the goal is to obtain cards that sum to as
    near as possible to 21 without going over.  They're playing against a fixed
    dealer.
    Face cards (Jack, Queen, King) have point value 10.
    Aces can either count as 11 or 1, and it's called 'usable' at 11.
    This game is placed with an infinite deck (or with replacement).
    The game starts with each (player and dealer) having one face up and one
    face down card.
    The player can request additional cards (hit=1) until they decide to stop
    (stick=0) or exceed 21 (bust).
    After the player sticks, the dealer reveals their facedown card, and draws
    until their sum is 17 or greater.  If the dealer goes bust the player wins.
    If neither player nor dealer busts, the outcome (win, lose, draw) is
    decided by whose sum is closer to 21.  The reward for winning is +1,
    drawing is 0, and losing is -1.
    The observation of a 3-tuple of: the players current sum,
    the dealer's one showing card (1-10 where 1 is ace),
    and whether or not the player holds a usable ace (0 or 1).
    This environment corresponds to the version of the blackjack problem
    described in Example 5.1 in Reinforcement Learning: An Introduction
    by Sutton and Barto (1998).
    http://incompleteideas.net/sutton/book/the-book.html
    """
    
    # ACTION SPACES
    # 0- Stick
    # 1- Hit
    # 2- Double Down
    # 3- Split
    # 4- Surrender
    
    def __init__(self, numdecks = 4, natural=False, counting = False):
        print('HELLO')
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(32), # Possible hands player can have
            spaces.Discrete(11), # Possible up card dealer has
            spaces.Discrete(2), # True or False- Usable Ace
            spaces.Discrete(2))) # True or False- Can Double Down
        if counting:
            self.observation_space = spaces.Tuple((
                spaces.Discrete(32), # Possible hands player can have
                spaces.Discrete(11), # Possible up card dealer has
                spaces.Discrete(2), # True or False- Usable Ace
                spaces.Discrete(2), # True or False- Can Double Down
                spaces.Box(-100,100,dtype=int)))
        self._seed()
        self.actionstaken = 0 # Every move increases actions taken by 1

        # Flag to payout 1.5 on a "natural" blackjack win, like casino rules
        # Ref: http://www.bicyclecards.com/how-to-play/blackjack/
        self.natural = natural
        # Start the first game
        self.numdecks = numdecks
        
        
        self.decks = CARDS * self.numdecks
        random.shuffle(self.decks)

        self.running_count = 0
        self.counting = counting
        # print(self.decks)
        self._reset()


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        
        return [seed]

    def _step(self, action):
        assert self.action_space.contains(action)
        
        # CHECK IF WE HAVE ENOUGH CARDS
        if self._deck_is_out(self.decks):
            self.decks = CARDS * self.numdecks
            random.shuffle(self.decks)
            self.running_count = 0
            # print('RESET')
        # print(self.decks)
        
        if action == 0:  # stick: play out the dealers hand, and score
            done = True
            while sum_hand(self.dealer) < 17:
                self.dealer.append(draw_card(self.decks))
            reward = cmp(score(self.player), score(self.dealer))
            if is_natural(self.player) and reward == 1:
                reward = 1.5
            
            ddown = False
            self.actionstaken += 1  

            for i in range(1, len(self.dealer)):
                self.running_count += count(self.dealer[i])

        elif action == 1:  # hit: add a card to players hand and return
            self.player.append(draw_card(self.decks))
            self.running_count += count(self.player[-1])

            if is_bust(self.player):
                done = True
                reward = -1
                ddown = False
                self.actionstaken += 1
            else:
                done = False
                reward = 0
                ddown = False
                self.actionstaken += 1


                  

        elif action == 2: # double down: add a card to players hand and return
            assert(len(self.player) == 2)

            self.player.append(draw_card(self.decks))
            self.running_count += count(self.player[-1])

            if is_bust(self.player):
                done = True
                reward = -2
                ddown = False
                self.actionstaken += 1            
            else:
                while sum_hand(self.dealer) < 17:
                    self.dealer.append(draw_card(self.decks))
                reward = 2 * cmp(score(self.player), score(self.dealer))
                done = True
                ddown = False
                self.actionstaken += 1            
            
            for i in range(1, len(self.dealer)):
                self.running_count += count(self.dealer[i])


        # elif action == 3:
        #     assert(self.player[0] == self.player[1] and len(self.player) == 2)
        #     reward = 0
        #     done = False
        #     logging.info(type(self.player))
        #     logging.info(self.player)
        # else:
        #     ddown = False
        #     reward = 0
        #     done = False
        #     self.actionstaken += 1   
        # print(list(self.dealer))
        return self._get_obs(), reward, done, {}, ddown

    def _get_obs(self):
        # RETURNS: (PLAYER HANDS, DEALER UP CARD, USABLE ACE, CAN DOUBLE DOWN)
        if self.counting:
            return tuple(sorted(self.player)), self.dealer[0], usable_ace(self.player), can_double_down(self.player,  self.actionstaken), self.running_count
        # return tuple(sorted(self.player)), self.dealer[0], usable_ace(self.player), can_double_down(self.player,  self.actionstaken)
        return sum(self.player), self.dealer[0], usable_ace(self.player), can_double_down(self.player,  self.actionstaken)
    def _get_dealer_hand(self):
        return self.dealer
    
    def _deck_is_out(self, inputdeck):
        return len(inputdeck) < self.numdecks * len(CARDS) * 0.1


    def _reset(self):
        self.actionstaken = 0         

        # print(len(self.decks))
        self.dealer = draw_hand(self.decks)
        self.player = draw_hand(self.decks)
        
        self.running_count += count(self.dealer[0]) + count(self.player[0]) + count(self.player[1])


        return self._get_obs()