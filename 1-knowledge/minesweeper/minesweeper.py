import itertools
import random
from copy import deepcopy


class Minesweeper():
    """
    Minesweeper game representation
    """

    def __init__(self, height=8, width=8, mines=8):

        # Set initial width, height, and number of mines
        self.height = height
        self.width = width
        self.mines = set()

        # Initialize an empty field with no mines
        self.board = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                row.append(False)
            self.board.append(row)

        # Add mines randomly
        while len(self.mines) != mines:
            i = random.randrange(height)
            j = random.randrange(width)
            if not self.board[i][j]:
                self.mines.add((i, j))
                self.board[i][j] = True

        # At first, player has found no mines
        self.mines_found = set()

    def print(self):
        """
        Prints a text-based representation
        of where mines are located.
        """
        for i in range(self.height):
            print("--" * self.width + "-")
            for j in range(self.width):
                if self.board[i][j]:
                    print("|X", end="")
                else:
                    print("| ", end="")
            print("|")
        print("--" * self.width + "-")

    def is_mine(self, cell):
        i, j = cell
        return self.board[i][j]

    def nearby_mines(self, cell):
        """
        Returns the number of mines that are
        within one row and column of a given cell,
        not including the cell itself.
        """

        # Keep count of nearby mines
        count = 0

        # Loop over all cells within one row and column
        for i in range(cell[0] - 1, cell[0] + 2):
            for j in range(cell[1] - 1, cell[1] + 2):

                # Ignore the cell itself
                if (i, j) == cell:
                    continue

                # Update count if cell in bounds and is mine
                if 0 <= i < self.height and 0 <= j < self.width:
                    if self.board[i][j]:
                        count += 1

        return count

    def won(self):
        """
        Checks if all mines have been flagged.
        """
        return self.mines_found == self.mines


class Sentence():
    """
    Logical statement about a Minesweeper game
    A sentence consists of a set of board cells,
    and a count of the number of those cells which are mines.
    """

    def __init__(self, cells, count):
        self.cells = set(cells)
        self.count = count

    def __eq__(self, other):
        return self.cells == other.cells and self.count == other.count

    def __str__(self):
        return f"{self.cells} = {self.count}"

    def known_mines(self):
        """
        If the count of number of self.cells that are mines 
        is equal to the 
        number of self.cells
        Then the board cells are mines

        Returns the set of all cells in self.cells known to be mines.
        """
        if self.count == len(self.cells):
            return self.cells

    def known_safes(self):
        """
        If the count of number of self.cells that are mines
        is equal to
        0
        Then the board cells are safe

        Returns the set of all cells in self.cells known to be safe.
        """
        if self.count == 0:
            return self.cells


    def mark_mine(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be a mine.

        First check if cell is one of the cells in the sentence
        If cell is in sentence:
            1. Update the sentence so that cell is no longer inside
            2. Represent a logically correct sentence that cell is a mine
        Otherwise, do nothing
        """
        if cell not in self.cells:
            return
        
        self.cells.remove(cell)
        self.count -= 1

    def mark_safe(self, cell):
        """
        Updates internal knowledge representation given the fact that
        a cell is known to be safe.

        First check if cell is one of the cells in the sentence
        If cell is in sentence:
            1. Update the sentence so that cell is no longer inside
            2. Represent a logically correct sentence that cell is safe
        Otherwise, do nothing
        """
        if cell not in self.cells:
            return
        self.cells.remove(cell)


class MinesweeperAI():
    """
    Minesweeper game player
    """

    def __init__(self, height=8, width=8):

        # Set initial height and width
        self.height = height
        self.width = width

        # Keep track of which cells have been clicked on
        self.moves_made = set()

        # Keep track of cells known to be safe or mines
        self.mines = set()
        self.safes = set()

        # List of sentences about the game known to be true
        self.knowledge = []

    def mark_mine(self, cell):
        """
        Marks a cell as a mine, and updates all knowledge
        to mark that cell as a mine as well.
        """
        self.mines.add(cell)
        for sentence in self.knowledge:
            sentence.mark_mine(cell)

    def mark_safe(self, cell):
        """
        Marks a cell as safe, and updates all knowledge
        to mark that cell as safe as well.
        """
        self.safes.add(cell)
        for sentence in self.knowledge:
            sentence.mark_safe(cell)

    def add_knowledge(self, cell, count):
        """
        Called when the Minesweeper board tells us, for a given
        safe cell, how many neighboring cells have mines in them.

        This function should:
            1) mark the cell as a move that has been made
            2) mark the cell as safe
            3) add a new sentence to the AI's knowledge base
               based on the value of `cell` and `count`
            4) mark any additional cells as safe or as mines
               if it can be concluded based on the AI's knowledge base
            5) add any new sentences to the AI's knowledge base
               if they can be inferred from existing knowledge
        """
        # 1) Mark the call as a move that has been made
        self.moves_made.add(cell)
        # 2) Mark the cell as safe
        self.mark_safe(cell)
        # 3) Add a new sentence to the AI's knowledge based on the value of `cell` and `count`
        self.add_new_sentence_to_kb(cell, count)
        # 4) Mark any additional cells as safe or mines based on knowledge base
        self.mark_cells_based_on_kb()
        # 5) Add new sentences to knowledge base if can be inferred
        self.add_inferred_sentences_to_kb()

    def make_safe_move(self):
        """
        Returns a safe cell to choose on the Minesweeper board.
        The move must be known to be safe, and not already a move
        that has been made.

        This function may use the knowledge in self.mines, self.safes
        and self.moves_made, but should not modify any of those values.
        """
        if len(self.safes) > 0:
            for cell in self.safes - self.moves_made:
                return cell
        return None

    def make_random_move(self):
        """
        Returns a move to make on the Minesweeper board.
        Should choose randomly among cells that:
            1) have not already been chosen, and
            2) are not known to be mines
        """
        possible_moves = []

        for i in range(self.height):
            for j in range(self.width):
                possible_move = (i, j)
                if possible_move not in self.moves_made and possible_move not in self.mines:
                    possible_moves.append(possible_move)

        if len(possible_moves) > 0:
            random_idx = random.randrange(len(possible_moves))
            random_move = possible_moves[random_idx]
            return random_move
        return None


    def add_new_sentence_to_kb(self, cell, count):
        '''
        Add new sentence to knowledge base based on the cell's neighbourhood and number of mines.
        Only cells with undetermined states are included in the new sentence
        '''
        neighbours = self.get_neighbours(cell)
        neighbours_temp = deepcopy(neighbours)

        for neighbour in neighbours_temp:
            if neighbour in self.mines or neighbour in self.safes:
                neighbours.remove(neighbour)
                if neighbour in self.mines:
                    count -= 1

        new_sentence = Sentence(neighbours, count)

        # Add to knowledge base if sentence is not empty
        if len(new_sentence.cells) > 0:
            self.knowledge.append(new_sentence)

    def mark_cells_based_on_kb(self):
        knowledge_temp = deepcopy(self.knowledge)

        for sentence in knowledge_temp:
            known_mines = sentence.known_mines()
            known_safes = sentence.known_safes()
            
            if known_mines is not None:
                for known_mine in known_mines:
                    self.mark_mine(known_mine)

            if known_safes is not None:
                for known_safe in known_safes:
                    self.mark_safe(known_safe)

    def add_inferred_sentences_to_kb(self):
        '''
        Add inferred sentences to knowledge base
        Iterate through pairs of sentences: 
            if either one is a subset of the other:
                A new sentence can be inferred using the difference between the two sentences
        '''
        knowledge_temp = deepcopy(self.knowledge)

        for sentence_1 in knowledge_temp:
            for sentence_2 in knowledge_temp:
                if sentence_1.__eq__(sentence_2):
                    continue

                if sentence_1.cells.issubset(sentence_2.cells):
                    inferred_cells = sentence_2.cells - sentence_1.cells
                    inferred_count = sentence_2.count - sentence_1.count
                    inferred_sentence = Sentence(inferred_cells, inferred_count)

                    if inferred_sentence not in self.knowledge:
                        self.knowledge.append(inferred_sentence)

                if sentence_2.cells.issubset(sentence_1.cells):
                    inferred_cells = sentence_1.cells - sentence_2.cells
                    inferred_count = sentence_1.count - sentence_2.count
                    inferred_sentence = Sentence(inferred_cells, inferred_count)

                    if inferred_sentence not in self.knowledge:
                        self.knowledge.append(inferred_sentence)
               

    def get_neighbours(self, cell):
        '''
        Return a set containing the neighbours of the cell passed in
        '''
        neighbours = set()

        for i in range(self.height):
            for j in range(self.width):
                if (i, j) == cell:
                    continue

                if abs(cell[0] - i) <= 1 and abs(cell[1] - j) <= 1:
                    neighbours.add((i, j))

        return neighbours
