import sys
from copy import deepcopy

from crossword import *

class CrosswordCreator():

    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy()
            for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont
        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size,
             self.crossword.height * cell_size),
            "black"
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border,
                     i * cell_size + cell_border),
                    ((j + 1) * cell_size - cell_border,
                     (i + 1) * cell_size - cell_border)
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        w, h = draw.textsize(letters[i][j], font=font)
                        draw.text(
                            (rect[0][0] + ((interior_size - w) / 2),
                             rect[0][1] + ((interior_size - h) / 2) - 10),
                            letters[i][j], fill="black", font=font
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        domains_temp = deepcopy(self.domains)

        for var in domains_temp:
            var_len = var.length

            for word in domains_temp[var]:
                if len(word) != var_len:
                    self.domains[var].remove(word)


    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        """
        x_overlap, y_overlap = self.crossword.overlaps[x, y]
        revised = False
        domains_temp = deepcopy(self.domains)

        if x_overlap:
            for x_word in domains_temp[x]:
                is_consistent = False
                for y_word in self.domains[y]:
                    if x_word[x_overlap] == y_word[y_overlap]:
                        is_consistent = True
                        break
                if not is_consistent:
                    self.domains[x].remove(x_word)
                    revised = True

        return revised

        raise NotImplementedError

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        
        if arcs is None:
            arcs = []
            for var_x in self.domains:
                for var_y in self.crossword.neighbors(var_x):
                    if self.crossword.overlaps[var_x, var_y]:
                        arcs.append((var_x, var_y))
        
        while len(arcs) > 0:
            var_x, var_y = arcs.pop()

            if self.revise(var_x, var_y):
                # No solution if x domain is empty after revision wrt y
                if len(self.domains[var_x]) == 0:
                    return False

                # Add all neighbors of x to arcs
                for neighbor in self.crossword.neighbors(var_x):
                    if neighbor != var_y:
                        arcs.append((neighbor, var_x))
        
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for var in self.domains:
            if var in assignment:
                return True
        return False

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        words = [*assignment.values()]
        distinct = (len(words) == len(set(words)))

        if not distinct:
            return False

        for var in assignment:
            consistent_length = var.length == len(assignment[var])
            if not consistent_length:
                return False
            for neighbor in self.crossword.neighbors(var):
                if neighbor in assignment:
                    x_overlap, y_overlap = self.crossword.overlaps[var, neighbor]
                    if assignment[var][x_overlap] != assignment[neighbor][y_overlap]:
                        return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        ruled_out = {word:0 for word in self.domains[var]}

        neighbors = self.crossword.neighbors(var)
        for word in self.domains[var]:
            for neighbor in neighbors:
                if neighbor in assignment:
                    continue
                else:
                    x_overlap, y_overlap = self.crossword.overlaps[var, neighbor]
                    for neighbor_word in self.domains[neighbor]:
                        if word[x_overlap] != neighbor_word[y_overlap]:
                            ruled_out[word] += 1
        
        return sorted([x for x in ruled_out], key = lambda x:ruled_out[x])

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        unassigned_vars = list(set(self.domains.keys()) - set(assignment.keys()))
        res = unassigned_vars
        res.sort(key = lambda x: (len(self.domains[x]), -len(self.crossword.neighbors(x))))
        #res = sorted(unassigned_vars, key=lambda x:(len(self.domains[x]), -len(self.crossword.neighbors(x))))
        return res[0]

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """
        if len(assignment) == len(self.domains):
            return assignment

        var = self.select_unassigned_variable(assignment)

        for word in self.order_domain_values(var, assignment):
            assignment_temp = assignment.copy()
            assignment_temp[var] = word
            if self.consistent(assignment_temp):
                res = self.backtrack(assignment_temp)
                if res:
                    return res

        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
