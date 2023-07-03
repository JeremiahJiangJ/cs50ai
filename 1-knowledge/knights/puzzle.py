from logic import *

'''
Knight tells truth
Knave tells lies
'''
AKnight = Symbol("A is a Knight")
AKnave = Symbol("A is a Knave")

BKnight = Symbol("B is a Knight")
BKnave = Symbol("B is a Knave")

CKnight = Symbol("C is a Knight")
CKnave = Symbol("C is a Knave")

# Puzzle 0
# A says "I am both a knight and a knave."
'''
1. A is either a knight or a knave, A cannot be both simultaneously
2. If A is a knight, A is both a knight and a knave
3. If A is a knave, A is not both a knight and a knave

So A is a knave since A cannot be both a knight and a knave
Note:
2 & 3 can be combined into a biconditional rather than using two implications
since AKnight == Not(AKnave)
'''
statement0 = And(AKnight,AKnave)
knowledge0 = And(
    #And(Or(AKnight, AKnave), Not(statement0)),
    Biconditional(AKnight, Not(AKnave)),
    #Implication(AKnight, statement0),
    #Implication(AKnave, Not(statement0))
    Biconditional(AKnight, statement0)
)

# Puzzle 1
# A says "We are both knaves."
# B says nothing.
'''
1. A is either a knight or a knave, A cannot be both simultaneously
2. B is either a knight or a knave, B cannot be both simultaneously
3. If A is a knight, A and B are both knaves
4. If A is a knave, A and B are not both knaves

So A is a knave and B is a knight since A cannot be both a knight and a knave
Note:
3 & 4 can be combined into a biconditional rather than using two implications
since AKnight == Not(AKnave)
'''
statement1 = And(AKnave, BKnave)
knowledge1 = And(
    #And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    #And(Or(BKnight, BKnave), Not(And(BKnight, BKnave))),
    Biconditional(AKnight, Not(AKnave)),
    Biconditional(BKnight, Not(BKnave)),
    #Implication(AKnight, statement1),
    #Implication(AKnave, Not(statement1))
    Biconditional(AKnight, statement1)
)

# Puzzle 2
# A says "We are the same kind."
# B says "We are of different kinds."
'''
1. A is either a knight or a knave, A cannot be both simultaneously
2. B is either a knight or a knave, B cannot be both simultaneously
3. If A is a knight, A and B are both knights/knaves
4. If A is a knave, A and B are different from each other
5. If B is a knight, A and B are different from each other
6. If B is a knave, A and B are both knights/knaves

So A is a Knave and B is a Knight since if A is a knight, then B is a knight but B says they are not the same (i.e lie)
Note:
3 & 4 can combine into 1 biconditional
5 & 6 can combine into 1 biconditional
'''
statement_two_by_A = Or(And(AKnave, BKnave), And(AKnight, BKnight))
statement_two_by_B = Or(And(AKnave, BKnight), And(AKnight, BKnave))
knowledge2 = And(
    #And(Or(AKnight, AKnave), Not(And(AKnight, AKnave))),
    #And(Or(BKnight, BKnave), Not(And(BKnight, BKnave))),
    Biconditional(AKnight, Not(AKnave)),
    Biconditional(BKnight, Not(BKnave)),
    #Implication(AKnight, statement_two_by_A),
    #Implication(AKnave, Not(statement_two_by_A)),
    #Implication(BKnight, statement_two_by_B),
    #Implication(BKnight, Not(statement_two_by_B))
    Biconditional(AKnight, statement_two_by_A),
    Biconditional(BKnight, statement_two_by_B)
)

# Puzzle 3
# A says either "I am a knight." or "I am a knave.", but you don't know which.
# B says "A said 'I am a knave'."
# B says "C is a knave."
# C says "A is a knight."
'''
1. A is either a knight or a knave, A cannot be both simultaneously
2. B is either a knight or a knave, B cannot be both simultaneously
3. C is either a knight or a knave, C cannot be both simultaneously
4. If B is a knight, A said he is a knave and C is a knave
    If A is a knight, then A is a knave
    If A is a knave, then A is a knight
5. If B is a knave, A said he is a knight and C is a knight
    If A is a knight, then A is a knight
    If A is a knave, then A is knave
6. If C is a knave, then A is a knave
7. If C is a knight, then A is a knight

So B is a Knave, A and C are knights since A cannot be both a knight and a knave
'''
statement_three_by_B = And(AKnave, CKnave)
statement_three_by_C = AKnight
knowledge3 = And(
    Biconditional(AKnight, Not(AKnave)),
    Biconditional(BKnight, Not(BKnave)),
    Biconditional(CKnight, Not(CKnave)),
    # If B is a Knight, C is a Knave and A is both a Knight and a Knave
    Implication(BKnight, CKnave), Implication(BKnight,Biconditional(AKnight, AKnave)),
    # If B is a Knave, C is a Knight and A is either a knight or a knave
    Implication(BKnave, CKnight), Implication(BKnave, Biconditional(AKnight, Not(AKnave))),
    Biconditional(CKnight, AKnight)
)


def main():
    symbols = [AKnight, AKnave, BKnight, BKnave, CKnight, CKnave]
    puzzles = [
        ("Puzzle 0", knowledge0),
        ("Puzzle 1", knowledge1),
        ("Puzzle 2", knowledge2),
        ("Puzzle 3", knowledge3)
    ]
    for puzzle, knowledge in puzzles:
        print(puzzle)
        if len(knowledge.conjuncts) == 0:
            print("    Not yet implemented.")
        else:
            for symbol in symbols:
                if model_check(knowledge, symbol):
                    print(f"    {symbol}")


if __name__ == "__main__":
    main()
