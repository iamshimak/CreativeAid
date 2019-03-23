from pattern.en import *

"""
conjugate(verb, 
    tense = PRESENT,        # INFINITIVE, PRESENT, PAST, FUTURE
   person = 3,              # 1, 2, 3 or None
   number = SINGULAR,       # SG, PL
     mood = INDICATIVE,     # INDICATIVE, IMPERATIVE, CONDITIONAL, SUBJUNCTIVE
   aspect = IMPERFECTIVE,   # IMPERFECTIVE, PERFECTIVE, PROGRESSIVE 
  negated = False,          # True or False
    parse = True)
"""

if __name__ == '__main__':
    print(conjugate('googled', tense=PARTICIPLE, parse=True))
