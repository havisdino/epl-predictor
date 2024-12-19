import os
import sys

from utils.data import TEAMS


teams = sorted(TEAMS)
team_shorthands = sorted("ars ast che eve lei liv mci mun new sou tot wes".split(" "))

assert(len(teams) == len(team_shorthands))


def execute(index):
    name = teams[index]
    name_shorthand = team_shorthands[index]
    
    cmd = f"python crawl_fbref.py -s jsonl/{name_shorthand}.jsonl -t \"{name}\" -u urls/{name_shorthand}.txt"
    return cmd


if __name__ == "__main__":
    i = int(sys.argv[1])
    cmd = execute(i)
    print(cmd)
    os.system(cmd)