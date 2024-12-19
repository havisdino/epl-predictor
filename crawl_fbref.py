from argparse import ArgumentParser
import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm


TEAMS = [
    "Arsenal", "Aston Villa", "Chelsea",
    "Everton", "Leicester City", "Liverpool",
    "Manchester City", "Manchester Utd", "Newcastle Utd",
    "Southampton", "Tottenham", "West Ham"
]
INDICES = [i for i in range(len(TEAMS))]
TEAM_TO_INDEX = dict(zip(TEAMS, INDICES))


def crawl(url, team):
    assert team in TEAMS
    
    data = []
    
    response = requests.get(url)
    soup = BeautifulSoup(response.text, "html.parser")
    
    table = soup.find("table", {"id": "matchlogs_for"})

    for tr in table.find_all("tr"):
        opponent = tr.find(attrs={"data-stat": "opponent"}).get_text(strip=True)
        
        if opponent not in TEAMS:
            continue
        
        date = tr.find(attrs={"data-stat": "date"}).get_text(strip=True)
        venue = tr.find(attrs={"data-stat": "venue"}).get_text(strip=True)
        result = tr.find(attrs={"data-stat": "result"}).get_text(strip=True)
        
        sample = dict(
            date=date,
            venue=venue,
            result=result,
            team=team,
            opponent=opponent
        )
        
        data.append(sample)
    
    return data


def main(args):
    assert args.team in TEAMS
    
    with open(args.urls) as url_file, open(args.save_path, "w") as jsonl_file:
        for url in (bar := tqdm(url_file)):
            try:
                bar.set_description(f"Crawling `{url}`")
                data = crawl(url, args.team)
            except Exception as e:
                print(e.with_traceback)()
                continue
            for sample in data:
                jsonl_file.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--save-path", "-s")
    parser.add_argument("--team", "-t")
    parser.add_argument("--urls", "-u")
    
    main(parser.parse_args())
    