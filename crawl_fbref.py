import json
from bs4 import BeautifulSoup
import cloudscraper
from tqdm import tqdm
from argparse import ArgumentParser


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
    
    scraper = cloudscraper.create_scraper()
    r = scraper.get(url)
    soup = BeautifulSoup(r.text, "html.parser")
    
    table = soup.find("table", {"id": "matchlogs_for"})

    for tr in table.find_all("tr"):
        opponent = tr.find(attrs={"data-stat": "opponent"}).get_text(strip=True)
        
        if opponent not in TEAMS:
            continue
        
        date = tr.find(attrs={"data-stat": "date"}).get_text(strip=True)
        venue = tr.find(attrs={"data-stat": "venue"}).get_text(strip=True)
        result = tr.find(attrs={"data-stat": "result"}).get_text(strip=True)
        goals_for = tr.find(attrs={"data-stat": "goals_for"}).get_text(strip=True)
        goals_against = tr.find(attrs={"data-stat": "goals_against"}).get_text(strip=True)
        
        sample = dict(
            date=date,
            venue=venue,
            result=result,
            team=team,
            opponent=opponent,
            goals_for=int(goals_for) if goals_for.isdigit() else 0,
            goals_against=int(goals_against) if goals_against.isdigit() else 0
        )
        
        data.append(sample)
    
    return data


def main(args):
    assert args.team in TEAMS
    
    with open(args.urls) as url_file, open(args.save_path, "w") as jsonl_file:
        for url in (bar := tqdm(url_file)):
            url = url.strip()
            try:
                bar.set_description(f"Crawling from `{url}`")
                data = crawl(url, args.team)
            except Exception as e:
                print(e)
                continue
            for sample in data:
                jsonl_file.write(json.dumps(sample) + "\n")


if __name__ == "__main__":
    parser = ArgumentParser()
    
    parser.add_argument("--save-path", "-s")
    parser.add_argument("--team", "-t")
    parser.add_argument("--urls", "-u")
    
    main(parser.parse_args())
    