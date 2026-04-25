import json
import time
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin
from urllib.request import Request, urlopen


BASE_URL = "https://api.chess.com/pub/"


@dataclass
class ChessComClient:
    user_agent: str
    base_url: str = BASE_URL
    delay_seconds: float = 0.1

    def get_json(self, path_or_url: str) -> Dict:
        url = (
            path_or_url
            if path_or_url.startswith("http://") or path_or_url.startswith("https://")
            else urljoin(self.base_url, path_or_url.lstrip("/"))
        )
        req = Request(url, headers={"User-Agent": self.user_agent})
        try:
            with urlopen(req) as resp:
                payload = resp.read().decode("utf-8")
            if self.delay_seconds > 0:
                time.sleep(self.delay_seconds)
            return json.loads(payload)
        except HTTPError as exc:
            raise RuntimeError(f"HTTP error fetching {url}: {exc.code} {exc.reason}") from exc
        except URLError as exc:
            raise RuntimeError(f"Network error fetching {url}: {exc.reason}") from exc

    def get_tournament(self, tournament_id: str) -> Dict:
        return self.get_json(f"tournament/{tournament_id}")

    def get_round_games(self, tournament_id: str, round_num: int, group_num: int) -> Dict:
        return self.get_json(f"tournament/{tournament_id}/{round_num}/{group_num}")


def iter_round_group_pairs(tournament_payload: Dict) -> Iterator[tuple]:
    rounds: List = tournament_payload.get("rounds", [])
    for round_url in rounds:
        round_payload = round_url if isinstance(round_url, dict) else {}
        if not round_payload and isinstance(round_url, str):
            # For compatibility if rounds are URLs instead of inline objects.
            continue
        groups = round_payload.get("groups", [])
        round_num = int(round_payload.get("number", 0))
        for group in groups:
            group_num = int(group.get("number", 0))
            yield round_num, group_num


def normalize_white_result(game: Dict) -> str:
    white_result = game.get("white", {}).get("result")
    black_result = game.get("black", {}).get("result")

    if white_result == "win":
        return "win"
    if black_result == "win":
        return "loss"
    return "draw"


def extract_game_row(tournament_id: str, round_num: int, group_num: int, game: Dict) -> Dict:
    white = game.get("white", {})
    black = game.get("black", {})
    return {
        "tournament_id": tournament_id,
        "round": round_num,
        "group": group_num,
        "game_url": game.get("url"),
        "time_class": game.get("time_class"),
        "time_control": game.get("time_control"),
        "rated": game.get("rated"),
        "rules": game.get("rules"),
        "white_username": white.get("username"),
        "white_rating": white.get("rating"),
        "white_result_raw": white.get("result"),
        "black_username": black.get("username"),
        "black_rating": black.get("rating"),
        "black_result_raw": black.get("result"),
        "eco": game.get("eco"),
        "start_time": game.get("start_time"),
        "end_time": game.get("end_time"),
        "result_white_pov": normalize_white_result(game),
    }

