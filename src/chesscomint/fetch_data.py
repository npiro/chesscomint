import argparse
import json
from pathlib import Path
from typing import Dict, List, Set

from tqdm import tqdm
from chesscomint.api import ChessComClient, extract_game_row


DEFAULT_TOURNAMENTS = [
    "titled-tuesday-blitz-february-10-2026-6221327",
    "titled-tuesday-blitz-march-10-2026-6277141",
]
def fetch_tournament_games(client: ChessComClient, tournament_id: str) -> List[Dict]:
    tournament_payload = client.get_tournament(tournament_id)
    rows: List[Dict] = []

    for round_ref in tournament_payload.get("rounds", []):
        if isinstance(round_ref, str):
            round_payload = client.get_json(round_ref)
            round_number = int(round_payload.get("number", 0))
            groups = round_payload.get("groups", [])
        else:
            round_payload = round_ref
            round_number = int(round_payload.get("number", 0))
            groups = round_payload.get("groups", [])

        for group_ref in groups:
            if isinstance(group_ref, str):
                group_payload = client.get_json(group_ref)
                group_number = int(group_payload.get("number", 0))
            else:
                group_payload = group_ref
                group_number = int(group_payload.get("number", 0))
                if "games" not in group_payload and group_payload.get("url"):
                    group_payload = client.get_json(group_payload["url"])

            for game in group_payload.get("games", []):
                rows.append(extract_game_row(tournament_id, round_number, group_number, game))

    return rows


def extract_player_usernames(rows: List[Dict]) -> List[str]:
    usernames: Set[str] = set()
    for row in rows:
        white_username = row.get("white_username")
        black_username = row.get("black_username")
        if white_username:
            usernames.add(white_username)
        if black_username:
            usernames.add(black_username)
    return sorted(usernames)


def fetch_player_rows(client: ChessComClient, usernames: List[str], tournament_ids: List[str]) -> List[Dict]:
    player_rows: List[Dict] = []
    for username in tqdm(usernames, desc="Fetching player rows"):
        profile_payload = None
        stats_payload = None
        profile_error = None
        stats_error = None

        try:
            profile_payload = client.get_json(f"player/{username}")
        except RuntimeError as exc:
            profile_error = str(exc)

        try:
            stats_payload = client.get_json(f"player/{username}/stats")
        except RuntimeError as exc:
            stats_error = str(exc)

        player_rows.append(
            {
                "username": username,
                "tournament_ids": tournament_ids,
                "profile": profile_payload,
                "stats": stats_payload,
                "profile_error": profile_error,
                "stats_error": stats_error,
            }
        )
    return player_rows


def write_jsonl(rows: List[Dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_jsonl(input_path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Fetch Chess.com Titled Tuesday games.")
    parser.add_argument(
        "--tournament",
        action="append",
        dest="tournaments",
        help="Tournament id to fetch. Can be provided multiple times.",
    )
    parser.add_argument(
        "--output",
        default="data/raw/titled_tuesday_games.jsonl",
        help="Path to output JSONL file.",
    )
    parser.add_argument(
        "--players-output",
        default="data/raw/titled_tuesday_players.jsonl",
        help="Path to player metadata+stats JSONL file.",
    )
    parser.add_argument(
        "--players-only",
        action="store_true",
        help="Only fetch player metadata/stats from an existing games JSONL input.",
    )
    parser.add_argument(
        "--games-input",
        help="Path to an existing games JSONL file used by --players-only mode.",
    )
    parser.add_argument(
        "--user-agent",
        default="chesscomint-interview-exercise",
        help="User-Agent header required by Chess.com API.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    client = ChessComClient(user_agent=args.user_agent)
    all_rows: List[Dict] = []
    tournaments: List[str]

    if args.players_only:
        if not args.games_input:
            parser.error("--games-input is required when --players-only is set.")
        games_input_path = Path(args.games_input)
        if not games_input_path.exists():
            parser.error(f"--games-input file does not exist: {games_input_path}")
        all_rows = read_jsonl(games_input_path)
        tournaments = sorted({row.get("tournament_id") for row in all_rows if row.get("tournament_id")})
        print(f"Loaded {len(all_rows)} game rows from {games_input_path}")
    else:
        tournaments = args.tournaments or DEFAULT_TOURNAMENTS
        for tid in tournaments:
            rows = fetch_tournament_games(client, tid)
            all_rows.extend(rows)
            print(f"Fetched {len(rows)} games from {tid}")

        output_path = Path(args.output)
        write_jsonl(all_rows, output_path)
        print(f"Wrote {len(all_rows)} rows to {output_path}")

    usernames = extract_player_usernames(all_rows)
    player_rows = fetch_player_rows(client, usernames, tournaments)
    players_output_path = Path(args.players_output)
    write_jsonl(player_rows, players_output_path)
    print(f"Wrote {len(player_rows)} player rows to {players_output_path}")


if __name__ == "__main__":
    main()

