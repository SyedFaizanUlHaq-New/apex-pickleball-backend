from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

app = FastAPI(title="APEX Pickleball Generator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Player(BaseModel):
    name: str
    level: float
    gender: str = Field(description="H ou F")
    anchor: bool = False
    fixedCourt: Optional[int] = None
    incompatibleWith: List[str] = Field(default_factory=list)
    lockedInAllRounds: bool = False


class Settings(BaseModel):
    courts: int = 12
    rounds: int = 6
    levelTolerance: float = 0.4
    maxPartnerRepeat: int = 0
    maxOpponentRepeat: int = 2
    minimumMixedMatches: int = 2


class GenerateScheduleRequest(BaseModel):
    eventId: str
    players: List[Player]
    settings: Settings


class MatchOut(BaseModel):
    round: int
    time: str
    court: int
    teamA: List[str]
    teamB: List[str]
    qualityFlags: List[str] = Field(default_factory=list)


ROUND_TIMES = ["10:00", "10:20", "10:40", "11:00", "11:20", "11:40"]


def pair_key(a: str, b: str) -> Tuple[str, str]:
    return tuple(sorted((a, b)))


def avg_level(players: List[Player]) -> float:
    if not players:
        return 0.0
    return sum(p.level for p in players) / len(players)


def is_mixed(team: List[Player]) -> bool:
    genders = {p.gender for p in team}
    return "H" in genders and "F" in genders


def split_teams_best(
    group: List[Player],
    partner_counts: Dict[Tuple[str, str], int],
    opponent_counts: Dict[Tuple[str, str], int],
    settings: Settings,
) -> Tuple[List[Player], List[Player], List[str], float]:
    best_score = float("-inf")
    best_result = None

    candidates = [
        ([group[0], group[1]], [group[2], group[3]]),
        ([group[0], group[2]], [group[1], group[3]]),
        ([group[0], group[3]], [group[1], group[2]]),
    ]

    for team_a, team_b in candidates:
        score = 100.0
        flags: List[str] = []

        # Priorité 1: jamais même partenaire
        partner_repeat_a = partner_counts[pair_key(team_a[0].name, team_a[1].name)]
        partner_repeat_b = partner_counts[pair_key(team_b[0].name, team_b[1].name)]
        if partner_repeat_a > settings.maxPartnerRepeat:
            score -= 1000
            flags.append("Partenaires répétés équipe A")
        if partner_repeat_b > settings.maxPartnerRepeat:
            score -= 1000
            flags.append("Partenaires répétés équipe B")

        # Priorité 2: équilibre de niveau
        level_gap = abs(avg_level(team_a) - avg_level(team_b))
        score -= level_gap * 100
        if level_gap > 0.25:
            flags.append("Équipes possiblement déséquilibrées")

        # Priorité 3: adversaires répétés
        for pa in team_a:
            for pb in team_b:
                repeats = opponent_counts[pair_key(pa.name, pb.name)]
                if repeats >= settings.maxOpponentRepeat:
                    score -= 200
                    flags.append(f"Adversaires répétés: {pa.name} vs {pb.name}")
                else:
                    score -= repeats * 20

        # Bonus mixte
        if is_mixed(team_a) or is_mixed(team_b):
            score += 10

        if score > best_score:
            best_score = score
            best_result = (team_a, team_b, list(dict.fromkeys(flags)), score)

    if best_result is None:
        raise ValueError("Impossible de créer les équipes")

    return best_result


def score_companion(
    anchor: Player,
    candidate: Player,
    partner_counts: Dict,
    opponent_counts: Dict,
    round_index: int,
) -> float:
    """Score pour choisir le meilleur compagnon pour un joueur ancré — favorise la variété."""
    score = 100.0

    # Pénalité forte si déjà partenaire
    score -= partner_counts[pair_key(anchor.name, candidate.name)] * 500

    # Pénalité si déjà adversaire souvent
    score -= opponent_counts[pair_key(anchor.name, candidate.name)] * 50

    # Légère pénalité si niveau très différent
    level_gap = abs(anchor.level - candidate.level)
    score -= level_gap * 30

    # Bonus mixte
    if anchor.gender != candidate.gender:
        score += 15

    # Petit aléatoire pour éviter toujours le même ordre
    score += random.uniform(0, 5)

    return score


def generate_groups_for_round(
    available: List[Player],
    settings: Settings,
    partner_counts: Dict[Tuple[str, str], int],
    opponent_counts: Dict[Tuple[str, str], int],
    round_index: int,
) -> Tuple[List[MatchOut], List[str]]:
    warnings: List[str] = []
    matches: List[MatchOut] = []

    anchored = [p for p in available if p.anchor and p.fixedCourt is not None]
    others = [p for p in available if p not in anchored]

    used_names = set()
    court_to_group: Dict[int, List[Player]] = {}

    # Joueurs ancrés sur leur terrain fixe
    for p in anchored:
        if p.name in used_names:
            continue
        court_to_group.setdefault(p.fixedCourt, []).append(p)
        used_names.add(p.name)

    # Pour chaque terrain ancré, choisir les 3 meilleurs compagnons avec rotation maximale
    for court, group in sorted(court_to_group.items()):
        anchor = group[0]
        while len(group) < 4:
            candidates = [o for o in others if o.name not in used_names]
            if not candidates:
                break
            # Trier par score de variété (évite répétitions)
            candidates.sort(
                key=lambda o: score_companion(anchor, o, partner_counts, opponent_counts, round_index),
                reverse=True,
            )
            chosen = candidates[0]
            group.append(chosen)
            used_names.add(chosen.name)

    # Groupes libres — mélange par niveau mais avec rotation
    remaining = [o for o in others if o.name not in used_names]

    # Mélange léger pour éviter toujours le même ordre de niveau
    remaining.sort(key=lambda p: (-p.level + random.uniform(-0.1, 0.1)))

    next_courts = [c for c in range(2, settings.courts + 20) if c not in court_to_group]

    for court in next_courts:
        if len(remaining) < 4:
            break

        seed = remaining.pop(0)
        group = [seed]

        # Choisir les 3 meilleurs compagnons avec variété
        remaining.sort(
            key=lambda o: score_companion(seed, o, partner_counts, opponent_counts, round_index),
            reverse=True,
        )
        while len(group) < 4 and remaining:
            group.append(remaining.pop(0))

        court_to_group[court] = group

    # Générer les matchs
    for court in sorted(court_to_group):
        group = court_to_group[court]
        if len(group) < 4:
            warnings.append(f"Terrain {court}: groupe incomplet")
            continue

        team_a, team_b, flags, _score = split_teams_best(
            group, partner_counts, opponent_counts, settings
        )

        matches.append(
            MatchOut(
                round=round_index + 1,
                time=ROUND_TIMES[min(round_index, len(ROUND_TIMES) - 1)],
                court=court,
                teamA=[p.name for p in team_a],
                teamB=[p.name for p in team_b],
                qualityFlags=flags,
            )
        )

        # Mise à jour des compteurs
        partner_counts[pair_key(team_a[0].name, team_a[1].name)] += 1
        partner_counts[pair_key(team_b[0].name, team_b[1].name)] += 1
        for pa in team_a:
            for pb in team_b:
                opponent_counts[pair_key(pa.name, pb.name)] += 1

    return matches, warnings


def choose_active_players(players: List[Player], courts: int, round_index: int) -> List[Player]:
    max_active = courts * 4
    if len(players) <= max_active:
        return players

    sorted_players = sorted(
        players, key=lambda p: (not p.anchor, p.fixedCourt is None, -p.level, p.name)
    )
    overflow = len(players) - max_active

    anchors = [p for p in sorted_players if p.anchor]
    flexible = [p for p in sorted_players if not p.anchor]

    if overflow > len(flexible):
        raise ValueError("Trop de joueurs ancrés pour le nombre de terrains")

    bench_start = (round_index * overflow) % max(1, len(flexible))
    benched = set()
    for i in range(overflow):
        benched.add(flexible[(bench_start + i) % len(flexible)].name)

    active = anchors + [p for p in flexible if p.name not in benched]
    return active[:max_active]


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/generate-schedule")
def generate_schedule(req: GenerateScheduleRequest) -> Dict[str, object]:
    if len(req.players) < 4:
        raise HTTPException(status_code=400, detail="Au moins 4 joueurs sont requis")

    partner_counts: Dict[Tuple[str, str], int] = defaultdict(int)
    opponent_counts: Dict[Tuple[str, str], int] = defaultdict(int)

    all_matches: List[MatchOut] = []
    all_warnings: List[str] = []

    try:
        for round_index in range(req.settings.rounds):
            active_players = choose_active_players(
                req.players, req.settings.courts, round_index
            )
            round_matches, warnings = generate_groups_for_round(
                available=active_players,
                settings=req.settings,
                partner_counts=partner_counts,
                opponent_counts=opponent_counts,
                round_index=round_index,
            )
            all_matches.extend(round_matches)
            all_warnings.extend(warnings)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    partner_repeats = sum(1 for _pair, count in partner_counts.items() if count > 1)
    opponent_over_limit = sum(
        1 for _pair, count in opponent_counts.items() if count > req.settings.maxOpponentRepeat
    )

    return {
        "eventId": req.eventId,
        "matches": [m.model_dump() for m in all_matches],
        "summary": {
            "partnerRepeats": partner_repeats,
            "opponentOverLimit": opponent_over_limit,
            "totalMatches": len(all_matches),
        },
        "warnings": all_warnings,
    }

# Pour lancer en local:
# python -m uvicorn apex_pickleball_backend:app --reload