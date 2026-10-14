

Commandes :
    python Jeu_d_echecV4.py --train 20
    python Jeu_d_echecV4.py --train 100
    python Jeu_d_echecV4.py --eval 50
    python Jeu_d_echecV4.py
"""

import argparse
import math
import random
from collections import deque
from pathlib import Path

import chess
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PARAMÈTRES
# ============================================================

TAILLE_CASE = 90
LARGEUR = 8 * TAILLE_CASE
HAUTEUR_PLATEAU = 8 * TAILLE_CASE
HAUTEUR_INFOS = 100
HAUTEUR = HAUTEUR_PLATEAU + HAUTEUR_INFOS

FPS = 60
TEMPS_REFLEXION_IA = 500  # ms

# 12 types de pièces x 64 cases + tour + 4 droits de roque
TAILLE_ENTREE = 773

# Action = case_depart * 64 + case_arrivee
# 64 x 64 = 4096 actions possibles.
# Pour simplifier, une promotion est toujours faite en dame.
TAILLE_POLICY = 4096

FICHIER_MODELE = Path(__file__).resolve().with_name("modele_echecs_V4.pt")

TAUX_APPRENTISSAGE = 3e-4
TAILLE_BATCH = 128
TAILLE_REPLAY = 20000
BATCHS_PAR_PARTIE = 4
MAX_DEMI_COUPS = 240

POIDS_VALUE = 0.5
POIDS_ENTROPIE = 0.01


# ============================================================
# COULEURS
# ============================================================

CASE_CLAIRE = (170, 170, 170)
CASE_FONCEE = (90, 120, 80)
SELECTION = (246, 246, 105)
COUP_POSSIBLE = (80, 80, 80)
COUP_CAPTURE = (180, 50, 50)
FOND_INFOS = (30, 30, 30)
TEXTE = (240, 240, 240)
BLANC = (245, 245, 245)
NOIR = (25, 25, 25)

PIECES = {
    "P": "♙", "N": "♘", "B": "♗", "R": "♖", "Q": "♕", "K": "♔",
    "p": "♟", "n": "♞", "b": "♝", "r": "♜", "q": "♛", "k": "♚",
}


# ============================================================
# RÉSEAU POLICY + VALUE
# ============================================================

class ReseauEchecsV4(nn.Module):
    """
    Le réseau analyse une position puis produit :

    POLICY : 4096 scores de coups.
    VALUE  : un nombre entre -1 et +1.

    Le VALUE est vu du point de vue du joueur qui doit jouer :
        +1 = très favorable
         0 = équilibré
        -1 = très défavorable
    """

    def __init__(self):
        super().__init__()

        self.tronc = nn.Sequential(
            nn.Linear(TAILLE_ENTREE, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.policy = nn.Linear(128, TAILLE_POLICY)

        self.value = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        representation = self.tronc(x)
        logits_policy = self.policy(representation)
        valeur = self.value(representation)
        return logits_policy, valeur


# ============================================================
# ENCODAGE DU PLATEAU
# ============================================================

def encoder_plateau(board):
    """Transforme une position en 773 nombres."""

    x = torch.zeros(TAILLE_ENTREE, dtype=torch.float32)

    for case, piece in board.piece_map().items():
        offset = 0 if piece.color == chess.WHITE else 6
        type_piece = offset + piece.piece_type - 1
        x[type_piece * 64 + case] = 1.0

    x[768] = 1.0 if board.turn == chess.WHITE else -1.0
    x[769] = float(board.has_kingside_castling_rights(chess.WHITE))
    x[770] = float(board.has_queenside_castling_rights(chess.WHITE))
    x[771] = float(board.has_kingside_castling_rights(chess.BLACK))
    x[772] = float(board.has_queenside_castling_rights(chess.BLACK))

    return x


def encoder_coup(coup):
    return coup.from_square * 64 + coup.to_square


def coups_legaux_uniques(board):
    """
    Associe chaque indice d'action à un coup légal.

    Pour les promotions partageant le même départ/arrivée,
    on privilégie la promotion en dame.
    """

    actions = {}

    for coup in board.legal_moves:
        indice = encoder_coup(coup)

        if indice not in actions or coup.promotion == chess.QUEEN:
            actions[indice] = coup

    return actions


# ============================================================
# RÉSULTAT / SIGNAL D'APPRENTISSAGE
# ============================================================

def resultat_officiel_blancs(board):
    """+1 Blancs, -1 Noirs, 0 nul, None si partie non terminée."""

    outcome = board.outcome(claim_draw=True)

    if outcome is None:
        return None
    if outcome.winner == chess.WHITE:
        return 1.0
    if outcome.winner == chess.BLACK:
        return -1.0
    return 0.0


VALEUR_PIECE = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.2,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


def score_materiel_blancs(board):
    """
    Faible signal de secours si le self-play atteint la limite
    de coups sans résultat officiel.

    Ce score ne choisit pas directement les coups.
    """

    score = 0.0

    for piece_type, valeur in VALEUR_PIECE.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * valeur
        score -= len(board.pieces(piece_type, chess.BLACK)) * valeur

    return 0.35 * math.tanh(score / 8.0)


# ============================================================
# PRÉDICTION / CHOIX D'UN COUP
# ============================================================

def prediction_reseau(board, modele, device):
    x = encoder_plateau(board).unsqueeze(0).to(device)

    modele.eval()
    with torch.no_grad():
        logits, valeur = modele(x)

    return logits[0], float(valeur[0, 0].item())


def choisir_coup_reseau(
    board,
    modele,
    device,
    epsilon=0.0,
    temperature=0.0,
):
    """
    epsilon > 0 : chance de jouer totalement au hasard.
    temperature > 0 : échantillonnage stochastique de la policy.
    temperature = 0 : meilleur coup selon la policy.
    """

    actions = coups_legaux_uniques(board)

    if not actions:
        return None, None, 0.0, []

    indices_legaux = list(actions.keys())
    logits, valeur = prediction_reseau(board, modele, device)

    if epsilon > 0 and random.random() < epsilon:
        indice = random.choice(indices_legaux)
        return actions[indice], indice, valeur, indices_legaux

    logits_legaux = logits[indices_legaux]

    if temperature <= 0:
        position = int(torch.argmax(logits_legaux).item())
    else:
        probabilites = F.softmax(logits_legaux / temperature, dim=0)
        position = int(torch.multinomial(probabilites, 1).item())

    indice = indices_legaux[position]
    return actions[indice], indice, valeur, indices_legaux


# ============================================================
# SAUVEGARDE / CHARGEMENT
# ============================================================

def sauvegarder_modele(modele, parties_entrainees):
    torch.save(
        {
            "version": 4,
            "model_state_dict": modele.state_dict(),
            "parties_entrainees": parties_entrainees,
        },
        FICHIER_MODELE,
    )


def charger_modele(device):
    modele = ReseauEchecsV4().to(device)
    parties_entrainees = 0
    existe = False

    if FICHIER_MODELE.exists():
        checkpoint = torch.load(FICHIER_MODELE, map_location=device)
        modele.load_state_dict(checkpoint["model_state_dict"])
        parties_entrainees = int(checkpoint.get("parties_entrainees", 0))
        existe = True

    return modele, parties_entrainees, existe


# ============================================================
# SELF-PLAY
# ============================================================

def jouer_partie_self_play(modele, device, epsilon, temperature):
    """Le même réseau joue les Blancs et les Noirs."""

    board = chess.Board()
    trajectoire = []

    for _ in range(MAX_DEMI_COUPS):
        if board.is_game_over(claim_draw=True):
            break

        joueur = board.turn

        coup, action, _, indices_legaux = choisir_coup_reseau(
            board,
            modele,
            device,
            epsilon=epsilon,
            temperature=temperature,
        )

        if coup is None:
            break

        trajectoire.append(
            (
                encoder_plateau(board),
                action,
                joueur,
                indices_legaux,
            )
        )

        board.push(coup)

    resultat = resultat_officiel_blancs(board)

    if resultat is None:
        resultat = score_materiel_blancs(board)
        type_resultat = "limite"
    elif resultat > 0:
        type_resultat = "blancs"
    elif resultat < 0:
        type_resultat = "noirs"
    else:
        type_resultat = "nul"

    experiences = []

    for etat, action, joueur, indices_legaux in trajectoire:
        # Récompense du point de vue du joueur qui devait jouer.
        recompense = resultat if joueur == chess.WHITE else -resultat

        experiences.append(
            (etat, action, float(recompense), indices_legaux)
        )

    return experiences, resultat, type_resultat, board.ply()


# ============================================================
# APPRENTISSAGE ACTOR-CRITIC
# ============================================================

def apprendre_depuis_replay(modele, optimiseur, replay, device):
    if len(replay) < 8:
        return None

    modele.train()

    pertes_total = []
    pertes_policy = []
    pertes_value = []

    for _ in range(BATCHS_PAR_PARTIE):
        taille = min(TAILLE_BATCH, len(replay))
        batch = random.sample(replay, taille)

        etats = torch.stack([e[0] for e in batch]).to(device)
        actions = torch.tensor(
            [e[1] for e in batch],
            dtype=torch.long,
            device=device,
        )
        recompenses = torch.tensor(
            [e[2] for e in batch],
            dtype=torch.float32,
            device=device,
        )
        listes_legales = [e[3] for e in batch]

        logits, valeurs = modele(etats)
        valeurs = valeurs.squeeze(1)

        # Interdit au réseau de placer de la probabilité
        # sur les coups illégaux pendant l'apprentissage.
        logits_masques = torch.full_like(logits, -1e9)

        for ligne, indices in enumerate(listes_legales):
            logits_masques[ligne, indices] = logits[ligne, indices]

        log_probs = F.log_softmax(logits_masques, dim=1)
        probs = torch.softmax(logits_masques, dim=1)

        log_prob_action = log_probs.gather(
            1,
            actions.unsqueeze(1),
        ).squeeze(1)

        # Advantage = résultat réel - ce que le réseau prévoyait.
        avantage = recompenses - valeurs.detach()

        perte_policy = -(avantage * log_prob_action).mean()
        perte_value = F.mse_loss(valeurs, recompenses)

        entropie = -(probs * log_probs).sum(dim=1).mean()

        perte = (
            perte_policy
            + POIDS_VALUE * perte_value
            - POIDS_ENTROPIE * entropie
        )

        optimiseur.zero_grad()
        perte.backward()
        torch.nn.utils.clip_grad_norm_(modele.parameters(), 1.0)
        optimiseur.step()

        pertes_total.append(float(perte.item()))
        pertes_policy.append(float(perte_policy.item()))
        pertes_value.append(float(perte_value.item()))

    return {
        "total": sum(pertes_total) / len(pertes_total),
        "policy": sum(pertes_policy) / len(pertes_policy),
        "value": sum(pertes_value) / len(pertes_value),
    }


# ============================================================
# ENTRAÎNEMENT
# ============================================================

def entrainer_ia(nombre_parties, device):
    modele, deja_entrainees, existe = charger_modele(device)

    if existe:
        print(f"Modèle V4 chargé : {deja_entrainees} parties précédentes.")
    else:
        print("Nouveau réseau V4 créé avec des poids aléatoires.")

    print(f"Appareil utilisé : {device}")
    print("Architecture : réseau Policy + Value\n")

    optimiseur = torch.optim.Adam(
        modele.parameters(),
        lr=TAUX_APPRENTISSAGE,
    )

    replay = deque(maxlen=TAILLE_REPLAY)

    nb_blancs = 0
    nb_noirs = 0
    nb_nuls = 0
    nb_limite = 0

    for numero in range(1, nombre_parties + 1):
        progression = (numero - 1) / max(1, nombre_parties - 1)

        epsilon = 0.30 * (1.0 - progression) + 0.05 * progression
        temperature = 1.20 * (1.0 - progression) + 0.55 * progression

        experiences, _, type_resultat, demi_coups = jouer_partie_self_play(
            modele,
            device,
            epsilon,
            temperature,
        )

        replay.extend(experiences)

        pertes = apprendre_depuis_replay(
            modele,
            optimiseur,
            replay,
            device,
        )

        if type_resultat == "blancs":
            nb_blancs += 1
        elif type_resultat == "noirs":
            nb_noirs += 1
        elif type_resultat == "nul":
            nb_nuls += 1
        else:
            nb_limite += 1

        total_entraine = deja_entrainees + numero

        if numero == 1 or numero % 10 == 0 or numero == nombre_parties:
            if pertes is None:
                perte_txt = "perte=n/a"
            else:
                perte_txt = (
                    f"total={pertes['total']:.4f} | "
                    f"policy={pertes['policy']:.4f} | "
                    f"value={pertes['value']:.4f}"
                )

            print(
                f"Partie {numero:5d}/{nombre_parties} | "
                f"eps={epsilon:.3f} | temp={temperature:.2f} | "
                f"{perte_txt} | coups={demi_coups:3d} | "
                f"B={nb_blancs} N={nb_noirs} "
                f"nuls={nb_nuls} limite={nb_limite}"
            )

        if numero % 10 == 0:
            sauvegarder_modele(modele, total_entraine)

    total_final = deja_entrainees + nombre_parties
    sauvegarder_modele(modele, total_final)

    print("\nEntraînement terminé.")
    print(f"Modèle sauvegardé : {FICHIER_MODELE}")
    print(f"Total de parties d'entraînement : {total_final}")


# ============================================================
# ÉVALUATION CONTRE UN BOT ALÉATOIRE
# ============================================================

def evaluer_contre_aleatoire(modele, device, nombre_parties):
    victoires = 0
    defaites = 0
    nuls = 0

    for numero in range(nombre_parties):
        board = chess.Board()
        couleur_ia = chess.WHITE if numero % 2 == 0 else chess.BLACK

        for _ in range(MAX_DEMI_COUPS):
            if board.is_game_over(claim_draw=True):
                break

            if board.turn == couleur_ia:
                coup, _, _, _ = choisir_coup_reseau(
                    board,
                    modele,
                    device,
                    epsilon=0.0,
                    temperature=0.0,
                )
            else:
                coups = list(board.legal_moves)
                coup = random.choice(coups) if coups else None

            if coup is None:
                break

            board.push(coup)

        outcome = board.outcome(claim_draw=True)

        if outcome is None or outcome.winner is None:
            nuls += 1
        elif outcome.winner == couleur_ia:
            victoires += 1
        else:
            defaites += 1

    score = (victoires + 0.5 * nuls) / max(1, nombre_parties)

    print("\nÉvaluation contre un bot aléatoire")
    print(f"Parties : {nombre_parties}")
    print(f"Victoires IA : {victoires}")
    print(f"Défaites IA : {defaites}")
    print(f"Nuls : {nuls}")
    print(f"Score : {100 * score:.1f}%")


# ============================================================
# INTERFACE PYGAME
# ============================================================

def lancer_interface(modele, device, modele_entraine, parties_entrainees):
    pygame.init()

    fenetre = pygame.display.set_mode((LARGEUR, HAUTEUR))
    pygame.display.set_caption("Jeu d'échecs V4 - Humain contre IA")
    horloge = pygame.time.Clock()

    police_piece = pygame.font.SysFont("DejaVu Sans", 64)
    police_infos = pygame.font.SysFont("DejaVu Sans", 24)
    police_coordonnees = pygame.font.SysFont("DejaVu Sans", 16)

    plateau = chess.Board()
    case_selectionnee = None
    coups_legaux_selection = []
    derniere_evaluation_ia = None

    def position_vers_case(x, y):
        colonne = x // TAILLE_CASE
        ligne = y // TAILLE_CASE
        return chess.square(colonne, 7 - ligne)

    def case_vers_position(case):
        fichier = chess.square_file(case)
        rang = chess.square_rank(case)
        return fichier * TAILLE_CASE, (7 - rang) * TAILLE_CASE

    def dessiner_plateau():
        for ligne in range(8):
            for colonne in range(8):
                couleur = (
                    CASE_CLAIRE
                    if (ligne + colonne) % 2 == 0
                    else CASE_FONCEE
                )
                pygame.draw.rect(
                    fenetre,
                    couleur,
                    (
                        colonne * TAILLE_CASE,
                        ligne * TAILLE_CASE,
                        TAILLE_CASE,
                        TAILLE_CASE,
                    ),
                )

    def dessiner_coordonnees():
        lettres = "abcdefgh"

        for colonne in range(8):
            texte = police_coordonnees.render(lettres[colonne], True, NOIR)
            fenetre.blit(
                texte,
                (
                    colonne * TAILLE_CASE + TAILLE_CASE - 16,
                    HAUTEUR_PLATEAU - 20,
                ),
            )

        for ligne in range(8):
            texte = police_coordonnees.render(str(8 - ligne), True, NOIR)
            fenetre.blit(texte, (5, ligne * TAILLE_CASE + 4))

    def dessiner_selection():
        if case_selectionnee is None:
            return

        x, y = case_vers_position(case_selectionnee)
        surface = pygame.Surface((TAILLE_CASE, TAILLE_CASE), pygame.SRCALPHA)
        surface.fill((*SELECTION, 130))
        fenetre.blit(surface, (x, y))

    def dessiner_coups_possibles():
        for coup in coups_legaux_selection:
            x, y = case_vers_position(coup.to_square)
            centre = (x + TAILLE_CASE // 2, y + TAILLE_CASE // 2)

            if plateau.is_capture(coup):
                pygame.draw.circle(
                    fenetre,
                    COUP_CAPTURE,
                    centre,
                    TAILLE_CASE // 2 - 7,
                    5,
                )
            else:
                pygame.draw.circle(fenetre, COUP_POSSIBLE, centre, 10)

    def dessiner_pieces():
        for case in chess.SQUARES:
            piece = plateau.piece_at(case)

            if piece is None:
                continue

            symbole = PIECES[piece.symbol()]
            couleur = BLANC if piece.color == chess.WHITE else NOIR
            texte = police_piece.render(symbole, True, couleur)

            x, y = case_vers_position(case)
            rect = texte.get_rect(
                center=(x + TAILLE_CASE // 2, y + TAILLE_CASE // 2)
            )
            fenetre.blit(texte, rect)

    def texte_statut():
        if plateau.is_checkmate():
            if plateau.turn == chess.WHITE:
                return "Échec et mat - Victoire de l'IA"
            return "Échec et mat - Victoire du joueur"

        if plateau.is_stalemate():
            return "Pat - Match nul"

        if plateau.is_insufficient_material():
            return "Match nul - Matériel insuffisant"

        if plateau.is_seventyfive_moves():
            return "Match nul - Règle des 75 coups"

        if plateau.is_fivefold_repetition():
            return "Match nul - Répétition"

        texte = "Tour : joueur" if plateau.turn == chess.WHITE else "Tour : IA"

        if plateau.is_check():
            texte += " - ÉCHEC !"

        return texte

    def dessiner_infos():
        pygame.draw.rect(
            fenetre,
            FOND_INFOS,
            (0, HAUTEUR_PLATEAU, LARGEUR, HAUTEUR_INFOS),
        )

        fenetre.blit(
            police_infos.render(texte_statut(), True, TEXTE),
            (20, HAUTEUR_PLATEAU + 10),
        )

        if modele_entraine:
            mode = (
                f"V4 Policy+Value | {parties_entrainees} parties d'entraînement"
            )
        else:
            mode = "V4 non entraînée : bot aléatoire"

        fenetre.blit(
            police_coordonnees.render(mode, True, TEXTE),
            (20, HAUTEUR_PLATEAU + 50),
        )

        if derniere_evaluation_ia is None:
            ligne3 = "R : recommencer     ESC : quitter"
        else:
            ligne3 = (
                "VALUE avant le dernier coup IA : "
                f"{derniere_evaluation_ia:+.3f}"
            )

        fenetre.blit(
            police_coordonnees.render(ligne3, True, TEXTE),
            (20, HAUTEUR_PLATEAU + 73),
        )

    def dessiner():
        dessiner_plateau()
        dessiner_selection()
        dessiner_coups_possibles()
        dessiner_pieces()
        dessiner_coordonnees()
        dessiner_infos()
        pygame.display.flip()

    def coups_legaux_piece(case):
        return [
            coup
            for coup in plateau.legal_moves
            if coup.from_square == case
        ]

    def jouer_case(case):
        nonlocal case_selectionnee, coups_legaux_selection

        if plateau.turn != chess.WHITE:
            return False

        if case_selectionnee is None:
            piece = plateau.piece_at(case)

            if piece is None or piece.color != chess.WHITE:
                return False

            case_selectionnee = case
            coups_legaux_selection = coups_legaux_piece(case)
            return False

        piece = plateau.piece_at(case)

        if piece is not None and piece.color == chess.WHITE:
            case_selectionnee = case
            coups_legaux_selection = coups_legaux_piece(case)
            return False

        candidats = [
            coup
            for coup in coups_legaux_selection
            if coup.to_square == case
        ]

        coup_choisi = None

        if candidats:
            promotions_dame = [
                coup for coup in candidats if coup.promotion == chess.QUEEN
            ]
            coup_choisi = promotions_dame[0] if promotions_dame else candidats[0]

        if coup_choisi is not None:
            notation = plateau.san(coup_choisi)
            plateau.push(coup_choisi)
            print("Humain :", notation)
            coup_joue = True
        else:
            coup_joue = False

        case_selectionnee = None
        coups_legaux_selection = []
        return coup_joue

    def jouer_coup_ordinateur():
        nonlocal derniere_evaluation_ia

        if plateau.is_game_over(claim_draw=True):
            return

        if plateau.turn != chess.BLACK:
            return

        if not modele_entraine:
            coups = list(plateau.legal_moves)

            if not coups:
                return

            coup = random.choice(coups)
            notation = plateau.san(coup)
            plateau.push(coup)
            derniere_evaluation_ia = None
            print("IA aléatoire :", notation)
            return

        coup, _, valeur, _ = choisir_coup_reseau(
            plateau,
            modele,
            device,
            epsilon=0.0,
            temperature=0.0,
        )

        if coup is None:
            return

        notation = plateau.san(coup)
        derniere_evaluation_ia = valeur
        plateau.push(coup)

        print(
            f"IA V4 : {notation} | "
            f"VALUE avant coup = {valeur:+.3f}"
        )

    actif = True
    moment_coup_ia = None

    while actif:
        horloge.tick(FPS)

        if (
            plateau.turn == chess.BLACK
            and not plateau.is_game_over(claim_draw=True)
        ):
            if moment_coup_ia is None:
                moment_coup_ia = pygame.time.get_ticks() + TEMPS_REFLEXION_IA
            elif pygame.time.get_ticks() >= moment_coup_ia:
                jouer_coup_ordinateur()
                moment_coup_ia = None
        else:
            moment_coup_ia = None

        for evenement in pygame.event.get():
            if evenement.type == pygame.QUIT:
                actif = False

            elif evenement.type == pygame.KEYDOWN:
                if evenement.key == pygame.K_ESCAPE:
                    actif = False

                elif evenement.key == pygame.K_r:
                    plateau = chess.Board()
                    case_selectionnee = None
                    coups_legaux_selection = []
                    derniere_evaluation_ia = None
                    moment_coup_ia = None
                    print("\nNouvelle partie")

            elif evenement.type == pygame.MOUSEBUTTONDOWN:
                if evenement.button == 1:
                    x, y = evenement.pos

                    if (
                        y < HAUTEUR_PLATEAU
                        and plateau.turn == chess.WHITE
                        and not plateau.is_game_over(claim_draw=True)
                    ):
                        jouer_case(position_vers_case(x, y))

        dessiner()

    pygame.quit()


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Jeu d'échecs V4 : "
            "réseau Policy + Value "
            "et apprentissage par self-play."
        )
    )

    # ========================================================
    # ARGUMENT : ENTRAÎNEMENT
    # ========================================================

    parser.add_argument(
        "--train",
        type=int,
        default=0,
        metavar="N",
        help=(
            "entraîne l'IA pendant "
            "N parties de self-play"
        )
    )

    # ========================================================
    # ARGUMENT : ÉVALUATION
    # ========================================================

    parser.add_argument(
        "--eval",
        type=int,
        default=0,
        metavar="N",
        help=(
            "évalue l'IA pendant "
            "N parties contre un bot aléatoire"
        )
    )

    # ========================================================
    # ARGUMENT : JOUER CONTRE L'IA
    # ========================================================

    parser.add_argument(
        "--play",
        action="store_true",
        help=(
            "lance une partie "
            "Humain contre IA"
        )
    )

    args = parser.parse_args()


    # ========================================================
    # VÉRIFICATIONS DES ARGUMENTS
    # ========================================================

    if args.train < 0:

        raise SystemExit(
            "--train doit être positif."
        )


    if args.eval < 0:

        raise SystemExit(
            "--eval doit être positif."
        )


    # On évite de lancer plusieurs modes
    # en même temps.

    nombre_modes = 0

    if args.train > 0:
        nombre_modes += 1

    if args.eval > 0:
        nombre_modes += 1

    if args.play:
        nombre_modes += 1


    if nombre_modes > 1:

        raise SystemExit(
            "\nChoisis un seul mode à la fois :\n\n"
            "  --train N\n"
            "  --eval N\n"
            "  --play\n"
        )


    # ========================================================
    # CHOIX CPU / GPU
    # ========================================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print()

    print(
        "Appareil utilisé :",
        device
    )


    # ========================================================
    # MODE 1 : ENTRAÎNEMENT
    # ========================================================

    if args.train > 0:

        print()
        print(
            "=============================="
        )
        print(
            "      ENTRAÎNEMENT IA"
        )
        print(
            "=============================="
        )
        print()

        print(
            f"Nombre de parties demandées : "
            f"{args.train}"
        )

        print()

        entrainer_ia(
            args.train,
            device
        )

        return


    # ========================================================
    # CHARGEMENT DU MODÈLE
    # ========================================================

    (
        modele,
        parties_entrainees,
        modele_existant,
    ) = charger_modele(
        device
    )


    # ========================================================
    # MODE 2 : ÉVALUATION
    # ========================================================

    if args.eval > 0:

        if not modele_existant:

            raise SystemExit(
                "\nAucun modèle V4 entraîné trouvé.\n\n"
                "Commence par entraîner l'IA :\n\n"
                "python Jeu_d_echecV4.py --train 100\n"
            )


        print()
        print(
            "=============================="
        )
        print(
            "        ÉVALUATION IA"
        )
        print(
            "=============================="
        )
        print()

        print(
            f"Modèle entraîné sur "
            f"{parties_entrainees} parties."
        )

        print(
            f"Nombre de parties de test : "
            f"{args.eval}"
        )

        print()


        evaluer_contre_aleatoire(
            modele,
            device,
            args.eval
        )

        return


    # ========================================================
    # MODE 3 : HUMAIN CONTRE IA
    # ========================================================

    if args.play:

        if not modele_existant:

            raise SystemExit(
                "\nAucun modèle V4 entraîné trouvé.\n\n"
                "Commence par entraîner l'IA :\n\n"
                "python Jeu_d_echecV4.py --train 100\n"
            )


        print()
        print(
            "=============================="
        )
        print(
            "       HUMAIN CONTRE IA"
        )
        print(
            "=============================="
        )
        print()

        print(
            f"Modèle entraîné sur "
            f"{parties_entrainees} parties."
        )

        print(
            "Tu joues les Blancs."
        )

        print(
            "L'IA joue les Noirs."
        )

        print()


        lancer_interface(
            modele,
            device,
            modele_existant,
            parties_entrainees
        )

        return


    # ========================================================
    # AUCUN MODE CHOISI
    # ========================================================

    print()
    print(
        "=============================="
    )
    print(
        "       JEU D'ÉCHECS V4"
    )
    print(
        "=============================="
    )
    print()


    if modele_existant:

        print(
            f"Un modèle V4 existe déjà."
        )

        print(
            f"Il a été entraîné sur "
            f"{parties_entrainees} parties."
        )

    else:

        print(
            "Aucun modèle V4 entraîné "
            "n'a encore été trouvé."
        )


    print()

    print(
        "Choisis une action :"
    )

    print()

    print(
        "1. Entraîner l'IA"
    )

    print(
        "   python Jeu_d_echecV4.py --train 100"
    )

    print()

    print(
        "2. Jouer contre l'IA"
    )

    print(
        "   python Jeu_d_echecV4.py --play"
    )

    print()

    print(
        "3. Tester l'IA contre un bot aléatoire"
    )

    print(
        "   python Jeu_d_echecV4.py --eval 50"
    )

    print()


if __name__ == "__main__":
    main()
