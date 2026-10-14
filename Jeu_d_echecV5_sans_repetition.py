

Commandes :
    python3 Jeu_d_echecV5_sans_repetition.py --train 10
    python3 Jeu_d_echecV5_sans_repetition.py --train 100
    python3 Jeu_d_echecV5_sans_repetition.py --play
    python3 Jeu_d_echecV5_sans_repetition.py --eval 20
    python3 Jeu_d_echecV5_sans_repetition.py --play --sims 128
    python3 Jeu_d_echecV5_sans_repetition.py --train 50 --sims 16
    python3 Jeu_d_echecV5_sans_repetition.py --compare-v4 20
    python3 Jeu_d_echecV5_sans_repetition.py --compare-v4 100 --sims 96 --seed 42
"""

import argparse
import math
import os
import random
import time
from collections import Counter, deque
from pathlib import Path

import chess
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# PARAMÈTRES
# ============================================================

TAILLE_CASE = 90
LARGEUR = 8 * TAILLE_CASE
HAUTEUR_PLATEAU = 8 * TAILLE_CASE
HAUTEUR_INFOS = 115
HAUTEUR = HAUTEUR_PLATEAU + HAUTEUR_INFOS

FPS = 60
TEMPS_REFLEXION_IA = 250

# Même entrée et même architecture que la V4 :
# cela permet de réutiliser automatiquement ses poids.
TAILLE_ENTREE = 773
TAILLE_POLICY = 4096

DOSSIER_SCRIPT = Path(__file__).resolve().parent
FICHIER_MODELE = DOSSIER_SCRIPT / "modele_echecs_V5.pt"
FICHIER_MODELE_V4 = DOSSIER_SCRIPT / "modele_echecs_V4.pt"
FICHIER_REPLAY = DOSSIER_SCRIPT / "replay_echecs_V5.pt"

TAUX_APPRENTISSAGE = 3e-4
TAILLE_BATCH = 96
TAILLE_REPLAY = 6000
BATCHS_PAR_PARTIE = 4
MAX_DEMI_COUPS = 15000

POIDS_VALUE = 1.0
POIDS_ENTROPIE = 0.002

# MCTS
C_PUCT = 1.6
SIMULATIONS_TRAIN = 32
SIMULATIONS_PLAY = 96
SIMULATIONS_EVAL = 64

DIRICHLET_ALPHA = 0.30
DIRICHLET_EPSILON = 0.25

# Petit guide matériel pour rendre les premiers MCTS
# moins dépendants d'un réseau encore médiocre.
POIDS_MATERIEL_MCTS = 0.20

NB_THREADS = max(1, min(4, os.cpu_count() or 1))
torch.set_num_threads(NB_THREADS)


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

VALEUR_PIECE = {
    chess.PAWN: 1.0,
    chess.KNIGHT: 3.0,
    chess.BISHOP: 3.2,
    chess.ROOK: 5.0,
    chess.QUEEN: 9.0,
    chess.KING: 0.0,
}


# ============================================================
# RÉSEAU POLICY + VALUE
# ============================================================

class ReseauEchecsV5(nn.Module):

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
        logits = self.policy(representation)
        valeur = self.value(representation)
        return logits, valeur


# ============================================================
# ENCODAGE
# ============================================================

def encoder_plateau(board):
    """12 pièces x 64 cases + tour + 4 droits de roque = 773."""

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
    Un indice = départ x 64 + arrivée.
    Les sous-promotions partagent le même indice ;
    on privilégie la promotion en dame.
    """

    actions = {}

    for coup in board.legal_moves:
        indice = encoder_coup(coup)

        if indice not in actions or coup.promotion == chess.QUEEN:
            actions[indice] = coup

    return actions


def cle_position_repetition(board):
    """Même identité que python-chess, sans les compteurs de demi-coups."""

    return (
        board.pawns, board.knights, board.bishops,
        board.rooks, board.queens, board.kings,
        board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK],
        board.turn, board.clean_castling_rights(),
        board.ep_square if board.has_legal_en_passant() else None,
    )


def compter_positions_partie(board):
    """Compte les positions réelles et simulées sans modifier le plateau."""

    copie = board.copy(stack=True)
    occurrences = Counter()
    while True:
        occurrences[cle_position_repetition(copie)] += 1
        if not copie.move_stack:
            break
        copie.pop()
    return occurrences


def coups_autorises_v5(board):
    """Interdiction stricte, même si tous les autres coups sont moins bons.

    Priorité 1 : ne jamais créer une troisième occurrence si une alternative
    légale existe. Priorité 2 : éviter aussi de donner à l'adversaire un coup
    permettant de réclamer immédiatement cette nulle, si c'est possible.
    Si la répétition est forcée, garder les coups légaux plutôt que bloquer l'IA.
    """

    actions = coups_legaux_uniques(board)
    if len(actions) <= 1:
        return actions

    occurrences = compter_positions_partie(board)
    if not any(nombre >= 2 for nombre in occurrences.values()):
        return actions

    sans_troisieme = {}
    sans_nulle_immediate = {}
    for indice, coup in actions.items():
        board.push(coup)
        try:
            if occurrences[cle_position_repetition(board)] >= 2:
                continue
            sans_troisieme[indice] = coup

            # L'adversaire peut réclamer la nulle AVANT de jouer le coup
            # qui produirait la troisième occurrence. Ne pas lui offrir
            # cette possibilité lorsqu'une autre branche l'évite.
            repetition_possible = False
            for reponse in list(board.legal_moves):
                board.push(reponse)
                try:
                    repetition_possible = (
                        occurrences[cle_position_repetition(board)] >= 2
                    )
                finally:
                    board.pop()
                if repetition_possible:
                    break

            if not repetition_possible:
                sans_nulle_immediate[indice] = coup
        finally:
            board.pop()

    return sans_nulle_immediate or sans_troisieme or actions


# ============================================================
# RÉSULTAT / MATÉRIEL
# ============================================================

def issue_partie(board, couleurs_v5=()):
    """La V5 ne réclame pas une répétition qu'elle peut encore éviter.

    Une troisième occurrence déjà réalisée reste une nulle. Les autres
    règles de fin de partie et les réclamations de l'adversaire sont conservées.
    """

    outcome = board.outcome(claim_draw=True)
    if (
        outcome is not None
        and outcome.termination == chess.Termination.THREEFOLD_REPETITION
        and board.turn in couleurs_v5
        and not board.is_repetition(3)
    ):
        return None
    return outcome


def resultat_officiel_blancs(board, couleurs_v5=(chess.WHITE, chess.BLACK)):
    outcome = issue_partie(board, couleurs_v5)

    if outcome is None:
        return None
    if outcome.winner == chess.WHITE:
        return 1.0
    if outcome.winner == chess.BLACK:
        return -1.0
    return 0.0


def valeur_terminale_pour_joueur(board, joueur, couleurs_v5=()):
    outcome = issue_partie(board, couleurs_v5)

    if outcome is None or outcome.winner is None:
        return 0.0

    return 1.0 if outcome.winner == joueur else -1.0


def score_materiel_blancs(board):
    score = 0.0

    for piece_type, valeur in VALEUR_PIECE.items():
        score += len(board.pieces(piece_type, chess.WHITE)) * valeur
        score -= len(board.pieces(piece_type, chess.BLACK)) * valeur

    return math.tanh(score / 10.0)


def score_materiel_joueur(board, joueur):
    score = score_materiel_blancs(board)
    return score if joueur == chess.WHITE else -score


# ============================================================
# PRÉDICTION RÉSEAU
# ============================================================

def prediction_reseau(board, modele, device):

    x = encoder_plateau(board).unsqueeze(0).to(device)

    modele.eval()

    with torch.no_grad():
        logits, valeur = modele(x)

    return logits[0], float(valeur[0, 0].item())


# ============================================================
# MCTS
# ============================================================

class NoeudMCTS:

    def __init__(self, prior=0.0, to_play=None, coup=None):
        self.prior = float(prior)
        self.to_play = to_play
        self.coup = coup

        self.visit_count = 0
        self.value_sum = 0.0
        self.children = {}

    def valeur_moyenne(self):
        if self.visit_count == 0:
            return 0.0

        return self.value_sum / self.visit_count

    def est_developpe(self):
        return bool(self.children)


def developper_noeud(noeud, board, modele, device, couleurs_v5=None):
    """
    La POLICY crée les priorités des branches.
    La VALUE évalue la nouvelle position.
    """

    noeud.to_play = board.turn
    if couleurs_v5 is None:
        couleurs_v5 = (board.turn,)

    if issue_partie(board, couleurs_v5) is not None:
        return valeur_terminale_pour_joueur(board, board.turn, couleurs_v5)

    actions = (
        coups_autorises_v5(board)
        if board.turn in couleurs_v5
        else coups_legaux_uniques(board)
    )

    if not actions:
        return 0.0

    logits, valeur_reseau = prediction_reseau(board, modele, device)

    indices = list(actions.keys())
    probs = F.softmax(logits[indices], dim=0).cpu().tolist()

    prochain_joueur = not board.turn

    for indice, probabilite in zip(indices, probs):
        noeud.children[indice] = NoeudMCTS(
            prior=probabilite,
            to_play=prochain_joueur,
            coup=actions[indice],
        )

    # Petite aide pour les premiers entraînements.
    valeur_materielle = score_materiel_joueur(board, board.turn)

    valeur = (
        (1.0 - POIDS_MATERIEL_MCTS) * valeur_reseau
        + POIDS_MATERIEL_MCTS * valeur_materielle
    )

    return float(max(-1.0, min(1.0, valeur)))


def ajouter_bruit_dirichlet(racine):
    enfants = list(racine.children.values())

    if not enfants:
        return

    bruit = [
        random.gammavariate(DIRICHLET_ALPHA, 1.0)
        for _ in enfants
    ]

    total = sum(bruit)

    if total <= 0:
        return

    bruit = [x / total for x in bruit]

    for enfant, noise in zip(enfants, bruit):
        enfant.prior = (
            (1.0 - DIRICHLET_EPSILON) * enfant.prior
            + DIRICHLET_EPSILON * noise
        )


def score_puct(parent, enfant):
    """
    Q = qualité connue de la branche.
    U = envie d'explorer une branche prometteuse peu visitée.
    """

    # enfant.value est vu du point de vue du joueur suivant.
    # Pour le parent, il faut inverser le signe.
    q = -enfant.valeur_moyenne() if enfant.visit_count > 0 else 0.0

    u = (
        C_PUCT
        * enfant.prior
        * math.sqrt(parent.visit_count + 1.0)
        / (enfant.visit_count + 1.0)
    )

    return q + u


def selectionner_enfant(noeud):

    meilleur = -float("inf")
    candidats = []

    for indice, enfant in noeud.children.items():
        score = score_puct(noeud, enfant)

        if score > meilleur + 1e-12:
            meilleur = score
            candidats = [(indice, enfant)]

        elif abs(score - meilleur) <= 1e-12:
            candidats.append((indice, enfant))

    return random.choice(candidats)


def retropropager_mcts(chemin, valeur_feuille, joueur_feuille):

    for noeud in reversed(chemin):

        if noeud.to_play == joueur_feuille:
            valeur = valeur_feuille
        else:
            valeur = -valeur_feuille

        noeud.value_sum += valeur
        noeud.visit_count += 1


def lancer_mcts(
    board,
    modele,
    device,
    simulations,
    bruit_racine=False,
    couleurs_v5=None,
):
    """
    Une simulation :
    sélection -> expansion -> évaluation -> rétropropagation.
    """

    if couleurs_v5 is None:
        couleurs_v5 = (board.turn,)
    couleurs_v5 = tuple(couleurs_v5)

    racine = NoeudMCTS(prior=1.0, to_play=board.turn)

    developper_noeud(racine, board, modele, device, couleurs_v5)

    if bruit_racine:
        ajouter_bruit_dirichlet(racine)

    for _ in range(max(1, int(simulations))):

        # Conserver les coups précédents : une simulation doit détecter
        # les répétitions réelles de la partie, pas seulement les siennes.
        copie = board.copy(stack=True)
        noeud = racine
        chemin = [racine]

        # 1. SÉLECTION
        while (
            noeud.est_developpe()
            and issue_partie(copie, couleurs_v5) is None
        ):
            _, enfant = selectionner_enfant(noeud)
            copie.push(enfant.coup)
            noeud = enfant
            chemin.append(noeud)

        joueur_feuille = copie.turn

        # 2 + 3. EXPANSION / ÉVALUATION
        if issue_partie(copie, couleurs_v5) is not None:
            valeur = valeur_terminale_pour_joueur(
                copie,
                joueur_feuille,
                couleurs_v5,
            )
        else:
            valeur = developper_noeud(
                noeud,
                copie,
                modele,
                device,
                couleurs_v5,
            )

        # 4. RÉTROPROPAGATION
        retropropager_mcts(
            chemin,
            valeur,
            joueur_feuille,
        )

    return racine


def politique_mcts(racine):
    """
    La distribution des visites devient la nouvelle
    cible de POLICY pendant l'apprentissage.
    """

    indices = list(racine.children.keys())

    if not indices:
        return [], []

    visites = [
        float(racine.children[i].visit_count)
        for i in indices
    ]

    total = sum(visites)

    if total <= 0:
        priors = [racine.children[i].prior for i in indices]
        total_prior = sum(priors)

        if total_prior <= 0:
            probs = [1.0 / len(indices)] * len(indices)
        else:
            probs = [p / total_prior for p in priors]

    else:
        probs = [n / total for n in visites]

    return indices, probs


def choisir_action_visites(racine, temperature):
    indices = list(racine.children.keys())

    if not indices:
        return None

    visites = [
        racine.children[i].visit_count
        for i in indices
    ]

    if temperature <= 0:
        maximum = max(visites)

        candidats = [
            i
            for i, n in zip(indices, visites)
            if n == maximum
        ]

        return random.choice(candidats)

    puissance = 1.0 / temperature

    poids = [
        max(1e-8, float(n)) ** puissance
        for n in visites
    ]

    return random.choices(
        indices,
        weights=poids,
        k=1,
    )[0]


# ============================================================
# SAUVEGARDE / CHARGEMENT
# ============================================================

def sauvegarder_modele(modele, optimiseur, parties_entrainees):

    torch.save(
        {
            "version": 5,
            "model_state_dict": modele.state_dict(),
            "optimizer_state_dict": (
                optimiseur.state_dict()
                if optimiseur is not None
                else None
            ),
            "parties_entrainees": int(parties_entrainees),
        },
        FICHIER_MODELE,
    )


def sauvegarder_replay(replay):
    torch.save(list(replay), FICHIER_REPLAY)


def charger_modele(device):
    """
    Priorité :
    1. modèle V5 ;
    2. sinon, import des poids V4 compatibles ;
    3. sinon, nouveau réseau aléatoire.
    """

    modele = ReseauEchecsV5().to(device)

    parties = 0
    checkpoint = None
    existe_v5 = False
    importe_v4 = False

    if FICHIER_MODELE.exists():

        checkpoint = torch.load(
            FICHIER_MODELE,
            map_location=device,
            weights_only=False,
        )

        modele.load_state_dict(checkpoint["model_state_dict"])
        parties = int(checkpoint.get("parties_entrainees", 0))
        existe_v5 = True

    elif FICHIER_MODELE_V4.exists():

        ancien = torch.load(
            FICHIER_MODELE_V4,
            map_location=device,
            weights_only=False,
        )

        try:
            modele.load_state_dict(ancien["model_state_dict"])
            parties = int(ancien.get("parties_entrainees", 0))
            importe_v4 = True

        except Exception:
            parties = 0
            importe_v4 = False

    return modele, parties, checkpoint, existe_v5, importe_v4


def charger_replay():

    replay = deque(maxlen=TAILLE_REPLAY)

    if FICHIER_REPLAY.exists():
        donnees = torch.load(
            FICHIER_REPLAY,
            map_location="cpu",
            weights_only=False,
        )

        for experience in donnees[-TAILLE_REPLAY:]:
            replay.append(experience)

    return replay


def reset_v5():

    for fichier in (FICHIER_MODELE, FICHIER_REPLAY):
        if fichier.exists():
            fichier.unlink()


# ============================================================
# SELF-PLAY MCTS
# ============================================================

def jouer_partie_self_play(modele, device, simulations):

    board = chess.Board()
    trajectoire = []

    for _ in range(MAX_DEMI_COUPS):

        if issue_partie(board, (chess.WHITE, chess.BLACK)) is not None:
            break

        joueur = board.turn

        racine = lancer_mcts(
            board,
            modele,
            device,
            simulations,
            bruit_racine=True,
            couleurs_v5=(chess.WHITE, chess.BLACK),
        )

        indices, probabilites = politique_mcts(racine)

        if not indices:
            break

        # Beaucoup de diversité au début,
        # puis jeu de plus en plus déterministe.
        if board.ply() < 24:
            temperature = 1.0
        elif board.ply() < 60:
            temperature = 0.45
        else:
            temperature = 0.15

        action = choisir_action_visites(
            racine,
            temperature,
        )

        if action is None:
            break

        enfant = racine.children[action]

        trajectoire.append(
            {
                "etat": encoder_plateau(board).half(),
                "indices": [int(i) for i in indices],
                "probabilites": [float(p) for p in probabilites],
                "joueur": bool(joueur),
            }
        )

        board.push(enfant.coup)

    resultat = resultat_officiel_blancs(board)

    if resultat is None:
        # Partie coupée par MAX_DEMI_COUPS :
        # petit signal matériel, limité.
        resultat = 0.35 * score_materiel_blancs(board)
        type_resultat = "limite"

    elif resultat > 0:
        type_resultat = "blancs"

    elif resultat < 0:
        type_resultat = "noirs"

    else:
        type_resultat = "nul"

    experiences = []

    for entree in trajectoire:

        if entree["joueur"] == chess.WHITE:
            valeur = resultat
        else:
            valeur = -resultat

        experiences.append(
            {
                "etat": entree["etat"],
                "indices": entree["indices"],
                "probabilites": entree["probabilites"],
                "value": float(valeur),
            }
        )

    return experiences, type_resultat, board.ply()


# ============================================================
# APPRENTISSAGE
# ============================================================

def apprendre_depuis_replay(
    modele,
    optimiseur,
    replay,
    device,
):

    if len(replay) < 16:
        return None

    modele.train()

    pertes_totales = []
    pertes_policy = []
    pertes_value = []

    for _ in range(BATCHS_PAR_PARTIE):

        taille = min(TAILLE_BATCH, len(replay))
        batch = random.sample(replay, taille)

        etats = torch.stack(
            [e["etat"].float() for e in batch]
        ).to(device)

        cibles_value = torch.tensor(
            [e["value"] for e in batch],
            dtype=torch.float32,
            device=device,
        )

        logits, valeurs = modele(etats)
        valeurs = valeurs.squeeze(1)

        logits_masques = torch.full_like(logits, -1e9)
        cibles_policy = torch.zeros_like(logits)

        for ligne, experience in enumerate(batch):

            indices = torch.tensor(
                experience["indices"],
                dtype=torch.long,
                device=device,
            )

            probabilites = torch.tensor(
                experience["probabilites"],
                dtype=torch.float32,
                device=device,
            )

            logits_masques[ligne, indices] = logits[ligne, indices]
            cibles_policy[ligne, indices] = probabilites

        log_probs = F.log_softmax(logits_masques, dim=1)
        probs = torch.softmax(logits_masques, dim=1)

        perte_policy = -(
            cibles_policy * log_probs
        ).sum(dim=1).mean()

        perte_value = F.mse_loss(
            valeurs,
            cibles_value,
        )

        entropie = -(
            probs * log_probs
        ).sum(dim=1).mean()

        perte = (
            perte_policy
            + POIDS_VALUE * perte_value
            - POIDS_ENTROPIE * entropie
        )

        optimiseur.zero_grad()
        perte.backward()

        torch.nn.utils.clip_grad_norm_(
            modele.parameters(),
            1.0,
        )

        optimiseur.step()

        pertes_totales.append(float(perte.item()))
        pertes_policy.append(float(perte_policy.item()))
        pertes_value.append(float(perte_value.item()))

    return {
        "total": sum(pertes_totales) / len(pertes_totales),
        "policy": sum(pertes_policy) / len(pertes_policy),
        "value": sum(pertes_value) / len(pertes_value),
    }


# ============================================================
# ENTRAÎNEMENT
# ============================================================

def entrainer_ia(nombre_parties, device, simulations, reset=False):

    if reset:
        reset_v5()
        print("Ancien apprentissage V5 supprimé.\n")

    (
        modele,
        deja_entrainees,
        checkpoint,
        existe_v5,
        importe_v4,
    ) = charger_modele(device)

    optimiseur = torch.optim.Adam(
        modele.parameters(),
        lr=TAUX_APPRENTISSAGE,
    )

    if (
        existe_v5
        and checkpoint is not None
        and checkpoint.get("optimizer_state_dict") is not None
    ):
        optimiseur.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    replay = charger_replay()

    if existe_v5:
        print(
            f"Modèle V5 chargé : "
            f"{deja_entrainees} parties précédentes."
        )

    elif importe_v4:
        print(
            f"Poids V4 importés : "
            f"{deja_entrainees} parties précédentes."
        )
        print(
            "La V5 va maintenant affiner ce réseau "
            "avec des cibles produites par le MCTS."
        )

    else:
        print("Nouveau réseau V5 créé avec des poids aléatoires.")

    print(f"Appareil : {device}")
    print(f"Threads CPU : {NB_THREADS}")
    print(f"MCTS : {simulations} simulations par coup")
    print(f"Replay chargé : {len(replay)} positions\n")

    blancs = 0
    noirs = 0
    nuls = 0
    limites = 0

    for numero in range(1, nombre_parties + 1):

        experiences, resultat, demi_coups = jouer_partie_self_play(
            modele,
            device,
            simulations,
        )

        replay.extend(experiences)

        pertes = apprendre_depuis_replay(
            modele,
            optimiseur,
            replay,
            device,
        )

        if resultat == "blancs":
            blancs += 1
        elif resultat == "noirs":
            noirs += 1
        elif resultat == "nul":
            nuls += 1
        else:
            limites += 1

        total = deja_entrainees + numero

        if numero == 1 or numero % 5 == 0 or numero == nombre_parties:

            if pertes is None:
                perte_txt = "perte=n/a"
            else:
                perte_txt = (
                    f"total={pertes['total']:.4f} | "
                    f"policy={pertes['policy']:.4f} | "
                    f"value={pertes['value']:.4f}"
                )

            print(
                f"Partie {numero:4d}/{nombre_parties} | "
                f"{perte_txt} | "
                f"coups={demi_coups:3d} | "
                f"replay={len(replay):4d} | "
                f"B={blancs} N={noirs} "
                f"nuls={nuls} limite={limites}"
            )

        if numero % 5 == 0:
            sauvegarder_modele(modele, optimiseur, total)
            sauvegarder_replay(replay)

    total_final = deja_entrainees + nombre_parties

    sauvegarder_modele(
        modele,
        optimiseur,
        total_final,
    )

    sauvegarder_replay(replay)

    print("\nEntraînement V5 terminé.")
    print(f"Modèle : {FICHIER_MODELE}")
    print(f"Replay : {FICHIER_REPLAY}")
    print(f"Total affiché : {total_final} parties.")


# ============================================================
# COUP MCTS "SÉRIEUX"
# ============================================================

def choisir_coup_mcts_serieux(
    board,
    modele,
    device,
    simulations,
):

    racine = lancer_mcts(
        board,
        modele,
        device,
        simulations,
        bruit_racine=False,
    )

    action = choisir_action_visites(
        racine,
        temperature=0.0,
    )

    if action is None:
        return None, 0, 0.0

    enfant = racine.children[action]

    return (
        enfant.coup,
        enfant.visit_count,
        racine.valeur_moyenne(),
    )


# ============================================================
# ÉVALUATION CONTRE BOT ALÉATOIRE
# ============================================================

def evaluer_contre_aleatoire(
    modele,
    device,
    nombre_parties,
    simulations,
):

    victoires = 0
    defaites = 0
    nuls = 0

    for numero in range(nombre_parties):

        board = chess.Board()

        couleur_ia = (
            chess.WHITE
            if numero % 2 == 0
            else chess.BLACK
        )

        for _ in range(MAX_DEMI_COUPS):

            if issue_partie(board, (couleur_ia,)) is not None:
                break

            if board.turn == couleur_ia:

                coup, _, _ = choisir_coup_mcts_serieux(
                    board,
                    modele,
                    device,
                    simulations,
                )

            else:

                coups = list(board.legal_moves)
                coup = random.choice(coups) if coups else None

            if coup is None:
                break

            board.push(coup)

        outcome = issue_partie(board, (couleur_ia,))

        if outcome is None or outcome.winner is None:
            nuls += 1

        elif outcome.winner == couleur_ia:
            victoires += 1

        else:
            defaites += 1

        print(
            f"Test {numero + 1:3d}/{nombre_parties} | "
            f"V={victoires} D={defaites} N={nuls}"
        )

    score = (
        victoires + 0.5 * nuls
    ) / max(1, nombre_parties)

    print("\nÉvaluation V5 contre bot aléatoire")
    print(f"Victoires : {victoires}")
    print(f"Défaites : {defaites}")
    print(f"Nuls : {nuls}")
    print(f"Score : {100 * score:.1f}%")


# ============================================================
# COMPARAISON V5 CONTRE V4
# ============================================================

def charger_modele_comparaison(fichier, device, version):
    """Charge un checkpoint précis, sans créer de réseau non entraîné."""

    fichier = Path(fichier).expanduser()

    if not fichier.is_file():
        raise FileNotFoundError(
            f"Modèle V{version} introuvable : {fichier}. "
            f"Entraîne et sauvegarde d'abord la V{version}, "
            "ou précise son chemin avec --model-v4 / --model-v5."
        )

    checkpoint = torch.load(
        fichier,
        map_location=device,
        weights_only=True,
    )

    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint V{version} incompatible : {fichier}.")

    version_fichier = checkpoint.get("version")
    if version_fichier is not None and version_fichier != version:
        raise ValueError(
            f"Le fichier {fichier} indique V{version_fichier}, "
            f"alors qu'un modèle V{version} est attendu."
        )

    # V4 et V5 ont exactement les mêmes couches et le même encodage.
    # On réutilise cette classe pour charger les poids V4, sans importer
    # le script V4 ni sa dépendance graphique Pygame.
    modele = ReseauEchecsV5().to(device)
    modele.load_state_dict(checkpoint["model_state_dict"])
    modele.eval()

    return modele, int(checkpoint.get("parties_entrainees", 0))


def choisir_coup_v4_comparaison(board, modele, device):
    """Choix identique à la V4 : Policy, epsilon=0 et température=0."""

    actions = coups_legaux_uniques(board)
    if not actions:
        return None

    indices = list(actions)
    logits, _ = prediction_reseau(board, modele, device)
    position = int(torch.argmax(logits[indices]).item())
    return actions[indices[position]]


def preparer_ouverture_comparaison(rng, demi_coups):
    """Une ouverture légale par paire, partagée par les deux parties."""

    board = chess.Board()

    for _ in range(demi_coups):
        candidats = list(coups_legaux_uniques(board).values())
        rng.shuffle(candidats)

        # L'ouverture commune ne doit pas déjà terminer la partie.
        for coup in candidats:
            board.push(coup)
            if not board.is_game_over(claim_draw=True):
                break
            board.pop()
        else:
            break

    return board


def comparer_v5_v4(
    modele_v5,
    modele_v4,
    device,
    nombre_parties,
    simulations,
    seed=42,
    ouverture_demi_coups=4,
    max_demi_coups=MAX_DEMI_COUPS,
):
    """Compare les joueurs ; seule la V5 applique le filtre anti-répétition."""

    if nombre_parties <= 0 or nombre_parties % 2:
        raise ValueError("La comparaison demande un nombre positif et pair de parties.")
    if simulations <= 0:
        raise ValueError("Le nombre de simulations doit être supérieur à 0.")
    if ouverture_demi_coups < 0 or max_demi_coups <= ouverture_demi_coups:
        raise ValueError("La limite doit dépasser le nombre de demi-coups d'ouverture.")

    modele_v5.eval()
    modele_v4.eval()
    rng_ouvertures = random.Random(seed)
    stats = {"v5": 0, "v4": 0, "nuls": 0, "limites": 0}
    temps = {"v5": 0.0, "v4": 0.0}
    decisions = {"v5": 0, "v4": 0}
    ouverture = None

    for numero in range(nombre_parties):
        if numero % 2 == 0:
            ouverture = preparer_ouverture_comparaison(
                rng_ouvertures,
                ouverture_demi_coups,
            )

        # Conserver l'historique pour les règles de répétition et de nulle.
        board = ouverture.copy(stack=True)
        couleur_v5 = chess.WHITE if numero % 2 == 0 else chess.BLACK
        couleur_txt = "Blancs" if couleur_v5 == chess.WHITE else "Noirs"
        random.seed(seed + numero)

        print(
            f"Partie {numero + 1}/{nombre_parties} | "
            f"V5 = {couleur_txt} | paire {numero // 2 + 1}...",
            flush=True,
        )

        while (
            board.ply() < max_demi_coups
            and issue_partie(board, (couleur_v5,)) is None
        ):
            joueur = "v5" if board.turn == couleur_v5 else "v4"
            debut = time.perf_counter()

            if joueur == "v5":
                coup, _, _ = choisir_coup_mcts_serieux(
                    board, modele_v5, device, simulations,
                )
            else:
                coup = choisir_coup_v4_comparaison(
                    board, modele_v4, device,
                )

            temps[joueur] += time.perf_counter() - debut
            decisions[joueur] += 1

            if coup is None or not board.is_legal(coup):
                raise RuntimeError(f"{joueur.upper()} n'a pas renvoyé de coup légal.")

            board.push(coup)

        outcome = issue_partie(board, (couleur_v5,))

        if outcome is None:
            stats["limites"] += 1
            resultat_txt = "arrêt à la limite (partie non terminée)"
        elif outcome.winner is None:
            stats["nuls"] += 1
            resultat_txt = f"nulle ({outcome.termination.name})"
        elif outcome.winner == couleur_v5:
            stats["v5"] += 1
            resultat_txt = "victoire V5"
        else:
            stats["v4"] += 1
            resultat_txt = "victoire V4"

        print(
            f"  {resultat_txt} | {board.ply()} demi-coups | "
            f"V5={stats['v5']} V4={stats['v4']} "
            f"Nuls={stats['nuls']} Limites={stats['limites']}",
            flush=True,
        )

    terminees = stats["v5"] + stats["v4"] + stats["nuls"]
    points_v5 = stats["v5"] + 0.5 * stats["nuls"]
    points_v4 = stats["v4"] + 0.5 * stats["nuls"]

    print("\n==============================")
    print("      BILAN V5 CONTRE V4")
    print("==============================")
    print(f"Parties prévues : {nombre_parties} ({nombre_parties // 2} par couleur)")
    print(f"Victoires V5 : {stats['v5']}")
    print(f"Victoires V4 : {stats['v4']}")
    print(f"Nulles officielles : {stats['nuls']}")
    print(f"Arrêts à la limite : {stats['limites']} (exclus du score)")

    if terminees:
        print(f"Points V5 : {points_v5:g}/{terminees}")
        print(f"Points V4 : {points_v4:g}/{terminees}")
        print(f"Score V5 sur les parties terminées : {100 * points_v5 / terminees:.1f}%")
    else:
        print("Score indisponible : aucune partie terminée.")

    for joueur in ("v5", "v4"):
        if decisions[joueur]:
            moyenne_ms = 1000 * temps[joueur] / decisions[joueur]
            print(f"Temps moyen par coup {joueur.upper()} : {moyenne_ms:.1f} ms")

    if stats["limites"]:
        print("Pour terminer davantage de parties, augmente --max-plies.")

    return stats


# ============================================================
# INTERFACE PYGAME
# ============================================================

def lancer_interface(
    modele,
    device,
    parties_entrainees,
    simulations,
):

    import pygame

    pygame.init()

    fenetre = pygame.display.set_mode(
        (LARGEUR, HAUTEUR)
    )

    pygame.display.set_caption(
        "Jeu d'échecs V5 - Humain contre IA MCTS"
    )

    horloge = pygame.time.Clock()

    police_piece = pygame.font.SysFont(
        "DejaVu Sans",
        64,
    )

    police_infos = pygame.font.SysFont(
        "DejaVu Sans",
        23,
    )

    police_coordonnees = pygame.font.SysFont(
        "DejaVu Sans",
        16,
    )

    plateau = chess.Board()
    case_selectionnee = None
    coups_legaux_selection = []
    derniere_info_ia = "IA prête."

    def position_vers_case(x, y):
        colonne = x // TAILLE_CASE
        ligne = y // TAILLE_CASE
        return chess.square(colonne, 7 - ligne)

    def case_vers_position(case):
        fichier = chess.square_file(case)
        rang = chess.square_rank(case)

        return (
            fichier * TAILLE_CASE,
            (7 - rang) * TAILLE_CASE,
        )

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
            texte = police_coordonnees.render(
                lettres[colonne],
                True,
                NOIR,
            )

            fenetre.blit(
                texte,
                (
                    colonne * TAILLE_CASE + TAILLE_CASE - 16,
                    HAUTEUR_PLATEAU - 20,
                ),
            )

        for ligne in range(8):
            texte = police_coordonnees.render(
                str(8 - ligne),
                True,
                NOIR,
            )

            fenetre.blit(
                texte,
                (5, ligne * TAILLE_CASE + 4),
            )

    def dessiner_selection():

        if case_selectionnee is None:
            return

        x, y = case_vers_position(case_selectionnee)

        surface = pygame.Surface(
            (TAILLE_CASE, TAILLE_CASE),
            pygame.SRCALPHA,
        )

        surface.fill((*SELECTION, 130))

        fenetre.blit(surface, (x, y))

    def dessiner_coups_possibles():

        for coup in coups_legaux_selection:

            x, y = case_vers_position(coup.to_square)

            centre = (
                x + TAILLE_CASE // 2,
                y + TAILLE_CASE // 2,
            )

            if plateau.is_capture(coup):
                pygame.draw.circle(
                    fenetre,
                    COUP_CAPTURE,
                    centre,
                    TAILLE_CASE // 2 - 7,
                    5,
                )
            else:
                pygame.draw.circle(
                    fenetre,
                    COUP_POSSIBLE,
                    centre,
                    10,
                )

    def dessiner_pieces():

        for case in chess.SQUARES:

            piece = plateau.piece_at(case)

            if piece is None:
                continue

            symbole = PIECES[piece.symbol()]
            couleur = BLANC if piece.color == chess.WHITE else NOIR

            texte = police_piece.render(
                symbole,
                True,
                couleur,
            )

            x, y = case_vers_position(case)

            rect = texte.get_rect(
                center=(
                    x + TAILLE_CASE // 2,
                    y + TAILLE_CASE // 2,
                )
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

        outcome = issue_partie(plateau, (chess.BLACK,))
        if outcome is not None:
            if outcome.termination == chess.Termination.THREEFOLD_REPETITION:
                return "Match nul - Répétition triple"
            if outcome.termination == chess.Termination.FIFTY_MOVES:
                return "Match nul - Règle des 50 coups"

        texte = (
            "Tour : joueur"
            if plateau.turn == chess.WHITE
            else "Tour : IA"
        )

        if plateau.is_check():
            texte += " - ÉCHEC !"

        return texte

    def dessiner_infos():

        pygame.draw.rect(
            fenetre,
            FOND_INFOS,
            (
                0,
                HAUTEUR_PLATEAU,
                LARGEUR,
                HAUTEUR_INFOS,
            ),
        )

        fenetre.blit(
            police_infos.render(
                texte_statut(),
                True,
                TEXTE,
            ),
            (
                20,
                HAUTEUR_PLATEAU + 9,
            ),
        )

        ligne2 = (
            f"V5 Policy+Value+MCTS | "
            f"{parties_entrainees} parties | "
            f"{simulations} simulations/coup"
        )

        fenetre.blit(
            police_coordonnees.render(
                ligne2,
                True,
                TEXTE,
            ),
            (
                20,
                HAUTEUR_PLATEAU + 48,
            ),
        )

        fenetre.blit(
            police_coordonnees.render(
                derniere_info_ia,
                True,
                TEXTE,
            ),
            (
                20,
                HAUTEUR_PLATEAU + 72,
            ),
        )

        fenetre.blit(
            police_coordonnees.render(
                "R : recommencer     ESC : quitter",
                True,
                TEXTE,
            ),
            (
                20,
                HAUTEUR_PLATEAU + 94,
            ),
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

        nonlocal case_selectionnee
        nonlocal coups_legaux_selection

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
                coup
                for coup in candidats
                if coup.promotion == chess.QUEEN
            ]

            coup_choisi = (
                promotions_dame[0]
                if promotions_dame
                else candidats[0]
            )

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

        nonlocal derniere_info_ia

        if issue_partie(plateau, (chess.BLACK,)) is not None:
            return

        if plateau.turn != chess.BLACK:
            return

        derniere_info_ia = "IA : calcul MCTS en cours..."
        dessiner()
        pygame.event.pump()

        coup, visites, valeur = choisir_coup_mcts_serieux(
            plateau,
            modele,
            device,
            simulations,
        )

        if coup is None:
            return

        notation = plateau.san(coup)
        plateau.push(coup)

        derniere_info_ia = (
            f"IA : {notation} | "
            f"branche choisie={visites} visites | "
            f"VALUE={valeur:+.3f}"
        )

        print(
            f"IA V5 : {notation} | "
            f"visites={visites} | "
            f"VALUE={valeur:+.3f}"
        )

    actif = True
    moment_coup_ia = None

    while actif:

        horloge.tick(FPS)

        if (
            plateau.turn == chess.BLACK
            and issue_partie(plateau, (chess.BLACK,)) is None
        ):

            if moment_coup_ia is None:
                moment_coup_ia = (
                    pygame.time.get_ticks()
                    + TEMPS_REFLEXION_IA
                )

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
                    moment_coup_ia = None
                    derniere_info_ia = "Nouvelle partie."
                    print("\nNouvelle partie")

            elif evenement.type == pygame.MOUSEBUTTONDOWN:

                if evenement.button == 1:

                    x, y = evenement.pos

                    if (
                        y < HAUTEUR_PLATEAU
                        and plateau.turn == chess.WHITE
                        and issue_partie(plateau, (chess.BLACK,)) is None
                    ):
                        jouer_case(
                            position_vers_case(x, y)
                        )

        dessiner()

    pygame.quit()


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Jeu d'échecs V5 : "
            "Policy + Value + MCTS + self-play."
        )
    )

    modes = parser.add_mutually_exclusive_group()

    modes.add_argument(
        "--train",
        type=int,
        default=0,
        metavar="N",
        help="entraîne l'IA pendant N parties",
    )

    modes.add_argument(
        "--eval",
        type=int,
        default=0,
        metavar="N",
        help="teste l'IA contre un bot aléatoire",
    )

    modes.add_argument(
        "--play",
        action="store_true",
        help="lance Humain contre IA",
    )

    modes.add_argument(
        "--compare-v4",
        type=int,
        default=None,
        metavar="N",
        help="compare V5 à V4 sur N parties (nombre positif et pair)",
    )

    parser.add_argument(
        "--sims",
        type=int,
        default=None,
        metavar="N",
        help="simulations MCTS par coup",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="repart de zéro pendant --train",
    )

    parser.add_argument(
        "--model-v4",
        type=Path,
        default=FICHIER_MODELE_V4,
        metavar="FICHIER",
        help="checkpoint V4 pour --compare-v4 (défaut : à côté du script)",
    )
    parser.add_argument(
        "--model-v5",
        type=Path,
        default=FICHIER_MODELE,
        metavar="FICHIER",
        help="checkpoint V5 pour --compare-v4 (défaut : à côté du script)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="graine des ouvertures et du MCTS pour --compare-v4",
    )
    parser.add_argument(
        "--opening-plies",
        type=int,
        default=4,
        metavar="N",
        help="demi-coups aléatoires par ouverture commune (comparaison, défaut : 4)",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=MAX_DEMI_COUPS,
        metavar="N",
        help="limite de demi-coups par partie pour --compare-v4",
    )

    args = parser.parse_args()

    if args.train < 0 or args.eval < 0:
        raise SystemExit("Les nombres de parties doivent être positifs.")

    if args.sims is not None and args.sims <= 0:
        raise SystemExit("--sims doit être supérieur à 0.")

    if args.reset and args.train <= 0:
        raise SystemExit("--reset s'utilise avec --train.")

    if args.compare_v4 is not None:
        if args.compare_v4 <= 0 or args.compare_v4 % 2:
            raise SystemExit("--compare-v4 demande un nombre positif et pair de parties.")
        if args.opening_plies < 0:
            raise SystemExit("--opening-plies doit être supérieur ou égal à 0.")
        if args.max_plies <= args.opening_plies:
            raise SystemExit("--max-plies doit dépasser --opening-plies.")

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nAppareil utilisé : {device}")

    # --------------------------------------------------------
    # COMPARAISON : deux checkpoints indépendants, sans reprise
    # automatique des poids V4 à la place d'un modèle V5 manquant.
    # --------------------------------------------------------

    if args.compare_v4 is not None:
        try:
            modele_v5, parties_v5 = charger_modele_comparaison(
                args.model_v5, device, version=5,
            )
            modele_v4, parties_v4 = charger_modele_comparaison(
                args.model_v4, device, version=4,
            )
        except (OSError, ValueError, RuntimeError) as erreur:
            raise SystemExit(str(erreur)) from None

        simulations = args.sims if args.sims is not None else SIMULATIONS_EVAL
        print("\nComparaison V5 (MCTS) contre V4 (Policy directe)")
        print(f"V5 : {args.model_v5} | compteur : {parties_v5} parties")
        print(f"V4 : {args.model_v4} | compteur : {parties_v4} parties")
        print(f"MCTS V5 : {simulations} simulations/coup | graine : {args.seed}")
        print("V5 : troisième répétition interdite si un autre coup légal existe.")
        print("V5 : évite aussi les réclamations immédiates adverses si possible.")
        print(f"Ouverture commune : {args.opening_plies} demi-coups aléatoires")
        print(f"Limite : {args.max_plies} demi-coups par partie\n")

        comparer_v5_v4(
            modele_v5,
            modele_v4,
            device,
            args.compare_v4,
            simulations,
            seed=args.seed,
            ouverture_demi_coups=args.opening_plies,
            max_demi_coups=args.max_plies,
        )
        return

    # --------------------------------------------------------
    # ENTRAÎNEMENT
    # --------------------------------------------------------

    if args.train > 0:

        simulations = (
            args.sims
            if args.sims is not None
            else SIMULATIONS_TRAIN
        )

        print("\n==============================")
        print("      ENTRAÎNEMENT V5")
        print("==============================")
        print(f"Parties : {args.train}")
        print(f"MCTS : {simulations} simulations/coup\n")

        entrainer_ia(
            args.train,
            device,
            simulations,
            reset=args.reset,
        )

        return

    # --------------------------------------------------------
    # CHARGEMENT
    # --------------------------------------------------------

    (
        modele,
        parties,
        _,
        existe_v5,
        importe_v4,
    ) = charger_modele(device)

    modele_disponible = existe_v5 or importe_v4

    # --------------------------------------------------------
    # ÉVALUATION
    # --------------------------------------------------------

    if args.eval > 0:

        if not modele_disponible:
            raise SystemExit(
                "\nAucun modèle V4/V5 disponible.\n"
                "Commence par :\n"
                "python3 Jeu_d_echecV5_sans_repetition.py --train 10\n"
            )

        simulations = (
            args.sims
            if args.sims is not None
            else SIMULATIONS_EVAL
        )

        print("\n==============================")
        print("       ÉVALUATION V5")
        print("==============================")
        print(f"MCTS : {simulations} simulations/coup\n")

        evaluer_contre_aleatoire(
            modele,
            device,
            args.eval,
            simulations,
        )

        return

    # --------------------------------------------------------
    # HUMAIN CONTRE IA
    # --------------------------------------------------------

    if args.play:

        if not modele_disponible:
            raise SystemExit(
                "\nAucun modèle V4/V5 disponible.\n"
                "Commence par :\n"
                "python3 Jeu_d_echecV5_sans_repetition.py --train 10\n"
            )

        simulations = (
            args.sims
            if args.sims is not None
            else SIMULATIONS_PLAY
        )

        print("\n==============================")
        print("       HUMAIN CONTRE IA")
        print("==============================")
        print(f"Ancien entraînement reconnu : {parties} parties")
        print(f"MCTS : {simulations} simulations/coup")
        print("Tu joues les Blancs.\n")

        lancer_interface(
            modele,
            device,
            parties,
            simulations,
        )

        return

    # --------------------------------------------------------
    # MENU
    # --------------------------------------------------------

    print("\n==============================")
    print("       JEU D'ÉCHECS V5")
    print("==============================")

    if existe_v5:
        print(f"Modèle V5 : {parties} parties.")
    elif importe_v4:
        print(
            f"Modèle V4 compatible détecté : "
            f"{parties} parties."
        )
    else:
        print("Aucun modèle entraîné détecté.")

    print("\nEntraîner :")
    print("  python3 Jeu_d_echecV5_sans_repetition.py --train 10")

    print("\nJouer :")
    print("  python3 Jeu_d_echecV5_sans_repetition.py --play")

    print("\nÉvaluer :")
    print("  python3 Jeu_d_echecV5_sans_repetition.py --eval 20")

    print("\nComparer à la V4 :")
    print("  python3 Jeu_d_echecV5_sans_repetition.py --compare-v4 20 --sims 64")

    print("\nTest rapide CPU :")
    print("  python3 Jeu_d_echecV5_sans_repetition.py --train 5 --sims 16")


if __name__ == "__main__":
    main()
