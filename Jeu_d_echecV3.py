

import pygame
import chess
import random


# ============================================================
# PARAMÈTRES
# ============================================================
TEMPS_REFLEXION_IA = 500  # millisecondes
TAILLE_CASE = 90

LARGEUR = 8 * TAILLE_CASE
HAUTEUR_PLATEAU = 8 * TAILLE_CASE

HAUTEUR_INFOS = 80

HAUTEUR = HAUTEUR_PLATEAU + HAUTEUR_INFOS

FPS = 60


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


# ============================================================
# SYMBOLES DES PIÈCES
# ============================================================

PIECES = {

    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",

    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}


# ============================================================
# INITIALISATION PYGAME
# ============================================================

pygame.init()

fenetre = pygame.display.set_mode(
    (
        LARGEUR,
        HAUTEUR
    )
)

pygame.display.set_caption(
    "Jeu d'échecs"
)

horloge = pygame.time.Clock()


# ============================================================
# POLICES
# ============================================================

police_piece = pygame.font.SysFont(
    "DejaVu Sans",
    64
)

police_infos = pygame.font.SysFont(
    "DejaVu Sans",
    24
)

police_coordonnees = pygame.font.SysFont(
    "DejaVu Sans",
    16
)


# ============================================================
# PLATEAU
# ============================================================

plateau = chess.Board()


# ============================================================
# VARIABLES DE SÉLECTION
# ============================================================

case_selectionnee = None

coups_legaux_selection = []


# ============================================================
# CONVERSION ÉCRAN -> CASE ÉCHECS
# ============================================================

def position_vers_case(x, y):
    """
    Convertit une position écran en case python-chess.

    En bas à gauche : a1
    En haut à droite : h8
    """

    colonne = x // TAILLE_CASE

    ligne = y // TAILLE_CASE

    rang = 7 - ligne

    fichier = colonne

    return chess.square(
        fichier,
        rang
    )


# ============================================================
# CONVERSION CASE ÉCHECS -> ÉCRAN
# ============================================================

def case_vers_position(case):
    """
    Convertit une case python-chess
    en coordonnées écran.
    """

    fichier = chess.square_file(
        case
    )

    rang = chess.square_rank(
        case
    )

    colonne = fichier

    ligne = 7 - rang

    x = colonne * TAILLE_CASE
    y = ligne * TAILLE_CASE

    return x, y


# ============================================================
# AFFICHAGE DU PLATEAU
# ============================================================

def dessiner_plateau():
    """
    Dessine les 64 cases du plateau.
    """

    for ligne in range(8):

        for colonne in range(8):

            x = colonne * TAILLE_CASE
            y = ligne * TAILLE_CASE

            if (ligne + colonne) % 2 == 0:

                couleur = CASE_CLAIRE

            else:

                couleur = CASE_FONCEE

            pygame.draw.rect(
                fenetre,
                couleur,
                (
                    x,
                    y,
                    TAILLE_CASE,
                    TAILLE_CASE
                )
            )


# ============================================================
# AFFICHAGE DES COORDONNÉES
# ============================================================

def dessiner_coordonnees():

    lettres = "abcdefgh"

    for colonne in range(8):

        texte = police_coordonnees.render(
            lettres[colonne],
            True,
            NOIR
        )

        x = (
            colonne * TAILLE_CASE
            + TAILLE_CASE
            - 16
        )

        y = (
            HAUTEUR_PLATEAU
            - 20
        )

        fenetre.blit(
            texte,
            (
                x,
                y
            )
        )


    for ligne in range(8):

        nombre = str(
            8 - ligne
        )

        texte = police_coordonnees.render(
            nombre,
            True,
            NOIR
        )

        x = 5

        y = (
            ligne * TAILLE_CASE
            + 4
        )

        fenetre.blit(
            texte,
            (
                x,
                y
            )
        )


# ============================================================
# AFFICHAGE DE LA CASE SÉLECTIONNÉE
# ============================================================

def dessiner_selection():

    if case_selectionnee is None:
        return

    x, y = case_vers_position(
        case_selectionnee
    )

    surface = pygame.Surface(
        (
            TAILLE_CASE,
            TAILLE_CASE
        ),
        pygame.SRCALPHA
    )

    surface.fill(
        (
            SELECTION[0],
            SELECTION[1],
            SELECTION[2],
            130
        )
    )

    fenetre.blit(
        surface,
        (
            x,
            y
        )
    )


# ============================================================
# AFFICHAGE DES COUPS POSSIBLES
# ============================================================

def dessiner_coups_possibles():

    for coup in coups_legaux_selection:

        destination = coup.to_square

        x, y = case_vers_position(
            destination
        )

        centre = (
            x + TAILLE_CASE // 2,
            y + TAILLE_CASE // 2
        )

        # Capture
        if plateau.piece_at(
            destination
        ) is not None:

            pygame.draw.circle(
                fenetre,
                COUP_CAPTURE,
                centre,
                TAILLE_CASE // 2 - 7,
                5
            )

        # Case libre
        else:

            pygame.draw.circle(
                fenetre,
                COUP_POSSIBLE,
                centre,
                10
            )


# ============================================================
# AFFICHAGE DES PIÈCES
# ============================================================

def dessiner_pieces():

    for case in chess.SQUARES:

        piece = plateau.piece_at(
            case
        )

        if piece is None:
            continue

        symbole = PIECES[
            piece.symbol()
        ]

        couleur = (
            BLANC
            if piece.color == chess.WHITE
            else NOIR
        )

        texte = police_piece.render(
            symbole,
            True,
            couleur
        )

        x, y = case_vers_position(
            case
        )

        rect = texte.get_rect(
            center=(
                x + TAILLE_CASE // 2,
                y + TAILLE_CASE // 2
            )
        )

        fenetre.blit(
            texte,
            rect
        )


# ============================================================
# STATUT DE LA PARTIE
# ============================================================

def texte_statut():

    if plateau.is_checkmate():

        if plateau.turn == chess.WHITE:

            return "Échec et mat - Victoire des Noirs"

        else:

            return "Échec et mat - Victoire des Blancs"


    if plateau.is_stalemate():

        return "Pat - Match nul"


    if plateau.is_insufficient_material():

        return "Match nul - Matériel insuffisant"


    if plateau.is_seventyfive_moves():

        return "Match nul - Règle des 75 coups"


    if plateau.is_fivefold_repetition():

        return "Match nul - Répétition"


    if plateau.turn == chess.WHITE:

        texte = "Tour : Blancs"

    else:

        texte = "Tour : Noirs"


    if plateau.is_check():

        texte += " - ÉCHEC !"


    return texte


# ============================================================
# AFFICHAGE DES INFORMATIONS
# ============================================================

def dessiner_infos():

    pygame.draw.rect(
        fenetre,
        FOND_INFOS,
        (
            0,
            HAUTEUR_PLATEAU,
            LARGEUR,
            HAUTEUR_INFOS
        )
    )

    statut = texte_statut()

    texte = police_infos.render(
        statut,
        True,
        TEXTE
    )

    fenetre.blit(
        texte,
        (
            20,
            HAUTEUR_PLATEAU + 12
        )
    )


    texte2 = police_coordonnees.render(
        "R : recommencer     ESC : quitter",
        True,
        TEXTE
    )

    fenetre.blit(
        texte2,
        (
            20,
            HAUTEUR_PLATEAU + 48
        )
    )


# ============================================================
# AFFICHAGE COMPLET
# ============================================================

def dessiner():

    dessiner_plateau()

    dessiner_selection()

    dessiner_coups_possibles()

    dessiner_pieces()

    dessiner_coordonnees()

    dessiner_infos()

    pygame.display.flip()


# ============================================================
# COUPS LÉGAUX D'UNE PIÈCE
# ============================================================

def coups_legaux_piece(case):

    coups = []

    for coup in plateau.legal_moves:

        if coup.from_square == case:

            coups.append(
                coup
            )

    return coups


# ============================================================
# JOUER UNE CASE
# ============================================================

def jouer_case(case):

    global case_selectionnee
    global coups_legaux_selection

    # ========================================================
    # AUCUNE PIÈCE SÉLECTIONNÉE
    # ========================================================

    if case_selectionnee is None:

        piece = plateau.piece_at(
            case
        )

        if piece is None:
            return False

        # On ne permet au joueur humain
        # de jouer que les Blancs

        if plateau.turn != chess.WHITE:
            return False

        if piece.color != chess.WHITE:
            return False

        case_selectionnee = case

        coups_legaux_selection = (
            coups_legaux_piece(
                case
            )
        )

        return False


    # ========================================================
    # UNE PIÈCE EST DÉJÀ SÉLECTIONNÉE
    # ========================================================

    piece = plateau.piece_at(
        case
    )

    # Sélection d'une autre pièce blanche

    if (
        piece is not None
        and piece.color == chess.WHITE
        and plateau.turn == chess.WHITE
    ):

        case_selectionnee = case

        coups_legaux_selection = (
            coups_legaux_piece(
                case
            )
        )

        return False


    # ========================================================
    # RECHERCHE DU COUP CHOISI
    # ========================================================

    coup_choisi = None

    for coup in coups_legaux_selection:

        if coup.to_square == case:

            coup_choisi = coup

            break


    # ========================================================
    # PROMOTION EN DAME
    # ========================================================

    if coup_choisi is None:

        piece_depart = plateau.piece_at(
            case_selectionnee
        )

        if (
            piece_depart is not None
            and piece_depart.piece_type == chess.PAWN
        ):

            coup_promotion = chess.Move(
                case_selectionnee,
                case,
                promotion=chess.QUEEN
            )

            if coup_promotion in plateau.legal_moves:

                coup_choisi = coup_promotion


    coup_joue = False


    # ========================================================
    # JOUER LE COUP HUMAIN
    # ========================================================

    if coup_choisi is not None:

        notation = plateau.san(
            coup_choisi
        )

        plateau.push(
            coup_choisi
        )

        print(
            "Humain :",
            notation
        )

        coup_joue = True


    # ========================================================
    # RESET SÉLECTION
    # ========================================================

    case_selectionnee = None

    coups_legaux_selection = []


    return coup_joue

    # ========================================================
    # UNE PIÈCE EST DÉJÀ SÉLECTIONNÉE
    # ========================================================

    piece = plateau.piece_at(
        case
    )


    # Cliquer sur une autre pièce du même camp
    # change la sélection

    if (
        piece is not None
        and piece.color == plateau.turn
    ):

        case_selectionnee = case

        coups_legaux_selection = (
            coups_legaux_piece(
                case
            )
        )

        return


    # ========================================================
    # RECHERCHE DU COUP CHOISI
    # ========================================================

    coup_choisi = None

    for coup in coups_legaux_selection:

        if coup.to_square == case:

            coup_choisi = coup

            break


    # ========================================================
    # PROMOTION AUTOMATIQUE EN DAME
    # ========================================================

    if coup_choisi is None:

        piece_depart = plateau.piece_at(
            case_selectionnee
        )

        if (
            piece_depart is not None
            and piece_depart.piece_type == chess.PAWN
        ):

            coup_promotion = chess.Move(
                case_selectionnee,
                case,
                promotion=chess.QUEEN
            )

            if coup_promotion in plateau.legal_moves:

                coup_choisi = coup_promotion


    # ========================================================
    # JOUER LE COUP
    # ========================================================

    if coup_choisi is not None:

        print(
            "Coup joué :",
            plateau.san(
                coup_choisi
            )
        )

        plateau.push(
            coup_choisi
        )


    # ========================================================
    # RÉINITIALISATION DE LA SÉLECTION
    # ========================================================

    case_selectionnee = None

    coups_legaux_selection = []

# ============================================================
# JOUEUR ORDINATEUR
# ============================================================

def jouer_coup_ordinateur():
    """
    L'ordinateur choisit aléatoirement
    un coup parmi tous les coups légaux.

    Pour l'instant :
        humain = Blancs
        ordinateur = Noirs
    """

    # Partie terminée
    if plateau.is_game_over():
        return

    # L'ordinateur joue uniquement les Noirs
    if plateau.turn != chess.BLACK:
        return

    # Liste de tous les coups autorisés
    coups_possibles = list(
        plateau.legal_moves
    )

    if len(coups_possibles) == 0:
        return

    # Choix aléatoire
    coup = random.choice(
        coups_possibles
    )

    # On récupère la notation avant de jouer le coup
    notation = plateau.san(
        coup
    )

    # L'ordinateur joue
    plateau.push(
        coup
    )

    print(
        "Ordinateur :",
        notation
    )
# ============================================================
# BOUCLE PRINCIPALE
# ============================================================

def main():

    global plateau
    global case_selectionnee
    global coups_legaux_selection

    actif = True

    # Moment auquel l'ordinateur devra jouer
    moment_coup_ia = None


    while actif:

        horloge.tick(FPS)


        # ====================================================
        # TOUR DE L'ORDINATEUR
        # ====================================================

        if (
            plateau.turn == chess.BLACK
            and not plateau.is_game_over()
        ):

            # On démarre le délai de réflexion
            if moment_coup_ia is None:

                moment_coup_ia = (
                    pygame.time.get_ticks()
                    + TEMPS_REFLEXION_IA
                )

            # Le délai est terminé :
            # l'ordinateur joue
            elif pygame.time.get_ticks() >= moment_coup_ia:

                jouer_coup_ordinateur()

                moment_coup_ia = None


        else:

            moment_coup_ia = None


        # ====================================================
        # ÉVÉNEMENTS
        # ====================================================

        for evenement in pygame.event.get():


            # Fermer la fenêtre
            if evenement.type == pygame.QUIT:

                actif = False


            # =================================================
            # CLAVIER
            # =================================================

            elif evenement.type == pygame.KEYDOWN:

                # ESC = quitter
                if evenement.key == pygame.K_ESCAPE:

                    actif = False


                # R = recommencer
                elif evenement.key == pygame.K_r:

                    plateau = chess.Board()

                    case_selectionnee = None

                    coups_legaux_selection = []

                    moment_coup_ia = None

                    print("\nNouvelle partie")


            # =================================================
            # SOURIS
            # =================================================

            elif evenement.type == pygame.MOUSEBUTTONDOWN:

                if evenement.button == 1:

                    x, y = evenement.pos

                    # Le joueur humain ne peut jouer
                    # que pendant le tour des Blancs
                    if (
                        y < HAUTEUR_PLATEAU
                        and plateau.turn == chess.WHITE
                        and not plateau.is_game_over()
                    ):

                        case = position_vers_case(
                            x,
                            y
                        )

                        jouer_case(
                            case
                        )


        # Mise à jour de l'affichage
        dessiner()


    pygame.quit()


# ============================================================
# LANCEMENT DU PROGRAMME
# ============================================================

if __name__ == "__main__":

    main()